#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train a Kimodo diffusion student via progressive distillation.

Default setup:
  - Teacher: official Kimodo-G1-RP-v1 (same arch as G1-SEED)
  - Data: BONES-SEED G1 CSV + text (LLM2Vec cache preferred)
  - Student init: copy of teacher weights
  - Conditions: text + synthetic constraints + CFG dropout (aligned with official recipe)
  - Stable loss (v2): x0-space PD match + Min-SNR + GT/teacher x0 anchors; formal lr=1e-5

Local 16GB (Stage 100→50):
  python -m kimodo.distill.scripts.train_pd --local --formal --stage 100to50 ...

Single H100 80GB:
  python -m kimodo.distill.scripts.train_pd --server --stage 100to50 ...

4x H100 (DDP):
  torchrun --standalone --nproc_per_node=4 -m kimodo.distill.scripts.train_pd \\
    --server --server-4gpu --stage 100to50 ...
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from kimodo.distill.config_utils import load_distill_config
from kimodo.distill.lr_schedule import compute_lr
from kimodo.distill.pd_loss import progressive_distill_batch_step
from kimodo.distill.schedule import resolve_stage
from kimodo.distill.wandb_log import DistillWandbLogger
from kimodo.distill.weights import (
    copy_weights,
    freeze_module,
    load_denoiser_weights,
    resolve_teacher_weights,
)
from kimodo.model.diffusion import DDIMSampler, Diffusion
from kimodo.train.build import build_denoiser, build_motion_rep
from kimodo.train.checkpoint import save_training_checkpoint
from kimodo.train.collate import collate_motion_batch
from kimodo.train.dataset import G1SeedTrainingDataset
from kimodo.train.distributed import (
    barrier,
    global_batch_size,
    init_distributed,
    is_main_process,
    print_rank0,
    unwrap_module,
    wrap_distributed,
)
from kimodo.train.text_cache import (
    count_missing_embeddings,
    precompute_text_embeddings,
    resolve_text_cache_dir,
)
from kimodo.train.text_embedding import TextEmbeddingProvider


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Progressive distillation for Kimodo-G1 diffusion.")
    parser.add_argument("--config", type=str, default="pd_g1_rp_teacher")
    parser.add_argument(
        "--recipe",
        type=str,
        default=None,
        help="Named training recipe overlay under kimodo/distill/config/ "
        "(e.g. pd_g1_stage1_100to50_local, pd_g1_stage2_50to25_local). "
        "Merged after --local; replaces --formal when set.",
    )
    parser.add_argument("--smoke", action="store_true", help="Tiny model + few steps.")
    parser.add_argument("--local", action="store_true", help="Merge pd_g1_local.yaml.")
    parser.add_argument(
        "--formal",
        action="store_true",
        help="Merge formal recipe: local→pd_g1_formal_local; with --server→pd_g1_formal_server.",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Merge pd_g1_formal_server.yaml (single H100 80GB formal Stage1).",
    )
    parser.add_argument(
        "--server-4gpu",
        action="store_true",
        help="Also merge pd_g1_formal_server_4gpu.yaml; launch with torchrun --nproc_per_node=4.",
    )
    parser.add_argument("--stage", type=str, default=None, help="e.g. 100to50")
    parser.add_argument("--teacher-steps", type=int, default=None)
    parser.add_argument("--student-steps", type=int, default=None)
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        required=False,
        default=None,
        help="Official RP (or SEED) checkpoint dir / model.safetensors. "
        "Required unless --smoke with random teacher.",
    )
    parser.add_argument(
        "--init-student-from",
        type=Path,
        default=None,
        help="Optional student init dir (defaults to teacher when distill.init_student_from_teacher).",
    )
    parser.add_argument("--data-root", type=Path, default=Path("datasets/bones-seed"))
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--stats-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pd_g1"))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--frame-crop",
        type=str,
        default=None,
        choices=["random", "prefix"],
        help="Clip windowing: random (teacher-aligned) or prefix. Default from config.",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Ablation: zero text embeddings (keeps 50-token architecture).",
    )
    parser.add_argument(
        "--text-mode",
        type=str,
        default="encoder",
        choices=["encoder", "dummy"],
        help="encoder=LLM2Vec (or text cache); dummy=zeros. Default: encoder.",
    )
    parser.add_argument(
        "--text-cache-dir",
        type=Path,
        default=None,
        help="Override text_cache.dir (precomputed LLM2Vec embeddings).",
    )
    parser.add_argument(
        "--precompute-text-cache",
        action="store_true",
        help="Precompute missing text embeddings before training.",
    )
    parser.add_argument(
        "--precompute-text-only",
        action="store_true",
        help="Only build text cache then exit (do not load teacher/student). "
        "Use on an empty GPU/CPU before PD training.",
    )
    parser.add_argument(
        "--text-encoder-device",
        type=str,
        default=None,
        help="Device for LLM2Vec during cache precompute. "
        "Default: cuda for --server/--server-4gpu, else cpu. "
        "Do NOT use cuda while teacher/student are already loaded on a tight GPU.",
    )
    parser.add_argument(
        "--constraint-prob",
        type=float,
        default=None,
        help="Override training.constraint_prob (default 0.8 from config).",
    )
    parser.add_argument(
        "--pd-match-space",
        type=str,
        default=None,
        choices=["x0", "xt"],
        help="PD match in inverted-x0 space (stable) or raw x_{t-2} (legacy).",
    )
    parser.add_argument(
        "--snr-gamma",
        type=float,
        default=None,
        help="Min-SNR-γ; <=0 disables. Default from config (5).",
    )
    parser.add_argument(
        "--pd-jump-weight",
        type=float,
        default=None,
        help="Weight on PD jump-match term.",
    )
    parser.add_argument(
        "--diffuse-anchor-weight",
        type=float,
        default=None,
        help="Weight on GT x0 diffusion anchor (keeps pretrained sampling alive).",
    )
    parser.add_argument(
        "--teacher-x0-weight",
        type=float,
        default=None,
        help="Weight on soft MSE(student_x0, teacher_x0@t).",
    )
    parser.add_argument(
        "--constraint-anchor-weight",
        type=float,
        default=None,
        help="Weight on MSE(student_x0, observed) over motion_mask (constraint follow).",
    )
    parser.add_argument(
        "--max-keyframes",
        type=int,
        default=None,
        help="Override training.max_keyframes for synthetic constraints.",
    )
    parser.add_argument(
        "--save-step0",
        action="store_true",
        help="Save step_0 (pure init copy) before training for travel regression.",
    )
    parser.add_argument(
        "--no-save-step0",
        action="store_true",
        help="Skip step_0 export even if config distill.save_step0 is true.",
    )
    parser.add_argument(
        "--teacher-dtype",
        type=str,
        default=None,
        choices=["fp32", "bf16", "fp16"],
        help="Frozen teacher weight dtype. Default: fp32 on --server, bf16 otherwise.",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="kimodo-pd-g1")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--resume-student-from", type=Path, default=None)
    return parser.parse_args()


def _next_batch(data_iter, loader, sampler, epoch: int):
    try:
        return next(data_iter), data_iter, epoch
    except StopIteration:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)
        data_iter = iter(loader)
        return next(data_iter), data_iter, epoch


def _resolve_amp_dtype(name: str) -> torch.dtype:
    key = name.lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported amp dtype: {name}")


def _build_pd_dataloader(
    dataset: G1SeedTrainingDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    distributed: bool,
) -> tuple[DataLoader, Optional[DistributedSampler]]:
    drop_last = len(dataset) >= batch_size
    loader_kwargs: dict = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_motion_batch,
        "drop_last": drop_last,
    }
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=drop_last)
        loader_kwargs["sampler"] = sampler
        loader_kwargs["shuffle"] = False
    else:
        loader_kwargs["shuffle"] = True
    if pin_memory:
        loader_kwargs["pin_memory"] = True
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs), sampler


def _export_student_checkpoint(
    *,
    ckpt_dir: Path,
    bare_student,
    cfg,
    stats_path: Path,
    teacher_steps: int,
    student_steps: int,
    student_init_desc: str,
    export_cfg_type: str,
) -> None:
    training_export = {
        "generative_paradigm": "diffusion",
        "num_base_steps": int(cfg.num_base_steps),
        "cfg_type": export_cfg_type,
        "distill": {
            "teacher_steps": teacher_steps,
            "student_steps": student_steps,
            "recommended_diffusion_steps": student_steps,
            "student_init": student_init_desc,
            "pd_match_space": str(cfg.distill.get("pd_match_space", "x0")),
            "snr_gamma": float(cfg.distill.get("snr_gamma", 5.0)),
            "pd_jump_weight": float(cfg.distill.get("pd_jump_weight", 1.0)),
            "diffuse_anchor_weight": float(cfg.distill.get("diffuse_anchor_weight", 0.25)),
            "teacher_x0_weight": float(cfg.distill.get("teacher_x0_weight", 0.1)),
            "constraint_anchor_weight": float(cfg.distill.get("constraint_anchor_weight", 0.0)),
            "root_xz_weight": float(cfg.distill.get("root_xz_weight", 0.0)),
            "ee_pos_weight": float(cfg.distill.get("ee_pos_weight", 0.0)),
            "contact_weight": float(cfg.distill.get("contact_weight", 0.0)),
            "skate_weight": float(cfg.distill.get("skate_weight", 0.0)),
        },
    }
    save_training_checkpoint(
        output_dir=ckpt_dir,
        denoiser=bare_student,
        denoiser_cfg=OmegaConf.to_container(cfg.denoiser, resolve=True),
        training_cfg=training_export,
        stats_dir=stats_path,
    )
    export_cfg_path = ckpt_dir / "config.yaml"
    export_cfg = OmegaConf.load(export_cfg_path)
    export_cfg.recommended_diffusion_steps = student_steps
    OmegaConf.save(export_cfg, export_cfg_path)


def main() -> int:
    args = parse_args()
    distributed, rank, world_size, local_rank = init_distributed()
    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)

    if args.server_4gpu:
        args.server = True
    if args.teacher_dtype is None:
        args.teacher_dtype = "fp32" if args.server else "bf16"
    if args.text_encoder_device is None:
        args.text_encoder_device = "cuda" if args.server else "cpu"

    cfg = load_distill_config(args.config)
    if args.smoke:
        cfg = OmegaConf.merge(cfg, load_distill_config("pd_g1_smoke"))
        if args.max_files is None:
            args.max_files = 4
        args.no_text = True
        args.text_mode = "dummy"
    if args.local:
        cfg = OmegaConf.merge(cfg, load_distill_config("pd_g1_local"))
    if args.recipe:
        cfg = OmegaConf.merge(cfg, load_distill_config(args.recipe))
    elif args.formal and not args.server:
        # Local formal recipe (16GB). On H100, --server already loads formal_server.
        cfg = OmegaConf.merge(cfg, load_distill_config("pd_g1_formal_local"))
    if args.server:
        cfg = OmegaConf.merge(cfg, load_distill_config("pd_g1_formal_server"))
    if args.server_4gpu:
        cfg = OmegaConf.merge(cfg, load_distill_config("pd_g1_formal_server_4gpu"))

    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.grad_accum_steps is not None:
        cfg.training.gradient_accumulation_steps = args.grad_accum_steps
    if args.max_frames is not None:
        cfg.training.max_frames = args.max_frames
    if args.frame_crop is not None:
        cfg.training.frame_crop = args.frame_crop
    if args.max_steps is not None:
        cfg.training.max_steps = args.max_steps
    if args.num_workers is not None:
        cfg.training.num_workers = args.num_workers
    if args.lr is not None:
        cfg.training.lr = args.lr
    if args.constraint_prob is not None:
        cfg.training.constraint_prob = args.constraint_prob
    if args.pd_match_space is not None:
        cfg.distill.pd_match_space = args.pd_match_space
    if args.snr_gamma is not None:
        cfg.distill.snr_gamma = args.snr_gamma
    if args.pd_jump_weight is not None:
        cfg.distill.pd_jump_weight = args.pd_jump_weight
    if args.diffuse_anchor_weight is not None:
        cfg.distill.diffuse_anchor_weight = args.diffuse_anchor_weight
    if args.teacher_x0_weight is not None:
        cfg.distill.teacher_x0_weight = args.teacher_x0_weight
    if args.constraint_anchor_weight is not None:
        cfg.distill.constraint_anchor_weight = args.constraint_anchor_weight
    if args.max_keyframes is not None:
        cfg.training.max_keyframes = args.max_keyframes

    if args.no_text:
        args.text_mode = "dummy"
        if "cfg_dropout" in cfg.training:
            cfg.training.cfg_dropout.text_only = 0.0

    use_text = (not args.no_text) and args.text_mode == "encoder"
    text_cache_dir = resolve_text_cache_dir(cfg, args.text_cache_dir) if use_text else None
    text_cache_cfg = (
        OmegaConf.to_container(cfg.get("text_cache", {}), resolve=True) if "text_cache" in cfg else {}
    )
    auto_precompute = bool(text_cache_cfg.get("auto_precompute", False)) or args.precompute_text_cache
    use_text_cache = text_cache_dir is not None and use_text

    teacher_steps, student_steps = resolve_stage(
        args.stage,
        teacher_steps=args.teacher_steps if args.teacher_steps is not None else cfg.distill.teacher_steps,
        student_steps=args.student_steps if args.student_steps is not None else cfg.distill.student_steps,
    )
    cfg.distill.teacher_steps = teacher_steps
    cfg.distill.student_steps = student_steps

    if args.server_4gpu and not distributed:
        print_rank0(
            rank,
            "WARNING: --server-4gpu without torchrun runs single-GPU; "
            "use: torchrun --standalone --nproc_per_node=4 -m kimodo.distill.scripts.train_pd ...",
        )

    if not args.data_root.is_dir():
        print_rank0(rank, f"ERROR: data root not found: {args.data_root}")
        return 1

    default_stats_candidates = [
        Path("checkpoints/Kimodo-G1-SEED-v1.hf/stats/motion"),
        Path("checkpoints/Kimodo-G1-SEED-v1/stats/motion"),
    ]
    stats_path = args.stats_path
    if stats_path is None:
        for cand in default_stats_candidates:
            if cand.is_dir():
                stats_path = cand
                break
    if stats_path is None or not Path(stats_path).is_dir():
        print_rank0(rank, "ERROR: provide --stats-path pointing to stats/motion (mean.npy/std.npy).")
        return 1

    train_cfg = cfg.training
    distill_cfg = cfg.distill
    dataloader_cfg = (
        OmegaConf.to_container(cfg.get("dataloader", {}), resolve=True) if "dataloader" in cfg else {}
    )
    grad_accum = max(1, int(train_cfg.get("gradient_accumulation_steps", 1) or 1))
    batch_size = int(train_cfg.batch_size)
    effective_global = global_batch_size(batch_size, world_size, grad_accum)

    pd_match_space = str(distill_cfg.get("pd_match_space", "x0"))
    snr_gamma = float(distill_cfg.get("snr_gamma", 5.0))
    pd_jump_weight = float(distill_cfg.get("pd_jump_weight", 1.0))
    diffuse_anchor_weight = float(distill_cfg.get("diffuse_anchor_weight", 0.25))
    teacher_x0_weight = float(distill_cfg.get("teacher_x0_weight", 0.1))
    constraint_anchor_weight = float(distill_cfg.get("constraint_anchor_weight", 0.0))
    root_xz_weight = float(distill_cfg.get("root_xz_weight", 0.0))
    ee_pos_weight = float(distill_cfg.get("ee_pos_weight", 0.0))
    contact_weight = float(distill_cfg.get("contact_weight", 0.0))
    skate_weight = float(distill_cfg.get("skate_weight", 0.0))
    constraint_mix = None
    if "constraint_mix" in train_cfg:
        constraint_mix = OmegaConf.to_container(train_cfg.constraint_mix, resolve=True)
    save_step0 = bool(distill_cfg.get("save_step0", True))
    if args.save_step0:
        save_step0 = True
    if args.no_save_step0:
        save_step0 = False

    print_rank0(
        rank,
        f"PD stage {teacher_steps}->{student_steps} | device={device} | "
        f"rank={rank}/{world_size} | batch/gpu={batch_size} accum={grad_accum} "
        f"global_batch={effective_global} max_frames={train_cfg.max_frames} "
        f"crop={str(train_cfg.get('frame_crop', 'random'))} | "
        f"text_mode={args.text_mode} text_cache={'on' if use_text_cache else 'off'} | "
        f"constraint_prob={float(train_cfg.get('constraint_prob', 0.0))} "
        f"max_kf={int(train_cfg.get('max_keyframes', 4))} | "
        f"teacher_dtype={args.teacher_dtype} | "
        f"match={pd_match_space} snr_γ={snr_gamma:g} "
        f"w_pd={pd_jump_weight:g} w_diff={diffuse_anchor_weight:g} "
        f"w_tx0={teacher_x0_weight:g} w_cons={constraint_anchor_weight:g} "
        f"w_root={root_xz_weight:g} w_ee={ee_pos_weight:g} "
        f"w_fc={contact_weight:g} w_skate={skate_weight:g}",
    )

    # Dataset + text cache BEFORE loading denoisers (LLM2Vec must not share VRAM with PD).
    cpu_motion_rep = build_motion_rep(cfg.denoiser, stats_path=str(stats_path))
    denoiser_cfg = OmegaConf.to_container(cfg.denoiser, resolve=True)
    llm_dim = int(denoiser_cfg["llm_shape"][-1])
    if denoiser_cfg.get("num_text_tokens_override") is not None:
        num_tokens = int(denoiser_cfg["num_text_tokens_override"])
    else:
        num_tokens = int(denoiser_cfg["llm_shape"][0])
    encoder_cfg = OmegaConf.to_container(cfg.text_encoder, resolve=True) if "text_encoder" in cfg else None

    dataset = G1SeedTrainingDataset(
        args.data_root,
        metadata_path=args.metadata,
        split_path=args.split_path,
        max_files=args.max_files,
        max_frames=int(train_cfg.max_frames),
        frame_crop=str(train_cfg.get("frame_crop", "random")),
        fps=int(train_cfg.fps),
        source_fps=float(train_cfg.source_fps),
        motion_rep=cpu_motion_rep,
        normalize=True,
        require_text=use_text,
        text_cache_dir=text_cache_dir if use_text_cache else None,
        text_cache_num_tokens=num_tokens,
        text_cache_llm_dim=llm_dim,
    )
    if len(dataset) == 0:
        print_rank0(rank, "ERROR: empty dataset")
        return 1

    if use_text_cache:
        assert text_cache_dir is not None
        missing = count_missing_embeddings(dataset.samples, text_cache_dir)
        if missing > 0:
            if not auto_precompute and not getattr(args, "precompute_text_only", False):
                print_rank0(
                    rank,
                    f"ERROR: {missing} text embeddings missing under {text_cache_dir}.\n"
                    "Build cache first, or use --server (auto_precompute) / --precompute-text-only.",
                )
                return 1
            # Only rank 0 builds the cache; others wait.
            if is_main_process(rank):
                enc_device = str(args.text_encoder_device)
                os.environ["TEXT_ENCODER_DEVICE"] = enc_device
                enc_cfg = dict(encoder_cfg or {})
                enc_cfg["device"] = enc_device
                print_rank0(
                    rank,
                    f"Precomputing {missing} missing text embeddings on {enc_device} "
                    f"into {text_cache_dir} (before loading teacher/student)...",
                )
                if enc_device == "cpu" and missing > 1000:
                    print_rank0(
                        rank,
                        f"WARNING: {missing} clips on CPU is slow. "
                        "On H100 use --server (default cuda) or --text-encoder-device cuda.",
                    )
                precompute_text_embeddings(
                    dataset.samples,
                    cache_dir=text_cache_dir,
                    encoder_cfg=enc_cfg,
                    num_tokens=num_tokens,
                    llm_dim=llm_dim,
                    device=torch.device(enc_device),
                )
                # Free Llama weights before loading teacher/student on the same GPUs.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            barrier()
            # Re-count after barrier so non-rank0 see completed files.
            missing_after = count_missing_embeddings(dataset.samples, text_cache_dir)
            if missing_after > 0:
                print_rank0(
                    rank,
                    f"ERROR: still missing {missing_after} text embeddings after precompute.",
                )
                return 1
        print_rank0(rank, f"Using precomputed text embeddings from {text_cache_dir}")

    if getattr(args, "precompute_text_only", False):
        print_rank0(rank, "Precompute-only done; exiting before PD training.")
        if distributed and dist.is_initialized():
            dist.destroy_process_group()
        return 0

    # Load teacher & student only after text cache is ready.
    teacher = build_denoiser(cfg, stats_path=str(stats_path), device=device)
    student = build_denoiser(cfg, stats_path=str(stats_path), device=device)

    student_init_desc = "random (smoke)"
    if args.smoke and args.teacher_checkpoint is None:
        print_rank0(rank, "SMOKE: random teacher/student (no official weights).")
        copy_weights(teacher, student)
        student_init_desc = "copy of randomly-init teacher"
    else:
        if args.teacher_checkpoint is None:
            print_rank0(
                rank,
                "ERROR: --teacher-checkpoint is required (unless --smoke / --precompute-text-only).",
            )
            return 1
        weights = resolve_teacher_weights(args.teacher_checkpoint)
        print_rank0(rank, f"Loading teacher from {weights}")
        load_denoiser_weights(teacher, weights)
        if args.resume_student_from is not None:
            print_rank0(rank, f"Resume student from {args.resume_student_from}")
            load_denoiser_weights(student, resolve_teacher_weights(args.resume_student_from))
            student_init_desc = f"resume weights: {args.resume_student_from}"
        else:
            init_path = args.init_student_from
            if init_path is None and distill_cfg.get("init_student_from_teacher", True):
                init_path = args.teacher_checkpoint
            if init_path is not None:
                print_rank0(rank, f"Init student from {init_path}")
                load_denoiser_weights(student, resolve_teacher_weights(init_path))
                student_init_desc = f"copy of teacher/init: {init_path}"
            else:
                copy_weights(teacher, student)
                student_init_desc = "copy of loaded teacher state_dict"
    print_rank0(rank, f"Student initialization: {student_init_desc}")

    freeze_module(teacher)
    teacher_dtype_name = str(args.teacher_dtype).lower()
    if device.type == "cuda" and teacher_dtype_name in {"bf16", "bfloat16"}:
        teacher.to(dtype=torch.bfloat16)
        print_rank0(rank, "Teacher cast to bfloat16 (frozen).")
    elif device.type == "cuda" and teacher_dtype_name in {"fp16", "float16", "half"}:
        teacher.to(dtype=torch.float16)
        print_rank0(rank, "Teacher cast to float16 (frozen).")
    student.train()

    if distributed:
        student = wrap_distributed(student, device=device, local_rank=local_rank)

    if device.type == "cuda" and is_main_process(rank):
        torch.cuda.empty_cache()
        free_b, total_b = torch.cuda.mem_get_info()
        print_rank0(
            rank,
            f"CUDA mem after load: free={free_b/1024**3:.2f}GiB / total={total_b/1024**3:.2f}GiB",
        )
        if free_b < 3.5 * (1024**3) and not args.server:
            print_rank0(
                rank,
                "WARNING: <3.5GiB free after loading models. Kill other GPU processes "
                "(nvidia-smi) or use --max-frames 32 --constraint-prob 0.",
            )

    diffusion = Diffusion(num_base_steps=int(cfg.num_base_steps)).to(device)
    sampler = DDIMSampler(diffusion)
    bare_student = unwrap_module(student)

    num_workers = int(train_cfg.num_workers)
    pin_memory = bool(dataloader_cfg.get("pin_memory", device.type == "cuda"))
    persistent_workers = bool(dataloader_cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = int(dataloader_cfg.get("prefetch_factor", 2))
    loader, data_sampler = _build_pd_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        distributed=distributed,
    )
    if data_sampler is not None:
        data_sampler.set_epoch(0)

    text_provider_mode = "dummy" if use_text_cache or args.no_text else args.text_mode
    text_provider = TextEmbeddingProvider(
        num_tokens=num_tokens,
        llm_dim=llm_dim,
        device=device,
        mode=text_provider_mode,
        encoder_cfg=None if text_provider_mode != "encoder" else encoder_cfg,
    )

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(train_cfg.lr),
        weight_decay=float(train_cfg.weight_decay),
    )

    use_amp = bool(distill_cfg.get("amp", False)) and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(str(distill_cfg.get("amp_dtype", "bfloat16")))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    if is_main_process(rank):
        args.output_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    wandb_logger = DistillWandbLogger(enabled=args.wandb and is_main_process(rank))
    run_cfg = OmegaConf.to_container(cfg, resolve=True)
    run_cfg["student_init"] = student_init_desc
    run_cfg["world_size"] = world_size
    run_cfg["global_batch"] = effective_global
    run_cfg["cli"] = {
        "text_mode": args.text_mode,
        "no_text": args.no_text,
        "max_frames": int(train_cfg.max_frames),
        "frame_crop": str(train_cfg.get("frame_crop", "random")),
        "teacher_checkpoint": str(args.teacher_checkpoint) if args.teacher_checkpoint else None,
        "teacher_dtype": args.teacher_dtype,
        "server": args.server,
        "server_4gpu": args.server_4gpu,
        "pd_match_space": pd_match_space,
        "snr_gamma": snr_gamma,
        "pd_jump_weight": pd_jump_weight,
        "diffuse_anchor_weight": diffuse_anchor_weight,
        "teacher_x0_weight": teacher_x0_weight,
        "constraint_anchor_weight": constraint_anchor_weight,
        "root_xz_weight": root_xz_weight,
        "ee_pos_weight": ee_pos_weight,
        "contact_weight": contact_weight,
        "skate_weight": skate_weight,
        "constraint_mix": constraint_mix,
    }
    wandb_logger.init(
        project=args.wandb_project,
        run_name=args.wandb_run_name or f"pd_{teacher_steps}to{student_steps}",
        config=run_cfg,
        output_dir=str(args.output_dir),
    )

    max_steps = int(train_cfg.max_steps)
    log_every = int(train_cfg.log_every)
    save_every = int(train_cfg.save_every)
    cfg_dropout = OmegaConf.to_container(train_cfg.get("cfg_dropout", {}), resolve=True)
    constraint_prob = float(train_cfg.get("constraint_prob", 0.0))
    max_keyframes = int(train_cfg.get("max_keyframes", 4))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    base_lr = float(train_cfg.lr)
    lr_schedule = str(train_cfg.get("lr_schedule", "cosine"))
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.1))
    export_cfg_type = str(cfg.get("cfg_type", "separated"))

    if save_step0 and is_main_process(rank) and args.resume_student_from is None:
        step0_dir = args.output_dir / "step_0"
        if not (step0_dir / "model.safetensors").is_file():
            _export_student_checkpoint(
                ckpt_dir=step0_dir,
                bare_student=bare_student,
                cfg=cfg,
                stats_path=stats_path,
                teacher_steps=teacher_steps,
                student_steps=student_steps,
                student_init_desc=student_init_desc,
                export_cfg_type=export_cfg_type,
            )
            print_rank0(rank, f"Saved init regression checkpoint {step0_dir}")
    barrier()

    print_rank0(
        rank,
        f"LR plan: warmup={warmup_steps} → peak={base_lr:g} → schedule={lr_schedule} "
        f"(min_ratio={min_lr_ratio}) over max_steps={max_steps}",
    )

    data_iter = iter(loader)
    data_epoch = 0
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    running_loss = 0.0

    pbar = None
    if is_main_process(rank):
        pbar = tqdm(range(1, max_steps + 1), desc="pd", leave=True)
        step_iter = pbar
    else:
        step_iter = range(1, max_steps + 1)

    for step in step_iter:
        lr_now = compute_lr(
            step,
            base_lr=base_lr,
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            schedule=lr_schedule,
            min_lr_ratio=min_lr_ratio,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        loss_accum = 0.0
        metrics_last: dict = {}
        for _ in range(grad_accum):
            batch, data_iter, data_epoch = _next_batch(data_iter, loader, data_sampler, data_epoch)
            with torch.amp.autocast(
                device.type,
                enabled=use_amp,
                dtype=amp_dtype if use_amp else torch.float32,
            ):
                loss, metrics = progressive_distill_batch_step(
                    teacher,
                    student,
                    bare_student.motion_rep,
                    diffusion,
                    sampler,
                    batch,
                    teacher_steps,
                    text_provider,
                    cfg_dropout=cfg_dropout,
                    constraint_prob=constraint_prob,
                    max_keyframes=max_keyframes,
                    constraint_mix=constraint_mix,
                    pd_match_space=pd_match_space,
                    snr_gamma=snr_gamma,
                    pd_jump_weight=pd_jump_weight,
                    diffuse_anchor_weight=diffuse_anchor_weight,
                    teacher_x0_weight=teacher_x0_weight,
                    constraint_anchor_weight=constraint_anchor_weight,
                    root_xz_weight=root_xz_weight,
                    ee_pos_weight=ee_pos_weight,
                    contact_weight=contact_weight,
                    skate_weight=skate_weight,
                )
                loss = loss / grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_accum += float(loss.detach().item()) * grad_accum
            metrics_last = metrics

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.9 * running_loss + 0.1 * loss_accum if step > 1 else loss_accum
        if pbar is not None:
            pbar.set_postfix(
                loss=f"{loss_accum:.2e}",
                avg=f"{running_loss:.2e}",
                grad=f"{float(grad_norm):.2e}",
                lr=f"{lr_now:.2e}",
            )

        if step % log_every == 0 or step == 1:
            elapsed = time.time() - t0
            log_payload = {
                "loss": loss_accum,
                "loss_avg": running_loss,
                "loss_pd": float(metrics_last.get("loss_pd", loss_accum)),
                "loss_diffuse": float(metrics_last.get("loss_diffuse", 0.0)),
                "loss_teacher_x0": float(metrics_last.get("loss_teacher_x0", 0.0)),
                "loss_constraint": float(metrics_last.get("loss_constraint", 0.0)),
                "loss_root": float(metrics_last.get("loss_root", 0.0)),
                "loss_ee": float(metrics_last.get("loss_ee", 0.0)),
                "loss_contact": float(metrics_last.get("loss_contact", 0.0)),
                "loss_skate": float(metrics_last.get("loss_skate", 0.0)),
                "constraint_frac": float(metrics_last.get("constraint_frac", 0.0)),
                "grad_norm": float(grad_norm),
                "lr": lr_now,
                "elapsed_sec": elapsed,
                "t_mean": float(metrics_last["t_mean"]),
                "x_pred_norm": float(metrics_last["x_pred_norm"]),
                "x_tgt_norm": float(metrics_last["x_tgt_norm"]),
                "pred_x0_norm": float(metrics_last.get("pred_x0_norm", 0.0)),
                "gt_x0_norm": float(metrics_last.get("gt_x0_norm", 0.0)),
                "snr_w_mean": float(metrics_last.get("snr_w_mean", 1.0)),
                "teacher_steps": teacher_steps,
                "student_steps": student_steps,
                "max_frames": int(train_cfg.max_frames),
                "constraint_prob": constraint_prob,
                "effective_batch": effective_global,
                "world_size": world_size,
            }
            print_rank0(
                rank,
                f"[step {step}/{max_steps}] loss={loss_accum:.3e} avg={running_loss:.3e} "
                f"pd={float(metrics_last.get('loss_pd', 0)):.3e} "
                f"diff={float(metrics_last.get('loss_diffuse', 0)):.3e} "
                f"cons={float(metrics_last.get('loss_constraint', 0)):.3e} "
                f"root={float(metrics_last.get('loss_root', 0)):.3e} "
                f"ee={float(metrics_last.get('loss_ee', 0)):.3e} "
                f"grad={float(grad_norm):.3e} lr={lr_now:.3e} "
                f"t_mean={float(metrics_last['t_mean']):.1f} "
                f"x0n={float(metrics_last.get('pred_x0_norm', 0)):.3e} "
                f"elapsed={_format_duration(elapsed)}",
            )
            wandb_logger.log_step(step, log_payload)

        if step % save_every == 0 or step == max_steps:
            barrier()
            if is_main_process(rank):
                ckpt_dir = args.output_dir / f"step_{step}"
                _export_student_checkpoint(
                    ckpt_dir=ckpt_dir,
                    bare_student=bare_student,
                    cfg=cfg,
                    stats_path=stats_path,
                    teacher_steps=teacher_steps,
                    student_steps=student_steps,
                    student_init_desc=student_init_desc,
                    export_cfg_type=export_cfg_type,
                )
                print_rank0(rank, f"Saved {ckpt_dir}")
                wandb_logger.log_checkpoint(step, str(ckpt_dir))
            barrier()

    wandb_logger.finish()
    print_rank0(rank, "Progressive distillation finished.")
    if distributed and dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
