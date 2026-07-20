#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train Kimodo G1 Flow Matching model on BONES-SEED."""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from kimodo.model.flow_matching import FlowMatchingLoss
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
from kimodo.train.flow_train import flow_matching_batch_step
from kimodo.train.stats_compute import compute_motion_stats, save_motion_stats
from kimodo.train.text_cache import (
    count_missing_embeddings,
    precompute_text_embeddings,
    resolve_text_cache_dir,
)
from kimodo.train.text_embedding import TextEmbeddingProvider
from kimodo.train.training_progress import (
    EpochSchedule,
    StepEpochProgress,
    format_epoch_progress,
    resolve_save_every_steps,
)
from kimodo.train.resource_guard import (
    DataLoaderTuning,
    ResourceGuard,
    cap_dataloader_at_startup,
)
from kimodo.train.utils import apply_no_text_overrides, load_train_config
from kimodo.train.wandb_log import WandbLogger


def _format_duration(seconds: float) -> str:
    """Format seconds as HhMMmSSs or MmSSs."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Kimodo G1 with Flow Matching.")
    parser.add_argument("--config", type=str, default="fm_g1_seed", help="Config name under kimodo/train/config/")
    parser.add_argument("--smoke", action="store_true", help="Use smoke overrides (small model, few steps).")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Merge fm_g1_seed_local.yaml (batch_size=2, num_workers=0; full model size).",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Merge fm_g1_seed_server.yaml (H100 80GB: batch_size=48, num_workers=12).",
    )
    parser.add_argument(
        "--server-4gpu",
        action="store_true",
        help="Merge fm_g1_seed_server_4gpu.yaml for 4x H100 DDP (launch with torchrun).",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size (per GPU).")
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="Override training.gradient_accumulation_steps.",
    )
    parser.add_argument("--num-workers", type=int, default=None, help="Override training.num_workers.")
    parser.add_argument("--max-frames", type=int, default=None, help="Override training.max_frames.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override training.max_steps.")
    parser.add_argument("--data-root", type=Path, default=Path("datasets/bones-seed"))
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--stats-path", type=Path, default=None, help="Existing stats/motion directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fm_g1_seed"))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--text-mode",
        type=str,
        default="dummy",
        choices=["encoder", "dummy"],
        help="Text encoding mode; dummy uses zero embeddings for fast smoke tests.",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Train without text input: zero text tokens, no LLM2Vec, constraint-only CFG.",
    )
    parser.add_argument("--compute-stats", action="store_true", help="Recompute stats from training data.")
    parser.add_argument("--stats-batches", type=int, default=50, help="Max batches for stats computation.")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default="kimodo-fm-g1")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume model weights from a prior training checkpoint dir (e.g. outputs/.../step_10000).",
    )
    parser.add_argument(
        "--text-cache-dir",
        type=Path,
        default=None,
        help="Directory of precomputed LLM2Vec embeddings (overrides config text_cache.dir).",
    )
    parser.add_argument(
        "--precompute-text-cache",
        action="store_true",
        help="Precompute missing text embeddings before training (also when text_cache.auto_precompute is true).",
    )
    return parser.parse_args()


def _parse_resume_step(resume_dir: Path) -> int:
    name = resume_dir.name
    if not name.startswith("step_"):
        raise ValueError(f"--resume-from must be a step_N directory, got: {resume_dir}")
    return int(name.split("_", 1)[1])


def _next_batch(data_iter, loader, sampler, epoch: int):
    """Fetch next batch, restarting the loader and advancing sampler epoch when exhausted."""
    try:
        return next(data_iter), data_iter, epoch
    except StopIteration:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)
        data_iter = iter(loader)
        return next(data_iter), data_iter, epoch


def _build_motion_dataloader(
    dataset: G1SeedTrainingDataset,
    *,
    per_gpu_batch: int,
    tuning: DataLoaderTuning,
    pin_memory: bool,
    distributed: bool,
) -> tuple[DataLoader, Optional[DistributedSampler], bool]:
    sampler = None
    loader_kwargs: dict = {
        "batch_size": per_gpu_batch,
        "num_workers": tuning.num_workers,
        "collate_fn": collate_motion_batch,
        "drop_last": len(dataset) >= per_gpu_batch,
    }
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=loader_kwargs["drop_last"])
        loader_kwargs["sampler"] = sampler
        loader_kwargs["shuffle"] = False
    else:
        loader_kwargs["shuffle"] = True
    if pin_memory:
        loader_kwargs["pin_memory"] = True
    if tuning.num_workers > 0:
        loader_kwargs["persistent_workers"] = tuning.persistent_workers
        loader_kwargs["prefetch_factor"] = tuning.prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)
    return loader, sampler, bool(loader_kwargs["drop_last"])


def main() -> int:
    args = parse_args()
    distributed, rank, world_size, local_rank = init_distributed()
    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)

    cfg = load_train_config(args.config)
    if args.smoke:
        smoke_cfg = load_train_config("fm_g1_seed_smoke")
        cfg = OmegaConf.merge(cfg, smoke_cfg)
        if args.max_files is None:
            args.max_files = 4
    if args.local:
        local_cfg = load_train_config("fm_g1_seed_local")
        cfg = OmegaConf.merge(cfg, local_cfg)
    if args.server or args.server_4gpu:
        cfg = OmegaConf.merge(cfg, load_train_config("fm_g1_seed_server"))
    if args.server_4gpu:
        cfg = OmegaConf.merge(cfg, load_train_config("fm_g1_seed_server_4gpu"))
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.grad_accum_steps is not None:
        cfg.training.gradient_accumulation_steps = args.grad_accum_steps
    if args.num_workers is not None:
        cfg.training.num_workers = args.num_workers
    if args.max_frames is not None:
        cfg.training.max_frames = args.max_frames
    if args.max_steps is not None:
        cfg.training.max_steps = args.max_steps
    if args.no_text:
        if args.text_mode == "encoder":
            print_rank0(rank, "WARNING: --no-text forces --text-mode dummy (LLM2Vec is not used).")
        args.text_mode = "dummy"
        cfg = apply_no_text_overrides(cfg)
    use_text_training = args.text_mode == "encoder" and not args.no_text
    if (args.server or args.server_4gpu) and use_text_training:
        cfg = OmegaConf.merge(cfg, load_train_config("fm_g1_seed_server_text"))
    if args.server_4gpu and use_text_training:
        cfg = OmegaConf.merge(cfg, load_train_config("fm_g1_seed_server_4gpu_text"))
    train_cfg = cfg.training
    grad_accum_steps = max(1, int(train_cfg.get("gradient_accumulation_steps", 1) or 1))
    per_gpu_batch = int(train_cfg.batch_size)
    effective_global_batch = global_batch_size(per_gpu_batch, world_size, grad_accum_steps)
    text_cache_dir = resolve_text_cache_dir(cfg, args.text_cache_dir)
    use_text_cache = text_cache_dir is not None and not args.no_text and args.text_mode == "encoder"
    text_cache_cfg = OmegaConf.to_container(cfg.get("text_cache", {}), resolve=True) if "text_cache" in cfg else {}
    auto_precompute = bool(text_cache_cfg.get("auto_precompute", False)) or args.precompute_text_cache

    if args.server_4gpu and not distributed:
        print_rank0(
            rank,
            "WARNING: --server-4gpu without torchrun runs single-GPU; "
            "use: torchrun --standalone --nproc_per_node=4 -m kimodo.train.scripts.train_fm ...",
        )

    if not args.data_root.is_dir():
        print_rank0(rank, f"ERROR: data root not found: {args.data_root}")
        return 1

    default_stats = Path("checkpoints/Kimodo-G1-SEED-v1/stats/motion")
    stats_path = args.stats_path
    if stats_path is None and default_stats.is_dir():
        stats_path = default_stats

    stats_workdir = args.output_dir / "stats_work"
    if is_main_process(rank) and (args.compute_stats or stats_path is None):
        stats_motion_rep = build_denoiser(cfg, stats_path=None, device=torch.device("cpu")).motion_rep
        stats_dataset = G1SeedTrainingDataset(
            args.data_root,
            metadata_path=args.metadata,
            split_path=args.split_path,
            max_files=args.max_files,
            max_frames=train_cfg.max_frames,
            frame_crop="prefix",
            fps=train_cfg.fps,
            source_fps=train_cfg.source_fps,
            motion_rep=stats_motion_rep,
            normalize=False,
            require_text=not args.no_text,
        )
        compute_motion_stats(
            stats_dataset,
            stats_motion_rep,
            batch_size=per_gpu_batch,
            num_workers=train_cfg.num_workers,
            max_batches=args.stats_batches if args.smoke else None,
        )
        stats_path = save_motion_stats(stats_motion_rep, stats_workdir)
        print_rank0(rank, f"Computed stats at {stats_path}")
    elif stats_path is not None:
        stats_workdir = Path(stats_path)
    else:
        stats_workdir = args.output_dir / "stats_work"
    barrier()
    if stats_path is None:
        stats_path = stats_workdir

    denoiser = build_denoiser(cfg, stats_path=str(stats_path), device=device)
    motion_rep = denoiser.motion_rep

    start_step = 1
    if args.resume_from is not None:
        resume_dir = args.resume_from.resolve()
        weights_path = resume_dir / "model.safetensors"
        if not weights_path.is_file():
            print_rank0(rank, f"ERROR: resume checkpoint missing model.safetensors: {weights_path}")
            return 1
        from safetensors.torch import load_file

        denoiser.load_state_dict(load_file(str(weights_path)))
        start_step = _parse_resume_step(resume_dir) + 1
        print_rank0(rank, f"Resumed weights from {resume_dir}; continuing at step {start_step}")

    if distributed:
        denoiser = wrap_distributed(denoiser, device=device, local_rank=local_rank)

    # Dataset preprocessing must stay on CPU: DataLoader workers cannot safely use
    # CUDA tensors after fork (denoiser.motion_rep lives on GPU when --device cuda).
    cpu_motion_rep = build_motion_rep(cfg.denoiser, stats_path=str(stats_path))

    denoiser_cfg = OmegaConf.to_container(cfg.denoiser, resolve=True)
    llm_dim = denoiser_cfg["llm_shape"][-1]
    if denoiser_cfg.get("num_text_tokens_override") is not None:
        num_tokens = denoiser_cfg["num_text_tokens_override"]
    else:
        num_tokens = denoiser_cfg["llm_shape"][0]
    encoder_cfg = OmegaConf.to_container(cfg.text_encoder, resolve=True) if "text_encoder" in cfg else None

    dataset = G1SeedTrainingDataset(
        args.data_root,
        metadata_path=args.metadata,
        split_path=args.split_path,
        max_files=args.max_files,
        max_frames=train_cfg.max_frames,
        frame_crop=str(train_cfg.get("frame_crop", "random")),
        fps=train_cfg.fps,
        source_fps=train_cfg.source_fps,
        motion_rep=cpu_motion_rep,
        normalize=True,
        require_text=not args.no_text,
        text_cache_dir=text_cache_dir if use_text_cache else None,
        text_cache_num_tokens=num_tokens,
        text_cache_llm_dim=llm_dim,
    )

    if use_text_cache:
        assert text_cache_dir is not None
        missing = count_missing_embeddings(dataset.samples, text_cache_dir)
        if missing > 0:
            if not auto_precompute:
                print_rank0(
                    rank,
                    f"ERROR: {missing} text embeddings missing under {text_cache_dir}. "
                    "Run with --precompute-text-cache or set text_cache.auto_precompute: true.",
                )
                return 1
            if is_main_process(rank):
                precompute_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
                print_rank0(rank, f"Precomputing {missing} missing text embeddings into {text_cache_dir} ...")
                precompute_text_embeddings(
                    dataset.samples,
                    cache_dir=text_cache_dir,
                    encoder_cfg=encoder_cfg,
                    num_tokens=num_tokens,
                    llm_dim=llm_dim,
                    device=precompute_device,
                )
        barrier()
        print_rank0(rank, f"Using precomputed text embeddings from {text_cache_dir}")

    requested_workers = int(train_cfg.num_workers)
    num_workers = requested_workers
    dl_cfg = OmegaConf.to_container(cfg.get("dataloader", {}), resolve=True) if "dataloader" in cfg else {}
    pin_memory = bool(dl_cfg.get("pin_memory", device.type == "cuda"))
    per_gpu_batch = min(per_gpu_batch, len(dataset))
    requested_prefetch = int(dl_cfg.get("prefetch_factor", 2))
    rg_cfg = OmegaConf.to_container(cfg.get("resource_guard", {}), resolve=True) if "resource_guard" in cfg else {}
    if rg_cfg.get("enabled", False):
        tuning, rg_msg = cap_dataloader_at_startup(
            requested_workers=requested_workers,
            requested_prefetch=requested_prefetch,
            world_size=world_size,
            reserve_gb=float(rg_cfg.get("reserve_gb", 16.0)),
            gb_per_worker=float(rg_cfg.get("gb_per_worker", 2.5)),
            min_workers=int(rg_cfg.get("min_num_workers", 0)),
            min_prefetch=int(rg_cfg.get("min_prefetch_factor", 1)),
            mem_prefetch_cap_ratio=float(rg_cfg.get("mem_prefetch_cap_ratio", 0.70)),
        )
        print_rank0(rank, rg_msg)
    else:
        tuning = DataLoaderTuning(
            num_workers=requested_workers,
            prefetch_factor=requested_prefetch,
            persistent_workers=bool(dl_cfg.get("persistent_workers", True)) and requested_workers > 0,
        )
    num_workers = tuning.num_workers
    resource_guard = ResourceGuard.from_config(rg_cfg if rg_cfg else None, world_size=world_size, initial=tuning)
    resource_guard.requested_workers = requested_workers
    resource_guard.requested_prefetch = requested_prefetch

    loader, sampler, drop_last = _build_motion_dataloader(
        dataset,
        per_gpu_batch=per_gpu_batch,
        tuning=resource_guard.current_tuning(),
        pin_memory=pin_memory,
        distributed=distributed,
    )
    epoch_schedule = EpochSchedule.from_training(
        num_samples=len(dataset),
        global_batch_size=effective_global_batch,
        max_steps=int(train_cfg.max_steps),
        micro_batches_per_epoch=len(loader),
        grad_accum_steps=grad_accum_steps,
        drop_last=drop_last,
    )
    save_every_epochs_cfg = train_cfg.get("save_every_epochs")
    save_every = resolve_save_every_steps(
        save_every_steps=int(train_cfg.save_every) if train_cfg.get("save_every") is not None else None,
        save_every_epochs=float(save_every_epochs_cfg) if save_every_epochs_cfg is not None else None,
        steps_per_epoch=epoch_schedule.steps_per_epoch,
    )
    data_epoch = 0
    if sampler is not None:
        sampler.set_epoch(data_epoch)
    data_iter = iter(loader)

    text_provider_mode = "dummy" if use_text_cache else args.text_mode
    text_provider = TextEmbeddingProvider(
        num_tokens=num_tokens,
        llm_dim=llm_dim,
        device=device,
        mode=text_provider_mode,
        encoder_cfg=None if use_text_cache else encoder_cfg,
    )

    flow_cfg = cfg.flow_matching
    flow_loss = FlowMatchingLoss(sigma=flow_cfg.sigma, matcher=flow_cfg.matcher)
    optimizer = torch.optim.AdamW(
        denoiser.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    warmup_steps = int(train_cfg.get("warmup_steps", 0) or 0)
    lr_scheduler = None
    if warmup_steps > 0:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps),
        )

    if not args.no_text and args.text_mode == "dummy":
        print_rank0(
            rank,
            "WARNING: training with text but --text-mode dummy (zero embeddings). "
            "Use --text-mode encoder for text-conditioned models.",
        )

    wandb_logger = WandbLogger(args.wandb and is_main_process(rank))
    if args.wandb and is_main_process(rank):
        wandb_logger.init(
            project=args.wandb_project,
            run_name=args.wandb_run_name,
            config={
                "config": args.config,
                "no_text": args.no_text,
                "text_mode": args.text_mode,
                "device": str(device),
                "distributed": distributed,
                "world_size": world_size,
                "per_gpu_batch": per_gpu_batch,
                "grad_accum_steps": grad_accum_steps,
                "global_batch_size": effective_global_batch,
                "clips": len(dataset),
                "steps_per_epoch": epoch_schedule.steps_per_epoch,
                "samples_per_epoch": epoch_schedule.samples_per_epoch,
                "total_epochs": epoch_schedule.total_epochs,
                "save_every_steps": save_every,
                **OmegaConf.to_container(cfg, resolve=True),
            },
            output_dir=str(args.output_dir),
        )

    denoiser.train()
    text_tokens = unwrap_module(denoiser).root_model.num_text_tokens
    max_steps = int(train_cfg.max_steps)
    log_every = int(train_cfg.log_every)
    print_rank0(
        rank,
        f"Device: {device} | rank: {rank}/{world_size} | clips: {len(dataset)} "
        f"| feat_dim: {motion_rep.motion_rep_dim} | max_steps: {max_steps} | start_step: {start_step} "
        f"| text_mode: {args.text_mode} | text_cache: {text_cache_dir if use_text_cache else 'off'} "
        f"| no_text: {args.no_text} | text_tokens: {text_tokens} "
        f"| per_gpu_batch: {per_gpu_batch} | grad_accum: {grad_accum_steps} "
        f"| global_batch: {effective_global_batch} | lr: {train_cfg.lr} "
        f"| steps/epoch: {epoch_schedule.steps_per_epoch} "
        f"| epochs≈{epoch_schedule.total_epochs:.1f} | save_every: {save_every} steps "
        f"| num_workers: {num_workers} | prefetch: {tuning.prefetch_factor} "
        f"| resource_guard: {'on' if resource_guard.cfg.enabled else 'off'} "
        f"| stats_dir: {stats_workdir}",
    )

    if lr_scheduler is not None and start_step > 1:
        for _ in range(start_step - 1):
            lr_scheduler.step()

    start_time = time.perf_counter()
    progress = None
    if is_main_process(rank):
        progress = tqdm(
            total=max_steps - start_step + 1,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
        )

    step_range = range(start_step, max_steps + 1)
    optimizer.zero_grad(set_to_none=True)
    for step in step_range:
        step_loss = 0.0
        step_metrics = None
        epoch_progress = StepEpochProgress.at_step(step, epoch_schedule)
        for _micro in range(grad_accum_steps):
            prev_data_epoch = data_epoch
            batch, data_iter, data_epoch = _next_batch(data_iter, loader, sampler, data_epoch)
            if data_epoch > prev_data_epoch:
                print_rank0(
                    rank,
                    f"--- dataloader epoch {data_epoch} started "
                    f"(optimizer {format_epoch_progress(epoch_progress)}) ---",
                )
                if resource_guard.consume_reload_request():
                    tuning = resource_guard.current_tuning()
                    num_workers = tuning.num_workers
                    del loader
                    loader, sampler, drop_last = _build_motion_dataloader(
                        dataset,
                        per_gpu_batch=per_gpu_batch,
                        tuning=tuning,
                        pin_memory=pin_memory,
                        distributed=distributed,
                    )
                    if sampler is not None:
                        sampler.set_epoch(data_epoch)
                    data_iter = iter(loader)
                    print_rank0(
                        rank,
                        f"resource_guard: reloaded DataLoader "
                        f"(workers={tuning.num_workers}, prefetch={tuning.prefetch_factor})",
                    )
                    barrier()
            loss, metrics = flow_matching_batch_step(
                denoiser,
                motion_rep,
                batch,
                flow_loss,
                text_provider,
                cfg_dropout=OmegaConf.to_container(train_cfg.cfg_dropout, resolve=True),
                constraint_prob=train_cfg.constraint_prob,
                max_keyframes=train_cfg.max_keyframes,
            )

            if not torch.isfinite(loss):
                print_rank0(rank, f"ERROR: non-finite loss at step {step}: {loss.item()}")
                return 1

            (loss / grad_accum_steps).backward()
            step_loss = metrics["loss"].item()
            step_metrics = metrics

        grad_norm = torch.nn.utils.clip_grad_norm_(denoiser.parameters(), train_cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if lr_scheduler is not None:
            lr_scheduler.step()

        if progress is not None:
            progress.update(1)

        elapsed = time.perf_counter() - start_time
        steps_done = step - start_step + 1
        eta = elapsed / max(steps_done, 1) * (max_steps - step)

        if progress is not None and (step % log_every == 0 or step == 1):
            assert step_metrics is not None
            progress.set_postfix(
                epoch=f"{epoch_progress.epoch:.2f}/{epoch_progress.total_epochs:.0f}",
                loss=f"{step_loss:.4f}",
                grad=f"{float(grad_norm):.4f}",
                elapsed=_format_duration(elapsed),
                eta=_format_duration(eta),
                refresh=False,
            )
            tqdm.write(
                f"step {step}/{max_steps} | {format_epoch_progress(epoch_progress)} "
                f"| sampler_epoch={data_epoch} "
                f"loss={step_loss:.6f} "
                f"t_mean={step_metrics['t_mean'].item():.3f} "
                f"grad_norm={float(grad_norm):.4f} "
                f"elapsed={_format_duration(elapsed)} "
                f"eta={_format_duration(eta)}"
            )
            wandb_logger.log_step(
                step,
                loss=step_loss,
                grad_norm=float(grad_norm),
                t_mean=step_metrics["t_mean"].item(),
                ut_norm=step_metrics["ut_norm"].item(),
                v_norm=step_metrics["v_norm"].item(),
                lr=optimizer.param_groups[0]["lr"],
                elapsed_sec=elapsed,
                epoch=epoch_progress.epoch,
                epoch_index=epoch_progress.epoch_index,
                step_in_epoch=epoch_progress.step_in_epoch,
                steps_per_epoch=epoch_progress.steps_per_epoch,
                sampler_epoch=data_epoch,
            )

        guard_msg = resource_guard.check(step)
        if guard_msg is not None:
            print_rank0(rank, guard_msg)

        if step % save_every == 0:
            barrier()
            if is_main_process(rank):
                bare_denoiser = unwrap_module(denoiser)
                denoiser.train()
                with torch.inference_mode():
                    ckpt_dir = save_training_checkpoint(
                        output_dir=args.output_dir / f"step_{step}",
                        denoiser=bare_denoiser,
                        denoiser_cfg=denoiser_cfg,
                        training_cfg=OmegaConf.to_container(cfg, resolve=True),
                        stats_dir=stats_workdir,
                    )
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print_rank0(
                    rank,
                    f"saved checkpoint to {ckpt_dir} ({format_epoch_progress(epoch_progress)})",
                )
                wandb_logger.log_checkpoint(step, str(ckpt_dir), epoch=epoch_progress.epoch)
            barrier()
            denoiser.train()

    if max_steps % save_every != 0:
        barrier()
        if is_main_process(rank):
            bare_denoiser = unwrap_module(denoiser)
            with torch.inference_mode():
                ckpt_dir = save_training_checkpoint(
                    output_dir=args.output_dir / f"step_{max_steps}",
                    denoiser=bare_denoiser,
                    denoiser_cfg=denoiser_cfg,
                    training_cfg=OmegaConf.to_container(cfg, resolve=True),
                    stats_dir=stats_workdir,
                )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print_rank0(
                rank,
                f"saved final checkpoint to {ckpt_dir} "
                f"({format_epoch_progress(StepEpochProgress.at_step(max_steps, epoch_schedule))})",
            )
            wandb_logger.log_checkpoint(
                max_steps,
                str(ckpt_dir),
                epoch=StepEpochProgress.at_step(max_steps, epoch_schedule).epoch,
            )
        barrier()

    total_time = time.perf_counter() - start_time
    print_rank0(rank, f"Training completed in {_format_duration(total_time)}")
    wandb_logger.finish()

    if is_main_process(rank):
        bare_denoiser = unwrap_module(denoiser)
        bare_denoiser.eval()
        with torch.no_grad():
            batch = next(iter(loader))
            if "text_feat" in batch:
                text_feat = batch["text_feat"].to(device)
                text_pad_mask = batch["text_pad_mask"].to(device)
            else:
                text_feat, text_pad_mask = text_provider.encode(batch["texts"])
            pad_mask = batch["pad_mask"].to(device)
            b, t, d = batch["feats"].shape
            x = torch.randn(b, t, d, device=device)
            dt = 1.0 / 5
            heading = torch.zeros(b, device=device)
            for i in range(5):
                t_batch = torch.full((b,), 1.0 - i * dt, device=device)
                v = bare_denoiser(
                    x,
                    pad_mask,
                    text_feat,
                    text_pad_mask,
                    t_batch,
                    first_heading_angle=heading,
                )
                x = x - dt * v
            print_rank0(rank, f"inference smoke ok: output shape {tuple(x.shape)}")

        if args.smoke:
            with torch.inference_mode():
                ckpt_dir = save_training_checkpoint(
                    output_dir=args.output_dir / "smoke_final",
                    denoiser=bare_denoiser,
                    denoiser_cfg=denoiser_cfg,
                    training_cfg=OmegaConf.to_container(cfg, resolve=True),
                    stats_dir=stats_workdir,
                )
            print_rank0(rank, f"saved smoke checkpoint to {ckpt_dir}")

    if distributed:
        dist.destroy_process_group()

    print_rank0(rank, "TRAIN PASSED" if args.smoke else "Training finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
