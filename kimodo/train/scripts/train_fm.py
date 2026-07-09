#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train Kimodo G1 Flow Matching model on BONES-SEED."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from kimodo.model.flow_matching import FlowMatchingLoss
from kimodo.train.build import build_denoiser, build_motion_rep
from kimodo.train.checkpoint import save_training_checkpoint
from kimodo.train.collate import collate_motion_batch
from kimodo.train.dataset import G1SeedTrainingDataset
from kimodo.train.flow_train import flow_matching_batch_step
from kimodo.train.stats_compute import compute_motion_stats, save_motion_stats
from kimodo.train.text_embedding import TextEmbeddingProvider
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
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size.")
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
    return parser.parse_args()


def _parse_resume_step(resume_dir: Path) -> int:
    name = resume_dir.name
    if not name.startswith("step_"):
        raise ValueError(f"--resume-from must be a step_N directory, got: {resume_dir}")
    return int(name.split("_", 1)[1])


def main() -> int:
    args = parse_args()
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
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.max_frames is not None:
        cfg.training.max_frames = args.max_frames
    if args.max_steps is not None:
        cfg.training.max_steps = args.max_steps
    if args.no_text:
        if args.text_mode == "encoder":
            print("WARNING: --no-text forces --text-mode dummy (LLM2Vec is not used).", file=sys.stderr)
        args.text_mode = "dummy"
        cfg = apply_no_text_overrides(cfg)
    train_cfg = cfg.training

    if not args.data_root.is_dir():
        print(f"ERROR: data root not found: {args.data_root}", file=sys.stderr)
        return 1

    default_stats = Path("checkpoints/Kimodo-G1-SEED-v1/stats/motion")
    stats_path = args.stats_path
    if stats_path is None and default_stats.is_dir():
        stats_path = default_stats

    stats_workdir = args.output_dir / "stats_work"
    if args.compute_stats or stats_path is None:
        stats_motion_rep = build_denoiser(cfg, stats_path=None, device=torch.device("cpu")).motion_rep
        stats_dataset = G1SeedTrainingDataset(
            args.data_root,
            metadata_path=args.metadata,
            split_path=args.split_path,
            max_files=args.max_files,
            max_frames=train_cfg.max_frames,
            fps=train_cfg.fps,
            source_fps=train_cfg.source_fps,
            motion_rep=stats_motion_rep,
            normalize=False,
            require_text=not args.no_text,
        )
        compute_motion_stats(
            stats_dataset,
            stats_motion_rep,
            batch_size=train_cfg.batch_size,
            num_workers=train_cfg.num_workers,
            max_batches=args.stats_batches if args.smoke else None,
        )
        stats_path = save_motion_stats(stats_motion_rep, stats_workdir)
        print(f"Computed stats at {stats_path}")
    elif stats_path is not None:
        stats_workdir = Path(stats_path)

    denoiser = build_denoiser(cfg, stats_path=str(stats_path), device=device)
    motion_rep = denoiser.motion_rep

    start_step = 1
    if args.resume_from is not None:
        resume_dir = args.resume_from.resolve()
        weights_path = resume_dir / "model.safetensors"
        if not weights_path.is_file():
            print(f"ERROR: resume checkpoint missing model.safetensors: {weights_path}", file=sys.stderr)
            return 1
        from safetensors.torch import load_file

        denoiser.load_state_dict(load_file(str(weights_path)))
        start_step = _parse_resume_step(resume_dir) + 1
        print(f"Resumed weights from {resume_dir}; continuing at step {start_step}")

    # Dataset preprocessing must stay on CPU: DataLoader workers cannot safely use
    # CUDA tensors after fork (denoiser.motion_rep lives on GPU when --device cuda).
    cpu_motion_rep = build_motion_rep(cfg.denoiser, stats_path=str(stats_path))

    dataset = G1SeedTrainingDataset(
        args.data_root,
        metadata_path=args.metadata,
        split_path=args.split_path,
        max_files=args.max_files,
        max_frames=train_cfg.max_frames,
        fps=train_cfg.fps,
        source_fps=train_cfg.source_fps,
        motion_rep=cpu_motion_rep,
        normalize=True,
        require_text=not args.no_text,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(train_cfg.batch_size, len(dataset)),
        shuffle=True,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_motion_batch,
        drop_last=len(dataset) >= train_cfg.batch_size,
    )
    data_iter = itertools.cycle(loader)

    denoiser_cfg = OmegaConf.to_container(cfg.denoiser, resolve=True)
    llm_dim = denoiser_cfg["llm_shape"][-1]
    if denoiser_cfg.get("num_text_tokens_override") is not None:
        num_tokens = denoiser_cfg["num_text_tokens_override"]
    else:
        num_tokens = denoiser_cfg["llm_shape"][0]
    encoder_cfg = OmegaConf.to_container(cfg.text_encoder, resolve=True) if "text_encoder" in cfg else None
    text_provider = TextEmbeddingProvider(
        num_tokens=num_tokens,
        llm_dim=llm_dim,
        device=device,
        mode=args.text_mode,
        encoder_cfg=encoder_cfg,
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
        print(
            "WARNING: training with text but --text-mode dummy (zero embeddings). "
            "Use --text-mode encoder for text-conditioned models.",
            file=sys.stderr,
        )

    wandb_logger = WandbLogger(args.wandb)
    if args.wandb:
        wandb_logger.init(
            project=args.wandb_project,
            run_name=args.wandb_run_name,
            config={
                "config": args.config,
                "no_text": args.no_text,
                "text_mode": args.text_mode,
                "device": str(device),
                "clips": len(dataset),
                **OmegaConf.to_container(cfg, resolve=True),
            },
            output_dir=str(args.output_dir),
        )

    denoiser.train()
    text_tokens = denoiser.root_model.num_text_tokens
    max_steps = int(train_cfg.max_steps)
    log_every = int(train_cfg.log_every)
    print(
        f"Device: {device} | clips: {len(dataset)} | feat_dim: {motion_rep.motion_rep_dim} "
        f"| max_steps: {max_steps} | start_step: {start_step} | text_mode: {args.text_mode} "
        f"| no_text: {args.no_text} | text_tokens: {text_tokens} "
        f"| num_workers: {train_cfg.num_workers} | stats_dir: {stats_workdir}"
    )

    if lr_scheduler is not None and start_step > 1:
        for _ in range(start_step - 1):
            lr_scheduler.step()

    start_time = time.perf_counter()
    progress = tqdm(range(start_step, max_steps + 1), desc="Training", unit="step", dynamic_ncols=True)

    for step in progress:
        batch = next(data_iter)
        optimizer.zero_grad(set_to_none=True)
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
            tqdm.write(f"ERROR: non-finite loss at step {step}: {loss.item()}")
            return 1

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(denoiser.parameters(), train_cfg.grad_clip)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        elapsed = time.perf_counter() - start_time
        steps_done = step - start_step + 1
        eta = elapsed / max(steps_done, 1) * (max_steps - step)

        if step % log_every == 0 or step == 1:
            progress.set_postfix(
                loss=f"{metrics['loss'].item():.4f}",
                grad=f"{float(grad_norm):.4f}",
                elapsed=_format_duration(elapsed),
                eta=_format_duration(eta),
                refresh=False,
            )
            tqdm.write(
                f"step {step}/{max_steps} "
                f"loss={metrics['loss'].item():.6f} "
                f"t_mean={metrics['t_mean'].item():.3f} "
                f"grad_norm={float(grad_norm):.4f} "
                f"elapsed={_format_duration(elapsed)} "
                f"eta={_format_duration(eta)}"
            )
            wandb_logger.log_step(
                step,
                loss=metrics["loss"].item(),
                grad_norm=float(grad_norm),
                t_mean=metrics["t_mean"].item(),
                ut_norm=metrics["ut_norm"].item(),
                v_norm=metrics["v_norm"].item(),
                lr=optimizer.param_groups[0]["lr"],
                elapsed_sec=elapsed,
            )

        if step % int(train_cfg.save_every) == 0:
            ckpt_dir = save_training_checkpoint(
                output_dir=args.output_dir / f"step_{step}",
                denoiser=denoiser.cpu(),
                denoiser_cfg=denoiser_cfg,
                training_cfg=OmegaConf.to_container(cfg, resolve=True),
                stats_dir=stats_workdir,
            )
            denoiser.to(device).train()
            tqdm.write(f"saved checkpoint to {ckpt_dir}")
            wandb_logger.log_checkpoint(step, str(ckpt_dir))

    save_every = int(train_cfg.save_every)
    if max_steps % save_every != 0:
        ckpt_dir = save_training_checkpoint(
            output_dir=args.output_dir / f"step_{max_steps}",
            denoiser=denoiser.cpu(),
            denoiser_cfg=denoiser_cfg,
            training_cfg=OmegaConf.to_container(cfg, resolve=True),
            stats_dir=stats_workdir,
        )
        denoiser.to(device).train()
        print(f"saved final checkpoint to {ckpt_dir}")
        wandb_logger.log_checkpoint(max_steps, str(ckpt_dir))

    total_time = time.perf_counter() - start_time
    print(f"Training completed in {_format_duration(total_time)}")
    wandb_logger.finish()

    denoiser.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        text_feat, text_pad_mask = text_provider.encode(batch["texts"])
        pad_mask = batch["pad_mask"].to(device)
        b, t, d = batch["feats"].shape
        x = torch.randn(b, t, d, device=device)
        dt = 1.0 / 5
        heading = torch.zeros(b, device=device)
        for i in range(5):
            t_batch = torch.full((b,), 1.0 - i * dt, device=device)
            v = denoiser(
                x,
                pad_mask,
                text_feat,
                text_pad_mask,
                t_batch,
                first_heading_angle=heading,
            )
            x = x - dt * v
        print(f"inference smoke ok: output shape {tuple(x.shape)}")

    if args.smoke:
        ckpt_dir = save_training_checkpoint(
            output_dir=args.output_dir / "smoke_final",
            denoiser=denoiser.cpu(),
            denoiser_cfg=denoiser_cfg,
            training_cfg=OmegaConf.to_container(cfg, resolve=True),
            stats_dir=stats_workdir,
        )
        denoiser.to(device)
        print(f"saved smoke checkpoint to {ckpt_dir}")

    print("TRAIN PASSED" if args.smoke else "Training finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
