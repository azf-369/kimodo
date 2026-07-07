#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quick validation for Flow Matching training checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from kimodo.exports.motion_io import save_kimodo_npz
from kimodo.model.flow_matching import FlowMatchingLoss
from kimodo.model.loading import instantiate_from_dict
from kimodo.train.build import build_motion_rep
from kimodo.train.collate import collate_motion_batch
from kimodo.train.dataset import G1SeedTrainingDataset
from kimodo.train.flow_train import flow_matching_batch_step
from kimodo.train.text_embedding import TextEmbeddingProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an FM training checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint dir containing config.yaml and model.safetensors",
    )
    parser.add_argument("--data-root", type=Path, default=Path("datasets/bones-seed"))
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ode-steps", type=int, default=20, help="ODE steps for generation smoke test")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save a generated motion .npz",
    )
    parser.add_argument("--max-files", type=int, default=4, help="Clips used for loss eval")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    ckpt_dir = args.checkpoint.resolve()

    config_path = ckpt_dir / "config.yaml"
    weights_path = ckpt_dir / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        print(f"ERROR: checkpoint must contain config.yaml and model.safetensors: {ckpt_dir}", file=sys.stderr)
        return 1

    cfg = OmegaConf.load(config_path)
    model = instantiate_from_dict(
        OmegaConf.to_container(cfg, resolve=True),
        overrides={"device": str(device), "text_encoder": None},
    )
    model.eval()

    denoiser = model.denoiser.model
    motion_rep = model.motion_rep
    no_text = denoiser.root_model.num_text_tokens == 0
    print(f"Loaded checkpoint: {ckpt_dir}")
    print(
        f"generative_paradigm={model.generative_paradigm} "
        f"cfg_type={cfg.get('cfg_type')} text_tokens={denoiser.root_model.num_text_tokens}"
    )

    stats_path = ckpt_dir / "stats" / "motion"
    cpu_motion_rep = build_motion_rep(cfg.denoiser, stats_path=str(stats_path))
    dataset = G1SeedTrainingDataset(
        args.data_root,
        split_path=args.split_path,
        max_files=args.max_files,
        max_frames=120,
        motion_rep=cpu_motion_rep,
        normalize=True,
        require_text=not no_text,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_motion_batch)
    batch = next(iter(loader))

    denoiser_cfg = OmegaConf.to_container(cfg.denoiser, resolve=True)
    llm_dim = denoiser_cfg["llm_shape"][-1]
    num_tokens = denoiser_cfg.get("num_text_tokens_override")
    if num_tokens is None:
        num_tokens = denoiser_cfg["llm_shape"][0]

    text_provider = TextEmbeddingProvider(
        num_tokens=num_tokens,
        llm_dim=llm_dim,
        device=device,
        mode="dummy",
    )
    flow_loss = FlowMatchingLoss()

    with torch.no_grad():
        cfg_dropout = None
        if not no_text:
            cfg_dropout = {"uncond": 0.1, "text_only": 0.1, "constraint_only": 0.1}
        loss, metrics = flow_matching_batch_step(
            denoiser,
            motion_rep,
            batch,
            flow_loss,
            text_provider,
            cfg_dropout=cfg_dropout,
            constraint_prob=0.8,
            max_keyframes=4,
        )
    print(f"eval loss={loss.item():.6f} t_mean={metrics['t_mean'].item():.3f}")

    num_frames = int(batch["lengths"][0].item())
    pad_mask = batch["pad_mask"].to(device)
    text_feat, text_pad_mask = text_provider.encode(batch["texts"])
    heading = torch.zeros(1, device=device)

    with torch.no_grad():
        generated = model._generate_flow_matching(
            batch_size=1,
            max_frames=num_frames,
            num_steps=args.ode_steps,
            pad_mask=pad_mask,
            text_feat=text_feat,
            text_pad_mask=text_pad_mask,
            first_heading_angle=heading,
            motion_mask=None,
            observed_motion=None,
            cfg_weight=1.0,
            cfg_type=cfg.get("cfg_type", "nocfg"),
            progress_bar=lambda x: x,
        )

    decoded = motion_rep.inverse(generated, is_normalized=True, return_numpy=True)
    print(f"generation ok: feats={tuple(generated.shape)} frames={num_frames} ode_steps={args.ode_steps}")
    print(f"decoded keys: {list(decoded.keys())}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_kimodo_npz(str(args.output), decoded)
        print(f"saved motion to {args.output}")

    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
