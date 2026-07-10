#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Precompute LLM2Vec text embeddings for Kimodo G1 Flow Matching training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from kimodo.train.build import build_motion_rep
from kimodo.train.dataset import G1SeedTrainingDataset
from kimodo.train.text_cache import count_missing_embeddings, precompute_text_embeddings, resolve_text_cache_dir
from kimodo.train.utils import load_train_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute LLM2Vec embeddings for G1 training.")
    parser.add_argument("--config", type=str, default="fm_g1_seed")
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("datasets/bones-seed"))
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--text-cache-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true", help="Recompute all embeddings, not only missing ones.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    cfg = load_train_config(args.config)
    if args.server:
        cfg = OmegaConf.merge(cfg, load_train_config("fm_g1_seed_server"))
        cfg = OmegaConf.merge(cfg, load_train_config("fm_g1_seed_server_text"))

    text_cache_dir = resolve_text_cache_dir(cfg, args.text_cache_dir)
    if text_cache_dir is None:
        print("ERROR: text cache is not enabled. Set text_cache.enabled/dir in config or pass --text-cache-dir.", file=sys.stderr)
        return 1

    denoiser_cfg = OmegaConf.to_container(cfg.denoiser, resolve=True)
    llm_dim = denoiser_cfg["llm_shape"][-1]
    num_tokens = denoiser_cfg.get("num_text_tokens_override") or denoiser_cfg["llm_shape"][0]
    encoder_cfg = OmegaConf.to_container(cfg.text_encoder, resolve=True) if "text_encoder" in cfg else None
    if encoder_cfg is None:
        print("ERROR: text_encoder config is missing.", file=sys.stderr)
        return 1

    train_cfg = cfg.training
    motion_rep = build_motion_rep(cfg.denoiser, stats_path=None)
    dataset = G1SeedTrainingDataset(
        args.data_root,
        metadata_path=args.metadata,
        split_path=args.split_path,
        max_files=args.max_files,
        max_frames=train_cfg.max_frames,
        fps=train_cfg.fps,
        source_fps=train_cfg.source_fps,
        motion_rep=motion_rep,
        normalize=False,
        require_text=True,
    )

    missing = count_missing_embeddings(dataset.samples, text_cache_dir)
    total = len(dataset.samples)
    print(f"clips: {total} | cache_dir: {text_cache_dir} | missing: {missing}")
    if missing == 0 and not args.force:
        print("All embeddings already cached.")
        return 0

    computed = precompute_text_embeddings(
        dataset.samples,
        cache_dir=text_cache_dir,
        encoder_cfg=encoder_cfg,
        num_tokens=num_tokens,
        llm_dim=llm_dim,
        device=device,
        skip_existing=not args.force,
    )
    print(f"Precomputed {computed} embeddings into {text_cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
