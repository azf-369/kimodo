# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Training helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from kimodo.motion_rep import KimodoMotionRep
from kimodo.motion_rep.stats import Stats
from kimodo.skeleton import build_skeleton


def build_motion_rep_with_identity_stats(nbjoints: int = 34, fps: int = 30) -> KimodoMotionRep:
    """Build KimodoMotionRep with mean=0/std=1 stats for training without a checkpoint."""
    skeleton = build_skeleton(nbjoints)
    motion_rep = KimodoMotionRep(skeleton, fps)

    def _identity_stats(dim: int) -> Stats:
        stats = Stats(load=False)
        stats.register_from_tensors(torch.zeros(dim), torch.ones(dim))
        return stats

    motion_rep.global_root_stats = _identity_stats(motion_rep.global_root_dim)
    motion_rep.local_root_stats = _identity_stats(motion_rep.local_root_dim)
    motion_rep.body_stats = _identity_stats(motion_rep.body_dim)

    mean = torch.cat([motion_rep.global_root_stats.mean, motion_rep.body_stats.mean])
    std = torch.cat([motion_rep.global_root_stats.std, motion_rep.body_stats.std])
    motion_rep.stats = Stats(load=False)
    motion_rep.stats.register_from_tensors(mean, std)
    return motion_rep


def load_train_config(config_name: str = "fm_g1_seed") -> OmegaConf:
    config_dir = Path(__file__).resolve().parent / "config"
    return OmegaConf.load(config_dir / f"{config_name}.yaml")


def apply_no_text_overrides(cfg: OmegaConf) -> OmegaConf:
    """Remove text prefix from denoiser and disable text-related CFG training."""
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.denoiser.num_text_tokens_override = 0
    cfg.cfg_type = "nocfg"
    if "cfg_dropout" in cfg.training:
        cfg.training.cfg_dropout.uncond = cfg.training.cfg_dropout.get("uncond", 0.1)
        cfg.training.cfg_dropout.text_only = 0.0
        cfg.training.cfg_dropout.constraint_only = 0.0
    return cfg
