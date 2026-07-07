# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build training objects from OmegaConf configs."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from kimodo.model.loading import instantiate_from_dict
from kimodo.model.twostage_denoiser import TwostageDenoiser
from kimodo.motion_rep import KimodoMotionRep


def resolve_stats_path(stats_path: str | None, checkpoint_dir: str | Path | None) -> str | None:
    if stats_path is None:
        return None
    if "${" in stats_path and checkpoint_dir is not None:
        return str(Path(checkpoint_dir) / "stats" / "motion")
    return stats_path


def build_motion_rep(denoiser_cfg: DictConfig, stats_path: str | None) -> KimodoMotionRep:
    motion_rep_cfg = OmegaConf.to_container(denoiser_cfg.motion_rep, resolve=True)
    motion_rep_cfg["stats_path"] = stats_path
    return instantiate_from_dict(motion_rep_cfg)


def build_denoiser(
    cfg: DictConfig,
    *,
    stats_path: str | None,
    device: torch.device,
) -> TwostageDenoiser:
    denoiser_cfg = OmegaConf.to_container(cfg.denoiser, resolve=True)
    denoiser_cfg.pop("ckpt_path", None)
    motion_rep = build_motion_rep(cfg.denoiser, stats_path)

    denoiser_kwargs = {
        key: value
        for key, value in denoiser_cfg.items()
        if key not in {"_target_", "motion_rep", "ckpt_path"}
    }
    denoiser = TwostageDenoiser(motion_rep=motion_rep, **denoiser_kwargs)
    return denoiser.to(device)
