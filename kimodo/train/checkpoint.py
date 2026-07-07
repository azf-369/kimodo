# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint export compatible with kimodo.model.load_model."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file


def _denoiser_export_config(denoiser_cfg: dict[str, Any], checkpoint_dir: Path) -> dict[str, Any]:
    """Build denoiser config for inference loading."""
    cfg = OmegaConf.to_container(OmegaConf.create(denoiser_cfg), resolve=True)
    cfg.pop("ckpt_path", None)
    motion_rep = cfg.setdefault("motion_rep", {})
    motion_rep["stats_path"] = str(checkpoint_dir / "stats" / "motion")
    return cfg


def save_training_checkpoint(
    *,
    output_dir: str | Path,
    denoiser: torch.nn.Module,
    denoiser_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    stats_dir: str | Path,
) -> Path:
    """Save config.yaml, model.safetensors, and motion stats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_dest = output_dir / "stats" / "motion"
    stats_dest.mkdir(parents=True, exist_ok=True)
    for sub in ("global_root", "local_root", "body"):
        src = Path(stats_dir) / sub
        dst = stats_dest / sub
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("mean.npy", "std.npy"):
            shutil.copy2(src / name, dst / name)

    model_path = output_dir / "model.safetensors"
    save_file(denoiser.state_dict(), model_path)

    export_cfg = {
        "_target_": "kimodo.model.Kimodo",
        "generative_paradigm": training_cfg.get("generative_paradigm", "flow_matching"),
        "num_base_steps": training_cfg.get("num_base_steps", 1000),
        "cfg_type": training_cfg.get("cfg_type", "separated"),
        "denoiser": {
            **_denoiser_export_config(denoiser_cfg, output_dir),
            "ckpt_path": str(model_path),
        },
    }

    OmegaConf.save(OmegaConf.create(export_cfg), output_dir / "config.yaml")
    return output_dir
