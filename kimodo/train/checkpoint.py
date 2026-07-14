# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint export compatible with kimodo.model.load_model."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file


def state_dict_on_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy weights to CPU without moving the live module off its training device."""
    return {key: tensor.detach().cpu() for key, tensor in module.state_dict().items()}


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

    model_path = output_dir / "model.safetensors"
    cpu_state = state_dict_on_cpu(denoiser)
    save_file(cpu_state, model_path)
    del cpu_state

    stats_dest = output_dir / "stats" / "motion"
    stats_dest.mkdir(parents=True, exist_ok=True)
    stats_root = Path(stats_dir)
    for sub in ("global_root", "local_root", "body"):
        src = stats_root / sub
        dst = stats_dest / sub
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("mean.npy", "std.npy"):
            src_file = src / name
            if not src_file.is_file():
                raise FileNotFoundError(
                    f"Missing stats file: {src_file}. "
                    "Ensure checkpoints/Kimodo-G1-SEED-v1/stats/motion exists "
                    "or pass --stats-path to a valid stats directory."
                )
            dst_file = dst / name
            if dst_file.exists() or dst_file.is_symlink():
                dst_file.unlink()
            try:
                os.symlink(src_file.resolve(), dst_file)
            except OSError:
                shutil.copy2(src_file, dst_file)

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
