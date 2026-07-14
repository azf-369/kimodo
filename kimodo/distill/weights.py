# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint loading helpers for progressive distillation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn

from kimodo.model.loading import load_checkpoint_state_dict


def resolve_teacher_weights(path: Union[str, Path]) -> Path:
    """Resolve a teacher directory or safetensors file to model.safetensors."""
    path = Path(path).expanduser().resolve()
    if path.is_file() and path.name.endswith(".safetensors"):
        return path
    if path.is_dir():
        candidate = path / "model.safetensors"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Teacher weights not found under: {path}")


def load_denoiser_weights(denoiser: nn.Module, weights_path: Union[str, Path]) -> None:
    """Load Kimodo denoiser weights; strips optional ``denoiser.backbone.`` prefix."""
    state = load_checkpoint_state_dict(weights_path)
    cleaned = {key.replace("denoiser.backbone.", ""): val for key, val in state.items()}
    missing, unexpected = denoiser.load_state_dict(cleaned, strict=False)
    # Official exports are bare TwostageDenoiser keys; allow empty missing for exact match.
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading {weights_path}: {unexpected[:8]}...")
    if missing:
        raise RuntimeError(f"Missing keys when loading {weights_path}: {missing[:8]}...")


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


def copy_weights(src: nn.Module, dst: nn.Module) -> None:
    dst.load_state_dict(src.state_dict())
