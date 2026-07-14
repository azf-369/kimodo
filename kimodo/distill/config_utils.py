# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config helpers for kimodo.distill (do not depend on FM config names)."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def load_distill_config(config_name: str = "pd_g1_rp_teacher") -> OmegaConf:
    config_dir = Path(__file__).resolve().parent / "config"
    path = config_dir / f"{config_name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Distill config not found: {path}")
    return OmegaConf.load(path)
