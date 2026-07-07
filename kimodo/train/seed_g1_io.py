# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load BONES-SEED G1 CSV (header + euler root) into Kimodo motion dict."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.skeleton.registry import build_skeleton


def load_bones_seed_g1_csv(
    path: str,
    source_fps: float = 120.0,
    *,
    mujoco_rest_zero: bool = False,
) -> Dict[str, torch.Tensor]:
    """Load a BONES-SEED G1 CSV into a Kimodo motion dict.

    SEED files use a header row and columns:
    Frame, root_translate (cm), root_rotate euler XYZ (deg), 29 joint dofs (deg).
    """
    raw = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] < 36:
        raise ValueError(f"Expected SEED G1 CSV with >=36 numeric columns; got {raw.shape}")

    # Numeric block: translate(3) + euler(3) + joints(29) = 35; optional leading Frame column
    if raw.shape[1] == 36:
        data = raw[:, 1:]
    else:
        data = raw[:, :35]

    root_trans_cm = data[:, :3]
    root_euler_deg = data[:, 3:6]
    joint_deg = data[:, 6:]

    root_trans_m = root_trans_cm / 100.0
    root_quat_xyzw = Rotation.from_euler("xyz", root_euler_deg, degrees=True).as_quat()
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]
    joint_rad = np.deg2rad(joint_deg)

    qpos = np.concatenate([root_trans_m, root_quat_wxyz, joint_rad], axis=1).astype(np.float64)
    sk = build_skeleton(34)
    converter = MujocoQposConverter(sk)
    return converter.qpos_to_motion_dict(qpos, source_fps, mujoco_rest_zero=mujoco_rest_zero)
