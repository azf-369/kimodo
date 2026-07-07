# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic full-body constraints for Flow Matching training."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from kimodo.constraints import FullBodyConstraintSet
from kimodo.motion_rep import KimodoMotionRep


def sample_training_constraints(
    motion_rep: KimodoMotionRep,
    feats: Tensor,
    lengths: Tensor,
    *,
    constraint_prob: float = 0.8,
    max_keyframes: int = 4,
    device: Optional[str] = None,
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Build batched constraint tensors from ground-truth normalized features."""
    if constraint_prob <= 0:
        return None, None

    batch_size, max_len, _ = feats.shape
    device = device or str(feats.device)
    skeleton = motion_rep.skeleton

    constraints_per_sample: list[list[FullBodyConstraintSet]] = []
    has_any = False

    for b in range(batch_size):
        length = int(lengths[b].item())
        if length < 1 or torch.rand(1).item() > constraint_prob:
            constraints_per_sample.append([])
            continue

        num_keyframes = int(torch.randint(1, min(max_keyframes, length) + 1, (1,)).item())
        frame_indices = torch.sort(torch.randperm(length, device=feats.device)[:num_keyframes])[0]

        decoded = motion_rep.inverse(feats[b : b + 1, :length], is_normalized=True)
        posed_joints = decoded["posed_joints"][0, frame_indices]
        global_rot_mats = decoded["global_rot_mats"][0, frame_indices]
        smooth_root_2d = decoded["smooth_root_pos"][0, frame_indices][..., [0, 2]]

        constraints_per_sample.append(
            [
                FullBodyConstraintSet(
                    skeleton,
                    frame_indices.to(device=feats.device),
                    posed_joints,
                    global_rot_mats,
                    smooth_root_2d=smooth_root_2d,
                )
            ]
        )
        has_any = True

    if not has_any:
        return None, None

    return motion_rep.create_conditions_from_constraints_batched(
        constraints_per_sample,
        lengths,
        to_normalize=True,
        device=device,
    )
