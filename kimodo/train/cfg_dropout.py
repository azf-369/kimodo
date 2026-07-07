# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Classifier-free guidance dropout for separated CFG training."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def apply_separated_cfg_dropout(
    text_feat: Tensor,
    text_pad_mask: Tensor,
    motion_mask: Optional[Tensor],
    observed_motion: Optional[Tensor],
    *,
    p_uncond: float = 0.1,
    p_text_only: float = 0.1,
    p_constraint_only: float = 0.1,
) -> tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    """Apply training-time dropout aligned with separated CFG at inference.

    Modes per sample (mutually exclusive draw):
    - full: text + constraints
    - uncond: zero text mask, zero constraints
    - text_only: text kept, constraints dropped
    - constraint_only: text dropped, constraints kept
    """
    if motion_mask is None or observed_motion is None:
        return text_feat, text_pad_mask, motion_mask, observed_motion

    batch_size = text_feat.shape[0]
    device = text_feat.device

    probs = torch.tensor(
        [1.0 - p_uncond - p_text_only - p_constraint_only, p_uncond, p_text_only, p_constraint_only],
        device=device,
    )
    probs = probs / probs.sum()
    modes = torch.multinomial(probs, batch_size, replacement=True)

    text_pad_mask = text_pad_mask.clone()
    motion_mask = motion_mask.clone()
    observed_motion = observed_motion.clone()

    for i in range(batch_size):
        mode = int(modes[i].item())
        if mode == 1:
            text_pad_mask[i] = False
            motion_mask[i] = False
            observed_motion[i] = 0
        elif mode == 2:
            motion_mask[i] = False
            observed_motion[i] = 0
        elif mode == 3:
            text_pad_mask[i] = False

    return text_feat, text_pad_mask, motion_mask, observed_motion
