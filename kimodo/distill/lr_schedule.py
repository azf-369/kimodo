# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Learning-rate schedule for progressive distillation."""

from __future__ import annotations

import math


def compute_lr(
    step: int,
    *,
    base_lr: float,
    max_steps: int,
    warmup_steps: int = 0,
    schedule: str = "cosine",
    min_lr_ratio: float = 0.1,
) -> float:
    """Warmup then constant or cosine decay.

    Args:
        step: 1-indexed optimizer step.
        base_lr: Peak learning rate after warmup.
        max_steps: Total training steps.
        warmup_steps: Linear ramp length (0 disables warmup).
        schedule: ``cosine`` (default) or ``constant``.
        min_lr_ratio: Floor as a fraction of ``base_lr`` for cosine.
    """
    if step < 1:
        step = 1
    warmup_steps = max(0, int(warmup_steps))
    if warmup_steps > 0 and step <= warmup_steps:
        return float(base_lr) * (step / float(warmup_steps))

    if schedule == "constant":
        return float(base_lr)

    if schedule != "cosine":
        raise ValueError(f"Unknown lr schedule: {schedule}")

    denom = max(1, int(max_steps) - warmup_steps)
    progress = (step - warmup_steps) / float(denom)
    progress = min(1.0, max(0.0, progress))
    min_lr = float(base_lr) * float(min_lr_ratio)
    # Cosine from base_lr → min_lr
    return min_lr + 0.5 * (float(base_lr) - min_lr) * (1.0 + math.cos(math.pi * progress))
