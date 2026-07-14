# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage schedule helpers for progressive distillation (N -> N/2)."""

from __future__ import annotations

from typing import Optional

# Default progressive stages aligned with Kimodo's default 100-step DDIM.
STAGE_SCHEDULE: tuple[tuple[int, int], ...] = (
    (100, 50),
    (50, 25),
    (24, 12),  # prefer even teacher steps; 25 is odd so use 24→12
    (12, 6),
    (8, 4),
)


def resolve_stage(
    stage: Optional[str] = None,
    *,
    teacher_steps: Optional[int] = None,
    student_steps: Optional[int] = None,
) -> tuple[int, int]:
    """Resolve (teacher_steps, student_steps).

    Args:
        stage: String like ``"100to50"`` or ``"100->50"``.
        teacher_steps / student_steps: Explicit overrides (both required if stage is None).
    """
    if stage is not None:
        text = stage.lower().replace("->", "to").replace("_", "to")
        if "to" not in text:
            raise ValueError(f"Invalid stage '{stage}', expected e.g. '100to50'")
        left, right = text.split("to", 1)
        t_steps, s_steps = int(left), int(right)
    else:
        if teacher_steps is None or student_steps is None:
            raise ValueError("Provide --stage or both --teacher-steps and --student-steps")
        t_steps, s_steps = int(teacher_steps), int(student_steps)

    if t_steps < 2:
        raise ValueError(f"teacher_steps must be >= 2, got {t_steps}")
    if s_steps != t_steps // 2:
        raise ValueError(
            f"student_steps must be teacher_steps // 2 ({t_steps // 2}), got {s_steps}"
        )
    if t_steps % 2 != 0:
        raise ValueError(f"teacher_steps must be even for 2-step distillation, got {t_steps}")
    return t_steps, s_steps
