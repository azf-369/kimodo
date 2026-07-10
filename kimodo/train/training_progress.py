# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Epoch / step progress helpers for Kimodo FM training."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EpochSchedule:
    """Derived epoch schedule from dataset size and global batch."""

    num_samples: int
    global_batch_size: int
    steps_per_epoch: int
    samples_per_epoch: int
    total_epochs: float
    max_steps: int

    @classmethod
    def from_training(
        cls,
        *,
        num_samples: int,
        global_batch_size: int,
        max_steps: int,
        micro_batches_per_epoch: int | None = None,
        grad_accum_steps: int = 1,
        drop_last: bool = True,
    ) -> EpochSchedule:
        if micro_batches_per_epoch is not None:
            steps_per_epoch = max(1, micro_batches_per_epoch // max(grad_accum_steps, 1))
            if micro_batches_per_epoch % max(grad_accum_steps, 1) != 0:
                steps_per_epoch += 1
            samples_per_epoch = min(
                num_samples,
                steps_per_epoch * global_batch_size,
            )
        elif drop_last and global_batch_size > 0:
            steps_per_epoch = max(1, num_samples // global_batch_size)
            samples_per_epoch = steps_per_epoch * global_batch_size
        else:
            steps_per_epoch = max(1, math.ceil(num_samples / max(global_batch_size, 1)))
            samples_per_epoch = num_samples

        total_epochs = max_steps / steps_per_epoch
        return cls(
            num_samples=num_samples,
            global_batch_size=global_batch_size,
            steps_per_epoch=steps_per_epoch,
            samples_per_epoch=samples_per_epoch,
            total_epochs=total_epochs,
            max_steps=max_steps,
        )


@dataclass(frozen=True)
class StepEpochProgress:
    """Epoch progress for a given optimizer step (1-indexed)."""

    step: int
    steps_per_epoch: int
    epoch: float
    epoch_index: int
    step_in_epoch: int
    epoch_progress: float
    total_epochs: float

    @classmethod
    def at_step(cls, step: int, schedule: EpochSchedule) -> StepEpochProgress:
        spe = max(1, schedule.steps_per_epoch)
        epoch_index = (step - 1) // spe + 1
        step_in_epoch = (step - 1) % spe + 1
        epoch = step / spe
        epoch_progress = step_in_epoch / spe
        return cls(
            step=step,
            steps_per_epoch=spe,
            epoch=epoch,
            epoch_index=epoch_index,
            step_in_epoch=step_in_epoch,
            epoch_progress=epoch_progress,
            total_epochs=schedule.total_epochs,
        )


def resolve_save_every_steps(
    *,
    save_every_steps: int | None,
    save_every_epochs: float | None,
    steps_per_epoch: int,
) -> int:
    """Resolve checkpoint interval in optimizer steps."""
    if save_every_epochs is not None and save_every_epochs > 0:
        return max(1, int(round(save_every_epochs * steps_per_epoch)))
    if save_every_steps is not None and save_every_steps > 0:
        return int(save_every_steps)
    return 5000


def format_epoch_progress(progress: StepEpochProgress) -> str:
    return (
        f"epoch {progress.epoch:.2f}/{progress.total_epochs:.1f} "
        f"({progress.epoch_index}, step {progress.step_in_epoch}/{progress.steps_per_epoch})"
    )
