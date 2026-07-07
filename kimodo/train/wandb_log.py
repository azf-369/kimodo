# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional Weights & Biases logging helpers for FM training."""

from __future__ import annotations

from typing import Any, Optional


class WandbLogger:
    """Thin wrapper so training runs without wandb installed unless enabled."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._wandb = None
        self._loss_window: list[float] = []
        self._window_size = 50

    def init(
        self,
        *,
        project: str,
        run_name: Optional[str],
        config: dict[str, Any],
        output_dir: str,
    ) -> None:
        if not self.enabled:
            return
        import wandb

        self._wandb = wandb
        wandb.init(project=project, name=run_name, config=config, dir=output_dir)

    def log_step(
        self,
        step: int,
        *,
        loss: float,
        grad_norm: float,
        t_mean: float,
        ut_norm: float,
        v_norm: float,
        lr: float,
        elapsed_sec: float,
    ) -> None:
        if not self.enabled:
            return

        self._loss_window.append(loss)
        if len(self._loss_window) > self._window_size:
            self._loss_window.pop(0)
        loss_smooth = sum(self._loss_window) / len(self._loss_window)

        self._wandb.log(
            {
                "train/loss": loss,
                "train/loss_smooth": loss_smooth,
                "train/grad_norm": grad_norm,
                "train/t_mean": t_mean,
                "train/ut_norm": ut_norm,
                "train/v_norm": v_norm,
                "train/lr": lr,
                "train/elapsed_sec": elapsed_sec,
            },
            step=step,
        )

    def log_checkpoint(self, step: int, checkpoint_dir: str) -> None:
        if not self.enabled:
            return
        self._wandb.log({"checkpoint/step": step}, step=step)
        self._wandb.summary["last_checkpoint"] = checkpoint_dir

    def finish(self) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.finish()
