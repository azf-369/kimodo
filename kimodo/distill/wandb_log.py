# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wandb logging for progressive distillation (independent of FM trainer fields)."""

from __future__ import annotations

from typing import Any, Optional


class DistillWandbLogger:
    """Optional W&B logger for PD training."""

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

    def log_step(self, step: int, metrics: dict[str, Any]) -> None:
        if not self.enabled:
            return
        loss = float(metrics.get("loss", 0.0))
        self._loss_window.append(loss)
        if len(self._loss_window) > self._window_size:
            self._loss_window.pop(0)
        payload = {
            "train/loss": loss,
            "train/loss_smooth": sum(self._loss_window) / len(self._loss_window),
            **{f"train/{k}": v for k, v in metrics.items() if k != "loss"},
        }
        self._wandb.log(payload, step=step)

    def log_checkpoint(self, step: int, checkpoint_dir: str) -> None:
        if not self.enabled:
            return
        self._wandb.log({"checkpoint/step": step}, step=step)
        self._wandb.summary["last_checkpoint"] = checkpoint_dir

    def finish(self) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.finish()
