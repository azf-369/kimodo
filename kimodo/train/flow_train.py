# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Flow Matching training step helpers."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kimodo.model.flow_matching import FlowMatchingLoss, apply_motion_constraints
from kimodo.motion_rep import KimodoMotionRep
from kimodo.train.cfg_dropout import apply_separated_cfg_dropout
from kimodo.train.constraint_synth import sample_training_constraints


def _randomize_batch_heading(
    motion_rep: KimodoMotionRep,
    x1: Tensor,
) -> tuple[Tensor, Tensor]:
    """Rotate canonicalized clips to a random frame-0 heading (official training aug)."""
    batch_size = x1.shape[0]
    device = x1.device
    first_heading_angle = torch.rand(batch_size, device=device) * (2 * math.pi)
    x1 = motion_rep.unnormalize(x1)
    rotated: list[Tensor] = []
    for i in range(batch_size):
        rotated.append(motion_rep.rotate_to(x1[i : i + 1], first_heading_angle[i : i + 1]))
    x1 = torch.cat(rotated, dim=0)
    x1 = motion_rep.normalize(x1)
    return x1, first_heading_angle


def _masked_mse_loss(pred: Tensor, target: Tensor, pad_mask: Tensor) -> Tensor:
    """MSE averaged over valid (non-padded) frames only."""
    frame_mask = pad_mask.unsqueeze(-1).expand_as(pred)
    if not frame_mask.any():
        return F.mse_loss(pred, target)
    return F.mse_loss(pred[frame_mask], target[frame_mask])


def flow_matching_train_step(
    denoiser: nn.Module,
    x1: Tensor,
    pad_mask: Tensor,
    flow_loss: FlowMatchingLoss,
    *,
    text_feat: Optional[Tensor] = None,
    text_pad_mask: Optional[Tensor] = None,
    first_heading_angle: Optional[Tensor] = None,
    motion_mask: Optional[Tensor] = None,
    observed_motion: Optional[Tensor] = None,
) -> tuple[Tensor, dict]:
    """One OT-CFM training step; denoiser predicts velocity v."""
    batch_size = x1.shape[0]
    device = x1.device

    if text_feat is None:
        root_model = denoiser.root_model
        num_tokens = root_model.num_text_tokens
        llm_dim = root_model.embed_text.in_features
        text_feat = torch.zeros(batch_size, num_tokens, llm_dim, device=device)
        text_pad_mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)

    if first_heading_angle is None:
        first_heading_angle = torch.zeros(batch_size, device=device)

    t, xt, ut = flow_loss.sample_path(x1)
    xt = apply_motion_constraints(xt, motion_mask, observed_motion)

    v_pred = denoiser(
        xt,
        pad_mask,
        text_feat,
        text_pad_mask,
        t,
        first_heading_angle=first_heading_angle,
        motion_mask=motion_mask,
        observed_motion=observed_motion,
    )

    loss = _masked_mse_loss(v_pred, ut, pad_mask)
    metrics = {
        "loss": loss.detach(),
        "t_mean": t.mean().detach(),
        "ut_norm": ut.norm(dim=-1).mean().detach(),
        "v_norm": v_pred.norm(dim=-1).mean().detach(),
    }
    return loss, metrics


def flow_matching_batch_step(
    denoiser: nn.Module,
    motion_rep: KimodoMotionRep,
    batch: dict,
    flow_loss: FlowMatchingLoss,
    text_provider,
    *,
    cfg_dropout: Optional[dict] = None,
    constraint_prob: float = 0.8,
    max_keyframes: int = 4,
) -> tuple[Tensor, dict]:
    """Full training step with text, synthetic constraints, and CFG dropout."""
    device = next(denoiser.parameters()).device
    x1 = batch["feats"].to(device)
    pad_mask = batch["pad_mask"].to(device)
    lengths = batch["lengths"].to(device)

    first_heading_angle: Optional[Tensor] = None
    if getattr(denoiser.root_model, "input_first_heading_angle", False):
        x1, first_heading_angle = _randomize_batch_heading(motion_rep, x1)

    if "text_feat" in batch and "text_pad_mask" in batch:
        text_feat = batch["text_feat"].to(device)
        text_pad_mask = batch["text_pad_mask"].to(device)
    else:
        text_feat, text_pad_mask = text_provider.encode(batch["texts"])
    motion_mask, observed_motion = sample_training_constraints(
        motion_rep,
        x1,
        lengths,
        constraint_prob=constraint_prob,
        max_keyframes=max_keyframes,
        device=str(device),
    )

    if cfg_dropout is not None:
        text_feat, text_pad_mask, motion_mask, observed_motion = apply_separated_cfg_dropout(
            text_feat,
            text_pad_mask,
            motion_mask,
            observed_motion,
            p_uncond=cfg_dropout.get("uncond", 0.1),
            p_text_only=cfg_dropout.get("text_only", 0.1),
            p_constraint_only=cfg_dropout.get("constraint_only", 0.1),
        )

    return flow_matching_train_step(
        denoiser,
        x1,
        pad_mask,
        flow_loss,
        text_feat=text_feat,
        text_pad_mask=text_pad_mask,
        first_heading_angle=first_heading_angle,
        motion_mask=motion_mask,
        observed_motion=observed_motion,
    )
