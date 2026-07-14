# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Progressive distillation training step (Salimans & Ho style, x0 + DDIM)."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kimodo.distill.ddim_ops import (
    ddim_jump_two,
    prepare_schedule,
    predict_x0,
    q_sample_on_schedule,
    teacher_two_ddim_steps,
)
from kimodo.model.diffusion import DDIMSampler, Diffusion
from kimodo.model.flow_matching import apply_motion_constraints
from kimodo.motion_rep import KimodoMotionRep
from kimodo.train.cfg_dropout import apply_separated_cfg_dropout
from kimodo.train.constraint_synth import sample_training_constraints


def _denoiser_core(denoiser: nn.Module) -> nn.Module:
    return denoiser.module if hasattr(denoiser, "module") else denoiser


def _masked_mse_loss(pred: Tensor, target: Tensor, pad_mask: Tensor) -> Tensor:
    frame_mask = pad_mask.unsqueeze(-1).expand_as(pred)
    if not frame_mask.any():
        return F.mse_loss(pred, target)
    return F.mse_loss(pred[frame_mask], target[frame_mask])


def _randomize_batch_heading(
    motion_rep: KimodoMotionRep,
    x1: Tensor,
) -> tuple[Tensor, Tensor]:
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


def _zeros_text(denoiser: nn.Module, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
    root_model = _denoiser_core(denoiser).root_model
    num_tokens = root_model.num_text_tokens
    llm_dim = root_model.embed_text.in_features
    text_feat = torch.zeros(batch_size, num_tokens, llm_dim, device=device)
    text_pad_mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)
    return text_feat, text_pad_mask


def progressive_distill_step(
    teacher: nn.Module,
    student: nn.Module,
    diffusion: Diffusion,
    sampler: DDIMSampler,
    x0: Tensor,
    pad_mask: Tensor,
    teacher_steps: int,
    *,
    text_feat: Optional[Tensor] = None,
    text_pad_mask: Optional[Tensor] = None,
    first_heading_angle: Optional[Tensor] = None,
    motion_mask: Optional[Tensor] = None,
    observed_motion: Optional[Tensor] = None,
) -> tuple[Tensor, dict]:
    """One PD step: student one-jump matches two frozen teacher DDIM steps.

    Teacher and student share the teacher ``N``-step schedule time embedding at the
    starting index ``t``; the student DDIM update jumps from ``t`` to ``t-2``.
    """
    if teacher_steps < 2 or teacher_steps % 2 != 0:
        raise ValueError(f"teacher_steps must be even and >= 2, got {teacher_steps}")

    batch_size = x0.shape[0]
    device = x0.device

    if text_feat is None or text_pad_mask is None:
        text_feat, text_pad_mask = _zeros_text(student, batch_size, device)
    if first_heading_angle is None:
        first_heading_angle = torch.zeros(batch_size, device=device)

    use_timesteps, map_tensor = prepare_schedule(diffusion, teacher_steps)
    # Need t >= 2 so that a two-step jump to t-2 is valid.
    t_idx = torch.randint(2, teacher_steps, (batch_size,), device=device, dtype=torch.long)

    noise = torch.randn_like(x0)
    x_t = q_sample_on_schedule(diffusion, x0, t_idx, noise)
    x_t = apply_motion_constraints(x_t, motion_mask, observed_motion)

    with torch.no_grad():
        x_tgt = teacher_two_ddim_steps(
            teacher,
            diffusion,
            sampler,
            x_t,
            t_idx,
            pad_mask=pad_mask,
            text_feat=text_feat,
            text_pad_mask=text_pad_mask,
            first_heading_angle=first_heading_angle,
            motion_mask=motion_mask,
            observed_motion=observed_motion,
            num_steps=teacher_steps,
        )

    # Re-prepare teacher schedule (teacher path mutates buffers via DDIMSampler).
    prepare_schedule(diffusion, teacher_steps)
    t_base = map_tensor[t_idx]
    pred_x0 = predict_x0(
        student,
        x_t,
        t_base,
        pad_mask=pad_mask,
        text_feat=text_feat,
        text_pad_mask=text_pad_mask,
        first_heading_angle=first_heading_angle,
        motion_mask=motion_mask,
        observed_motion=observed_motion,
    )
    x_pred = ddim_jump_two(diffusion, x_t, pred_x0, t_idx)
    x_pred = apply_motion_constraints(x_pred, motion_mask, observed_motion)

    loss = _masked_mse_loss(x_pred, x_tgt, pad_mask)
    metrics = {
        "loss": loss.detach(),
        "t_mean": t_idx.float().mean().detach(),
        "x_pred_norm": x_pred.detach().norm(dim=-1).mean(),
        "x_tgt_norm": x_tgt.detach().norm(dim=-1).mean(),
        "teacher_steps": torch.tensor(float(teacher_steps), device=device),
        "student_steps": torch.tensor(float(teacher_steps // 2), device=device),
    }
    return loss, metrics


def progressive_distill_batch_step(
    teacher: nn.Module,
    student: nn.Module,
    motion_rep: KimodoMotionRep,
    diffusion: Diffusion,
    sampler: DDIMSampler,
    batch: dict,
    teacher_steps: int,
    text_provider,
    *,
    cfg_dropout: Optional[dict] = None,
    constraint_prob: float = 0.0,
    max_keyframes: int = 4,
) -> tuple[Tensor, dict]:
    """Full PD step with optional heading aug, constraints, and CFG dropout."""
    device = next(student.parameters()).device
    x0 = batch["feats"].to(device)
    pad_mask = batch["pad_mask"].to(device)
    lengths = batch["lengths"].to(device)

    first_heading_angle: Optional[Tensor] = None
    if getattr(_denoiser_core(student).root_model, "input_first_heading_angle", False):
        x0, first_heading_angle = _randomize_batch_heading(motion_rep, x0)

    if "text_feat" in batch and "text_pad_mask" in batch:
        text_feat = batch["text_feat"].to(device)
        text_pad_mask = batch["text_pad_mask"].to(device)
    else:
        text_feat, text_pad_mask = text_provider.encode(batch["texts"])

    motion_mask, observed_motion = sample_training_constraints(
        motion_rep,
        x0,
        lengths,
        constraint_prob=constraint_prob,
        max_keyframes=max_keyframes,
        device=str(device),
    )

    if cfg_dropout is not None and any(float(cfg_dropout.get(k, 0.0)) > 0 for k in ("uncond", "text_only", "constraint_only")):
        text_feat, text_pad_mask, motion_mask, observed_motion = apply_separated_cfg_dropout(
            text_feat,
            text_pad_mask,
            motion_mask,
            observed_motion,
            p_uncond=cfg_dropout.get("uncond", 0.0),
            p_text_only=cfg_dropout.get("text_only", 0.0),
            p_constraint_only=cfg_dropout.get("constraint_only", 0.0),
        )

    return progressive_distill_step(
        teacher,
        student,
        diffusion,
        sampler,
        x0,
        pad_mask,
        teacher_steps,
        text_feat=text_feat,
        text_pad_mask=text_pad_mask,
        first_heading_angle=first_heading_angle,
        motion_mask=motion_mask,
        observed_motion=observed_motion,
    )
