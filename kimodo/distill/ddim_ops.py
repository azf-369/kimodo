# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DDIM helpers for progressive distillation (mirrors Kimodo inference)."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from kimodo.model.diffusion import DDIMSampler, Diffusion
from kimodo.model.flow_matching import apply_motion_constraints


def prepare_schedule(diffusion: Diffusion, num_steps: int) -> tuple[Tensor, Tensor]:
    """Return (use_timesteps, map_tensor) and refresh diffusion buffers for num_steps."""
    use_timesteps, map_tensor = diffusion.space_timesteps(num_steps)
    diffusion.calc_diffusion_vars(use_timesteps)
    return use_timesteps, map_tensor


def q_sample_on_schedule(
    diffusion: Diffusion,
    x0: Tensor,
    t_idx: Tensor,
    noise: Optional[Tensor] = None,
) -> Tensor:
    """Forward-diffuse x0 at subsampled schedule indices ``t_idx`` (current buffers)."""
    if noise is None:
        noise = torch.randn_like(x0)
    return (
        diffusion.sqrt_alphas_cumprod[t_idx, None, None] * x0
        + diffusion.sqrt_one_minus_alphas_cumprod[t_idx, None, None] * noise
    )


def predict_x0(
    denoiser: nn.Module,
    x: Tensor,
    t_base: Tensor,
    *,
    pad_mask: Tensor,
    text_feat: Tensor,
    text_pad_mask: Tensor,
    first_heading_angle: Optional[Tensor],
    motion_mask: Optional[Tensor],
    observed_motion: Optional[Tensor],
    match_param_dtype: bool = False,
) -> Tensor:
    """Denoiser predicts clean motion x0 at base timestep indices ``t_base``."""
    if match_param_dtype:
        dtype = next(denoiser.parameters()).dtype

        def _cast(t: Optional[Tensor]) -> Optional[Tensor]:
            if t is None or not torch.is_floating_point(t):
                return t
            return t.to(dtype=dtype)

        x = _cast(x)
        text_feat = _cast(text_feat)
        first_heading_angle = _cast(first_heading_angle)
        # motion_mask often float 0/1 in concat mode
        motion_mask = _cast(motion_mask)
        observed_motion = _cast(observed_motion)

    return denoiser(
        x,
        pad_mask,
        text_feat,
        text_pad_mask,
        t_base,
        first_heading_angle=first_heading_angle,
        motion_mask=motion_mask,
        observed_motion=observed_motion,
    )


def ddim_step(
    sampler: DDIMSampler,
    use_timesteps: Tensor,
    x_t: Tensor,
    pred_x0: Tensor,
    t_idx: Tensor,
) -> Tensor:
    """One deterministic DDIM step at subsampled indices ``t_idx``."""
    return sampler(use_timesteps, x_t, pred_x0, t_idx)


def ddim_jump_two(
    diffusion: Diffusion,
    x_t: Tensor,
    pred_x0: Tensor,
    t_idx: Tensor,
) -> Tensor:
    """Compress two DDIM steps into one update from ``t`` to ``t-2`` (η=0).

    Uses the same ε reconstruction as ``DDIMSampler``, but targets
    ``alphas_cumprod_prev[t-1]`` (noise level after two steps).
    """
    eps = (
        diffusion.sqrt_recip_alphas_cumprod[t_idx, None, None] * x_t - pred_x0
    ) / diffusion.sqrt_recipm1_alphas_cumprod[t_idx, None, None]
    # After two steps from t, alpha bar equals alphas_cumprod_prev[t-1].
    alpha_bar_tm2 = diffusion.alphas_cumprod_prev[t_idx - 1, None, None]
    return pred_x0 * torch.sqrt(alpha_bar_tm2) + torch.sqrt(1.0 - alpha_bar_tm2) * eps


def invert_ddim_jump_two_x0(
    diffusion: Diffusion,
    x_t: Tensor,
    x_tm2: Tensor,
    t_idx: Tensor,
) -> Tensor:
    """Solve for ``pred_x0`` such that ``ddim_jump_two(x_t, pred_x0, t) == x_tm2``.

    Matches Progressive Distillation in the network's native x0 output space, which
    is better-conditioned than raw noisy-state MSE at high noise.
    """
    # alpha_bar after two DDIM steps from t equals alphas_cumprod_prev[t-1].
    a_tm2 = diffusion.alphas_cumprod_prev[t_idx - 1].clamp(min=1e-8, max=1.0 - 1e-8)
    A = torch.sqrt(a_tm2)[:, None, None]
    B = torch.sqrt(1.0 - a_tm2)[:, None, None]
    Sr = diffusion.sqrt_recip_alphas_cumprod[t_idx][:, None, None]
    Sm = diffusion.sqrt_recipm1_alphas_cumprod[t_idx][:, None, None].clamp(min=1e-8)
    denom = A - B / Sm
    denom = torch.where(denom.abs() < 1e-6, torch.full_like(denom, 1e-6), denom)
    return (x_tm2 - (B * Sr / Sm) * x_t) / denom


def teacher_two_ddim_steps(
    teacher: nn.Module,
    diffusion: Diffusion,
    sampler: DDIMSampler,
    x_t: Tensor,
    t_idx: Tensor,
    *,
    pad_mask: Tensor,
    text_feat: Tensor,
    text_pad_mask: Tensor,
    first_heading_angle: Optional[Tensor],
    motion_mask: Optional[Tensor],
    observed_motion: Optional[Tensor],
    num_steps: int,
) -> Tensor:
    """Frozen teacher executes two DDIM steps: t -> t-1 -> t-2."""
    use_timesteps, map_tensor = prepare_schedule(diffusion, num_steps)
    x = apply_motion_constraints(x_t, motion_mask, observed_motion)

    t_base = map_tensor[t_idx]
    pred1 = predict_x0(
        teacher,
        x,
        t_base,
        pad_mask=pad_mask,
        text_feat=text_feat,
        text_pad_mask=text_pad_mask,
        first_heading_angle=first_heading_angle,
        motion_mask=motion_mask,
        observed_motion=observed_motion,
        match_param_dtype=True,
    )
    x = ddim_step(sampler, use_timesteps, x, pred1.float(), t_idx)
    x = apply_motion_constraints(x, motion_mask, observed_motion)

    t1 = t_idx - 1
    t_base1 = map_tensor[t1]
    pred2 = predict_x0(
        teacher,
        x,
        t_base1,
        pad_mask=pad_mask,
        text_feat=text_feat,
        text_pad_mask=text_pad_mask,
        first_heading_angle=first_heading_angle,
        motion_mask=motion_mask,
        observed_motion=observed_motion,
        match_param_dtype=True,
    )
    x = ddim_step(sampler, use_timesteps, x, pred2.float(), t1)
    return apply_motion_constraints(x, motion_mask, observed_motion)
