# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Flow Matching path sampling (torchcfm) and ODE integration for inference."""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
from torch import Tensor, nn


def sinusoidal_timestep_embedding(timesteps: Tensor, dim: int) -> Tensor:
    """Sinusoidal embedding for continuous t in [0, 1]. Returns [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = torch.nn.functional.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class FlowMatchingLoss(nn.Module):
    """OT-CFM training path sampler via torchcfm."""

    def __init__(self, sigma: float = 0.0, matcher: str = "ot_cfm"):
        super().__init__()
        self.sigma = sigma
        self.matcher = self._build_matcher(matcher, sigma)

    @staticmethod
    def _build_matcher(matcher: str, sigma: float):
        from torchcfm.conditional_flow_matching import (
            ConditionalFlowMatcher,
            ExactOptimalTransportConditionalFlowMatcher,
        )

        if matcher == "ot_cfm":
            return ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        if matcher == "cfm":
            return ConditionalFlowMatcher(sigma=sigma)
        raise ValueError(f"Unknown flow matcher: {matcher}")

    def sample_path(self, x1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Sample (t, xt, ut) for conditional flow matching loss.

        Args:
            x1: Data samples [B, *dim].

        Returns:
            t [B], xt [B, *dim], ut [B, *dim]
        """
        x0 = torch.randn_like(x1)
        t, xt, ut = self.matcher.sample_location_and_conditional_flow(x0, x1)
        return t, xt, ut


def apply_motion_constraints(
    x: Tensor,
    motion_mask: Optional[Tensor],
    observed_motion: Optional[Tensor],
) -> Tensor:
    """Hard inpainting: overwrite constrained dimensions."""
    if motion_mask is None or observed_motion is None:
        return x
    # create_conditions returns a Boolean mask; arithmetic needs float/int.
    mask = motion_mask.to(dtype=x.dtype)
    return x * (1 - mask) + observed_motion.to(dtype=x.dtype) * mask


class EulerODESolver:
    """Deterministic Euler integration from t=1 (noise) to t=0 (data)."""

    def integrate(
        self,
        velocity_fn: Callable[[Tensor, Tensor], Tensor],
        x_init: Tensor,
        num_steps: int,
        motion_mask: Optional[Tensor] = None,
        observed_motion: Optional[Tensor] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Tensor:
        """Integrate dx/dt = -v(x, t) from t=1 toward t=0.

        Args:
            velocity_fn: Callable (x, t_batch) -> velocity [B, T, D].
            x_init: Initial state at t=1, typically N(0, I).
            num_steps: Number of Euler steps.
            motion_mask: Optional constraint mask.
            observed_motion: Optional constrained values.
            progress_callback: Optional hook called each step.

        Returns:
            x at t≈0 with shape [B, T, D].
        """
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")

        x = x_init
        dt = 1.0 / num_steps
        device = x.device
        batch_size = x.shape[0]

        for i in range(num_steps):
            t_val = 1.0 - i * dt
            t_batch = torch.full((batch_size,), t_val, device=device, dtype=x.dtype)
            x = apply_motion_constraints(x, motion_mask, observed_motion)
            v = velocity_fn(x, t_batch)
            x = x - dt * v
            if progress_callback is not None:
                progress_callback(i, num_steps)

        return apply_motion_constraints(x, motion_mask, observed_motion)
