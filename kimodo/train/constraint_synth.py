# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic constraints for FM / PD training (fullbody, root2d, EE hands/feet)."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from kimodo.constraints import (
    EndEffectorConstraintSet,
    FullBodyConstraintSet,
    Root2DConstraintSet,
)
from kimodo.motion_rep import KimodoMotionRep

# Default mix for constraint-first PD (EE / root / feet).
DEFAULT_CONSTRAINT_MIX = {
    "fullbody": 0.15,
    "root2d": 0.30,
    "ee_hands": 0.35,
    "ee_feet": 0.20,
}


def _sample_mode(mix: dict[str, float], device: torch.device) -> str:
    keys = list(mix.keys())
    probs = torch.tensor([float(mix[k]) for k in keys], device=device, dtype=torch.float32)
    probs = probs / probs.sum().clamp(min=1e-8)
    return keys[int(torch.multinomial(probs, 1).item())]


def _pick_frames(length: int, max_n: int, device, *, dense: bool = False) -> Tensor:
    if length < 1:
        return torch.zeros(0, device=device, dtype=torch.long)
    if dense:
        # Dense root paths (match demo root2d / waypoints).
        n = min(length, max(16, max_n * 4))
        if n >= length:
            return torch.arange(length, device=device, dtype=torch.long)
        stride = max(1, length // n)
        idx = torch.arange(0, length, stride, device=device, dtype=torch.long)[:n]
        return idx
    n = int(torch.randint(1, min(max_n, length) + 1, (1,)).item())
    return torch.sort(torch.randperm(length, device=device)[:n])[0]


def _pick_contact_frames(foot_contacts: Tensor, length: int, max_n: int, device) -> Tensor:
    """Prefer frames with any foot contact for feet-EE / skate-aligned constraints."""
    fc = foot_contacts[:length]
    if fc.dim() == 2 and fc.shape[-1] >= 1:
        any_c = (fc > 0.5).any(dim=-1)
        cand = torch.where(any_c)[0]
        if cand.numel() >= 1:
            n = int(torch.randint(1, min(max_n, int(cand.numel())) + 1, (1,)).item())
            sel = cand[torch.randperm(cand.numel(), device=device)[:n]]
            return torch.sort(sel)[0]
    return _pick_frames(length, max_n, device, dense=False)


def _is_fullbody_only(mix: dict) -> bool:
    """True when the mix is Stage-1 style: only FullBody constraints."""
    active = {k for k, v in mix.items() if float(v) > 0}
    return active == {"fullbody"}


def sample_training_constraints(
    motion_rep: KimodoMotionRep,
    feats: Tensor,
    lengths: Tensor,
    *,
    constraint_prob: float = 0.8,
    max_keyframes: int = 4,
    device: Optional[str] = None,
    constraint_mix: Optional[dict] = None,
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Build batched constraint tensors from ground-truth normalized features.

    Returns:
        ``(motion_mask, observed_motion)`` — same order as
        ``apply_motion_constraints`` / CFG dropout / FM & PD trainers.

    When ``constraint_mix`` is FullBody-only (e.g. ``{fullbody: 1.0}``), sampling
    matches Stage-1 ``formal_cons`` / ``stage_100to50_best`` exactly: random
    keyframe count on ``feats.device``, FullBodyConstraintSet only.
    """
    if constraint_prob <= 0:
        return None, None

    mix = dict(DEFAULT_CONSTRAINT_MIX if constraint_mix is None else constraint_mix)
    batch_size, max_len, _ = feats.shape
    out_device = device or str(feats.device)
    skeleton = motion_rep.skeleton
    fullbody_only = _is_fullbody_only(mix)
    # Mixed modes keep constraint tensors on CPU (demo / EE loaders); FullBody-only
    # stays on feats.device to match Stage-1 formal_cons.
    idx_device = feats.device if fullbody_only else torch.device("cpu")

    constraints_per_sample: list[list] = []
    has_any = False

    for b in range(batch_size):
        length = int(lengths[b].item())
        if length < 1 or torch.rand(1).item() > constraint_prob:
            constraints_per_sample.append([])
            continue

        if fullbody_only:
            # Stage-1 path (pd-100to50-formal-cons-v4 / stage_100to50_best).
            num_keyframes = int(torch.randint(1, min(max_keyframes, length) + 1, (1,)).item())
            frame_indices = torch.sort(torch.randperm(length, device=feats.device)[:num_keyframes])[0]
            decoded = motion_rep.inverse(feats[b : b + 1, :length], is_normalized=True)
            posed_joints = decoded["posed_joints"][0, frame_indices]
            global_rot_mats = decoded["global_rot_mats"][0, frame_indices]
            smooth_root_2d = decoded["smooth_root_pos"][0, frame_indices][..., [0, 2]]
            constraints_per_sample.append(
                [
                    FullBodyConstraintSet(
                        skeleton,
                        frame_indices.to(device=feats.device),
                        posed_joints,
                        global_rot_mats,
                        smooth_root_2d=smooth_root_2d,
                    )
                ]
            )
            has_any = True
            continue

        mode = _sample_mode(mix, feats.device)
        decoded = motion_rep.inverse(feats[b : b + 1, :length], is_normalized=True)
        posed_joints = decoded["posed_joints"][0].detach().cpu()
        global_rot_mats = decoded["global_rot_mats"][0].detach().cpu()
        smooth_root_2d_full = decoded["smooth_root_pos"][0][..., [0, 2]].detach().cpu()
        foot_contacts = decoded.get("foot_contacts")
        if foot_contacts is not None:
            foot_contacts = foot_contacts[0].detach().cpu()

        if mode == "root2d":
            frame_indices = _pick_frames(length, max_keyframes, idx_device, dense=True)
            cons = Root2DConstraintSet(
                skeleton,
                frame_indices,
                smooth_root_2d_full[frame_indices],
            )
            constraints_per_sample.append([cons])
        elif mode == "ee_hands":
            frame_indices = _pick_frames(length, max_keyframes, idx_device, dense=False)
            # Match demo EE style: one or both hands.
            if torch.rand(1).item() < 0.5:
                joint_names = ["LeftHand"] if torch.rand(1).item() < 0.5 else ["RightHand"]
            else:
                joint_names = ["LeftHand", "RightHand"]
            cons = EndEffectorConstraintSet(
                skeleton,
                frame_indices,
                posed_joints[frame_indices],
                global_rot_mats[frame_indices],
                smooth_root_2d_full[frame_indices],
                joint_names=joint_names,
            )
            constraints_per_sample.append([cons])
        elif mode == "ee_feet":
            if foot_contacts is not None:
                frame_indices = _pick_contact_frames(foot_contacts, length, max_keyframes, idx_device)
            else:
                frame_indices = _pick_frames(length, max_keyframes, idx_device, dense=False)
            if torch.rand(1).item() < 0.5:
                joint_names = ["LeftFoot"] if torch.rand(1).item() < 0.5 else ["RightFoot"]
            else:
                joint_names = ["LeftFoot", "RightFoot"]
            cons = EndEffectorConstraintSet(
                skeleton,
                frame_indices,
                posed_joints[frame_indices],
                global_rot_mats[frame_indices],
                smooth_root_2d_full[frame_indices],
                joint_names=joint_names,
            )
            constraints_per_sample.append([cons])
        else:
            frame_indices = _pick_frames(length, max_keyframes, idx_device, dense=False)
            cons = FullBodyConstraintSet(
                skeleton,
                frame_indices,
                posed_joints[frame_indices],
                global_rot_mats[frame_indices],
                smooth_root_2d=smooth_root_2d_full[frame_indices],
            )
            constraints_per_sample.append([cons])
        has_any = True

    if not has_any:
        return None, None

    observed_motion, motion_mask = motion_rep.create_conditions_from_constraints_batched(
        constraints_per_sample,
        lengths,
        to_normalize=True,
        device=out_device,
    )
    # Callers expect (mask, observed); create_conditions returns (observed, mask).
    return motion_mask.to(dtype=observed_motion.dtype), observed_motion
