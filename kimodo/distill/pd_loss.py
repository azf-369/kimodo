# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Progressive distillation training step (Salimans & Ho style, x0 + DDIM).

Stabilized recipe (post frozen-motion diagnosis):
  - Match in **x0 space** (invert jump target) instead of raw noisy ``x_{t-2}`` MSE
  - Optional **Min-SNR-γ** weighting
  - Optional **GT x0 diffusion anchor** so pretrained sampling is not destroyed
  - Optional soft match to teacher's same-t x0 prediction
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from kimodo.distill.ddim_ops import (
    ddim_jump_two,
    invert_ddim_jump_two_x0,
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


def _masked_mse_loss(
    pred: Tensor,
    target: Tensor,
    pad_mask: Tensor,
    weight: Optional[Tensor] = None,
) -> Tensor:
    """Frame-masked MSE; optional per-sample ``weight`` of shape (B,)."""
    frame_mask = pad_mask.unsqueeze(-1).expand_as(pred)
    if not frame_mask.any():
        err = (pred - target).pow(2).mean(dim=(1, 2))
    else:
        # Mean over valid (frame, feat) per batch element, then weight, then mean.
        diff2 = (pred - target).pow(2)
        valid = frame_mask.to(dtype=diff2.dtype)
        denom = valid.sum(dim=(1, 2)).clamp(min=1.0)
        err = (diff2 * valid).sum(dim=(1, 2)) / denom
    if weight is not None:
        err = err * weight.reshape(-1).to(dtype=err.dtype)
    return err.mean()


def _constraint_mse_loss(
    pred: Tensor,
    observed: Tensor,
    motion_mask: Tensor,
    pad_mask: Tensor,
) -> Tensor:
    """MSE only on constrained (motion_mask>0) locs ∩ valid frames.

    Sparse keyframes are ~1–4 frames; without this term uniform PD/diffuse MSE
    nearly ignores constraint dims, and eval constraint_* metrics degrade.
    """
    mask = motion_mask.to(dtype=pred.dtype) * pad_mask.unsqueeze(-1).to(dtype=pred.dtype)
    denom = mask.sum().clamp(min=1.0)
    return ((pred - observed).pow(2) * mask).sum() / denom


def _channel_mse(
    pred: Tensor,
    target: Tensor,
    pad_mask: Tensor,
    channel_slice: slice,
    *,
    sub_idx: Optional[list[int]] = None,
) -> Tensor:
    """Frame-masked MSE on a feature slice (optional sub-indices within the slice)."""
    p = pred[..., channel_slice]
    t = target[..., channel_slice]
    if sub_idx is not None:
        idx = torch.tensor(sub_idx, device=pred.device, dtype=torch.long)
        p = p.index_select(-1, idx)
        t = t.index_select(-1, idx)
    return _masked_mse_loss(p, t, pad_mask)


def _ee_joint_indices(motion_rep: KimodoMotionRep) -> list[int]:
    """Hand + foot position joint indices (demo EE set) for local_joints_positions."""
    sk = motion_rep.skeleton
    names = ["LeftHand", "RightHand", "LeftFoot", "RightFoot"]
    idxs: list[int] = []
    for n in names:
        try:
            _, pos_names = sk.expand_joint_names([n])
        except Exception:
            continue
        for jn in pos_names:
            if jn in sk.bone_index:
                idxs.append(int(sk.bone_index[jn]))
    # Stable unique order.
    return sorted(set(idxs))


def _ee_local_pos_mse(
    pred: Tensor,
    gt: Tensor,
    pad_mask: Tensor,
    motion_rep: KimodoMotionRep,
) -> Tensor:
    """MSE on EE local joint XYZ (hands+feet) vs GT, all valid frames."""
    if "local_joints_positions" not in motion_rep.slice_dict:
        return pred.new_zeros(())
    jidx = _ee_joint_indices(motion_rep)
    if not jidx:
        return pred.new_zeros(())
    sl = motion_rep.slice_dict["local_joints_positions"]
    nbj = int(motion_rep.skeleton.nbjoints)
    pred_j = pred[..., sl].reshape(pred.shape[0], pred.shape[1], nbj, 3)
    gt_j = gt[..., sl].reshape(gt.shape[0], gt.shape[1], nbj, 3)
    idx = torch.tensor(jidx, device=pred.device, dtype=torch.long)
    p = pred_j.index_select(2, idx).reshape(pred.shape[0], pred.shape[1], -1)
    t = gt_j.index_select(2, idx).reshape(gt.shape[0], gt.shape[1], -1)
    return _masked_mse_loss(p, t, pad_mask)


def _foot_skate_feature_loss(
    pred: Tensor,
    gt: Tensor,
    pad_mask: Tensor,
    motion_rep: KimodoMotionRep,
) -> Tensor:
    """Penalize horizontal foot velocity when GT says the foot is in contact.

    Operates in **unnormalized** feature space (contacts ~0/1, velocities physical-ish).
    Aligns with FootSkateFromContacts / FootSkateRatio intent without FK.
    """
    if "foot_contacts" not in motion_rep.slice_dict or "velocities" not in motion_rep.slice_dict:
        return pred.new_zeros(())

    pred_u = motion_rep.unnormalize(pred)
    gt_u = motion_rep.unnormalize(gt)
    fc_sl = motion_rep.slice_dict["foot_contacts"]
    vel_sl = motion_rep.slice_dict["velocities"]
    contacts = gt_u[..., fc_sl].clamp(0.0, 1.0)  # [B,T,4]
    nbj = int(motion_rep.skeleton.nbjoints)
    vel = pred_u[..., vel_sl].reshape(pred.shape[0], pred.shape[1], nbj, 3)
    fidx = list(motion_rep.skeleton.foot_joint_idx)
    if len(fidx) != contacts.shape[-1]:
        # Fallback: match min length.
        n = min(len(fidx), contacts.shape[-1])
        fidx = fidx[:n]
        contacts = contacts[..., :n]
    foot_vel = vel.index_select(2, torch.tensor(fidx, device=pred.device, dtype=torch.long))
    horiz = foot_vel[..., 0].pow(2) + foot_vel[..., 2].pow(2)  # [B,T,F]
    frame = pad_mask.to(dtype=pred.dtype).unsqueeze(-1)
    w = contacts * frame
    denom = w.sum().clamp(min=1.0)
    return (horiz * w).sum() / denom


def _min_snr_weight(diffusion: Diffusion, t_idx: Tensor, snr_gamma: float) -> Tensor:
    """Per-sample Min-SNR-γ weight for **x0-space** losses (Hang et al.).

    ``w = min(SNR, γ) / SNR`` with SNR = ᾱ / (1-ᾱ). γ<=0 disables (all ones).
    """
    if snr_gamma is None or float(snr_gamma) <= 0:
        return torch.ones(t_idx.shape[0], device=t_idx.device, dtype=torch.float32)
    alpha = diffusion.alphas_cumprod[t_idx].clamp(min=1e-8, max=1.0 - 1e-8)
    snr = alpha / (1.0 - alpha)
    gamma = float(snr_gamma)
    return (torch.minimum(snr, snr.new_full((), gamma)) / snr).to(dtype=torch.float32)


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
    motion_rep: Optional[KimodoMotionRep] = None,
    text_feat: Optional[Tensor] = None,
    text_pad_mask: Optional[Tensor] = None,
    first_heading_angle: Optional[Tensor] = None,
    motion_mask: Optional[Tensor] = None,
    observed_motion: Optional[Tensor] = None,
    pd_match_space: str = "x0",
    snr_gamma: float = 5.0,
    pd_jump_weight: float = 1.0,
    diffuse_anchor_weight: float = 0.25,
    teacher_x0_weight: float = 0.1,
    constraint_anchor_weight: float = 0.0,
    root_xz_weight: float = 0.0,
    ee_pos_weight: float = 0.0,
    contact_weight: float = 0.0,
    skate_weight: float = 0.0,
) -> tuple[Tensor, dict]:
    """One PD step: student one-jump matches two frozen teacher DDIM steps.

    Extra anchors (when ``motion_rep`` is set):
      - **root_xz**: MSE on smooth_root XZ (root path)
      - **ee_pos**: MSE on hand/foot local joint XYZ
      - **contact**: MSE on foot_contact channels
      - **skate**: contact-weighted horizontal foot velocity (anti-skate)
    """
    if teacher_steps < 2 or teacher_steps % 2 != 0:
        raise ValueError(f"teacher_steps must be even and >= 2, got {teacher_steps}")
    match_space = str(pd_match_space).lower().strip()
    if match_space not in {"x0", "xt"}:
        raise ValueError(f"pd_match_space must be 'x0' or 'xt', got {pd_match_space!r}")

    batch_size = x0.shape[0]
    device = x0.device

    if text_feat is None or text_pad_mask is None:
        text_feat, text_pad_mask = _zeros_text(student, batch_size, device)
    if first_heading_angle is None:
        first_heading_angle = torch.zeros(batch_size, device=device)

    _, map_tensor = prepare_schedule(diffusion, teacher_steps)
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
        # Teacher same-t x0 (soft distill target); refresh schedule after teacher path.
        prepare_schedule(diffusion, teacher_steps)
        t_base = map_tensor[t_idx]
        teacher_pred_x0 = predict_x0(
            teacher,
            x_t,
            t_base,
            pad_mask=pad_mask,
            text_feat=text_feat,
            text_pad_mask=text_pad_mask,
            first_heading_angle=first_heading_angle,
            motion_mask=motion_mask,
            observed_motion=observed_motion,
            match_param_dtype=True,
        ).float()
        if match_space == "x0":
            prepare_schedule(diffusion, teacher_steps)
            x0_tgt = invert_ddim_jump_two_x0(diffusion, x_t, x_tgt, t_idx)

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

    snr_w = _min_snr_weight(diffusion, t_idx, float(snr_gamma))

    if match_space == "x0":
        loss_pd = _masked_mse_loss(pred_x0, x0_tgt, pad_mask, weight=snr_w)
    else:
        loss_pd = _masked_mse_loss(x_pred, x_tgt, pad_mask, weight=snr_w)

    loss_diffuse = pred_x0.new_zeros(())
    if float(diffuse_anchor_weight) > 0:
        loss_diffuse = _masked_mse_loss(pred_x0, x0, pad_mask)

    loss_teacher_x0 = pred_x0.new_zeros(())
    if float(teacher_x0_weight) > 0:
        loss_teacher_x0 = _masked_mse_loss(pred_x0, teacher_pred_x0, pad_mask, weight=snr_w)

    loss_constraint = pred_x0.new_zeros(())
    cons_frac = pred_x0.new_zeros(())
    if (
        float(constraint_anchor_weight) > 0
        and motion_mask is not None
        and observed_motion is not None
        and motion_mask.to(dtype=pred_x0.dtype).sum() > 0
    ):
        loss_constraint = _constraint_mse_loss(pred_x0, observed_motion, motion_mask, pad_mask)
        cons_frac = (motion_mask.to(dtype=pred_x0.dtype) * pad_mask.unsqueeze(-1).to(dtype=pred_x0.dtype)).mean()

    loss_root = pred_x0.new_zeros(())
    loss_ee = pred_x0.new_zeros(())
    loss_contact = pred_x0.new_zeros(())
    loss_skate = pred_x0.new_zeros(())
    if motion_rep is not None:
        if float(root_xz_weight) > 0 and "smooth_root_pos" in motion_rep.slice_dict:
            loss_root = _channel_mse(
                pred_x0,
                x0,
                pad_mask,
                motion_rep.slice_dict["smooth_root_pos"],
                sub_idx=[0, 2],
            )
        if float(ee_pos_weight) > 0:
            loss_ee = _ee_local_pos_mse(pred_x0, x0, pad_mask, motion_rep)
        if float(contact_weight) > 0 and "foot_contacts" in motion_rep.slice_dict:
            loss_contact = _channel_mse(
                pred_x0,
                x0,
                pad_mask,
                motion_rep.slice_dict["foot_contacts"],
            )
        if float(skate_weight) > 0:
            loss_skate = _foot_skate_feature_loss(pred_x0, x0, pad_mask, motion_rep)

    loss = (
        float(pd_jump_weight) * loss_pd
        + float(diffuse_anchor_weight) * loss_diffuse
        + float(teacher_x0_weight) * loss_teacher_x0
        + float(constraint_anchor_weight) * loss_constraint
        + float(root_xz_weight) * loss_root
        + float(ee_pos_weight) * loss_ee
        + float(contact_weight) * loss_contact
        + float(skate_weight) * loss_skate
    )

    with torch.no_grad():
        pred_norm = pred_x0.detach().norm(dim=-1).mean()
        gt_norm = x0.detach().norm(dim=-1).mean()

    metrics = {
        "loss": loss.detach(),
        "loss_pd": loss_pd.detach(),
        "loss_diffuse": loss_diffuse.detach(),
        "loss_teacher_x0": loss_teacher_x0.detach(),
        "loss_constraint": loss_constraint.detach(),
        "loss_root": loss_root.detach(),
        "loss_ee": loss_ee.detach(),
        "loss_contact": loss_contact.detach(),
        "loss_skate": loss_skate.detach(),
        "constraint_frac": cons_frac.detach(),
        "t_mean": t_idx.float().mean().detach(),
        "x_pred_norm": x_pred.detach().norm(dim=-1).mean(),
        "x_tgt_norm": x_tgt.detach().norm(dim=-1).mean(),
        "pred_x0_norm": pred_norm,
        "gt_x0_norm": gt_norm,
        "snr_w_mean": snr_w.mean().detach(),
        "teacher_steps": torch.tensor(float(teacher_steps), device=device),
        "student_steps": torch.tensor(float(teacher_steps // 2), device=device),
        "pd_match_space": torch.tensor(1.0 if match_space == "x0" else 0.0, device=device),
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
    constraint_mix: Optional[dict] = None,
    pd_match_space: str = "x0",
    snr_gamma: float = 5.0,
    pd_jump_weight: float = 1.0,
    diffuse_anchor_weight: float = 0.25,
    teacher_x0_weight: float = 0.1,
    constraint_anchor_weight: float = 0.0,
    root_xz_weight: float = 0.0,
    ee_pos_weight: float = 0.0,
    contact_weight: float = 0.0,
    skate_weight: float = 0.0,
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
        constraint_mix=constraint_mix,
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
        motion_rep=motion_rep,
        text_feat=text_feat,
        text_pad_mask=text_pad_mask,
        first_heading_angle=first_heading_angle,
        motion_mask=motion_mask,
        observed_motion=observed_motion,
        pd_match_space=pd_match_space,
        snr_gamma=snr_gamma,
        pd_jump_weight=pd_jump_weight,
        diffuse_anchor_weight=diffuse_anchor_weight,
        teacher_x0_weight=teacher_x0_weight,
        constraint_anchor_weight=constraint_anchor_weight,
        root_xz_weight=root_xz_weight,
        ee_pos_weight=ee_pos_weight,
        contact_weight=contact_weight,
        skate_weight=skate_weight,
    )
