# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compute and persist Kimodo motion normalization statistics."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from kimodo.motion_rep import KimodoMotionRep
from kimodo.motion_rep.stats import Stats
from kimodo.train.collate import collate_motion_batch
from kimodo.train.dataset import G1SeedTrainingDataset


def _online_mean_std(accum: dict, tensor: torch.Tensor, mask: torch.Tensor) -> None:
    """Update running mean/variance with masked frames."""
    flat = tensor[mask]
    if flat.numel() == 0:
        return
    n_new = flat.shape[0]
    mean_new = flat.mean(dim=0)
    var_new = flat.var(dim=0, unbiased=False)

    if accum["count"] == 0:
        accum["count"] = n_new
        accum["mean"] = mean_new
        accum["m2"] = var_new * n_new
        return

    n_old = accum["count"]
    mean_old = accum["mean"]
    delta = mean_new - mean_old
    n_total = n_old + n_new
    accum["mean"] = mean_old + delta * (n_new / n_total)
    accum["m2"] = accum["m2"] + var_new * n_new + delta.pow(2) * n_old * n_new / n_total
    accum["count"] = n_total


def _finalize_std(accum: dict) -> torch.Tensor:
    if accum["count"] < 2:
        return torch.ones_like(accum["mean"])
    return torch.sqrt(accum["m2"] / accum["count"]).clamp_min(1e-6)


def compute_motion_stats(
    dataset: G1SeedTrainingDataset,
    motion_rep: KimodoMotionRep,
    *,
    batch_size: int = 8,
    num_workers: int = 0,
    max_batches: int | None = None,
) -> KimodoMotionRep:
    """Compute global_root, local_root, and body stats from encoded training clips."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_motion_batch,
    )

    global_accum = {"count": 0, "mean": None, "m2": None}
    body_accum = {"count": 0, "mean": None, "m2": None}
    local_accum = {"count": 0, "mean": None, "m2": None}

    gr_dim = motion_rep.global_root_dim
    body_start = motion_rep.global_root_dim

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        feats = batch["feats"]
        pad_mask = batch["pad_mask"]
        lengths = batch["lengths"]
        b, t, _ = feats.shape
        frame_mask = pad_mask.unsqueeze(-1).expand(b, t, feats.shape[-1])

        global_block = feats[..., :gr_dim]
        body_block = feats[..., body_start:]
        _online_mean_std(global_accum, global_block, frame_mask[..., :gr_dim])
        _online_mean_std(body_accum, body_block, frame_mask[..., body_start:])

        local_root_feats = motion_rep.global_root_to_local_root(
            feats[..., :gr_dim],
            normalized=False,
            lengths=lengths,
        )
        local_mask = pad_mask.unsqueeze(-1).expand_as(local_root_feats)
        _online_mean_std(local_accum, local_root_feats, local_mask)

    motion_rep.global_root_stats = Stats(load=False)
    motion_rep.global_root_stats.register_from_tensors(global_accum["mean"], _finalize_std(global_accum))

    motion_rep.body_stats = Stats(load=False)
    motion_rep.body_stats.register_from_tensors(body_accum["mean"], _finalize_std(body_accum))

    motion_rep.local_root_stats = Stats(load=False)
    motion_rep.local_root_stats.register_from_tensors(local_accum["mean"], _finalize_std(local_accum))

    mean = torch.cat([motion_rep.global_root_stats.mean, motion_rep.body_stats.mean])
    std = torch.cat([motion_rep.global_root_stats.std, motion_rep.body_stats.std])
    motion_rep.stats = Stats(load=False)
    motion_rep.stats.register_from_tensors(mean, std)
    return motion_rep


def save_motion_stats(motion_rep: KimodoMotionRep, output_dir: str | Path) -> Path:
    """Save stats in the Kimodo checkpoint layout."""
    output_dir = Path(output_dir)
    for name, stats in (
        ("global_root", motion_rep.global_root_stats),
        ("local_root", motion_rep.local_root_stats),
        ("body", motion_rep.body_stats),
    ):
        folder = output_dir / name
        os.makedirs(folder, exist_ok=True)
        np.save(folder / "mean.npy", stats.mean.cpu().numpy())
        np.save(folder / "std.npy", stats.std.cpu().numpy())
    return output_dir
