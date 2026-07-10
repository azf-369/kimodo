# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Disk cache for LLM2Vec text embeddings used during Flow Matching training."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from tqdm.auto import tqdm

from kimodo.train.text_embedding import TextEmbeddingProvider

CACHE_VERSION = 1


def cache_path_for_rel_path(cache_dir: Path, rel_path: str) -> Path:
    """Map a dataset rel_path to a stable cache file."""
    safe = rel_path.replace("/", "__")
    return cache_dir / f"{safe}.pt"


def save_text_embedding(
    cache_path: Path,
    *,
    text_feat: Tensor,
    text_pad_mask: Tensor,
    text: str,
    rel_path: str,
) -> None:
    """Save a compact single-token embedding (LLM2Vec returns one vector per clip)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    valid = text_pad_mask.nonzero(as_tuple=False).squeeze(-1)
    if valid.numel() == 0:
        compact_feat = text_feat[:1].cpu()
    else:
        compact_feat = text_feat[valid].cpu()
    torch.save(
        {
            "version": CACHE_VERSION,
            "rel_path": rel_path,
            "text": text,
            "text_feat": compact_feat,
        },
        cache_path,
    )


def load_text_embedding(cache_path: Path, *, num_tokens: int, llm_dim: int) -> tuple[Tensor, Tensor]:
    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    compact_feat = data["text_feat"].float()
    if compact_feat.dim() == 1:
        compact_feat = compact_feat.unsqueeze(0)

    text_feat = torch.zeros(num_tokens, llm_dim, dtype=torch.float32)
    text_pad_mask = torch.zeros(num_tokens, dtype=torch.bool)
    tokens = min(compact_feat.shape[0], num_tokens)
    text_feat[:tokens] = compact_feat[:tokens]
    text_pad_mask[:tokens] = True
    return text_feat, text_pad_mask


def count_missing_embeddings(samples: list[dict], cache_dir: Path) -> int:
    missing = 0
    for sample in samples:
        cache_path = cache_path_for_rel_path(cache_dir, sample["rel_path"])
        if not cache_path.is_file():
            missing += 1
    return missing


def precompute_text_embeddings(
    samples: list[dict],
    *,
    cache_dir: Path,
    encoder_cfg: dict,
    num_tokens: int,
    llm_dim: int,
    device: torch.device,
    skip_existing: bool = True,
) -> int:
    """Encode all training prompts once and store them on disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    provider = TextEmbeddingProvider(
        num_tokens=num_tokens,
        llm_dim=llm_dim,
        device=device,
        mode="encoder",
        encoder_cfg=encoder_cfg,
    )

    to_compute: list[dict] = []
    for sample in samples:
        cache_path = cache_path_for_rel_path(cache_dir, sample["rel_path"])
        if skip_existing and cache_path.is_file():
            continue
        to_compute.append(sample)

    if not to_compute:
        return 0

    for sample in tqdm(to_compute, desc="Precomputing text embeddings", unit="clip"):
        text_feat, text_pad_mask = provider.encode([sample["text"]])
        save_text_embedding(
            cache_path_for_rel_path(cache_dir, sample["rel_path"]),
            text_feat=text_feat[0],
            text_pad_mask=text_pad_mask[0],
            text=sample["text"],
            rel_path=sample["rel_path"],
        )

    return len(to_compute)


def resolve_text_cache_dir(cfg, cli_cache_dir: Optional[Path]) -> Optional[Path]:
    if cli_cache_dir is not None:
        return cli_cache_dir
    text_cache_cfg = cfg.get("text_cache")
    if text_cache_cfg is None:
        return None
    if not bool(text_cache_cfg.get("enabled", False)):
        return None
    cache_dir = text_cache_cfg.get("dir")
    if cache_dir is None:
        return None
    return Path(cache_dir)
