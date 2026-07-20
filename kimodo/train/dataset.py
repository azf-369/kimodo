# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""G1 BONES-SEED motion dataset for Flow Matching training."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from kimodo.motion_rep import KimodoMotionRep
from kimodo.train.seed_g1_io import load_bones_seed_g1_csv


def _load_metadata_index(metadata_path: Path) -> pd.DataFrame:
    if metadata_path.suffix == ".parquet":
        df = pd.read_parquet(metadata_path)
    else:
        df = pd.read_csv(metadata_path)
    if "move_g1_path" not in df.columns:
        raise KeyError(f"metadata missing move_g1_path column: {metadata_path}")
    df = df.dropna(subset=["move_g1_path"]).copy()
    df["rel_path"] = df["move_g1_path"].astype(str)
    return df.set_index("rel_path", drop=False)


def _load_split_keys(split_path: Path) -> set[str]:
    keys: set[str] = set()
    for line in split_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        keys.add(f"g1/csv/{line}.csv" if not line.startswith("g1/") else line)
    return keys


class G1SeedTrainingDataset(Dataset):
    """Load G1 CSV clips from BONES-SEED with text prompts and optional train split."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        metadata_path: str | Path | None = None,
        split_path: str | Path | None = None,
        max_files: Optional[int] = None,
        max_frames: int = 300,
        frame_crop: str = "random",
        fps: int = 30,
        source_fps: float = 120.0,
        motion_rep: KimodoMotionRep | None = None,
        text_field: str = "content_natural_desc_4",
        normalize: bool = True,
        require_text: bool = True,
        text_cache_dir: str | Path | None = None,
        text_cache_num_tokens: int = 50,
        text_cache_llm_dim: int = 4096,
    ):
        self.data_root = Path(data_root)
        self.max_frames = max_frames
        crop = str(frame_crop).lower().strip()
        if crop not in {"random", "prefix"}:
            raise ValueError(f"frame_crop must be 'random' or 'prefix', got {frame_crop!r}")
        self.frame_crop = crop
        self.fps = fps
        self.source_fps = source_fps
        self.motion_rep = motion_rep
        self.text_field = text_field
        self.normalize = normalize
        self.require_text = require_text
        self.text_cache_dir = Path(text_cache_dir) if text_cache_dir is not None else None
        self.text_cache_num_tokens = text_cache_num_tokens
        self.text_cache_llm_dim = text_cache_llm_dim

        metadata_path = Path(metadata_path or self.data_root / "metadata" / "seed_metadata_v004.csv")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        metadata = _load_metadata_index(metadata_path)
        split_keys: Optional[set[str]] = None
        if split_path is not None:
            split_path = Path(split_path)
            if split_path.is_file():
                split_keys = _load_split_keys(split_path)

        samples: list[dict] = []
        for rel_path, row in metadata.iterrows():
            if split_keys is not None and rel_path not in split_keys:
                continue
            csv_path = self.data_root / rel_path
            if not csv_path.is_file():
                continue
            text = str(row.get(text_field, "") or "").strip()
            if self.require_text and not text:
                continue
            samples.append(
                {
                    "path": csv_path,
                    "text": text,
                    "rel_path": rel_path,
                }
            )

        if not samples:
            raise FileNotFoundError(
                "No training samples found. Check data_root, metadata, and optional split_path."
            )

        samples.sort(key=lambda item: str(item["path"]))
        if max_files is not None:
            samples = samples[:max_files]

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        if self.motion_rep is None:
            raise RuntimeError("motion_rep must be set before sampling")

        sample = self.samples[index]
        motion = load_bones_seed_g1_csv(str(sample["path"]), source_fps=self.source_fps)
        local_rot_mats = motion["local_rot_mats"]
        root_positions = motion["root_positions"]

        n_frames = int(local_rot_mats.shape[0])
        if n_frames > self.max_frames:
            if self.frame_crop == "random":
                start = int(torch.randint(0, n_frames - self.max_frames + 1, (1,)).item())
            else:
                start = 0
            end = start + self.max_frames
            local_rot_mats = local_rot_mats[start:end]
            root_positions = root_positions[start:end]

        feats = self.motion_rep(
            local_rot_mats,
            root_positions,
            to_normalize=self.normalize,
            to_canonicalize=True,
        )
        if feats.dim() == 3:
            length = feats.shape[1]
            feats = feats.squeeze(0)
        else:
            length = feats.shape[0]

        item = {
            "feats": feats.float(),
            "length": length,
            "path": str(sample["path"]),
            "text": sample["text"],
            "rel_path": sample["rel_path"],
        }
        if self.text_cache_dir is not None:
            from kimodo.train.text_cache import cache_path_for_rel_path, load_text_embedding

            cache_path = cache_path_for_rel_path(self.text_cache_dir, sample["rel_path"])
            if not cache_path.is_file():
                raise FileNotFoundError(f"Missing text embedding cache: {cache_path}")
            text_feat, text_pad_mask = load_text_embedding(
                cache_path,
                num_tokens=self.text_cache_num_tokens,
                llm_dim=self.text_cache_llm_dim,
            )
            item["text_feat"] = text_feat
            item["text_pad_mask"] = text_pad_mask
        return item


# Backward-compatible alias used by the initial smoke test.
G1SeedMotionDataset = G1SeedTrainingDataset
