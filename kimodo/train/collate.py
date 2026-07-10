# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batch collation for variable-length motion sequences."""

from __future__ import annotations

import torch


def collate_motion_batch(samples: list[dict]) -> dict:
    """Pad motion features to the max length in the batch."""
    max_len = max(s["length"] for s in samples)
    feat_dim = samples[0]["feats"].shape[-1]
    batch_size = len(samples)

    feats = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
    pad_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, sample in enumerate(samples):
        length = sample["length"]
        feats[i, :length] = sample["feats"]
        pad_mask[i, :length] = True
        lengths[i] = length

    batch = {
        "feats": feats,
        "pad_mask": pad_mask,
        "lengths": lengths,
        "paths": [s["path"] for s in samples],
        "texts": [s["text"] for s in samples],
    }
    if "text_feat" in samples[0]:
        num_tokens = samples[0]["text_feat"].shape[0]
        llm_dim = samples[0]["text_feat"].shape[-1]
        text_feat = torch.zeros(batch_size, num_tokens, llm_dim, dtype=torch.float32)
        text_pad_mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool)
        for i, sample in enumerate(samples):
            text_feat[i] = sample["text_feat"]
            text_pad_mask[i] = sample["text_pad_mask"]
        batch["text_feat"] = text_feat
        batch["text_pad_mask"] = text_pad_mask
    return batch
