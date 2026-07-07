# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Text embedding helpers for Kimodo training."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from kimodo.model.loading import instantiate_from_dict
from kimodo.sanitize import sanitize_texts


def pad_text_batch(
    encoded: list[Tensor],
    lengths: list[int],
    *,
    num_tokens: int,
    llm_dim: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Pad variable-length LLM2Vec outputs to fixed token count."""
    batch_size = len(encoded)
    text_feat = torch.zeros(batch_size, num_tokens, llm_dim, device=device)
    text_pad_mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)

    for i, (feat, length) in enumerate(zip(encoded, lengths)):
        if feat.dim() == 2:
            tokens = min(feat.shape[0], num_tokens)
            text_feat[i, :tokens] = feat[:tokens].to(device=device, dtype=text_feat.dtype)
            text_pad_mask[i, :tokens] = True
        else:
            tokens = min(int(length), num_tokens)
            vec = feat.reshape(-1, llm_dim)[:tokens]
            text_feat[i, :tokens] = vec.to(device=device, dtype=text_feat.dtype)
            text_pad_mask[i, :tokens] = True

    return text_feat, text_pad_mask


class TextEmbeddingProvider:
    """Encode prompts for training; supports live encoder or dummy zeros for smoke tests."""

    def __init__(
        self,
        *,
        num_tokens: int,
        llm_dim: int,
        device: torch.device,
        mode: str = "encoder",
        encoder_cfg: Optional[dict] = None,
    ):
        self.num_tokens = num_tokens
        self.llm_dim = llm_dim
        self.device = device
        self.mode = mode
        self.encoder = None

        if mode == "encoder":
            if encoder_cfg is None:
                raise ValueError("encoder_cfg is required when text mode is 'encoder'")
            self.encoder = instantiate_from_dict(encoder_cfg)
            if hasattr(self.encoder, "to"):
                self.encoder.to(device)
            if hasattr(self.encoder, "eval"):
                self.encoder.eval()

    def encode(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        batch_size = len(texts)
        if self.num_tokens == 0:
            return (
                torch.zeros(batch_size, 0, self.llm_dim, device=self.device),
                torch.zeros(batch_size, 0, dtype=torch.bool, device=self.device),
            )

        texts = sanitize_texts(texts)

        if self.mode == "dummy":
            text_feat = torch.zeros(batch_size, self.num_tokens, self.llm_dim, device=self.device)
            text_pad_mask = torch.zeros(batch_size, self.num_tokens, dtype=torch.bool, device=self.device)
            text_pad_mask[:, 0] = True
            return text_feat, text_pad_mask

        encoded: list[Tensor] = []
        lengths: list[int] = []
        for text in texts:
            feat, length = self.encoder(text)
            if not torch.is_tensor(feat):
                feat = torch.tensor(feat)
            if isinstance(length, int):
                length = [length]
            encoded.append(feat)
            lengths.append(int(length[0]) if isinstance(length, list) else int(length))

        return pad_text_batch(
            encoded,
            lengths,
            num_tokens=self.num_tokens,
            llm_dim=self.llm_dim,
            device=self.device,
        )
