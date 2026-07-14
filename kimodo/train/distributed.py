# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distributed training helpers (torchrun / DDP)."""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def is_distributed_env() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def init_distributed(backend: str = "nccl") -> tuple[bool, int, int, int]:
    """Initialize process group when launched via torchrun.

    Returns:
        (is_distributed, rank, world_size, local_rank)
    """
    if not is_distributed_env():
        return False, 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        init_kwargs: dict = {"backend": backend, "rank": rank, "world_size": world_size}
        if torch.cuda.is_available():
            init_kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(**init_kwargs)

    return True, rank, world_size, local_rank


def is_main_process(rank: int = 0) -> bool:
    return rank == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, DDP) else module


def wrap_distributed(
    module: torch.nn.Module,
    *,
    device: torch.device,
    local_rank: int,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    if device.type != "cuda":
        raise ValueError("DDP training requires CUDA devices.")
    # Static buffers only (e.g. positional encodings); skip per-forward broadcast
    # to avoid NCCL hangs if rank0 ever diverges during checkpoint I/O.
    return DDP(
        module,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused_parameters,
        broadcast_buffers=False,
    )


def global_batch_size(per_gpu_batch: int, world_size: int, grad_accum_steps: int = 1) -> int:
    return per_gpu_batch * max(world_size, 1) * max(grad_accum_steps, 1)


def scaled_lr(
    base_lr: float,
    *,
    base_batch: int,
    global_batch: int,
    scaling: str = "sqrt",
) -> float:
    """Scale learning rate from a reference (base_batch, base_lr) pair."""
    if base_batch <= 0 or global_batch <= 0:
        return base_lr
    ratio = global_batch / base_batch
    if scaling == "linear":
        return base_lr * ratio
    if scaling == "none":
        return base_lr
    return base_lr * (ratio**0.5)


def print_rank0(rank: int, message: str) -> None:
    if is_main_process(rank):
        print(message, flush=True)
