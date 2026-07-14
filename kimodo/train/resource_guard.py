# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host memory/CPU pressure monitoring and DataLoader auto-tuning."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _read_int_file(path: Path) -> Optional[int]:
    try:
        raw = path.read_text().strip()
    except OSError:
        return None
    if raw in {"max", "Max"}:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _cgroup_v2_base() -> Optional[Path]:
    for path in (Path("/sys/fs/cgroup"), Path(f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice")):
        if (path / "cgroup.controllers").is_file():
            return path
    return None


def _cgroup_memory_limit_bytes() -> Optional[int]:
    v2 = _cgroup_v2_base()
    if v2 is not None:
        for rel in ("memory.max", "memory/memory.max"):
            val = _read_int_file(v2 / rel)
            if val is not None:
                return val
    for path in (
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        Path("/sys/fs/cgroup/memory.limit_in_bytes"),
    ):
        val = _read_int_file(path)
        if val is not None and val < (1 << 62):
            return val
    return None


def _cgroup_memory_usage_bytes() -> Optional[int]:
    v2 = _cgroup_v2_base()
    if v2 is not None:
        for rel in ("memory.current", "memory/memory.current"):
            val = _read_int_file(v2 / rel)
            if val is not None:
                return val
    for path in (
        Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        Path("/sys/fs/cgroup/memory.usage_in_bytes"),
    ):
        val = _read_int_file(path)
        if val is not None:
            return val
    return None


def _meminfo_bytes() -> tuple[int, int, int]:
    """Return (total, available, used) in bytes from /proc/meminfo."""
    fields: dict[str, int] = {}
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            key, rest = line.split(":", 1)
            fields[key.strip()] = int(rest.strip().split()[0]) * 1024
    total = fields["MemTotal"]
    available = fields.get("MemAvailable", fields.get("MemFree", 0))
    used = max(0, total - available)
    return total, available, used


def _cpu_count_effective() -> int:
    quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    try:
        quota = int(quota_path.read_text().strip())
        period = int(period_path.read_text().strip())
        if quota > 0 and period > 0:
            return max(1, quota // period)
    except OSError:
        pass
    v2 = _cgroup_v2_base()
    if v2 is not None:
        try:
            lines = (v2 / "cpu.max").read_text().strip().split()
            if len(lines) == 2 and lines[0] != "max":
                quota = int(lines[0])
                period = int(lines[1])
                if quota > 0 and period > 0:
                    return max(1, quota // period)
        except OSError:
            pass
    return os.cpu_count() or 1


@dataclass(frozen=True)
class ResourceSnapshot:
    memory_total_bytes: int
    memory_used_bytes: int
    memory_used_ratio: float
    cpu_count_effective: int
    cgroup_limited: bool


def snapshot_resources() -> ResourceSnapshot:
    cgroup_limit = _cgroup_memory_limit_bytes()
    cgroup_used = _cgroup_memory_usage_bytes()
    host_total, host_available, host_used = _meminfo_bytes()

    if cgroup_limit is not None and cgroup_used is not None:
        total = cgroup_limit
        used = cgroup_used
        cgroup_limited = True
    else:
        total = host_total
        used = host_used
        cgroup_limited = False

    ratio = used / total if total > 0 else 0.0
    return ResourceSnapshot(
        memory_total_bytes=total,
        memory_used_bytes=used,
        memory_used_ratio=ratio,
        cpu_count_effective=_cpu_count_effective(),
        cgroup_limited=cgroup_limited,
    )


def _format_gb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.1f}GB"


@dataclass
class DataLoaderTuning:
    num_workers: int
    prefetch_factor: int
    persistent_workers: bool


def cap_dataloader_at_startup(
    *,
    requested_workers: int,
    requested_prefetch: int,
    world_size: int,
    reserve_gb: float = 16.0,
    gb_per_worker: float = 2.5,
    min_workers: int = 0,
    min_prefetch: int = 1,
    mem_prefetch_cap_ratio: float = 0.70,
) -> tuple[DataLoaderTuning, str]:
    """Pick safe DataLoader settings before workers are spawned."""
    snap = snapshot_resources()
    # Base worker RSS; prefetch buffers are accounted separately (lighter weight).
    worker_gb = gb_per_worker + 0.2 * max(0, requested_prefetch - 1)
    available_gb = max(0.0, (snap.memory_total_bytes - snap.memory_used_bytes) / (1024 ** 3) - reserve_gb)
    worker_budget = int(available_gb / worker_gb) if worker_gb > 0 else requested_workers
    total_worker_cap = max(min_workers, worker_budget)
    per_rank_cap = max(min_workers, total_worker_cap // max(world_size, 1))

    # Allow slight CPU oversubscription when I/O + preprocessing bound (common for CSV loads).
    cpu_cap = max(min_workers, int(snap.cpu_count_effective * 1.25) // max(world_size, 1))
    per_rank_cap = min(per_rank_cap, cpu_cap, requested_workers)

    cpu_total_workers = per_rank_cap * max(world_size, 1)
    if cpu_total_workers > int(snap.cpu_count_effective * 1.5):
        per_rank_cap = max(min_workers, snap.cpu_count_effective // max(world_size, 1))

    prefetch = requested_prefetch
    if per_rank_cap < requested_workers:
        prefetch = min(prefetch, max(min_prefetch, requested_prefetch - 1))
    if snap.memory_used_ratio > mem_prefetch_cap_ratio:
        prefetch = min(prefetch, max(min_prefetch, requested_prefetch - 2))

    tuning = DataLoaderTuning(
        num_workers=per_rank_cap,
        prefetch_factor=max(min_prefetch, prefetch) if per_rank_cap > 0 else 2,
        persistent_workers=per_rank_cap > 0,
    )
    msg = (
        f"resource_guard startup: mem {snap.memory_used_ratio * 100:.1f}% "
        f"({_format_gb(snap.memory_used_bytes)}/{_format_gb(snap.memory_total_bytes)}"
        f"{', cgroup' if snap.cgroup_limited else ''}), "
        f"cpu_effective={snap.cpu_count_effective}, "
        f"workers {requested_workers}->{tuning.num_workers}/rank, "
        f"prefetch {requested_prefetch}->{tuning.prefetch_factor}"
    )
    return tuning, msg


@dataclass
class ResourceGuardConfig:
    enabled: bool = True
    mem_caution_watermark: float = 0.78
    mem_high_watermark: float = 0.85
    mem_critical_watermark: float = 0.92
    check_every_steps: int = 50
    min_num_workers: int = 0
    min_prefetch_factor: int = 1


class ResourceGuard:
    """Monitor host pressure and request lighter DataLoader settings when needed."""

    def __init__(
        self,
        *,
        cfg: ResourceGuardConfig,
        world_size: int,
        initial: DataLoaderTuning,
    ):
        self.cfg = cfg
        self.world_size = world_size
        self.num_workers = initial.num_workers
        self.prefetch_factor = initial.prefetch_factor
        self.persistent_workers = initial.persistent_workers
        self.requested_workers = initial.num_workers
        self.requested_prefetch = initial.prefetch_factor
        self._reload_requested = False
        self._last_snapshot: Optional[ResourceSnapshot] = None

    @classmethod
    def from_config(cls, raw: Optional[dict], *, world_size: int, initial: DataLoaderTuning) -> ResourceGuard:
        if not raw:
            return cls(cfg=ResourceGuardConfig(enabled=False), world_size=world_size, initial=initial)
        cfg = ResourceGuardConfig(
            enabled=bool(raw.get("enabled", True)),
            mem_caution_watermark=float(raw.get("mem_caution_watermark", 0.78)),
            mem_high_watermark=float(raw.get("mem_high_watermark", 0.85)),
            mem_critical_watermark=float(raw.get("mem_critical_watermark", 0.92)),
            check_every_steps=int(raw.get("check_every_steps", 50)),
            min_num_workers=int(raw.get("min_num_workers", 0)),
            min_prefetch_factor=int(raw.get("min_prefetch_factor", 1)),
        )
        guard = cls(cfg=cfg, world_size=world_size, initial=initial)
        return guard

    def current_tuning(self) -> DataLoaderTuning:
        return DataLoaderTuning(
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def check(self, step: int) -> Optional[str]:
        if not self.cfg.enabled or step % self.cfg.check_every_steps != 0:
            return None

        snap = snapshot_resources()
        self._last_snapshot = snap
        action = None

        if snap.memory_used_ratio >= self.cfg.mem_critical_watermark:
            if self.num_workers > self.cfg.min_num_workers:
                self.num_workers = max(self.cfg.min_num_workers, self.num_workers // 2)
                action = "critical_mem_reduce_workers"
            elif self.prefetch_factor > self.cfg.min_prefetch_factor:
                self.prefetch_factor = max(self.cfg.min_prefetch_factor, self.prefetch_factor - 1)
                action = "critical_mem_reduce_prefetch"
        elif snap.memory_used_ratio >= self.cfg.mem_high_watermark:
            if self.prefetch_factor > self.cfg.min_prefetch_factor:
                self.prefetch_factor = max(self.cfg.min_prefetch_factor, self.prefetch_factor - 1)
                action = "high_mem_reduce_prefetch"
            elif self.num_workers > self.cfg.min_num_workers:
                self.num_workers = max(self.cfg.min_num_workers, self.num_workers - 1)
                action = "high_mem_reduce_workers"
        elif snap.memory_used_ratio >= self.cfg.mem_caution_watermark:
            if self.prefetch_factor > self.cfg.min_prefetch_factor:
                self.prefetch_factor = max(self.cfg.min_prefetch_factor, self.prefetch_factor - 1)
                action = "caution_mem_reduce_prefetch"

        if action is not None:
            self._reload_requested = True
            return (
                f"resource_guard step {step}: mem {snap.memory_used_ratio * 100:.1f}% "
                f"({_format_gb(snap.memory_used_bytes)}/{_format_gb(snap.memory_total_bytes)}) "
                f"-> {action}, workers {self.requested_workers}->{self.num_workers}, "
                f"prefetch {self.requested_prefetch}->{self.prefetch_factor} "
                f"(reload at next dataloader epoch)"
            )
        return None

    def consume_reload_request(self) -> bool:
        if not self._reload_requested:
            return False
        self._reload_requested = False
        return True
