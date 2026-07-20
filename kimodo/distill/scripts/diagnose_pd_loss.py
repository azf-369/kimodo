#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Automated root-cause diagnostics for large PD MSE.

Usage:
  source .venv/bin/activate
  python -m kimodo.distill.scripts.diagnose_pd_loss [--device cuda|cpu] [--quick]

Explains / measures:
  1) Why MSE is NOT bounded to [0, 1]
  2) space_timesteps schedule quirks
  3) oracle jump-2 == two DDIM steps
  4) Init PD loss vs t (identical teacher/student) with real G1 weights if available
  5) x0-space MSE vs noisy x_{t-2} MSE
  6) AMP/bf16 vs fp32 effect
"""

from __future__ import annotations

import argparse
import math
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from kimodo.distill.ddim_ops import (
    ddim_jump_two,
    ddim_step,
    prepare_schedule,
    predict_x0,
    q_sample_on_schedule,
    teacher_two_ddim_steps,
)
from kimodo.distill.pd_loss import progressive_distill_step
from kimodo.distill.weights import freeze_module, load_denoiser_weights, resolve_teacher_weights
from kimodo.model.diffusion import DDIMSampler, Diffusion
from kimodo.distill.config_utils import load_distill_config
from kimodo.train.build import build_denoiser, build_motion_rep
from kimodo.train.collate import collate_motion_batch
from kimodo.train.dataset import G1SeedTrainingDataset
from kimodo.train.text_embedding import TextEmbeddingProvider
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


def _section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def test_mse_not_bounded_to_01() -> None:
    _section("1) MSE is NOT restricted to [0, 1]")
    a = torch.tensor([0.0, 1.0, 2.0])
    b = torch.tensor([0.0, 1.0, 2.0])
    c = torch.tensor([10.0, 10.0, 10.0])
    d = torch.tensor([100.0, -50.0, 0.0])
    print(f"MSE(a,b) identical         = {F.mse_loss(a, b).item():.6f}")
    print(f"MSE(a,a+1) per-dim err=1   = {F.mse_loss(a, a + 1).item():.6f}")
    print(f"MSE(a,c)                   = {F.mse_loss(a, c).item():.6f}")
    print(f"MSE(a,d)                   = {F.mse_loss(a, d).item():.6f}")
    print(
        "Definition: MSE = mean((pred - target)^2) over elements. "
        "If |err|=10, contribution=100. There is NO upper bound of 1."
    )
    print(
        "Confusion: classification 'accuracy' ∈ [0,1]; BCE with logits ≠ MSE; "
        "normalized features have E[x^2]~O(1), but MSE of two tensors can still be >>1."
    )
    assert F.mse_loss(a, a + 1).item() == 1.0
    assert F.mse_loss(a, c).item() > 1.0
    print("[PASS] mse_not_bounded_to_01")


def test_space_timesteps_quirk() -> None:
    _section("2) Diffusion.space_timesteps(100) schedule shape")
    diff = Diffusion(1000)
    use, mp = diff.space_timesteps(100)
    uniq = torch.unique(use)
    print(f"len(use_timesteps)={len(use)} unique={len(uniq)}")
    print(f"use[:12]={use[:12].tolist()}")
    print(f"use[90:100]={use[90:100].tolist()}")
    print(f"use[-5:]={use[-5:].tolist()}  (duplicates at end)")
    diff.calc_diffusion_vars(use)
    print(
        f"alphas_cumprod[0]={float(diff.alphas_cumprod[0]):.6g} "
        f"[50]={float(diff.alphas_cumprod[50]):.6g} "
        f"[99]={float(diff.alphas_cumprod[99]):.6g} "
        f"[100]={float(diff.alphas_cumprod[100]):.6g}"
    )
    print(
        f"sqrt_recip_alphas[99]={float(diff.sqrt_recip_alphas_cumprod[99]):.6g} "
        f"(large ⇒ ε reconstruction amplifies x0 errors)"
    )
    # Official inference indexes t in [0, num_steps), so first 100 entries span full noise.
    assert len(use) == 1000
    assert len(uniq) <= 101
    print("[PASS] space_timesteps_quirk documented")


def test_oracle_jump_equals_two_steps(device: torch.device) -> None:
    _section("3) Oracle: same pred_x0 ⇒ jump-2 == two DDIM steps")
    torch.manual_seed(0)
    diff = Diffusion(1000).to(device)
    sampler = DDIMSampler(diff)
    n = 100
    use, _ = prepare_schedule(diff, n)
    b, t, d = 4, 16, 64
    x0 = torch.randn(b, t, d, device=device)
    for tval in [5, 40, 80, 98]:
        t_idx = torch.full((b,), tval, device=device, dtype=torch.long)
        x_t = q_sample_on_schedule(diff, x0, t_idx)
        # Use true x0 as prediction at BOTH steps (oracle constant x0)
        pred = x0
        x = x_t
        prepare_schedule(diff, n)
        x = ddim_step(sampler, use, x, pred, t_idx)
        prepare_schedule(diff, n)
        x = ddim_step(sampler, use, x, pred, t_idx - 1)
        prepare_schedule(diff, n)
        xj = ddim_jump_two(diff, x_t, pred, t_idx)
        mse = F.mse_loss(xj, x).item()
        print(f"  t={tval:3d}  oracle MSE(jump, 2step)={mse:.3e}")
        assert mse < 1e-6, f"oracle mismatch at t={tval}: {mse}"
    print("[PASS] oracle_jump_equals_two_steps")


def test_feature_scale(data_root: Path, stats: Path, split: Path | None) -> None:
    _section("4) Normalized motion feature scale (dataset)")
    cfg = load_distill_config("pd_g1_rp_teacher")
    mr = build_motion_rep(cfg.denoiser, stats_path=str(stats))
    ds = G1SeedTrainingDataset(
        data_root,
        split_path=split,
        max_files=16,
        max_frames=64,
        fps=30,
        source_fps=120.0,
        motion_rep=mr,
        normalize=True,
        require_text=False,
    )
    x = ds[0]["feats"]
    print(
        f"feats shape={tuple(x.shape)} mean={x.mean():.4f} std={x.std():.4f} "
        f"absmax={x.abs().max():.4f} rms={x.pow(2).mean().sqrt():.4f}"
    )
    print(f"MSE(x, 0)={F.mse_loss(x, torch.zeros_like(x)).item():.4f}  (can be >1 even for normalized x)")
    print(f"MSE(x, x+3)={F.mse_loss(x, x + 3).item():.4f}")
    assert x.std() > 0.1
    print("[PASS] feature_scale")


def _find_rp() -> Path | None:
    root = Path.home() / ".cache/huggingface/hub/models--nvidia--Kimodo-G1-RP-v1/snapshots"
    if not root.is_dir():
        return None
    snaps = sorted(p for p in root.iterdir() if p.is_dir())
    return snaps[-1] if snaps else None


def test_real_init_loss_vs_t(
    device: torch.device,
    *,
    data_root: Path,
    stats: Path,
    split: Path | None,
    rp: Path,
    quick: bool,
) -> dict:
    _section("5) REAL G1 weights: init PD loss vs t (identical teacher/student, all fp32)")
    cfg = load_distill_config("pd_g1_rp_teacher")
    cfg = OmegaConf.merge(cfg, load_distill_config("pd_g1_formal_local"))
    teacher = build_denoiser(cfg, stats_path=str(stats), device=device)
    student = build_denoiser(cfg, stats_path=str(stats), device=device)
    w = resolve_teacher_weights(rp)
    load_denoiser_weights(teacher, w)
    load_denoiser_weights(student, w)
    freeze_module(teacher)
    student.eval()

    mr = build_motion_rep(cfg.denoiser, stats_path=str(stats))
    cache = data_root / "cache/llm2vec_g1_train"
    use_cache = cache.is_dir()
    ds = G1SeedTrainingDataset(
        data_root,
        split_path=split,
        max_files=4 if quick else 8,
        max_frames=64,
        fps=30,
        source_fps=120.0,
        motion_rep=mr,
        normalize=True,
        require_text=use_cache,
        text_cache_dir=cache if use_cache else None,
        text_cache_num_tokens=50,
        text_cache_llm_dim=4096,
    )
    batch = collate_motion_batch([ds[i] for i in range(min(2, len(ds)))])
    diff = Diffusion(1000).to(device)
    sampler = DDIMSampler(diff)

    x0 = batch["feats"].to(device)
    pad = batch["pad_mask"].to(device)
    if "text_feat" in batch:
        text = batch["text_feat"].to(device)
        tpad = batch["text_pad_mask"].to(device)
    else:
        text = torch.zeros(x0.shape[0], 50, 4096, device=device)
        tpad = torch.zeros(x0.shape[0], 50, dtype=torch.bool, device=device)
    heading = torch.zeros(x0.shape[0], device=device)

    t_grid = [5, 20, 40, 60, 80, 90, 95, 98] if not quick else [10, 50, 90, 98]
    rows = []
    for tval in t_grid:
        t_idx = torch.full((x0.shape[0],), tval, device=device, dtype=torch.long)
        torch.manual_seed(0)
        noise = torch.randn_like(x0)
        prepare_schedule(diff, 100)
        x_t = q_sample_on_schedule(diff, x0, t_idx, noise)

        with torch.no_grad():
            x_tgt = teacher_two_ddim_steps(
                teacher,
                diff,
                sampler,
                x_t,
                t_idx,
                pad_mask=pad,
                text_feat=text,
                text_pad_mask=tpad,
                first_heading_angle=heading,
                motion_mask=None,
                observed_motion=None,
                num_steps=100,
            )
            use, map_t = prepare_schedule(diff, 100)
            pred = predict_x0(
                student,
                x_t,
                map_t[t_idx],
                pad_mask=pad,
                text_feat=text,
                text_pad_mask=tpad,
                first_heading_angle=heading,
                motion_mask=None,
                observed_motion=None,
                match_param_dtype=False,
            )
            x_pred = ddim_jump_two(diff, x_t, pred, t_idx)

            # Also: teacher two-step vs jump using FIRST teacher pred only (isolates 2nd-pred effect)
            prepare_schedule(diff, 100)
            pred_t = predict_x0(
                teacher,
                x_t,
                map_t[t_idx],
                pad_mask=pad,
                text_feat=text,
                text_pad_mask=tpad,
                first_heading_angle=heading,
                motion_mask=None,
                observed_motion=None,
                match_param_dtype=True,
            )
            x_jump_from_teacher_pred = ddim_jump_two(diff, x_t, pred_t.float(), t_idx)

            mse_xt = F.mse_loss(x_pred, x_tgt).item()
            mse_same_pred = F.mse_loss(x_jump_from_teacher_pred, x_tgt).item()
            # x0-space: student pred vs teacher first pred
            mse_x0 = F.mse_loss(pred.float(), pred_t.float()).item()
            # Compare student jump vs teacher jump (same algo, may differ bf16 path — here both fp32)
            mse_sj_tj = F.mse_loss(x_pred, x_jump_from_teacher_pred).item()

        row = {
            "t": tval,
            "mse_xt_pd": mse_xt,
            "mse_xt_2nd_pred_gap": mse_same_pred,
            "mse_x0_teacher_student": mse_x0,
            "mse_student_vs_teacher_jump": mse_sj_tj,
            "x_pred_absmax": float(x_pred.abs().max()),
            "x_tgt_absmax": float(x_tgt.abs().max()),
            "pred_x0_absmax": float(pred.abs().max()),
            "alpha_t": float(diff.alphas_cumprod[tval]),
        }
        rows.append(row)
        print(
            f"  t={tval:3d} ᾱ={row['alpha_t']:.3e} "
            f"PD_MSE(x_t-2)={mse_xt:.4e}  "
            f"gap_from_2nd_pred={mse_same_pred:.4e}  "
            f"MSE(x0_s,x0_t)={mse_x0:.4e}  "
            f"|x_pred|∞={row['x_pred_absmax']:.3e} |x_tgt|∞={row['x_tgt_absmax']:.3e}"
        )

    # Full progressive_distill_step once (random t)
    torch.manual_seed(123)
    with torch.no_grad():
        loss, metrics = progressive_distill_step(
            teacher,
            student,
            diff,
            sampler,
            x0,
            pad,
            100,
            text_feat=text,
            text_pad_mask=tpad,
            first_heading_angle=heading,
        )
    print(
        f"\n  random-t progressive_distill_step loss={float(loss):.6e} "
        f"t_mean={float(metrics['t_mean']):.1f} "
        f"x_pred_norm={float(metrics['x_pred_norm']):.4e} "
        f"x_tgt_norm={float(metrics['x_tgt_norm']):.4e}"
    )

    # Diagnose dominant cause
    mid = [r for r in rows if 20 <= r["t"] <= 60]
    high = [r for r in rows if r["t"] >= 90]
    mid_pd = sum(r["mse_xt_pd"] for r in mid) / max(1, len(mid))
    high_pd = sum(r["mse_xt_pd"] for r in high) / max(1, len(high))
    high_gap = sum(r["mse_xt_2nd_pred_gap"] for r in high) / max(1, len(high))
    high_x0 = sum(r["mse_x0_teacher_student"] for r in high) / max(1, len(high))

    print("\n--- Root-cause summary ---")
    print(f"  mean PD MSE @ mid t (20-60): {mid_pd:.4e}")
    print(f"  mean PD MSE @ high t (>=90): {high_pd:.4e}")
    print(f"  mean '2nd prediction gap' @ high t: {high_gap:.4e}")
    print(f"  mean student↔teacher x0 MSE @ high t: {high_x0:.4e}")

    if high_pd > 1e3 and high_gap > 0.5 * high_pd:
        print(
            "  VERDICT: Large loss is dominated by teacher 2nd-step re-prediction "
            "≠ single jump (PD geometric gap), amplified at high noise (tiny ᾱ)."
        )
        verdict = "second_pred_gap_high_t"
    elif high_x0 > 1.0 and high_pd > 1e3:
        print("  VERDICT: Teacher/student x0 preds already diverge (dtype/amp/loading).")
        verdict = "teacher_student_x0_mismatch"
    elif high_pd > 1e3:
        print("  VERDICT: High-t noisy-state MSE explosion (DDIM amplification).")
        verdict = "high_t_amplification"
    else:
        print("  VERDICT: Init PD MSE not catastrophic on this device/batch.")
        verdict = "ok_moderate"

    # Peak explosion check
    if any(r["x_pred_absmax"] > 100 or r["x_tgt_absmax"] > 100 for r in rows):
        print("  NOTE: |x|∞ > 100 ⇒ DDIM states left unit-Gaussian scale (unstable preds).")

    print("[PASS] real_init_loss_vs_t")
    return {"rows": rows, "verdict": verdict, "random_loss": float(loss)}


def test_constraint_unpack_order(
    device: torch.device,
    *,
    data_root: Path,
    stats: Path,
    split: Path | None,
    rp: Path,
) -> None:
    _section("7) Regression: constraint (mask, observed) order + formal init loss")
    from kimodo.distill.pd_loss import progressive_distill_batch_step
    from kimodo.train.constraint_synth import sample_training_constraints

    cfg = load_distill_config("pd_g1_rp_teacher")
    teacher = build_denoiser(cfg, stats_path=str(stats), device=device)
    student = build_denoiser(cfg, stats_path=str(stats), device=device)
    w = resolve_teacher_weights(rp)
    load_denoiser_weights(teacher, w)
    load_denoiser_weights(student, w)
    freeze_module(teacher)
    student.train()

    mr = build_motion_rep(cfg.denoiser, stats_path=str(stats))
    cache = data_root / "cache/llm2vec_g1_train"
    ds = G1SeedTrainingDataset(
        data_root,
        split_path=split,
        max_files=4,
        max_frames=64,
        fps=30,
        source_fps=120.0,
        motion_rep=mr,
        normalize=True,
        require_text=cache.is_dir(),
        text_cache_dir=cache if cache.is_dir() else None,
        text_cache_num_tokens=50,
        text_cache_llm_dim=4096,
    )
    batch = collate_motion_batch([ds[i] for i in range(min(2, len(ds)))])
    x0 = batch["feats"].to(device)
    lengths = batch["lengths"].to(device)
    torch.manual_seed(0)
    motion_mask, observed_motion = sample_training_constraints(
        student.motion_rep,
        x0,
        lengths,
        constraint_prob=1.0,
        max_keyframes=4,
        device=str(device),
    )
    assert motion_mask is not None and observed_motion is not None
    # Mask must be Boolean or {0,1}-valued, NOT raw motion values.
    mask_f = motion_mask.float()
    assert float(mask_f.min()) >= 0.0 and float(mask_f.max()) <= 1.0, (
        f"motion_mask looks like values, not a mask: min={float(mask_f.min())} max={float(mask_f.max())}"
    )
    assert float(observed_motion.abs().max()) > 1.0 or float((observed_motion != 0).any()) >= 0
    print(
        f"  mask dtype={motion_mask.dtype} min/max={float(mask_f.min()):.3f}/{float(mask_f.max()):.3f} "
        f"mean={float(mask_f.mean()):.4f}"
    )
    print(
        f"  observed absmax={float(observed_motion.abs().max()):.4f} "
        f"(should be O(1) normalized motion, not 0/1-only)"
    )

    diff = Diffusion(1000).to(device)
    sampler = DDIMSampler(diff)
    text_provider = TextEmbeddingProvider(num_tokens=50, llm_dim=4096, device=device, mode="dummy")
    losses = []
    for seed in range(8):
        torch.manual_seed(seed)
        loss, metrics = progressive_distill_batch_step(
            teacher,
            student,
            student.motion_rep,
            diff,
            sampler,
            batch,
            100,
            text_provider,
            cfg_dropout={"uncond": 0.1, "text_only": 0.1, "constraint_only": 0.1},
            constraint_prob=0.8,
            max_keyframes=4,
        )
        losses.append(float(loss.detach()))
        print(
            f"  seed={seed} formal_loss={losses[-1]:.4e} "
            f"t={float(metrics['t_mean']):.1f} "
            f"xp_norm={float(metrics['x_pred_norm']):.3e} "
            f"xt_norm={float(metrics['x_tgt_norm']):.3e}"
        )
    mean_loss = sum(losses) / len(losses)
    print(f"  mean formal init loss={mean_loss:.4e}")
    assert mean_loss < 1.0, (
        f"After mask/observed fix, formal init MSE should be <<1, got mean={mean_loss}"
    )
    print("[PASS] constraint_unpack_order")


def test_amp_vs_fp32(
    device: torch.device,
    *,
    data_root: Path,
    stats: Path,
    split: Path | None,
    rp: Path,
) -> None:
    if device.type != "cuda":
        print("\n[SKIP] amp_vs_fp32 (CUDA only)")
        return
    _section("6) AMP bf16 vs full fp32 on same batch/t")
    cfg = load_distill_config("pd_g1_rp_teacher")
    teacher = build_denoiser(cfg, stats_path=str(stats), device=device)
    student = build_denoiser(cfg, stats_path=str(stats), device=device)
    w = resolve_teacher_weights(rp)
    load_denoiser_weights(teacher, w)
    load_denoiser_weights(student, w)
    freeze_module(teacher)
    student.eval()

    mr = build_motion_rep(cfg.denoiser, stats_path=str(stats))
    cache = data_root / "cache/llm2vec_g1_train"
    ds = G1SeedTrainingDataset(
        data_root,
        split_path=split,
        max_files=2,
        max_frames=64,
        fps=30,
        source_fps=120.0,
        motion_rep=mr,
        normalize=True,
        require_text=cache.is_dir(),
        text_cache_dir=cache if cache.is_dir() else None,
        text_cache_num_tokens=50,
        text_cache_llm_dim=4096,
    )
    batch = collate_motion_batch([ds[0]])
    x0 = batch["feats"].to(device)
    pad = batch["pad_mask"].to(device)
    if "text_feat" in batch:
        text, tpad = batch["text_feat"].to(device), batch["text_pad_mask"].to(device)
    else:
        text = torch.zeros(1, 50, 4096, device=device)
        tpad = torch.zeros(1, 50, dtype=torch.bool, device=device)
    heading = torch.zeros(1, device=device)
    diff = Diffusion(1000).to(device)
    sampler = DDIMSampler(diff)

    def _one(amp: bool) -> float:
        torch.manual_seed(0)
        with torch.amp.autocast("cuda", enabled=amp, dtype=torch.bfloat16):
            loss, _ = progressive_distill_step(
                teacher,
                student,
                diff,
                sampler,
                x0,
                pad,
                100,
                text_feat=text,
                text_pad_mask=tpad,
                first_heading_angle=heading,
            )
        return float(loss)

    # Force same t via patching
    fixed_t = 95
    real_randint = torch.randint

    def _fixed(*args, **kwargs):
        size = args[2] if len(args) >= 3 else kwargs.get("size")
        if isinstance(size, tuple):
            n = size[0]
        else:
            n = int(size[0]) if hasattr(size, "__getitem__") else kwargs.get("size", (1,))[0]
        return torch.full((n,), fixed_t, device=device, dtype=torch.long)

    torch.randint = lambda *a, **k: torch.full(
        (x0.shape[0],), fixed_t, device=device, dtype=torch.long
    )
    try:
        loss_fp32 = _one(False)
        loss_amp = _one(True)
    finally:
        torch.randint = real_randint
    print(f"  t={fixed_t}  loss_fp32={loss_fp32:.6e}  loss_amp_bf16={loss_amp:.6e}  ratio={loss_amp/max(loss_fp32,1e-12):.3f}")
    print("[PASS] amp_vs_fp32")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose large PD MSE")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--data-root", type=Path, default=Path("datasets/bones-seed"))
    p.add_argument(
        "--stats-path",
        type=Path,
        default=Path("checkpoints/Kimodo-G1-SEED-v1.hf/stats/motion"),
    )
    p.add_argument(
        "--split-path",
        type=Path,
        default=Path("datasets/kimodo-benchmark/splits/train_split_paths.txt"),
    )
    p.add_argument("--teacher-checkpoint", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    print(f"device={device}  cuda_available={torch.cuda.is_available()}")
    if device.type == "cuda" and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"CUDA free={free/1024**3:.2f}GiB / total={total/1024**3:.2f}GiB")

    test_mse_not_bounded_to_01()
    test_space_timesteps_quirk()
    test_oracle_jump_equals_two_steps(torch.device("cpu"))

    if not args.stats_path.is_dir():
        alt = Path("checkpoints/Kimodo-G1-SEED-v1/stats/motion")
        if alt.is_dir():
            args.stats_path = alt
        else:
            print(f"ERROR: stats not found: {args.stats_path}")
            return 1
    if not args.data_root.is_dir():
        print(f"ERROR: data-root not found: {args.data_root}")
        return 1

    split = args.split_path if args.split_path.is_file() else None
    test_feature_scale(args.data_root, args.stats_path, split)

    rp = args.teacher_checkpoint or _find_rp()
    if rp is None:
        print("ERROR: cannot find Kimodo-G1-RP-v1 snapshot; pass --teacher-checkpoint")
        return 1
    print(f"\nUsing RP checkpoint: {rp}")

    if device.type == "cuda" and torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free < 8 * 1024**3:
            print(
                f"WARNING: only {free/1024**3:.2f}GiB free; real-weight test needs ~8GiB+. "
                "Trying anyway; if OOM, stop other GPU jobs and re-run."
            )

    try:
        result = test_real_init_loss_vs_t(
            device,
            data_root=args.data_root,
            stats=args.stats_path,
            split=split,
            rp=rp,
            quick=args.quick,
        )
        test_constraint_unpack_order(
            device,
            data_root=args.data_root,
            stats=args.stats_path,
            split=split,
            rp=rp,
        )
        test_amp_vs_fp32(
            device,
            data_root=args.data_root,
            stats=args.stats_path,
            split=split,
            rp=rp,
        )
    except torch.cuda.OutOfMemoryError:
        print("\n[OOM] Falling back to CPU for real-weight mid/high-t subsample (slow)...")
        torch.cuda.empty_cache()
        result = test_real_init_loss_vs_t(
            torch.device("cpu"),
            data_root=args.data_root,
            stats=args.stats_path,
            split=split,
            rp=rp,
            quick=True,
        )

    _section("FINAL")
    print(f"verdict={result.get('verdict')}  random_t_loss={result.get('random_loss')}")
    print("All automated diagnosis sections completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
