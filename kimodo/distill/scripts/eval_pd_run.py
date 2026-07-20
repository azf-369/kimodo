#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batch-eval all ``step_*`` checkpoints in one PD run and rank the best.

Reuses the same metrics / examples as ``eval_pd_vs_rp``. Loads the text encoder
and RP baseline once, then sweeps student checkpoints.

Example:
  export TEXT_ENCODER_DEVICE=cpu
  RP=~/.cache/huggingface/hub/models--nvidia--Kimodo-G1-RP-v1/snapshots/<hash>
  python -m kimodo.distill.scripts.eval_pd_run \\
    --run-dir outputs/pd_g1_rp_teacher_seed/stage_50to25_gentle \\
    --baseline-checkpoint "$RP" --steps 25 --device cuda

  # Faster smoke (1 example, every 2k steps):
  python -m kimodo.distill.scripts.eval_pd_run \\
    --run-dir outputs/pd_g1_rp_teacher_seed/stage_50to25_gentle \\
    --baseline-checkpoint "$RP" --steps 25 --max-examples 1 --every 2000
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Optional

import torch

from kimodo.distill.scripts.eval_pd_vs_rp import (
    METRIC_KEYS,
    _configure_eval_env,
    _load_shared_text_encoder,
    _mean_over_examples,
    _optional_path,
    _resolve_checkpoint_dir,
    discover_examples,
    evaluate_motion,
    generate_one,
    run_side,
)

# Higher weight = more influence on ranking. Sign: positive → prefer higher metric.
# Constraint-first default (matches project priority); skate is secondary.
DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "constraint_root2d_acc": 1.0,  # higher better
    "constraint_root2d_err": -1.5,  # lower better
    "constraint_fullbody_keyframe": -1.5,
    "constraint_end_effector": -1.0,
    "foot_skate_ratio": -0.35,
    "root_travel_m": 0.05,  # mild preference for healthy travel
}

STEP_DIR_RE = re.compile(r"^step_(\d+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Eval all step_* checkpoints in a PD run dir and pick the best.",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training output dir containing step_0, step_1000, ...",
    )
    p.add_argument("--baseline-model", type=str, default="g1-rp")
    p.add_argument("--baseline-checkpoint", type=_optional_path, default=None)
    p.add_argument(
        "--examples-dir",
        type=Path,
        default=Path("kimodo/assets/demo/examples/kimodo-g1-rp"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: outputs/pd_eval/<run_name>_sweep_s<steps>",
    )
    p.add_argument("--steps", type=int, default=25, help="DDIM steps for RP + all students.")
    p.add_argument("--post-processing", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--student-model-name", type=str, default="g1-rp")
    p.add_argument("--text-encoder-mode", type=str, default="local", choices=["local", "api", "auto"])
    p.add_argument("--text-encoder-device", type=str, default="cpu")
    p.add_argument("--min-step", type=int, default=None, help="Inclusive min step index.")
    p.add_argument("--max-step", type=int, default=None, help="Inclusive max step index.")
    p.add_argument(
        "--every",
        type=int,
        default=1,
        help="Only eval step indices divisible by N (step_0 always kept if present).",
    )
    p.add_argument(
        "--include-step0",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include step_0 (init copy of teacher). Default: yes.",
    )
    p.add_argument(
        "--save-npz",
        action="store_true",
        help="Save motion.npz per example (large). Default: metrics only.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip checkpoints that already have metrics.json under output-dir.",
    )
    p.add_argument(
        "--travel-floor",
        type=float,
        default=0.05,
        help="If root_travel_m < this, mark frozen and exclude from best (default 0.05).",
    )
    p.add_argument(
        "--score-mode",
        type=str,
        default="vs_rp",
        choices=["vs_rp", "absolute"],
        help="vs_rp: score deltas vs RP baseline; absolute: score raw student metrics.",
    )
    return p.parse_args()


def discover_step_checkpoints(
    run_dir: Path,
    *,
    min_step: Optional[int],
    max_step: Optional[int],
    every: int,
    include_step0: bool,
) -> list[tuple[int, Path]]:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    found: list[tuple[int, Path]] = []
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        m = STEP_DIR_RE.match(child.name)
        if not m:
            continue
        step = int(m.group(1))
        if not (child / "config.yaml").is_file() or not (child / "model.safetensors").is_file():
            print(f"WARNING: skip incomplete checkpoint {child}")
            continue
        if min_step is not None and step < min_step:
            continue
        if max_step is not None and step > max_step:
            continue
        if step == 0:
            if not include_step0:
                continue
        elif every > 1 and step % every != 0:
            continue
        found.append((step, child))

    found.sort(key=lambda x: x[0])
    if not found:
        raise FileNotFoundError(f"No valid step_* checkpoints under {run_dir}")
    return found


def _finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def compute_score(
    student: dict[str, float],
    baseline: dict[str, float],
    *,
    mode: str,
    weights: dict[str, float],
    travel_floor: float,
) -> dict[str, Any]:
    """Scalar score (higher better) + frozen flag."""
    travel = float(student.get("root_travel_m", float("nan")))
    frozen = _finite(travel) and travel < travel_floor

    parts: dict[str, float] = {}
    total = 0.0
    for key, w in weights.items():
        s = student.get(key, float("nan"))
        if not _finite(s):
            continue
        if mode == "vs_rp":
            b = baseline.get(key, float("nan"))
            if not _finite(b):
                continue
            contrib = w * (float(s) - float(b))
        else:
            contrib = w * float(s)
        parts[key] = contrib
        total += contrib

    if frozen:
        total -= 100.0  # hard penalty; also excluded from best below

    return {
        "score": total,
        "frozen": frozen,
        "score_parts": parts,
        "eligible_for_best": (not frozen) and bool(parts),
    }


def _fmt(x: float) -> str:
    if not _finite(x):
        return "nan"
    return f"{x:.4f}"


def _write_leaderboard_md(
    path: Path,
    *,
    run_dir: Path,
    steps: int,
    baseline_mean: dict[str, float],
    rows: list[dict[str, Any]],
    best: Optional[dict[str, Any]],
    score_mode: str,
) -> None:
    lines = [
        f"# PD run sweep: `{run_dir.name}` @ {steps} DDIM steps",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Score mode: **{score_mode}** (weights={DEFAULT_SCORE_WEIGHTS})",
        f"- Checkpoints evaluated: **{len(rows)}**",
        "",
    ]
    if best is not None:
        lines.extend(
            [
                "## Best checkpoint",
                "",
                f"- **`{best['name']}`** (`{best['checkpoint']}`)",
                f"- score = **{best['score']:.4f}**",
                "",
            ]
        )
    else:
        lines.extend(["## Best checkpoint", "", "- None eligible (all frozen or empty).", ""])

    cols = [
        "step",
        "score",
        "travel",
        "r2d_err",
        "r2d_acc",
        "fullbody",
        "ee",
        "skate",
        "frozen",
    ]
    lines.extend(
        [
            "## Leaderboard (sorted by score ↓)",
            "",
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join("---" for _ in cols) + " |",
        ]
    )
    ranked = sorted(rows, key=lambda r: (-r["score"], r["step"]))
    for r in ranked:
        m = r["mean_metrics"]
        mark = " **" if best is not None and r["name"] == best["name"] else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["step"]),
                    f"{r['score']:.4f}{mark}",
                    _fmt(m.get("root_travel_m", float("nan"))),
                    _fmt(m.get("constraint_root2d_err", float("nan"))),
                    _fmt(m.get("constraint_root2d_acc", float("nan"))),
                    _fmt(m.get("constraint_fullbody_keyframe", float("nan"))),
                    _fmt(m.get("constraint_end_effector", float("nan"))),
                    _fmt(m.get("foot_skate_ratio", float("nan"))),
                    "yes" if r["frozen"] else "",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## RP baseline (same steps)",
            "",
            "| Metric | RP |",
            "|--------|----|",
        ]
    )
    for k in ("root_travel_m",) + METRIC_KEYS:
        if k in baseline_mean:
            lines.append(f"| `{k}` | {_fmt(baseline_mean[k])} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Primary ranking is **constraint-first** (root2d / fullbody / EE); skate is down-weighted.",
            "- `step_0` is the Stage-1 init copy — useful baseline for whether Stage-2 helped.",
            "- Frozen rows (`travel` < floor) are penalized and not chosen as best.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def eval_student_checkpoint(
    *,
    tag: str,
    model_name: str,
    checkpoint: Path,
    example_dirs: list[Path],
    steps: int,
    post_processing: bool,
    device: str,
    output_dir: Path,
    fallback_seed: int,
    text_encoder: Any,
    save_npz: bool,
) -> dict[str, dict[str, float]]:
    """Like ``run_side`` but optionally skips writing motion.npz."""
    if save_npz:
        return run_side(
            tag=tag,
            model_name=model_name,
            checkpoint=checkpoint,
            example_dirs=example_dirs,
            steps=steps,
            post_processing=post_processing,
            device=device,
            output_dir=output_dir,
            fallback_seed=fallback_seed,
            text_encoder=text_encoder,
        )

    from kimodo.model.load_model import load_model

    print(f"\n=== [{tag}] load {model_name} @ {checkpoint} ===")
    model = load_model(
        modelname=model_name,
        device=device,
        checkpoint_path=str(checkpoint),
        text_encoder=text_encoder,
    )
    model.eval()
    results: dict[str, dict[str, float]] = {}
    for ex in example_dirs:
        name = ex.name
        print(f"[{tag}] {name}  steps={steps}")
        out = generate_one(
            model,
            ex,
            steps=steps,
            post_processing=post_processing,
            fallback_seed=fallback_seed,
        )
        results[name] = evaluate_motion(out, ex, model, torch.device(device))
        travel = results[name].get("root_travel_m", float("nan"))
        skate = results[name].get("foot_skate_ratio", float("nan"))
        print(f"    root_travel_m={travel:.4f}  foot_skate_ratio={skate:.4f}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def main() -> int:
    args = parse_args()
    _configure_eval_env(args)

    run_dir = args.run_dir.expanduser().resolve()
    try:
        checkpoints = discover_step_checkpoints(
            run_dir,
            min_step=args.min_step,
            max_step=args.max_step,
            every=max(1, int(args.every)),
            include_step0=bool(args.include_step0),
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    examples = discover_examples(args.examples_dir)
    if args.max_examples is not None:
        examples = examples[: max(0, args.max_examples)]
    if not examples:
        print("ERROR: no examples selected", file=sys.stderr)
        return 1

    try:
        baseline_ckpt = _resolve_checkpoint_dir(args.baseline_checkpoint, label="baseline")
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = Path("outputs/pd_eval") / f"{run_dir.name}_sweep_s{args.steps}"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    per_ckpt_dir = out_dir / "per_checkpoint"
    per_ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"PD run sweep | run={run_dir} | n_ckpt={len(checkpoints)} | steps={args.steps} | "
        f"device={args.device} | n_examples={len(examples)} | score_mode={args.score_mode}"
    )
    print("Checkpoints: " + ", ".join(f"step_{s}" for s, _ in checkpoints))

    text_encoder = _load_shared_text_encoder()

    # --- RP baseline once ---
    rp_metrics = run_side(
        tag="rp_baseline",
        model_name=args.baseline_model,
        checkpoint=baseline_ckpt,
        example_dirs=examples,
        steps=args.steps,
        post_processing=args.post_processing,
        device=args.device,
        output_dir=out_dir,
        fallback_seed=args.seed,
        text_encoder=text_encoder,
    )
    mean_rp = _mean_over_examples(rp_metrics)
    (out_dir / "rp_baseline.json").write_text(
        json.dumps({"mean_metrics": mean_rp, "per_example": rp_metrics}, indent=2),
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    for step, ckpt in checkpoints:
        metrics_path = per_ckpt_dir / f"step_{step}.json"
        if args.resume and metrics_path.is_file():
            print(f"\n=== resume: load cached {metrics_path.name} ===")
            cached = json.loads(metrics_path.read_text(encoding="utf-8"))
            mean_pd = cached["mean_metrics"]
            per_ex = cached.get("per_example", {})
        else:
            per_ex = eval_student_checkpoint(
                tag=f"step_{step}",
                model_name=args.student_model_name,
                checkpoint=ckpt,
                example_dirs=examples,
                steps=args.steps,
                post_processing=args.post_processing,
                device=args.device,
                output_dir=out_dir,
                fallback_seed=args.seed,
                text_encoder=text_encoder,
                save_npz=bool(args.save_npz),
            )
            mean_pd = _mean_over_examples(per_ex)
            payload = {
                "step": step,
                "checkpoint": str(ckpt),
                "mean_metrics": mean_pd,
                "per_example": per_ex,
                "delta_vs_rp": {
                    k: mean_pd.get(k, float("nan")) - mean_rp.get(k, float("nan"))
                    for k in sorted(set(METRIC_KEYS) | {"root_travel_m"})
                },
            }
            metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        scored = compute_score(
            mean_pd,
            mean_rp,
            mode=args.score_mode,
            weights=DEFAULT_SCORE_WEIGHTS,
            travel_floor=float(args.travel_floor),
        )
        row = {
            "step": step,
            "name": f"step_{step}",
            "checkpoint": str(ckpt),
            "mean_metrics": mean_pd,
            "delta_vs_rp": {
                k: mean_pd.get(k, float("nan")) - mean_rp.get(k, float("nan"))
                for k in sorted(set(METRIC_KEYS) | {"root_travel_m"})
            },
            **scored,
        }
        rows.append(row)
        print(
            f"  >> step_{step}: score={row['score']:.4f} "
            f"r2d_acc={mean_pd.get('constraint_root2d_acc', float('nan')):.4f} "
            f"fb={mean_pd.get('constraint_fullbody_keyframe', float('nan')):.4f} "
            f"skate={mean_pd.get('foot_skate_ratio', float('nan')):.4f}"
            + (" FROZEN" if row["frozen"] else "")
        )

    eligible = [r for r in rows if r["eligible_for_best"]]
    best = max(eligible, key=lambda r: (r["score"], -r["step"])) if eligible else None

    summary = {
        "run_dir": str(run_dir),
        "steps": args.steps,
        "score_mode": args.score_mode,
        "score_weights": DEFAULT_SCORE_WEIGHTS,
        "travel_floor": args.travel_floor,
        "baseline_model": args.baseline_model,
        "baseline_checkpoint": str(baseline_ckpt) if baseline_ckpt else None,
        "examples": [ex.name for ex in examples],
        "rp_baseline_mean": mean_rp,
        "checkpoints": rows,
        "best": best,
    }
    (out_dir / "ranking.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_leaderboard_md(
        out_dir / "leaderboard.md",
        run_dir=run_dir,
        steps=args.steps,
        baseline_mean=mean_rp,
        rows=rows,
        best=best,
        score_mode=args.score_mode,
    )

    print("\n" + "=" * 60)
    if best is None:
        print("No eligible best checkpoint.")
    else:
        print(f"BEST: {best['name']}  score={best['score']:.4f}")
        print(f"  path: {best['checkpoint']}")
        m = best["mean_metrics"]
        print(
            f"  travel={_fmt(m.get('root_travel_m', float('nan')))}  "
            f"r2d_err={_fmt(m.get('constraint_root2d_err', float('nan')))}  "
            f"r2d_acc={_fmt(m.get('constraint_root2d_acc', float('nan')))}  "
            f"fb={_fmt(m.get('constraint_fullbody_keyframe', float('nan')))}  "
            f"ee={_fmt(m.get('constraint_end_effector', float('nan')))}  "
            f"skate={_fmt(m.get('foot_skate_ratio', float('nan')))}"
        )
    print(f"Wrote {out_dir / 'leaderboard.md'}")
    print(f"Wrote {out_dir / 'ranking.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
