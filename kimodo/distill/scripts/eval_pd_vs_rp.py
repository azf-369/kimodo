#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Headless eval: PD student vs official G1-RP at the same denoise step count.

Default: no motion post-processing; ``--steps`` applies to both models.

Example:
  export TEXT_ENCODER_DEVICE=cpu
  RP=~/.cache/huggingface/hub/models--nvidia--Kimodo-G1-RP-v1/snapshots/<hash>
  python -m kimodo.distill.scripts.eval_pd_vs_rp \\
    --student-checkpoint outputs/pd_g1_rp_teacher_seed/stage_100to50_formal_cons/step_30000 \\
    --baseline-checkpoint "$RP" --steps 50 --device cuda \\
    --text-encoder-mode local --text-encoder-device cpu

  # Or omit --baseline-checkpoint to load g1-rp from HF cache automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import torch

from kimodo.constraints import load_constraints_lst
from kimodo.exports.motion_io import save_kimodo_npz
from kimodo.meta import parse_prompts_from_meta
from kimodo.metrics import (
    ContraintFollow,
    FootContactConsistency,
    FootSkateFromContacts,
    FootSkateFromHeight,
    FootSkateRatio,
    aggregate_metrics,
    clear_metrics,
    compute_metrics,
)
from kimodo.model.load_model import _build_local_text_encoder_conf, load_model
from kimodo.model.loading import instantiate_from_dict
from kimodo.tools import load_json, seed_everything, to_torch

METRIC_KEYS = (
    "foot_skate_from_pred_contacts",
    "foot_skate_from_height",
    "foot_skate_ratio",
    "foot_contact_consistency",
    "constraint_root2d_err",
    "constraint_root2d_acc",
    "constraint_fullbody_keyframe",
    "constraint_end_effector",
)


def _optional_path(value: str | None) -> Path | None:
    """argparse type: empty ``$RP`` must not become Path('.') → CWD."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare PD student checkpoint vs official RP at identical diffusion steps "
        "(headless; post-processing off by default).",
    )
    p.add_argument(
        "--student-checkpoint",
        type=Path,
        required=True,
        help="PD student dir with config.yaml + model.safetensors.",
    )
    p.add_argument(
        "--baseline-model",
        type=str,
        default="g1-rp",
        help="Official baseline short name (default: g1-rp).",
    )
    p.add_argument(
        "--baseline-checkpoint",
        type=_optional_path,
        default=None,
        help="Optional local RP checkpoint dir with config.yaml (skips HF). "
        "Empty / unset falls back to HF cache for --baseline-model.",
    )
    p.add_argument(
        "--examples-dir",
        type=Path,
        default=Path("kimodo/assets/demo/examples/kimodo-g1-rp"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: outputs/pd_eval/<student_stem>_vs_rp_s<steps>",
    )
    p.add_argument("--steps", type=int, default=50, help="DDIM steps for BOTH models (default: 50).")
    p.add_argument(
        "--post-processing",
        action="store_true",
        help="Enable motion post-processing (default: off).",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--seed", type=int, default=0, help="Fallback seed when meta has none.")
    p.add_argument(
        "--student-model-name",
        type=str,
        default="g1-rp",
        help="Registry name used only to select text-encoder / API wiring; weights come from checkpoint.",
    )
    p.add_argument(
        "--text-encoder-mode",
        type=str,
        default="local",
        choices=["local", "api", "auto"],
        help="Text encoder backend (default: local). Use local for headless eval without API.",
    )
    p.add_argument(
        "--text-encoder-device",
        type=str,
        default="cpu",
        help="Device for local LLM2Vec (default: cpu; keeps VRAM for denoiser).",
    )
    return p.parse_args()


def _configure_eval_env(args: argparse.Namespace) -> None:
    """Prefer local/offline text encoder so eval does not hang on API or HF hub."""
    os.environ["TEXT_ENCODER_DEVICE"] = args.text_encoder_device
    os.environ["TEXT_ENCODER_MODE"] = args.text_encoder_mode
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.is_dir():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _load_shared_text_encoder() -> Any:
    """Load LLM2Vec once; reused for baseline + student to save RAM and startup time."""
    print("Loading shared local text encoder (LLM2Vec)...")
    conf = _build_local_text_encoder_conf(text_encoder_fp32=False)
    encoder = instantiate_from_dict(conf)
    if hasattr(encoder, "eval"):
        encoder.eval()
    return encoder


def discover_examples(root: Path) -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Examples dir not found: {root}")
    found = sorted({m.parent for m in root.rglob("meta.json")})
    if not found:
        raise FileNotFoundError(f"No meta.json under {root}")
    return found


def _cfg_kwargs(meta: dict, model_cfg_type: str) -> dict[str, Any]:
    if model_cfg_type == "nocfg":
        return {"cfg_type": "nocfg"}
    cfg = meta.get("cfg")
    if isinstance(cfg, dict):
        if not cfg.get("enabled", True):
            return {"cfg_type": "nocfg"}
        return {
            "cfg_type": "separated",
            "cfg_weight": [
                float(cfg.get("text_weight", 2.0)),
                float(cfg.get("constraint_weight", 2.0)),
            ],
        }
    return {"cfg_type": model_cfg_type}


def generate_one(
    model,
    example_dir: Path,
    *,
    steps: int,
    post_processing: bool,
    fallback_seed: int,
) -> dict[str, Any]:
    meta = load_json(example_dir / "meta.json")
    texts, num_frames = parse_prompts_from_meta(meta, fps=model.fps)
    constraints_path = example_dir / "constraints.json"
    skeleton = model.motion_rep.skeleton
    constraint_lst = (
        load_constraints_lst(str(constraints_path), skeleton=skeleton)
        if constraints_path.is_file()
        else []
    )
    seed = meta.get("seed", fallback_seed)
    if seed is not None:
        seed_everything(int(seed))

    cfg_type = getattr(model.denoiser, "cfg_type", "separated")
    with torch.inference_mode():
        out = model(
            texts,
            num_frames,
            constraint_lst=constraint_lst,
            num_denoising_steps=int(steps),
            num_samples=1,
            multi_prompt=len(texts) > 1,
            num_transition_frames=5,
            post_processing=bool(post_processing),
            return_numpy=True,
            progress_bar=lambda x: x,
            **_cfg_kwargs(meta, cfg_type),
        )
    return out


def _root_travel_m(posed_joints) -> float:
    """Root XY path length in meters from posed joints (B,T,J,3) or (T,J,3)."""
    import numpy as np

    joints = np.asarray(posed_joints)
    if joints.ndim == 4:
        joints = joints[0]
    if joints.ndim != 3 or joints.shape[0] < 2:
        return 0.0
    root_xy = joints[:, 0, :2]
    return float(np.linalg.norm(np.diff(root_xy, axis=0), axis=1).sum())


def evaluate_motion(output: dict[str, Any], example_dir: Path, model, device: torch.device) -> dict[str, float]:
    skeleton = model.motion_rep.skeleton
    constraints_path = example_dir / "constraints.json"
    constraints_lst = (
        load_constraints_lst(str(constraints_path), skeleton=skeleton)
        if constraints_path.is_file()
        else []
    )
    posed = to_torch(output["posed_joints"], device=str(device))
    feet = to_torch(output["foot_contacts"], device=str(device))
    if posed.dim() == 3:
        posed = posed.unsqueeze(0)
        feet = feet.unsqueeze(0)
    lengths = torch.tensor([posed.shape[1]], dtype=torch.long, device=device)
    metrics_list = [
        FootSkateFromHeight(skeleton, model.fps),
        FootSkateFromContacts(skeleton, model.fps),
        FootSkateRatio(skeleton, model.fps),
        FootContactConsistency(skeleton, model.fps),
        ContraintFollow(skeleton),
    ]
    clear_metrics(metrics_list)
    compute_metrics(
        metrics_list,
        {
            "posed_joints": posed,
            "foot_contacts": feet,
            "lengths": lengths,
            "constraints_lst": [constraints_lst],
        },
    )
    aggregated = aggregate_metrics(metrics_list)

    def _scalar(t: torch.Tensor) -> float:
        return float("nan") if t.numel() == 0 else float(t.mean().item())

    out_m = {k: _scalar(aggregated[k]) for k in aggregated if k in METRIC_KEYS}
    # Primary freeze detector — do not trust skate alone.
    out_m["root_travel_m"] = _root_travel_m(output["posed_joints"])
    return out_m


def _save_npz(output: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    single = {
        k: (v[0] if hasattr(v, "shape") and getattr(v, "shape", None) is not None and len(v.shape) > 0 and v.shape[0] == 1 else v)
        for k, v in output.items()
    }
    save_kimodo_npz(str(path), single)


def run_side(
    *,
    tag: str,
    model_name: str,
    checkpoint: Optional[Path],
    example_dirs: list[Path],
    steps: int,
    post_processing: bool,
    device: str,
    output_dir: Path,
    fallback_seed: int,
    text_encoder: Any,
) -> dict[str, dict[str, float]]:
    print(f"\n=== [{tag}] load {model_name}" + (f" @ {checkpoint}" if checkpoint else " (HF/cache)") + " ===")
    model = load_model(
        modelname=model_name,
        device=device,
        checkpoint_path=str(checkpoint) if checkpoint is not None else None,
        text_encoder=text_encoder,
    )
    model.eval()
    results: dict[str, dict[str, float]] = {}
    for ex in example_dirs:
        name = ex.name
        print(f"[{tag}] {name}  steps={steps}  post_processing={post_processing}")
        out = generate_one(
            model,
            ex,
            steps=steps,
            post_processing=post_processing,
            fallback_seed=fallback_seed,
        )
        _save_npz(out, output_dir / tag / name / "motion.npz")
        results[name] = evaluate_motion(out, ex, model, torch.device(device))
        skate = results[name].get("foot_skate_ratio", float("nan"))
        travel = results[name].get("root_travel_m", float("nan"))
        print(f"    root_travel_m={travel:.4f}  foot_skate_ratio={skate:.4f}")
        if travel < 0.05:
            print("    WARNING: near-static motion (travel < 5cm) — do not trust skate")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def _mean_over_examples(per_ex: dict[str, dict[str, float]]) -> dict[str, float]:
    keys = set()
    for m in per_ex.values():
        keys.update(m.keys())
    out: dict[str, float] = {}
    for k in sorted(keys):
        vals = [m[k] for m in per_ex.values() if k in m and m[k] == m[k]]
        out[k] = sum(vals) / len(vals) if vals else float("nan")
    return out


def _resolve_checkpoint_dir(path: Optional[Path], *, label: str) -> Optional[Path]:
    """Validate a checkpoint folder; treat missing path as None (HF fallback).

    Common failure: ``--baseline-checkpoint "$RP"`` with RP unset → empty → CWD.
    """
    if path is None:
        return None
    ckpt = path.expanduser().resolve()
    if not (ckpt / "config.yaml").is_file():
        raise FileNotFoundError(
            f"{label} checkpoint is invalid: {ckpt}\n"
            f"  Expected config.yaml (+ model weights) in that directory.\n"
            f"  If you passed --baseline-checkpoint \"$RP\", export RP first, e.g.\n"
            f"    RP=~/.cache/huggingface/hub/models--nvidia--Kimodo-G1-RP-v1/snapshots/<hash>\n"
            f"  Or omit --baseline-checkpoint to load from Hugging Face cache."
        )
    return ckpt


def main() -> int:
    args = parse_args()
    _configure_eval_env(args)

    try:
        student_ckpt = _resolve_checkpoint_dir(args.student_checkpoint, label="student")
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    assert student_ckpt is not None
    if not (student_ckpt / "model.safetensors").is_file():
        print(f"ERROR: missing model.safetensors in {student_ckpt}", file=sys.stderr)
        return 1

    examples = discover_examples(args.examples_dir)
    if args.max_examples is not None:
        examples = examples[: max(0, args.max_examples)]
    if not examples:
        print("ERROR: no examples selected", file=sys.stderr)
        return 1

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = Path("outputs/pd_eval") / f"{student_ckpt.name}_vs_rp_s{args.steps}"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        baseline_ckpt = _resolve_checkpoint_dir(args.baseline_checkpoint, label="baseline")
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"PD vs RP headless eval | steps={args.steps} (both) | "
        f"post_processing={args.post_processing} | device={args.device} | "
        f"n_examples={len(examples)} | TEXT_ENCODER_MODE={os.environ.get('TEXT_ENCODER_MODE')} "
        f"| TEXT_ENCODER_DEVICE={os.environ.get('TEXT_ENCODER_DEVICE')} "
        f"| HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', '0')}"
    )

    text_encoder = _load_shared_text_encoder()

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
    student_metrics = run_side(
        tag="pd_student",
        model_name=args.student_model_name,
        checkpoint=student_ckpt,
        example_dirs=examples,
        steps=args.steps,
        post_processing=args.post_processing,
        device=args.device,
        output_dir=out_dir,
        fallback_seed=args.seed,
        text_encoder=text_encoder,
    )

    mean_rp = _mean_over_examples(rp_metrics)
    mean_pd = _mean_over_examples(student_metrics)

    summary = {
        "student_checkpoint": str(student_ckpt),
        "baseline_model": args.baseline_model,
        "baseline_checkpoint": str(baseline_ckpt) if baseline_ckpt else None,
        "steps": args.steps,
        "post_processing": args.post_processing,
        "examples": [ex.name for ex in examples],
        "mean_metrics": {
            "rp_baseline": mean_rp,
            "pd_student": mean_pd,
            "delta_student_minus_rp": {
                k: mean_pd.get(k, float("nan")) - mean_rp.get(k, float("nan"))
                for k in sorted(set(METRIC_KEYS) | {"root_travel_m"})
                if k in mean_rp or k in mean_pd
            },
        },
        "per_example": {
            name: {
                "rp_baseline": rp_metrics[name],
                "pd_student": student_metrics[name],
                "delta": {
                    k: student_metrics[name].get(k, float("nan")) - rp_metrics[name].get(k, float("nan"))
                    for k in sorted(set(METRIC_KEYS) | {"root_travel_m"})
                    if k in rp_metrics[name] or k in student_metrics[name]
                },
            }
            for name in rp_metrics
        },
    }
    (out_dir / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"# PD student vs RP @ {args.steps} DDIM steps",
        "",
        f"- Student: `{student_ckpt}`",
        f"- Baseline: `{args.baseline_model}`"
        + (f" (`{baseline_ckpt}`)" if baseline_ckpt else ""),
        f"- Steps (both): **{args.steps}**",
        f"- Post-processing: **{args.post_processing}**",
        "",
        "## Motion check (primary)",
        "",
        "| Metric | RP | Student | Δ (S−RP) |",
        "|--------|----|---------|----------|",
    ]
    for k in ("root_travel_m",):
        b, c = mean_rp.get(k, float("nan")), mean_pd.get(k, float("nan"))
        lines.append(f"| `{k}` | {b:.4f} | {c:.4f} | {c - b:+.4f} |")
    lines.extend(
        [
            "",
            "> `root_travel_m` ≪ RP (e.g. <0.05) ⇒ frozen / collapsed. Do not trust skate alone.",
            "",
            "## Mean metrics (lower skate / constraint err is better if travel is healthy)",
            "",
            "| Metric | RP | Student | Δ (S−RP) |",
            "|--------|----|---------|----------|",
        ]
    )
    for k in METRIC_KEYS:
        if k not in mean_rp and k not in mean_pd:
            continue
        b, c = mean_rp.get(k, float("nan")), mean_pd.get(k, float("nan"))
        d = c - b
        lines.append(f"| `{k}` | {b:.4f} | {c:.4f} | {d:+.4f} |")
    lines.extend(["", f"Artifacts under `{out_dir}` (npz + this report).", ""])
    (out_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"Wrote {out_dir / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
