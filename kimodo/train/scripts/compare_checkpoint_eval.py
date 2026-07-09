#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare a custom checkpoint against an official/baseline Kimodo model on G1 examples."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

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
from kimodo.model.flow_matching import FlowMatchingLoss
from kimodo.model.load_model import load_model
from kimodo.tools import load_json, seed_everything, to_torch
from kimodo.train.collate import collate_motion_batch
from kimodo.train.dataset import G1SeedTrainingDataset
from kimodo.train.flow_train import flow_matching_batch_step
from kimodo.train.text_embedding import TextEmbeddingProvider


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


@dataclass
class ExampleSpec:
    name: str
    path: Path
    texts: list[str]
    num_frames: list[int]
    seed: Optional[int]
    steps: int
    cfg_type: Optional[str]
    cfg_weight: Any
    has_constraints: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate on G1 demo examples and compare motion-quality metrics "
        "between a baseline model and a candidate checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Candidate checkpoint directory (config.yaml + model.safetensors).",
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default="g1-seed",
        help="Baseline model short name (default: official Kimodo-G1-SEED-v1).",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=None,
        help="Optional local baseline checkpoint dir; overrides HF download for baseline.",
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=Path("kimodo/assets/demo/examples/kimodo-g1-rp"),
        help="Root directory containing demo-style example subfolders (meta.json each).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/compare_eval"),
        help="Directory for generated motions and comparison reports.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--baseline-steps",
        type=int,
        default=None,
        help="Sampling steps for baseline (DDIM or ODE). Default: meta diffusion_steps or 100/20.",
    )
    parser.add_argument(
        "--candidate-steps",
        type=int,
        default=20,
        help="Sampling steps for candidate checkpoint (ODE for flow matching).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit number of examples (useful for quick smoke comparisons).",
    )
    parser.add_argument(
        "--eval-loss",
        action="store_true",
        help="Also compute FM training loss on held-out G1 clips (flow_matching models only).",
    )
    parser.add_argument("--data-root", type=Path, default=Path("datasets/bones-seed"))
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--loss-max-files", type=int, default=8)
    parser.add_argument("--loss-max-frames", type=int, default=120)
    return parser.parse_args()


def discover_examples(root: Path) -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Examples directory not found: {root}")
    examples = sorted({meta.parent for meta in root.rglob("meta.json")})
    if not examples:
        raise FileNotFoundError(f"No example folders with meta.json under {root}")
    return examples


def _resolve_cfg_from_meta(meta: dict, model_cfg_type: str) -> dict[str, Any]:
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


def load_example_spec(example_dir: Path, fps: float, default_steps: int) -> ExampleSpec:
    meta_path = example_dir / "meta.json"
    meta = load_json(meta_path)
    texts, num_frames = parse_prompts_from_meta(meta, fps=fps)
    steps = int(meta.get("diffusion_steps", default_steps))
    constraints_path = example_dir / "constraints.json"
    return ExampleSpec(
        name=example_dir.name,
        path=example_dir,
        texts=texts,
        num_frames=num_frames,
        seed=meta.get("seed"),
        steps=steps,
        cfg_type=None,
        cfg_weight=None,
        has_constraints=constraints_path.is_file(),
    )


def _default_steps_for_model(model) -> int:
    if getattr(model, "generative_paradigm", "diffusion") == "flow_matching":
        return 20
    return 100


def _model_cfg_type(model) -> str:
    return getattr(model.denoiser, "cfg_type", "separated")


def compute_fm_loss(
    model,
    checkpoint_dir: Path,
    *,
    data_root: Path,
    split_path: Optional[Path],
    max_files: int,
    max_frames: int,
    device: torch.device,
) -> Optional[float]:
    if getattr(model, "generative_paradigm", "diffusion") != "flow_matching":
        return None

    from omegaconf import OmegaConf

    from kimodo.train.build import build_motion_rep

    cfg = OmegaConf.load(checkpoint_dir / "config.yaml")
    stats_path = checkpoint_dir / "stats" / "motion"
    denoiser = model.denoiser.model
    motion_rep = model.motion_rep
    no_text = denoiser.root_model.num_text_tokens == 0

    cpu_motion_rep = build_motion_rep(cfg.denoiser, stats_path=str(stats_path))
    dataset = G1SeedTrainingDataset(
        data_root,
        split_path=split_path,
        max_files=max_files,
        max_frames=max_frames,
        motion_rep=cpu_motion_rep,
        normalize=True,
        require_text=not no_text,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_motion_batch)

    llm_dim = denoiser.root_model.embed_text.in_features
    num_tokens = denoiser.root_model.num_text_tokens
    text_provider = TextEmbeddingProvider(
        num_tokens=num_tokens,
        llm_dim=llm_dim,
        device=device,
        mode="dummy",
    )
    flow_loss = FlowMatchingLoss()
    losses: list[float] = []

    cfg_dropout = None
    if not no_text:
        cfg_dropout = {"uncond": 0.1, "text_only": 0.1, "constraint_only": 0.1}

    with torch.no_grad():
        for batch in loader:
            loss, _ = flow_matching_batch_step(
                denoiser,
                motion_rep,
                batch,
                flow_loss,
                text_provider,
                cfg_dropout=cfg_dropout,
                constraint_prob=0.8,
                max_keyframes=4,
            )
            losses.append(float(loss.item()))

    if not losses:
        return None
    return sum(losses) / len(losses)


def generate_for_example(
    model,
    spec: ExampleSpec,
    *,
    steps: int,
    skeleton,
    multi_prompt: bool,
) -> dict[str, Any]:
    constraints_path = spec.path / "constraints.json"
    constraint_lst = (
        load_constraints_lst(str(constraints_path), skeleton=skeleton) if spec.has_constraints else []
    )

    if spec.seed is not None:
        seed_everything(spec.seed)

    cfg_kwargs = _resolve_cfg_from_meta(load_json(spec.path / "meta.json"), _model_cfg_type(model))

    with torch.inference_mode():
        output = model(
            spec.texts,
            spec.num_frames,
            constraint_lst=constraint_lst,
            num_denoising_steps=steps,
            num_samples=1,
            multi_prompt=multi_prompt,
            num_transition_frames=5,
            post_processing=False,
            return_numpy=True,
            progress_bar=lambda x: x,
            **cfg_kwargs,
        )
    return output


def _scalar_metric(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return float("nan")
    return float(value.mean().item())


def evaluate_motion(
    output: dict[str, Any],
    *,
    constraints_lst: list,
    skeleton,
    fps: float,
    device: torch.device,
) -> dict[str, float]:
    posed_joints = to_torch(output["posed_joints"], device=str(device))
    foot_contacts = to_torch(output["foot_contacts"], device=str(device))
    if posed_joints.dim() == 3:
        posed_joints = posed_joints.unsqueeze(0)
        foot_contacts = foot_contacts.unsqueeze(0)

    lengths = torch.tensor([posed_joints.shape[1]], dtype=torch.long, device=device)
    metrics_list = [
        FootSkateFromHeight(skeleton, fps),
        FootSkateFromContacts(skeleton, fps),
        FootSkateRatio(skeleton, fps),
        FootContactConsistency(skeleton, fps),
        ContraintFollow(skeleton),
    ]
    clear_metrics(metrics_list)
    compute_metrics(
        metrics_list,
        {
            "posed_joints": posed_joints,
            "foot_contacts": foot_contacts,
            "lengths": lengths,
            "constraints_lst": [constraints_lst],
        },
    )
    aggregated = aggregate_metrics(metrics_list)
    return {key: _scalar_metric(aggregated[key]) for key in aggregated if key in METRIC_KEYS}


def _load_model_bundle(name: str, checkpoint_path: Optional[Path], device: str):
    print(f"Loading model: {name}" + (f" from {checkpoint_path}" if checkpoint_path else " (official/HF)"))
    model = load_model(
        modelname=name,
        device=device,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
    )
    model.eval()
    skeleton = model.motion_rep.skeleton
    return model, skeleton


def _save_motion(output: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    single = {
        key: (value[0] if hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] == 1 else value)
        for key, value in output.items()
    }
    save_kimodo_npz(str(path), single)


def _compare_metrics(baseline: dict[str, float], candidate: dict[str, float]) -> dict[str, dict[str, float]]:
    comparison: dict[str, dict[str, float]] = {}
    for key in METRIC_KEYS:
        b = baseline.get(key)
        c = candidate.get(key)
        if b is None and c is None:
            continue
        delta = None if b is None or c is None else c - b
        comparison[key] = {"baseline": b, "candidate": c, "delta": delta}
    return comparison


def _format_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Example | Metric | Baseline | Candidate | Delta |",
        "|---------|--------|----------|-----------|-------|",
    ]
    for row in rows:
        delta = row["delta"]
        delta_str = "—" if delta is None else f"{delta:+.4f}"
        b = row["baseline"]
        c = row["candidate"]
        b_str = "—" if b is None else f"{b:.4f}"
        c_str = "—" if c is None else f"{c:.4f}"
        lines.append(f"| {row['example']} | {row['metric']} | {b_str} | {c_str} | {delta_str} |")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    examples_root = args.examples_dir.resolve()
    candidate_ckpt = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (candidate_ckpt / "config.yaml").is_file():
        print(f"ERROR: missing config.yaml in {candidate_ckpt}", file=sys.stderr)
        return 1

    example_dirs = discover_examples(examples_root)
    if args.max_examples is not None:
        example_dirs = example_dirs[: args.max_examples]

    baseline_model, baseline_skel = _load_model_bundle(
        args.baseline_model,
        args.baseline_checkpoint,
        str(device),
    )
    baseline_default_steps = args.baseline_steps or _default_steps_for_model(baseline_model)
    baseline_specs = [
        load_example_spec(path, baseline_model.fps, baseline_default_steps) for path in example_dirs
    ]

    baseline_results: dict[str, Any] = {}
    for spec in baseline_specs:
        steps = args.baseline_steps or spec.steps
        multi_prompt = len(spec.texts) > 1
        print(f"[baseline] generating {spec.name} (steps={steps})")
        output = generate_for_example(
            baseline_model,
            spec,
            steps=steps,
            skeleton=baseline_skel,
            multi_prompt=multi_prompt,
        )
        motion_path = output_dir / "baseline" / spec.name / "motion.npz"
        _save_motion(output, motion_path)

        constraints = (
            load_constraints_lst(str(spec.path / "constraints.json"), skeleton=baseline_skel)
            if spec.has_constraints
            else []
        )
        metrics = evaluate_motion(
            output,
            constraints_lst=constraints,
            skeleton=baseline_skel,
            fps=baseline_model.fps,
            device=device,
        )
        baseline_results[spec.name] = metrics

    baseline_loss = None
    if args.eval_loss and args.baseline_checkpoint is not None:
        baseline_loss = compute_fm_loss(
            baseline_model,
            args.baseline_checkpoint.resolve(),
            data_root=args.data_root,
            split_path=args.split_path,
            max_files=args.loss_max_files,
            max_frames=args.loss_max_frames,
            device=device,
        )

    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    candidate_model, candidate_skel = _load_model_bundle(
        "g1-seed",
        candidate_ckpt,
        str(device),
    )
    candidate_default_steps = args.candidate_steps
    candidate_specs = [
        load_example_spec(path, candidate_model.fps, candidate_default_steps) for path in example_dirs
    ]

    candidate_results: dict[str, Any] = {}
    for spec in candidate_specs:
        steps = args.candidate_steps
        multi_prompt = len(spec.texts) > 1
        print(f"[candidate] generating {spec.name} (steps={steps})")
        output = generate_for_example(
            candidate_model,
            spec,
            steps=steps,
            skeleton=candidate_skel,
            multi_prompt=multi_prompt,
        )
        motion_path = output_dir / "candidate" / spec.name / "motion.npz"
        _save_motion(output, motion_path)

        constraints = (
            load_constraints_lst(str(spec.path / "constraints.json"), skeleton=candidate_skel)
            if spec.has_constraints
            else []
        )
        metrics = evaluate_motion(
            output,
            constraints_lst=constraints,
            skeleton=candidate_skel,
            fps=candidate_model.fps,
            device=device,
        )
        candidate_results[spec.name] = metrics

    candidate_loss = None
    if args.eval_loss:
        candidate_loss = compute_fm_loss(
            candidate_model,
            candidate_ckpt,
            data_root=args.data_root,
            split_path=args.split_path,
            max_files=args.loss_max_files,
            max_frames=args.loss_max_frames,
            device=device,
        )

    per_example: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    for spec in baseline_specs:
        name = spec.name
        comparison = _compare_metrics(baseline_results[name], candidate_results[name])
        per_example.append(
            {
                "example": name,
                "has_constraints": spec.has_constraints,
                "metrics": comparison,
            }
        )
        for metric, values in comparison.items():
            table_rows.append(
                {
                    "example": name,
                    "metric": metric,
                    "baseline": values["baseline"],
                    "candidate": values["candidate"],
                    "delta": values["delta"],
                }
            )

    summary = {
        "baseline_model": args.baseline_model,
        "baseline_checkpoint": str(args.baseline_checkpoint) if args.baseline_checkpoint else None,
        "candidate_checkpoint": str(candidate_ckpt),
        "examples_dir": str(examples_root),
        "baseline_steps": baseline_default_steps,
        "candidate_steps": args.candidate_steps,
        "fm_eval_loss": {
            "baseline": baseline_loss,
            "candidate": candidate_loss,
        },
        "per_example": per_example,
    }

    summary_path = output_dir / "comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = output_dir / "comparison.md"
    report_lines = [
        "# Checkpoint comparison report",
        "",
        f"- **Baseline**: `{args.baseline_model}`"
        + (f" (`{args.baseline_checkpoint}`)" if args.baseline_checkpoint else " (official/HF)"),
        f"- **Candidate**: `{candidate_ckpt}`",
        f"- **Examples**: `{examples_root}`",
        f"- **Baseline steps**: {baseline_default_steps}",
        f"- **Candidate steps**: {args.candidate_steps}",
        "",
    ]
    if args.eval_loss:
        report_lines.extend(
            [
                "## FM training loss (held-out G1 clips)",
                "",
                f"- Baseline: {baseline_loss if baseline_loss is not None else 'N/A (not flow_matching)'}",
                f"- Candidate: {candidate_loss if candidate_loss is not None else 'N/A'}",
                "",
            ]
        )
    report_lines.extend(
        [
            "## Per-example metrics (candidate − baseline)",
            "",
            _format_table(table_rows),
            "",
            "Lower foot-skate and constraint errors are better. "
            "Positive `constraint_root2d_acc` delta means candidate follows root constraints more often.",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nWrote {summary_path}")
    print(f"Wrote {report_path}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
