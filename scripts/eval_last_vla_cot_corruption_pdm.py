#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Last-VLA CoT corruption without future teacher leakage.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--chunk-cache-root", type=Path, required=True)
    parser.add_argument("--chunk-name-pattern", default="navtest_full_chunk_*")
    parser.add_argument("--metric-cache-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="fp32")
    parser.add_argument("--zero-all-cot", action="store_true")
    parser.add_argument("--zero-geometry-cot", action="store_true")
    parser.add_argument("--zero-dynamic-cot", action="store_true")
    parser.add_argument("--zero-ego-cot", action="store_true")
    parser.add_argument("--zero-action-refine-cot", action="store_true")
    parser.add_argument("--zero-coarse-prior", action="store_true")
    parser.add_argument("--use-raw-vlm-context-ablation", action="store_true")
    parser.add_argument("--drop-vlm-summary", action="store_true")
    parser.add_argument("--deterministic", action="store_true", default=True)
    return parser.parse_args()


def corruption_mode(args: argparse.Namespace) -> str:
    names = [
        "zero_all_cot",
        "zero_geometry_cot",
        "zero_dynamic_cot",
        "zero_ego_cot",
        "zero_action_refine_cot",
        "zero_coarse_prior",
        "use_raw_vlm_context_ablation",
        "drop_vlm_summary",
    ]
    active = [name for name in names if getattr(args, name)]
    return "+".join(active) if active else "none"


def install_hooks(planner, args: argparse.Namespace):
    if not getattr(planner.config, "use_last_vla", False):
        raise RuntimeError("Last-VLA corruption eval requires use_last_vla=True.")
    if args.use_raw_vlm_context_ablation:
        planner.config.last_vla_cot_bottleneck_mode = False
        planner.config.last_vla_raw_vlm_context_to_dit = True
        planner.last_vla_cot.config.cot_bottleneck_mode = False
        planner.last_vla_cot.config.raw_vlm_context_to_dit = True
    if args.drop_vlm_summary:
        planner.last_vla_cot.config.vlm_summary_tokens = 0

    if not any((args.zero_all_cot, args.zero_geometry_cot, args.zero_dynamic_cot, args.zero_ego_cot, args.zero_action_refine_cot, args.zero_coarse_prior)):
        return None

    original = planner.last_vla_cot.forward

    def wrapped(*forward_args, **forward_kwargs):
        out = original(*forward_args, **forward_kwargs)
        if args.zero_all_cot:
            out.cot_tokens.zero_()
            out.planner_context_tokens[:, : out.cot_tokens.shape[1]].zero_()
        for flag, key in (
            (args.zero_geometry_cot, "geometry"),
            (args.zero_dynamic_cot, "dynamic"),
            (args.zero_ego_cot, "ego"),
            (args.zero_action_refine_cot, "action_refine"),
        ):
            if flag and key in out.cot_tokens_by_step:
                out.cot_tokens_by_step[key].zero_()
        if args.zero_coarse_prior:
            out.coarse_traj_norm.zero_()
        return out

    planner.last_vla_cot.forward = wrapped
    return original


def _strip_train_only(action_input: Dict[str, Any]) -> Dict[str, Any]:
    blocked = ("target", "teacher_trajectory", "teacher_score", "gt_score", "oracle_best_of_k")
    return {key: value for key, value in action_input.items() if not any(token in key for token in blocked)}


def main() -> int:
    from scripts import eval_recogdrive_expert_pdm as base
    from navsim.agents.recogdrive.expert_cache import load_sample

    args = parse_args()
    cfg = base.load_yaml(args.config)
    planner = base.build_planner(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = base.dtype_from_precision(args.precision) if device.type == "cuda" else torch.float32
    planner = planner.to(device)
    if dtype != torch.float32:
        planner = planner.to(dtype=dtype)
    base.load_checkpoint(planner, args.checkpoint)
    planner.eval()
    install_hooks(planner, args)
    rows: List[Dict[str, Any]] = []
    l1_values: List[float] = []
    for chunk_dir, sample_path, record in base.sample_paths(base.chunk_dirs(args), args.max_samples):
        sample = load_sample(sample_path)
        vl_features, action_input = base.make_batch(sample, planner, device, dtype)
        action_input = type(action_input)(data=_strip_train_only(dict(action_input)))
        with torch.no_grad():
            output = planner.get_action(vl_features, action_input, deterministic=args.deterministic)
        pred = output["pred_traj"].detach().float().cpu().squeeze(0)
        gt = sample.get("trajectory")
        l1 = float((pred - gt.float()).abs().mean().item()) if isinstance(gt, torch.Tensor) else None
        if l1 is not None:
            l1_values.append(l1)
        rows.append(
            {
                "sample_token": str(sample.get("sample_token") or record.get("sample_token") or sample_path.stem),
                "chunk": chunk_dir.name,
                "corruption_mode": corruption_mode(args),
                "target_teacher_tokens_disabled_in_eval": True,
                "cot_bottleneck_active": bool(planner.config.last_vla_cot_bottleneck_mode),
                "raw_vlm_context_used": bool(planner.config.last_vla_raw_vlm_context_to_dit and not planner.config.last_vla_cot_bottleneck_mode),
                "trajectory_l1": l1,
            }
        )
    summary = {
        "corruption_mode": corruption_mode(args),
        "target_teacher_tokens_disabled_in_eval": True,
        "cot_bottleneck_active": bool(planner.config.last_vla_cot_bottleneck_mode),
        "raw_vlm_context_used": bool(planner.config.last_vla_raw_vlm_context_to_dit and not planner.config.last_vla_cot_bottleneck_mode),
        "mean_trajectory_l1": sum(l1_values) / len(l1_values) if l1_values else None,
        "num_samples": len(rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rows.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
