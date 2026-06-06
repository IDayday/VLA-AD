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

from scripts.last_vla_v2.pdm_scoring_utils import (
    PDM_COMPONENT_KEYS,
    TrajectoryScorer,
    average_score_dicts,
    json_default,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Last-VLA CoT corruption without future teacher leakage.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--chunk-cache-root", type=Path, required=True)
    parser.add_argument("--chunk-name-pattern", default="navtest_full_chunk_*")
    parser.add_argument("--metric-cache-dir", type=Path, default=None)
    parser.add_argument("--score-mode", choices=("pdm", "proxy"), default="pdm")
    parser.add_argument("--allow-proxy-scoring", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="fp32")
    parser.add_argument("--zero-all-cot", action="store_true")
    parser.add_argument("--zero-scene-cot", action="store_true")
    parser.add_argument("--zero-geometry-cot", action="store_true")
    parser.add_argument("--zero-dynamic-cot", action="store_true")
    parser.add_argument("--zero-fusion-cot", action="store_true")
    parser.add_argument("--zero-ego-cot", action="store_true")
    parser.add_argument("--zero-action-refine-cot", action="store_true")
    parser.add_argument("--zero-coarse-prior", action="store_true")
    parser.add_argument("--zero-cot-condition-branch", action="store_true")
    parser.add_argument("--raw-vlm-only", action="store_true")
    parser.add_argument("--cot-only-for-debug-only", action="store_true")
    parser.add_argument("--deterministic", action="store_true", default=True)
    return parser.parse_args()


def corruption_mode(args: argparse.Namespace) -> str:
    names = [
        "zero_all_cot",
        "zero_scene_cot",
        "zero_geometry_cot",
        "zero_dynamic_cot",
        "zero_fusion_cot",
        "zero_ego_cot",
        "zero_action_refine_cot",
        "zero_coarse_prior",
        "zero_cot_condition_branch",
        "raw_vlm_only",
        "cot_only_for_debug_only",
    ]
    active = [name for name in names if getattr(args, name)]
    return "+".join(active) if active else "none"


def install_hooks(planner, args: argparse.Namespace):
    if not getattr(planner.config, "use_last_vla", False):
        raise RuntimeError("Last-VLA corruption eval requires use_last_vla=True.")
    return None


def add_corruption_flags(data: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    data = dict(data)
    data["last_vla_corrupt_zero_all_cot"] = bool(args.zero_all_cot)
    data["last_vla_corrupt_zero_scene_cot"] = bool(args.zero_scene_cot)
    data["last_vla_corrupt_zero_geometry_cot"] = bool(args.zero_geometry_cot)
    data["last_vla_corrupt_zero_dynamic_cot"] = bool(args.zero_dynamic_cot)
    data["last_vla_corrupt_zero_fusion_cot"] = bool(args.zero_fusion_cot)
    data["last_vla_corrupt_zero_ego_cot"] = bool(args.zero_ego_cot)
    data["last_vla_corrupt_zero_action_refine_cot"] = bool(args.zero_action_refine_cot)
    data["last_vla_corrupt_zero_coarse_prior"] = bool(args.zero_coarse_prior)
    data["last_vla_corrupt_zero_cot_condition_branch"] = bool(args.zero_cot_condition_branch)
    data["last_vla_raw_vlm_only"] = bool(args.raw_vlm_only)
    data["last_vla_cot_only_for_debug_only"] = bool(args.cot_only_for_debug_only)
    return data


def _strip_train_only(action_input: Dict[str, Any]) -> Dict[str, Any]:
    blocked = ("target", "teacher_trajectory", "teacher_score", "gt_score", "oracle_best_of_k")
    return {key: value for key, value in action_input.items() if not any(token in key for token in blocked)}


def build_metrics_summary(
    *,
    args: argparse.Namespace,
    planner: Any,
    scorer: TrajectoryScorer,
    score_results: List[Dict[str, Any]],
    l1_values: List[float],
    num_rows: int,
) -> Dict[str, Any]:
    averaged = average_score_dicts(score_results)
    return {
        "score_mode": args.score_mode,
        "pdm_scoring_active": scorer.pdm_scoring_active,
        "proxy_scoring_active": scorer.proxy_scoring_active,
        "corruption_mode": corruption_mode(args),
        "target_teacher_tokens_disabled_in_eval": True,
        "cot_bottleneck_active": False,
        "raw_vlm_context_used": not bool(args.cot_only_for_debug_only),
        "cot_only_for_debug_only": bool(args.cot_only_for_debug_only),
        **{key: averaged.get(key) if args.score_mode == "pdm" else None for key in PDM_COMPONENT_KEYS},
        "trajectory_l1": averaged.get("trajectory_l1"),
        "mean_trajectory_l1": sum(l1_values) / len(l1_values) if l1_values else None,
        "proxy_score_mean": averaged.get("proxy_score") if args.score_mode == "proxy" else None,
        "num_samples": int(num_rows),
    }


def main() -> int:
    from scripts import eval_recogdrive_expert_pdm as base
    from navsim.agents.recogdrive.expert_cache import load_sample

    args = parse_args()
    if args.score_mode == "pdm" and args.metric_cache_dir is None:
        raise ValueError("--score-mode pdm requires --metric-cache-dir.")
    if args.score_mode == "proxy" and not args.allow_proxy_scoring:
        raise ValueError("--score-mode proxy requires --allow-proxy-scoring; proxy is smoke/debug only.")
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
    scorer = TrajectoryScorer(args.score_mode, args.metric_cache_dir)
    rows: List[Dict[str, Any]] = []
    l1_values: List[float] = []
    score_results: List[Dict[str, Any]] = []
    for chunk_dir, sample_path, record in base.sample_paths(base.chunk_dirs(args), args.max_samples):
        sample = load_sample(sample_path)
        vl_features, action_input = base.make_batch(sample, planner, device, dtype)
        action_data = add_corruption_flags(_strip_train_only(dict(action_input)), args)
        action_input = type(action_input)(data=action_data)
        with torch.no_grad():
            output = planner.get_action(vl_features, action_input, deterministic=args.deterministic)
        pred = output["pred_traj"].detach().float().cpu().squeeze(0)
        gt = sample.get("trajectory")
        l1 = float((pred - gt.float()).abs().mean().item()) if isinstance(gt, torch.Tensor) else None
        if l1 is not None:
            l1_values.append(l1)
        sample_token = str(sample.get("sample_token") or record.get("sample_token") or sample_path.stem)
        score = scorer.score(sample, sample_token, pred)
        score_results.append(score)
        components = {key: score.get(key) for key in PDM_COMPONENT_KEYS}
        rows.append(
            {
                "sample_token": sample_token,
                "chunk": chunk_dir.name,
                "score_mode": args.score_mode,
                "corruption_mode": corruption_mode(args),
                "target_teacher_tokens_disabled_in_eval": True,
                "cot_bottleneck_active": False,
                "raw_vlm_context_used": not bool(args.cot_only_for_debug_only),
                "cot_only_for_debug_only": bool(args.cot_only_for_debug_only),
                "trajectory_l1": l1,
                "proxy_score": score.get("proxy_score") if args.score_mode == "proxy" else None,
                **components,
            }
        )
    summary = build_metrics_summary(
        args=args,
        planner=planner,
        scorer=scorer,
        score_results=score_results,
        l1_values=l1_values,
        num_rows=len(rows),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rows.json").write_text(json.dumps(rows, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
