#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import eval_recogdrive_expert_pdm as base  # noqa: E402
from transformers.feature_extraction_utils import BatchFeature  # noqa: E402
from navsim.agents.recogdrive.expert_cache import load_sample  # noqa: E402
from navsim.common.dataclasses import Trajectory  # noqa: E402
from navsim.evaluate.pdm_score import pdm_score  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LaST-RD token corruption without exposing future targets.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/navsim"))
    parser.add_argument("--split", default="navtest")
    parser.add_argument("--feature-source", choices=("chunk", "disk", "chunk_or_online"), default="chunk")
    parser.add_argument("--chunk-cache-dir", type=Path, default=None)
    parser.add_argument("--chunk-cache-root", type=Path, default=None)
    parser.add_argument("--chunk-name-pattern", default="navtest_full_chunk_*")
    parser.add_argument("--metric-cache-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="fp32")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--allow-noop", action="store_true")
    parser.add_argument("--zero-jepa-dynamic", action="store_true")
    parser.add_argument("--shuffle-jepa-dynamic", action="store_true")
    parser.add_argument("--zero-vggt-geometry", action="store_true")
    parser.add_argument("--shuffle-vggt-geometry", action="store_true")
    parser.add_argument("--zero-ego-tokens", action="store_true")
    parser.add_argument("--shuffle-ego-tokens", action="store_true")
    parser.add_argument("--zero-risk-tokens", action="store_true")
    parser.add_argument("--shuffle-risk-tokens", action="store_true")
    parser.add_argument("--zero-all-last-rd", action="store_true")
    return parser.parse_args()


def corruption_modes(args: argparse.Namespace) -> List[str]:
    modes = []
    for name in (
        "zero_jepa_dynamic",
        "shuffle_jepa_dynamic",
        "zero_vggt_geometry",
        "shuffle_vggt_geometry",
        "zero_ego_tokens",
        "shuffle_ego_tokens",
        "zero_risk_tokens",
        "shuffle_risk_tokens",
        "zero_all_last_rd",
    ):
        if getattr(args, name):
            modes.append(name)
    return modes or ["none"]


def _shuffle_batch(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.shape[0] > 1:
        return tokens[torch.randperm(tokens.shape[0], device=tokens.device)]
    return tokens.flip(1)


def corrupt_action_input(action_input: BatchFeature, args: argparse.Namespace) -> BatchFeature:
    data = dict(action_input)
    if args.zero_all_last_rd or args.zero_jepa_dynamic:
        if "jepa_context_tokens" in data:
            data["jepa_context_tokens"] = torch.zeros_like(data["jepa_context_tokens"])
        data["last_rd_zero_dynamic_tokens"] = True
    elif args.shuffle_jepa_dynamic and "jepa_context_tokens" in data:
        data["jepa_context_tokens"] = _shuffle_batch(data["jepa_context_tokens"])
        data["last_rd_shuffle_dynamic_tokens"] = True

    if args.zero_all_last_rd or args.zero_vggt_geometry:
        for key in ("vggt_context_tokens", "vggt_geometry_tokens", "vggt_depth_tokens", "vggt_pointmap_tokens"):
            if key in data:
                data[key] = torch.zeros_like(data[key])
        data["last_rd_zero_geometry_tokens"] = True
    elif args.shuffle_vggt_geometry:
        for key in ("vggt_context_tokens", "vggt_geometry_tokens", "vggt_depth_tokens", "vggt_pointmap_tokens"):
            if key in data:
                data[key] = _shuffle_batch(data[key])
        data["last_rd_shuffle_geometry_tokens"] = True

    if args.zero_all_last_rd or args.zero_ego_tokens:
        data["last_rd_zero_ego_tokens"] = True
    elif args.shuffle_ego_tokens:
        data["last_rd_shuffle_ego_tokens"] = True
    if args.zero_all_last_rd or args.zero_risk_tokens:
        data["last_rd_zero_risk_tokens"] = True
    elif args.shuffle_risk_tokens:
        data["last_rd_shuffle_risk_tokens"] = True
    if args.zero_all_last_rd:
        data["last_rd_zero_all"] = True
    return BatchFeature(data=data)


def make_batch(sample: Dict[str, Any], planner, device: torch.device, dtype: torch.dtype, args: argparse.Namespace) -> Tuple[torch.Tensor, BatchFeature]:
    vl_features, action_input = base.make_batch(sample, planner, device, dtype)
    if "high_command_one_hot" in sample:
        action_input["high_command_one_hot"] = sample["high_command_one_hot"].float().unsqueeze(0).to(device=device, dtype=dtype)
    if "history_trajectory" in sample:
        action_input["history_trajectory"] = sample["history_trajectory"].float().unsqueeze(0).to(device=device, dtype=dtype)
    for key in ("vggt_geometry_tokens", "vggt_depth_tokens", "vggt_pointmap_tokens", "vggt_camera_tokens"):
        if key in sample:
            action_input[key] = sample[key].float().unsqueeze(0).to(device=device, dtype=dtype)
    return vl_features, corrupt_action_input(action_input, args)


def mean_or_none(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def main() -> int:
    args = parse_args()
    cfg = base.load_yaml(args.config)
    planner = base.build_planner(cfg)
    if not bool(getattr(planner.config, "use_last_rd", False)) and not args.allow_noop:
        raise RuntimeError("Corruption eval requires a checkpoint/config with use_last_rd=True. Pass --allow-noop to run anyway.")

    chunks = base.chunk_dirs(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = base.dtype_from_precision(args.precision) if device.type == "cuda" else torch.float32
    planner = planner.to(device)
    if dtype != torch.float32:
        planner = planner.to(dtype=dtype)
    base.load_checkpoint(planner, args.checkpoint)
    planner.eval()

    metric_cache_loader = base.build_metric_cache_loader(args.metric_cache_dir) if args.metric_cache_dir is not None else None
    pdm_tools = base.build_pdm_tools() if metric_cache_loader is not None else None
    paths = base.sample_paths(chunks, args.max_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    l1_values: List[float] = []
    pdm_values: Dict[str, List[float]] = {key: [] for key in ("score", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance")}
    missing_metric_cache = failed_pdm = 0
    for chunk_dir, sample_path, index_record in paths:
        sample = load_sample(sample_path)
        vl_features, action_input = make_batch(sample, planner, device, dtype, args)
        with torch.no_grad():
            output = planner.get_action(vl_features, action_input, deterministic=args.deterministic)
        pred = output["pred_traj"].detach().float().cpu().squeeze(0)
        row: Dict[str, Any] = {
            "sample_token": str(sample.get("sample_token", index_record.get("sample_token", sample_path.stem))),
            "scene_token": str(sample.get("scene_token", sample_path.stem)),
            "chunk": chunk_dir.name,
            "valid": None,
            "corruption_mode": "+".join(corruption_modes(args)),
        }
        if "trajectory" in sample:
            l1 = torch.nn.functional.l1_loss(pred, sample["trajectory"].float()).item()
            row["trajectory_l1"] = l1
            l1_values.append(l1)
        if metric_cache_loader is not None and pdm_tools is not None:
            token = row["sample_token"]
            if token not in metric_cache_loader.metric_cache_paths:
                row.update({"valid": False, "error": "missing_metric_cache"})
                missing_metric_cache += 1
            else:
                try:
                    metric_cache = metric_cache_loader.get_from_token(token)
                    future_sampling, simulator, scorer = pdm_tools
                    result = pdm_score(
                        metric_cache=metric_cache,
                        model_trajectory=Trajectory(poses=pred.numpy()),
                        future_sampling=future_sampling,
                        simulator=simulator,
                        scorer=scorer,
                    )
                    result_dict = asdict(result)
                    row.update({"valid": True, **result_dict})
                    for key, value in result_dict.items():
                        if key in pdm_values:
                            pdm_values[key].append(float(value))
                except Exception as exc:
                    failed_pdm += 1
                    row.update({"valid": False, "error": repr(exc)})
        rows.append(row)

    averages = {key: mean_or_none(values) for key, values in pdm_values.items()}
    metrics = {
        "corruption_mode": "+".join(corruption_modes(args)),
        "use_last_rd": bool(getattr(planner.config, "use_last_rd", False)),
        "target_teacher_tokens_disabled_in_eval": True,
        "num_samples": len(rows),
        "trajectory_l1": mean_or_none(l1_values),
        "num_pdm_valid": len(pdm_values["score"]),
        "num_pdm_missing_metric_cache": missing_metric_cache,
        "num_pdm_failed": failed_pdm,
        "PDMS": averages["score"],
        "NC": averages["no_at_fault_collisions"],
        "DAC": averages["drivable_area_compliance"],
        "TTC": averages["time_to_collision_within_bound"],
        "comfort": averages["comfort"],
        "EP": averages["ego_progress"],
        "DDC": averages["driving_direction_compliance"],
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "rows.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
