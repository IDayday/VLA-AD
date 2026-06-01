#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from navsim.agents.recogdrive.expert_cache import load_sample
from navsim.planning.script.run_training_recogdrive import ChunkCacheDataset, custom_collate_fn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small LaST-RD real chunk batch smoke without training.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--stage", choices=("stage1_5", "progressive_sft", "both"), default="both")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "reports" / "last_rd_real_batch_smoke.json")
    return parser.parse_args()


def build_dataset(cache_root: Path, include_targets: bool, use_last_rd: bool = True) -> ChunkCacheDataset:
    return ChunkCacheDataset(
        str(cache_root),
        include_expert_features=use_last_rd,
        include_expert_targets=include_targets,
        use_jepa=True,
        use_vggt=True,
        use_last_rd=use_last_rd,
        future_jepa_loss_weight=0.3,
        vggt_geometry_loss_weight=0.1,
    )


def take_batch(dataset: ChunkCacheDataset, max_samples: int):
    count = min(max_samples, len(dataset))
    return custom_collate_fn([dataset[idx] for idx in range(count)])


def sample_geometry_mode(sample: Dict[str, Any]) -> str:
    raw = sample.get("vggt_geometry_mode")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str) and raw in {"full_geometry", "patch_fallback", "no_geometry", "missing"}:
        return raw
    if "vggt_geometry_tokens" in sample:
        return "missing"
    if "vggt_context_tokens" in sample:
        return "patch_fallback"
    return "no_geometry"


def geometry_mode_distribution(dataset: ChunkCacheDataset, max_samples: int) -> Dict[str, int]:
    modes: Counter[str] = Counter()
    for _, sample_path, _ in dataset.records[:min(max_samples, len(dataset.records))]:
        modes[sample_geometry_mode(load_sample(sample_path))] += 1
    return dict(sorted(modes.items()))


def build_agent(stage: str) -> ReCogDriveAgent:
    stage1_5 = stage == "stage1_5"
    return ReCogDriveAgent(
        trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
        checkpoint_path="",
        allow_random_init=True,
        cache_hidden_state=True,
        dit_type="small",
        use_expert_features=not stage1_5,
        use_jepa=True,
        use_vggt=True,
        allow_expert_target_features=True,
        use_last_rd=True,
        last_rd_stage=stage,
        diffusion_loss_weight=0.0 if stage1_5 else 1.0,
        expert_alignment_weight=0.0,
        jepa_alignment_weight=0.0,
        vggt_alignment_weight=0.0,
        future_jepa_loss_weight=0.30 if stage1_5 else 0.10,
        vggt_geometry_loss_weight=0.10 if stage1_5 else 0.05,
        coarse_traj_loss_weight=0.50 if stage1_5 else 0.20,
        coarse_heading_loss_weight=0.10 if stage1_5 else 0.05,
        risk_loss_weight=0.0,
        policy_kd_loss_weight=0.0,
        policy_kd_mode="none",
        freeze_base_action_head=stage1_5,
        train_expert_only=stage1_5,
        freeze_expert=False,
    )


def scalar_dict(prediction: Dict[str, torch.Tensor], keys: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in keys:
        value = prediction.get(key)
        if isinstance(value, torch.Tensor):
            out[key] = float(value.detach().cpu().float().item())
    return out


def run_stage(stage: str, batch) -> Dict[str, Any]:
    features, targets, _ = batch
    agent = build_agent(stage)
    agent.train()
    target_keys_in_training_batch = sorted(key for key in features if "target" in key)
    prediction = agent.forward(dict(features), dict(targets))
    loss = prediction["loss"] if isinstance(prediction, dict) else prediction.loss
    loss.backward()
    trainable_counts = agent.count_trainable_parameters_by_group()
    stage1_5_scope_pass = True
    if stage == "stage1_5":
        stage1_5_scope_pass = (
            trainable_counts["last_rd"]["trainable"] > 0
            and trainable_counts["action_base"]["trainable"] == 0
            and trainable_counts["legacy_a4_expert"]["trainable"] == 0
            and trainable_counts["backbone"]["trainable"] == 0
        )
    report = {
        "stage": stage,
        "use_expert_features": bool(agent.use_expert_features),
        "use_last_rd": bool(agent.use_last_rd),
        "losses": scalar_dict(
            prediction,
            ["loss", "diffusion_loss", "future_jepa_loss", "vggt_geometry_loss", "coarse_traj_loss", "coarse_heading_loss", "risk_loss"],
        ),
        "loss_finite": bool(torch.isfinite(loss.detach()).item()),
        "trainable_parameter_counts": trainable_counts,
        "target_keys_in_training_batch": target_keys_in_training_batch,
        "target_tokens_present_for_training": any(key in features for key in ("jepa_target_tokens", "vggt_target_tokens", "vggt_geometry_target_tokens")),
        "stage1_5_trainable_scope_pass": stage1_5_scope_pass,
    }
    agent.eval()
    eval_features = {key: value for key, value in features.items() if "target" not in key}
    target_keys_in_get_action_batch = sorted(key for key in eval_features if "target" in key)
    with torch.no_grad():
        pred = agent.forward(dict(eval_features), targets=None)
    no_future_targets_pass = not target_keys_in_get_action_batch
    report["get_action"] = {
        "pred_traj_shape": list(pred["pred_traj"].shape),
        "pred_traj_finite": bool(torch.isfinite(pred["pred_traj"]).all().item()),
        "target_keys_in_get_action_batch": target_keys_in_get_action_batch,
        "target_tokens_absent_for_get_action": no_future_targets_pass,
        "get_action_no_future_targets_pass": no_future_targets_pass,
    }
    return report


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        dataset = build_dataset(args.cache_root, include_targets=True)
        batch = take_batch(dataset, args.max_samples)
    except Exception as exc:
        report = {
            "cache_root": str(args.cache_root),
            "max_samples": int(args.max_samples),
            "pass": False,
            "error": repr(exc),
            "stage1_5_trainable_scope_pass": False,
            "get_action_no_future_targets_pass": False,
        }
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 1
    features, _, _ = batch
    stages = ["stage1_5", "progressive_sft"] if args.stage == "both" else [args.stage]
    stage_reports = [run_stage(stage, batch) for stage in stages]
    stage1_5_scope_pass = all(item["stage1_5_trainable_scope_pass"] for item in stage_reports if item["stage"] == "stage1_5")
    get_action_no_future_targets_pass = all(item["get_action"]["get_action_no_future_targets_pass"] for item in stage_reports)
    all_losses_finite = all(item["loss_finite"] for item in stage_reports)
    report = {
        "cache_root": str(args.cache_root),
        "batch_shapes": {
            key: list(value.shape)
            for key, value in features.items()
            if isinstance(value, torch.Tensor)
        },
        "high_command_one_hot_shape": list(features["high_command_one_hot"].shape) if "high_command_one_hot" in features else None,
        "target_keys_in_training_batch": sorted(key for key in features if "target" in key),
        "target_keys_in_get_action_batch": [],
        "geometry_mode_distribution": geometry_mode_distribution(dataset, args.max_samples),
        "stage1_5_trainable_scope_pass": stage1_5_scope_pass,
        "get_action_no_future_targets_pass": get_action_no_future_targets_pass,
        "all_losses_finite": all_losses_finite,
        "pass": bool(stage1_5_scope_pass and get_action_no_future_targets_pass and all_losses_finite),
        "stages": stage_reports,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
