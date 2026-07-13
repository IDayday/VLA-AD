#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import lzma
import pickle
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.pareto_support import CandidateRecord
from navsim.agents.recogdrive.pareto_support.pareto_archive import (
    support_quality_metrics,
    support_reward_pass,
    support_semantic_pass,
)


def _load_record(path: Path) -> dict[str, Any]:
    if path.suffix == ".xz":
        with lzma.open(path, "rb") as f:
            return pickle.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)


def _iter_record_paths(root: Path):
    if root.is_file():
        yield root
        return
    yield from sorted(root.glob("*.pkl.xz"))
    yield from sorted(root.glob("*.pkl"))


def _as_array(value: Any, dtype=None) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def _source_bucket(source: str, external_names: set[str]) -> str:
    lower = source.lower()
    if lower == "gt":
        return "gt"
    if lower == "il":
        return "il"
    if lower.startswith("policy"):
        return "policy"
    if any(name and name in lower for name in external_names):
        return "external"
    if lower.startswith("progress"):
        return "progress"
    if "lateral" in lower:
        return "lateral"
    if lower.startswith("timing"):
        return "timing"
    return "other"


def audit_archive(
    support_archive_path: Path,
    *,
    external_source_names: set[str],
    max_records: int,
    rank: int,
    world_size: int,
    improvement_margin: float,
    selected_min_reward: float,
    max_first_xy_error_m: float,
    max_xy_turn_rad: float,
    max_early_xy_turn_rad: float,
    enable_semantic_gate: bool,
    allow_turn_class_mismatch: bool,
    max_semantic_final_heading_error_rad: float,
    max_semantic_path_angle_error_rad: float,
    max_semantic_endpoint_lateral_error_m: float,
    poor_gt_reward_threshold: float,
    min_candidate_count: int,
    reward_gate_mode: str,
    min_non_gt_improver_reward: float,
    gt_improver_margin: float,
    gt_improver_ref_max_reward: float,
) -> dict[str, Any]:
    all_paths = list(_iter_record_paths(support_archive_path))
    rank = int(rank)
    world_size = max(int(world_size), 1)
    paths = [path for idx, path in enumerate(all_paths) if idx % world_size == rank]
    if max_records > 0:
        paths = paths[:max_records]

    counts = Counter()
    version_counts = Counter()
    source_counts = Counter()
    selected_source_counts = Counter()
    source_bucket_counts = Counter()
    selected_source_bucket_counts = Counter()
    support_tag_counts = Counter()
    selected_support_tag_counts = Counter()
    warnings: list[str] = []

    candidate_counts: list[float] = []
    selected_counts: list[float] = []
    valid_ratios: list[float] = []
    selected_valid_ratios: list[float] = []
    pareto_ratios: list[float] = []
    selected_pareto_ratios: list[float] = []
    selected_source_diversity: list[float] = []
    feasibility_cost_mean: list[float] = []
    best_valid_minus_gt: list[float] = []
    best_valid_minus_il: list[float] = []
    best_selected_minus_gt: list[float] = []
    selected_min_rewards: list[float] = []
    selected_first_xy_error: list[float] = []
    selected_max_xy_turn: list[float] = []
    selected_early_xy_turn: list[float] = []
    selected_semantic_final_heading_error: list[float] = []
    selected_semantic_path_angle_error: list[float] = []
    selected_semantic_endpoint_lateral_error: list[float] = []

    examples: dict[str, list[dict[str, Any]]] = {
        "no_valid_candidate": [],
        "no_selected_improver": [],
        "low_selected_valid_ratio": [],
        "external_missing": [],
        "selected_low_reward": [],
        "selected_first_point_mismatch": [],
        "selected_local_kink": [],
        "selected_semantic_mismatch": [],
        "poor_gt_few_candidates": [],
    }
    semantic_cfg = {
        "support_semantic_enable": bool(enable_semantic_gate),
        "support_allow_turn_class_mismatch": bool(allow_turn_class_mismatch),
        "support_max_semantic_final_heading_error_rad": float(max_semantic_final_heading_error_rad),
        "support_max_semantic_path_angle_error_rad": float(max_semantic_path_angle_error_rad),
        "support_max_semantic_endpoint_lateral_error_m": float(max_semantic_endpoint_lateral_error_m),
    }
    reward_cfg = {
        "support_quality_exempt_gt": True,
        "support_min_non_gt_reward": float(selected_min_reward),
        "support_reward_gate_mode": str(reward_gate_mode),
        "support_min_non_gt_improver_reward": float(min_non_gt_improver_reward),
        "support_gt_improver_margin": float(gt_improver_margin),
        "support_gt_improver_ref_max_reward": float(gt_improver_ref_max_reward),
    }

    for path in paths:
        try:
            record = _load_record(path)
        except Exception as exc:
            counts["load_errors"] += 1
            if len(examples.setdefault("load_errors", [])) < 8:
                examples["load_errors"].append({"path": str(path), "error": str(exc)})
            continue

        counts["record_count"] += 1
        token = str(record.get("token", path.stem))
        version = int(record.get("version", 1))
        version_counts[str(version)] += 1

        rewards = _as_array(record.get("rewards", []), np.float32)
        sources = [str(item) for item in record.get("sources", [])]
        support_tags = [str(item) for item in record.get("support_tags", [""] * len(sources))]
        valid_mask = _as_array(record.get("valid_mask", np.zeros(len(sources), dtype=np.bool_)), np.bool_)
        pareto_mask = _as_array(record.get("pareto_front_mask", np.zeros(len(sources), dtype=np.bool_)), np.bool_)
        selected_mask = np.asarray([bool(tag) for tag in support_tags], dtype=np.bool_)
        candidates = _as_array(record.get("candidates", []), np.float32)
        if rewards.shape[:1] != (len(sources),):
            counts["shape_errors"] += 1
            continue
        if valid_mask.shape != rewards.shape:
            counts["shape_errors"] += 1
            continue
        if pareto_mask.shape != rewards.shape:
            counts["shape_errors"] += 1
            continue
        if selected_mask.shape != rewards.shape:
            counts["shape_errors"] += 1
            continue
        if candidates.shape[:1] != rewards.shape[:1] or candidates.ndim != 3 or candidates.shape[-1] != 3:
            counts["shape_errors"] += 1
            continue

        k = int(rewards.shape[0])
        s = int(selected_mask.sum())
        candidate_counts.append(float(k))
        selected_counts.append(float(s))
        valid_count = int(valid_mask.sum())
        selected_valid_count = int((valid_mask & selected_mask).sum())
        pareto_count = int(pareto_mask.sum())
        selected_pareto_count = int((pareto_mask & selected_mask).sum())

        valid_ratios.append(float(valid_count / max(k, 1)))
        selected_valid_ratios.append(float(selected_valid_count / max(s, 1)))
        pareto_ratios.append(float(pareto_count / max(k, 1)))
        selected_pareto_ratios.append(float(selected_pareto_count / max(s, 1)))

        if bool(record.get("has_valid_candidate", valid_count > 0)):
            counts["has_valid_candidate"] += 1
        elif len(examples["no_valid_candidate"]) < 8:
            examples["no_valid_candidate"].append({"token": token})

        gt_reward = float(record.get("gt_reward", 0.0))
        il_reward = float(record.get("il_reward", gt_reward))
        best_valid = float(record.get("best_valid_reward", rewards[valid_mask].max() if valid_count else rewards.max(initial=0.0)))
        best_selected = float(record.get("best_selected_reward", rewards[selected_mask].max() if s else best_valid))
        best_valid_minus_gt.append(best_valid - gt_reward)
        best_valid_minus_il.append(best_valid - il_reward)
        best_selected_minus_gt.append(best_selected - gt_reward)
        if best_valid > gt_reward + improvement_margin:
            counts["best_valid_above_gt"] += 1
        if best_valid > il_reward + improvement_margin:
            counts["best_valid_above_il"] += 1
        if best_selected > gt_reward + improvement_margin:
            counts["best_selected_above_gt"] += 1
        if s and bool(np.any(rewards[selected_mask] > gt_reward + improvement_margin)):
            counts["selected_has_improver_over_gt"] += 1
        elif len(examples["no_selected_improver"]) < 8:
            examples["no_selected_improver"].append({"token": token, "gt_reward": gt_reward, "best_selected": best_selected})

        selected_buckets = set()
        has_external = False
        selected_has_external = False
        for idx, source in enumerate(sources):
            bucket = _source_bucket(source, external_source_names)
            source_counts[source] += 1
            source_bucket_counts[bucket] += 1
            if bucket == "external":
                has_external = True
            if selected_mask[idx]:
                selected_source_counts[source] += 1
                selected_source_bucket_counts[bucket] += 1
                selected_buckets.add(bucket)
                if bucket == "external":
                    selected_has_external = True
        for tag, selected in zip(support_tags, selected_mask):
            if tag:
                support_tag_counts[tag] += 1
                if selected:
                    selected_support_tag_counts[tag] += 1
        selected_source_diversity.append(float(len(selected_buckets)))
        if len(selected_buckets) >= 2:
            counts["selected_source_diversity_ge2"] += 1
        if has_external:
            counts["external_candidate_scene"] += 1
        elif len(examples["external_missing"]) < 8:
            examples["external_missing"].append({"token": token})
        if selected_has_external:
            counts["external_selected_scene"] += 1

        if s and selected_valid_count / max(s, 1) < 0.90 and len(examples["low_selected_valid_ratio"]) < 8:
            examples["low_selected_valid_ratio"].append(
                {"token": token, "selected_count": s, "selected_valid_count": selected_valid_count}
            )

        selected_idx = np.flatnonzero(selected_mask)
        if selected_idx.size:
            selected_min_rewards.append(float(rewards[selected_idx].min()))
            non_gt_selected_idx = np.asarray(
                [idx for idx in selected_idx if str(sources[idx]).lower() != "gt"],
                dtype=np.int64,
            )
            low_reward_idx = []
            for idx in non_gt_selected_idx:
                ok = support_reward_pass(
                    CandidateRecord(
                        trajectory=candidates[idx],
                        source=sources[idx],
                        token=token,
                        components={},
                        reward=float(rewards[idx]),
                        feas={},
                        selection_score=float(rewards[idx]),
                    ),
                    gt_reward,
                    reward_cfg,
                )
                if not ok:
                    low_reward_idx.append(int(idx))
            low_reward_idx = np.asarray(low_reward_idx, dtype=np.int64)
            if low_reward_idx.size:
                counts["scenes_with_selected_low_reward"] += 1
                counts["selected_low_reward_count"] += int(low_reward_idx.size)
                if len(examples["selected_low_reward"]) < 8:
                    examples["selected_low_reward"].append(
                        {
                            "token": token,
                            "count": int(low_reward_idx.size),
                            "min_selected_reward": float(rewards[selected_idx].min()),
                        }
                    )
            gt_candidates = [idx for idx, source in enumerate(sources) if str(source).lower() == "gt"]
            ref_traj = candidates[gt_candidates[0]] if gt_candidates else candidates[0]
            first_errors = []
            max_turns = []
            early_turns = []
            semantic_final_heading_errors = []
            semantic_path_angle_errors = []
            semantic_endpoint_lateral_errors = []
            quality_check_idx = non_gt_selected_idx
            for idx in quality_check_idx:
                qm = support_quality_metrics(candidates[idx], ref_traj)
                first_errors.append(float(qm["first_xy_error_m"]))
                max_turns.append(float(qm["max_xy_turn_rad"]))
                early_turns.append(float(qm["early_xy_turn_rad"]))
                semantic_final_heading_errors.append(float(qm.get("semantic_final_heading_error_rad", 0.0)))
                semantic_path_angle_errors.append(float(qm.get("semantic_path_angle_error_rad", 0.0)))
                semantic_endpoint_lateral_errors.append(float(qm.get("semantic_endpoint_lateral_error_m", 0.0)))
            selected_first_xy_error.extend(first_errors)
            selected_max_xy_turn.extend(max_turns)
            selected_early_xy_turn.extend(early_turns)
            selected_semantic_final_heading_error.extend(semantic_final_heading_errors)
            selected_semantic_path_angle_error.extend(semantic_path_angle_errors)
            selected_semantic_endpoint_lateral_error.extend(semantic_endpoint_lateral_errors)
            bad_first = [
                idx for idx, value in zip(quality_check_idx, first_errors) if value > float(max_first_xy_error_m)
            ]
            bad_kink = [
                idx
                for idx, max_turn, early_turn in zip(quality_check_idx, max_turns, early_turns)
                if max_turn > float(max_xy_turn_rad) or early_turn > float(max_early_xy_turn_rad)
            ]
            bad_semantic = [
                idx
                for idx in quality_check_idx
                if not support_semantic_pass(candidates[idx], ref_traj, semantic_cfg)
            ]
            if bad_first:
                counts["scenes_with_selected_first_point_mismatch"] += 1
                counts["selected_first_point_mismatch_count"] += len(bad_first)
                if len(examples["selected_first_point_mismatch"]) < 8:
                    examples["selected_first_point_mismatch"].append(
                        {"token": token, "count": len(bad_first), "max_first_xy_error_m": float(max(first_errors))}
                    )
            if bad_kink:
                counts["scenes_with_selected_local_kink"] += 1
                counts["selected_local_kink_count"] += len(bad_kink)
                if len(examples["selected_local_kink"]) < 8:
                    examples["selected_local_kink"].append(
                        {
                            "token": token,
                            "count": len(bad_kink),
                            "max_xy_turn_rad": float(max(max_turns)),
                            "max_early_xy_turn_rad": float(max(early_turns)),
                        }
                    )
            if bad_semantic:
                counts["scenes_with_selected_semantic_mismatch"] += 1
                counts["selected_semantic_mismatch_count"] += len(bad_semantic)
                if len(examples["selected_semantic_mismatch"]) < 8:
                    examples["selected_semantic_mismatch"].append(
                        {
                            "token": token,
                            "count": len(bad_semantic),
                            "max_semantic_final_heading_error_rad": float(max(semantic_final_heading_errors or [0.0])),
                            "max_semantic_path_angle_error_rad": float(max(semantic_path_angle_errors or [0.0])),
                            "max_semantic_endpoint_lateral_error_m": float(max(semantic_endpoint_lateral_errors or [0.0])),
                        }
                    )
        if gt_reward < float(poor_gt_reward_threshold) and k < int(min_candidate_count):
            counts["poor_gt_few_candidates"] += 1
            if len(examples["poor_gt_few_candidates"]) < 8:
                examples["poor_gt_few_candidates"].append({"token": token, "gt_reward": gt_reward, "candidate_count": k})

        feasibility = record.get("feasibility", {})
        if isinstance(feasibility, dict) and "feas_cost" in feasibility:
            feas = _as_array(feasibility["feas_cost"], np.float32)
            if feas.size:
                feasibility_cost_mean.append(float(feas.mean()))

    n = max(int(counts["record_count"]), 1)
    selected_total = max(sum(selected_support_tag_counts.values()), 1)
    candidate_total = max(sum(source_bucket_counts.values()), 1)

    metrics = {
        "support_archive_path": str(support_archive_path),
        "rank": int(rank),
        "world_size": int(world_size),
        "record_count": int(counts["record_count"]),
        "load_errors": int(counts["load_errors"]),
        "shape_errors": int(counts["shape_errors"]),
        "version_counts": dict(version_counts),
        "candidate_count": _stats(candidate_counts),
        "selected_count": _stats(selected_counts),
        "valid_ratio": _stats(valid_ratios),
        "selected_valid_ratio": _stats(selected_valid_ratios),
        "pareto_front_ratio": _stats(pareto_ratios),
        "selected_pareto_ratio": _stats(selected_pareto_ratios),
        "selected_source_diversity": _stats(selected_source_diversity),
        "feasibility_cost_mean": _stats(feasibility_cost_mean),
        "best_valid_minus_gt": _stats(best_valid_minus_gt),
        "best_valid_minus_il": _stats(best_valid_minus_il),
        "best_selected_minus_gt": _stats(best_selected_minus_gt),
        "selected_min_reward": _stats(selected_min_rewards),
        "selected_first_xy_error_m": _stats(selected_first_xy_error),
        "selected_max_xy_turn_rad": _stats(selected_max_xy_turn),
        "selected_early_xy_turn_rad": _stats(selected_early_xy_turn),
        "selected_semantic_final_heading_error_rad": _stats(selected_semantic_final_heading_error),
        "selected_semantic_path_angle_error_rad": _stats(selected_semantic_path_angle_error),
        "selected_semantic_endpoint_lateral_error_m": _stats(selected_semantic_endpoint_lateral_error),
        "has_valid_candidate_ratio": float(counts["has_valid_candidate"] / n),
        "best_valid_above_gt_ratio": float(counts["best_valid_above_gt"] / n),
        "best_valid_above_il_ratio": float(counts["best_valid_above_il"] / n),
        "best_selected_above_gt_ratio": float(counts["best_selected_above_gt"] / n),
        "selected_has_improver_over_gt_ratio": float(counts["selected_has_improver_over_gt"] / n),
        "selected_source_diversity_ge2_ratio": float(counts["selected_source_diversity_ge2"] / n),
        "external_candidate_scene_ratio": float(counts["external_candidate_scene"] / n),
        "external_selected_scene_ratio": float(counts["external_selected_scene"] / n),
        "source_bucket_counts": dict(source_bucket_counts),
        "selected_source_bucket_counts": dict(selected_source_bucket_counts),
        "source_counts_top20": dict(source_counts.most_common(20)),
        "selected_source_counts_top20": dict(selected_source_counts.most_common(20)),
        "support_tag_counts": dict(support_tag_counts),
        "fallback_tag_ratio": float(selected_support_tag_counts.get("fallback_best", 0) / selected_total),
        "external_candidate_ratio": float(source_bucket_counts.get("external", 0) / candidate_total),
        "scenes_with_selected_low_reward_ratio": float(counts["scenes_with_selected_low_reward"] / n),
        "scenes_with_selected_first_point_mismatch_ratio": float(counts["scenes_with_selected_first_point_mismatch"] / n),
        "scenes_with_selected_local_kink_ratio": float(counts["scenes_with_selected_local_kink"] / n),
        "scenes_with_selected_semantic_mismatch_ratio": float(counts["scenes_with_selected_semantic_mismatch"] / n),
        "poor_gt_few_candidates_ratio": float(counts["poor_gt_few_candidates"] / n),
        "selected_low_reward_count": int(counts["selected_low_reward_count"]),
        "selected_first_point_mismatch_count": int(counts["selected_first_point_mismatch_count"]),
        "selected_local_kink_count": int(counts["selected_local_kink_count"]),
        "selected_semantic_mismatch_count": int(counts["selected_semantic_mismatch_count"]),
        "examples": {key: value for key, value in examples.items() if value},
    }

    if metrics["load_errors"] or metrics["shape_errors"]:
        warnings.append("archive contains load/shape errors")
    if metrics["has_valid_candidate_ratio"] < 0.99:
        warnings.append("valid candidate coverage is below 99%")
    if metrics["selected_valid_ratio"]["mean"] < 0.90:
        warnings.append("selected support contains too many invalid candidates")
    if metrics["scenes_with_selected_low_reward_ratio"] > 0.0:
        warnings.append("selected support contains trajectories below the configured PDMS/reward threshold")
    if metrics["scenes_with_selected_first_point_mismatch_ratio"] > 0.0:
        warnings.append("selected support contains trajectories whose first point is far from GT")
    if metrics["scenes_with_selected_local_kink_ratio"] > 0.0:
        warnings.append("selected support contains locally kinked trajectories")
    if metrics["scenes_with_selected_semantic_mismatch_ratio"] > 0.0:
        warnings.append("selected support contains trajectories with turn/heading semantics inconsistent with GT")
    if metrics["best_valid_above_gt_ratio"] < 0.30:
        warnings.append("best valid candidates rarely improve over GT")
    if metrics["fallback_tag_ratio"] > 0.50:
        warnings.append("fallback_best dominates selected support tags")
    if external_source_names and metrics["external_candidate_scene_ratio"] < 0.50:
        warnings.append("external candidate coverage is low; DDv2/DriveOR may not be contributing")
    metrics["warnings"] = warnings
    return metrics


def _write_markdown(metrics: dict[str, Any], path: Path) -> None:
    lines = [
        "# SG-FPS Support Archive Audit",
        "",
        f"- archive: `{metrics['support_archive_path']}`",
        f"- records: `{metrics['record_count']}`",
        f"- has valid candidate ratio: `{metrics['has_valid_candidate_ratio']:.6f}`",
        f"- selected valid ratio mean: `{metrics['selected_valid_ratio']['mean']:.6f}`",
        f"- best valid above GT ratio: `{metrics['best_valid_above_gt_ratio']:.6f}`",
        f"- best valid above IL ratio: `{metrics['best_valid_above_il_ratio']:.6f}`",
        f"- selected improver over GT ratio: `{metrics['selected_has_improver_over_gt_ratio']:.6f}`",
        f"- external candidate scene ratio: `{metrics['external_candidate_scene_ratio']:.6f}`",
        f"- external selected scene ratio: `{metrics['external_selected_scene_ratio']:.6f}`",
        f"- fallback tag ratio: `{metrics['fallback_tag_ratio']:.6f}`",
        f"- selected low reward scene ratio: `{metrics['scenes_with_selected_low_reward_ratio']:.6f}`",
        f"- selected first-point mismatch scene ratio: `{metrics['scenes_with_selected_first_point_mismatch_ratio']:.6f}`",
        f"- selected local kink scene ratio: `{metrics['scenes_with_selected_local_kink_ratio']:.6f}`",
        f"- selected semantic mismatch scene ratio: `{metrics['scenes_with_selected_semantic_mismatch_ratio']:.6f}`",
        f"- poor-GT/few-candidate scene ratio: `{metrics['poor_gt_few_candidates_ratio']:.6f}`",
        "",
        "## Source Buckets",
        "",
        "```json",
        json.dumps(metrics["source_bucket_counts"], indent=2, sort_keys=True),
        "```",
        "",
        "## Selected Source Buckets",
        "",
        "```json",
        json.dumps(metrics["selected_source_bucket_counts"], indent=2, sort_keys=True),
        "```",
        "",
        "## Support Tags",
        "",
        "```json",
        json.dumps(metrics["support_tag_counts"], indent=2, sort_keys=True),
        "```",
    ]
    if metrics.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in metrics["warnings"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SG-FPS v3 Pareto support archive effectiveness.")
    parser.add_argument("--support-archive-path", required=True)
    parser.add_argument("--external-source-names", default="diffusiondrivev2,drivor,driveor,ddv2")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--improvement-margin", type=float, default=1e-6)
    parser.add_argument("--selected-min-reward", type=float, default=0.90)
    parser.add_argument(
        "--reward-gate-mode",
        choices=["absolute", "absolute_or_gt_improver", "gt_relative"],
        default="absolute",
    )
    parser.add_argument("--min-non-gt-improver-pdms", type=float, default=0.70)
    parser.add_argument("--gt-improver-margin", type=float, default=0.05)
    parser.add_argument("--gt-improver-ref-max-pdms", type=float, default=0.90)
    parser.add_argument("--max-first-xy-error-m", type=float, default=1.0)
    parser.add_argument("--max-xy-turn-rad", type=float, default=1.2)
    parser.add_argument("--max-early-xy-turn-rad", type=float, default=1.0)
    parser.add_argument("--enable-semantic-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-turn-class-mismatch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-semantic-final-heading-error-rad", type=float, default=0.75)
    parser.add_argument("--max-semantic-path-angle-error-rad", type=float, default=0.75)
    parser.add_argument("--max-semantic-endpoint-lateral-error-m", type=float, default=4.0)
    parser.add_argument("--poor-gt-reward-threshold", type=float, default=0.85)
    parser.add_argument("--min-candidate-count", type=int, default=20)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument("--min-has-valid-candidate-ratio", type=float, default=0.99)
    parser.add_argument("--min-selected-valid-ratio", type=float, default=0.90)
    parser.add_argument("--min-best-valid-above-gt-ratio", type=float, default=0.30)
    parser.add_argument("--min-selected-source-diversity-ge2-ratio", type=float, default=0.80)
    parser.add_argument("--min-external-scene-ratio", type=float, default=0.0)
    parser.add_argument("--max-fallback-tag-ratio", type=float, default=0.50)
    args = parser.parse_args()

    external_names = {item.strip().lower() for item in args.external_source_names.replace(",", " ").split() if item.strip()}
    metrics = audit_archive(
        Path(args.support_archive_path),
        external_source_names=external_names,
        max_records=int(args.max_records),
        rank=int(args.rank),
        world_size=int(args.world_size),
        improvement_margin=float(args.improvement_margin),
        selected_min_reward=float(args.selected_min_reward),
        max_first_xy_error_m=float(args.max_first_xy_error_m),
        max_xy_turn_rad=float(args.max_xy_turn_rad),
        max_early_xy_turn_rad=float(args.max_early_xy_turn_rad),
        enable_semantic_gate=bool(args.enable_semantic_gate),
        allow_turn_class_mismatch=bool(args.allow_turn_class_mismatch),
        max_semantic_final_heading_error_rad=float(args.max_semantic_final_heading_error_rad),
        max_semantic_path_angle_error_rad=float(args.max_semantic_path_angle_error_rad),
        max_semantic_endpoint_lateral_error_m=float(args.max_semantic_endpoint_lateral_error_m),
        poor_gt_reward_threshold=float(args.poor_gt_reward_threshold),
        min_candidate_count=int(args.min_candidate_count),
        reward_gate_mode=str(args.reward_gate_mode),
        min_non_gt_improver_reward=float(args.min_non_gt_improver_pdms),
        gt_improver_margin=float(args.gt_improver_margin),
        gt_improver_ref_max_reward=float(args.gt_improver_ref_max_pdms),
    )

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(metrics, Path(args.output_md))

    print(json.dumps(metrics, indent=2, sort_keys=True))

    failures = []
    if metrics["record_count"] < int(args.min_records):
        failures.append(f"record_count {metrics['record_count']} < {args.min_records}")
    if metrics["has_valid_candidate_ratio"] < float(args.min_has_valid_candidate_ratio):
        failures.append(
            f"has_valid_candidate_ratio {metrics['has_valid_candidate_ratio']:.6f} < {args.min_has_valid_candidate_ratio}"
        )
    if metrics["selected_valid_ratio"]["mean"] < float(args.min_selected_valid_ratio):
        failures.append(
            f"selected_valid_ratio.mean {metrics['selected_valid_ratio']['mean']:.6f} < {args.min_selected_valid_ratio}"
        )
    if metrics["best_valid_above_gt_ratio"] < float(args.min_best_valid_above_gt_ratio):
        failures.append(
            f"best_valid_above_gt_ratio {metrics['best_valid_above_gt_ratio']:.6f} < {args.min_best_valid_above_gt_ratio}"
        )
    if metrics["selected_source_diversity_ge2_ratio"] < float(args.min_selected_source_diversity_ge2_ratio):
        failures.append(
            "selected_source_diversity_ge2_ratio "
            f"{metrics['selected_source_diversity_ge2_ratio']:.6f} < {args.min_selected_source_diversity_ge2_ratio}"
        )
    if metrics["external_candidate_scene_ratio"] < float(args.min_external_scene_ratio):
        failures.append(
            f"external_candidate_scene_ratio {metrics['external_candidate_scene_ratio']:.6f} < {args.min_external_scene_ratio}"
        )
    if metrics["fallback_tag_ratio"] > float(args.max_fallback_tag_ratio):
        failures.append(f"fallback_tag_ratio {metrics['fallback_tag_ratio']:.6f} > {args.max_fallback_tag_ratio}")

    if failures:
        print("SG-FPS support audit failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
