#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import lzma
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.pareto_support import (  # noqa: E402
    CandidateRecord,
    build_archive_record,
    select_feasible_pareto_support,
)


def _iter_paths(root: Path):
    if root.is_file():
        yield root
        return
    yield from sorted(root.glob("*.pkl.xz"))


def _load(path: Path) -> dict[str, Any]:
    with lzma.open(path, "rb") as f:
        record = pickle.load(f)
    if not isinstance(record, dict):
        raise TypeError(f"record must be dict, got {type(record).__name__}: {path}")
    return record


def _save_atomic(path: Path, record: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with lzma.open(tmp, "wb") as f:
            pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _candidate_records(record: dict[str, Any]) -> list[CandidateRecord]:
    trajectories = np.asarray(record["candidates"], dtype=np.float32)
    rewards = np.asarray(record.get("rewards", np.zeros(trajectories.shape[0])), dtype=np.float32)
    selection = np.asarray(record.get("selection_score", rewards), dtype=np.float32)
    sources = [str(source) for source in record.get("sources", ["unknown"] * trajectories.shape[0])]
    components = dict(record.get("components", {}))
    feasibility = dict(record.get("feasibility", {}))
    token = str(record.get("token", ""))
    out: list[CandidateRecord] = []
    for idx in range(trajectories.shape[0]):
        comp = {str(key): float(np.asarray(value, dtype=np.float32)[idx]) for key, value in components.items()}
        feas = {str(key): float(np.asarray(value, dtype=np.float32)[idx]) for key, value in feasibility.items()}
        out.append(
            CandidateRecord(
                trajectory=trajectories[idx],
                source=sources[idx],
                token=token,
                components=comp,
                reward=float(rewards[idx]),
                feas=feas,
                selection_score=float(selection[idx]),
            )
        )
    return out


def _reference_components(candidates: list[CandidateRecord]) -> dict[str, float]:
    gt = next((candidate for candidate in candidates if candidate.source.lower() == "gt"), None)
    return dict((gt or candidates[0]).components) if candidates else {}


def reselect_record(record: dict[str, Any], cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    stats = Counter()
    candidates = _candidate_records(record)
    if not candidates:
        stats["empty_records"] += 1
        return record, dict(stats)
    selection_cfg = dict(cfg)
    source_metadata = dict(record.get("build_metadata", {}) or {})
    if int(record.get("version", 0)) >= 4 and source_metadata:
        preserved_metadata = dict(source_metadata)
        preserved_metadata["reselected_from_raw_v4"] = bool(source_metadata.get("raw_internal_candidates", False))
        selection_cfg["support_build_metadata"] = preserved_metadata
        selection_cfg.setdefault(
            "support_v4_require_policy_reachability",
            bool(source_metadata.get("policy_reachability_required", False)),
        )
        selection_cfg.setdefault(
            "support_v4_max_policy_snsad",
            float(source_metadata.get("max_policy_snsad", 0.50)),
        )
    selected = select_feasible_pareto_support(candidates, _reference_components(candidates), selection_cfg)
    rebuilt = build_archive_record(
        str(record.get("token", candidates[0].token)),
        candidates,
        selected,
        ref=_reference_components(candidates),
        cfg=selection_cfg,
    )
    old_tags = [str(tag) for tag in record.get("support_tags", [])]
    new_tags = [str(tag) for tag in rebuilt.get("support_tags", [])]
    if old_tags != new_tags:
        stats["records_changed"] += 1
        stats["old_tag_count"] += sum(1 for tag in old_tags if tag)
        stats["new_tag_count"] += sum(1 for tag in new_tags if tag)
    stats["records_seen"] += 1
    stats["candidate_count"] += len(candidates)
    stats["selected_count"] += sum(1 for tag in new_tags if tag)
    stats["valid_count"] += int(np.asarray(rebuilt.get("valid_mask", []), dtype=np.bool_).sum())
    quality_mask = np.asarray(rebuilt.get("support_quality_mask", np.ones(len(candidates), dtype=np.bool_)), dtype=np.bool_)
    selected_mask = np.asarray([bool(tag) for tag in new_tags], dtype=np.bool_)
    stats["quality_pass_count"] += int(quality_mask.sum())
    stats["selected_quality_pass_count"] += int((quality_mask & selected_mask).sum())
    stats["selected_evaluator_valid_count"] += int((np.asarray(rebuilt.get("valid_mask", []), dtype=np.bool_) & selected_mask).sum())
    return rebuilt, dict(stats)


def _output_path(input_root: Path, output_root: Path | None, source_path: Path) -> Path:
    if output_root is None:
        return source_path
    if input_root.is_file():
        if output_root.suffix:
            return output_root
        return output_root / source_path.name
    return output_root / source_path.name


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reselect SG-FPS support from already evaluator-labeled candidates. "
            "This does not change trajectories or evaluator metrics."
        )
    )
    parser.add_argument("--support-archive-path", required=True)
    parser.add_argument(
        "--output-support-archive-path",
        default="",
        help="Optional output archive directory/file. If omitted, records are rewritten in place unless --dry-run is set.",
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--archive-version", type=int, default=3)
    parser.add_argument("--support-top-m", type=int, default=12)
    parser.add_argument("--ddc-min-absolute", type=float, default=0.95)
    parser.add_argument("--ddc-drop-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--ddc-gate-mode",
        choices=["absolute_and_ref", "absolute", "ref_relative", "relax_ref_below_min"],
        default="absolute_and_ref",
    )
    parser.add_argument("--relax-ddc-when-ref-below-min", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--feas-cost-max", type=float, default=0.10)
    parser.add_argument(
        "--feas-gate-mode",
        choices=["absolute", "ref_relative", "relax_ref_above_max", "disabled"],
        default="absolute",
    )
    parser.add_argument("--feas-cost-tolerance", type=float, default=0.03)
    parser.add_argument("--comfort-min", type=float, default=0.95)
    parser.add_argument(
        "--comfort-gate-mode",
        choices=["absolute", "ref_relative", "relax_ref_below_min", "disabled"],
        default="absolute",
    )
    parser.add_argument("--comfort-drop-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--selection-strategy",
        choices=["quota", "quality_pareto", "mode_pareto_v4"],
        default="quota",
    )
    parser.add_argument("--enable-train-quality-gate", action="store_true")
    parser.add_argument("--min-non-gt-pdms", type=float, default=0.0)
    parser.add_argument(
        "--reward-gate-mode",
        choices=["absolute", "absolute_or_gt_improver", "gt_relative"],
        default="absolute",
    )
    parser.add_argument("--min-non-gt-improver-pdms", type=float, default=0.70)
    parser.add_argument("--gt-improver-margin", type=float, default=0.05)
    parser.add_argument("--gt-improver-ref-max-pdms", type=float, default=0.90)
    parser.add_argument("--max-first-xy-error-m", type=float, default=float("inf"))
    parser.add_argument("--max-first-heading-error-rad", type=float, default=float("inf"))
    parser.add_argument("--max-xy-turn-rad", type=float, default=float("inf"))
    parser.add_argument("--max-early-xy-turn-rad", type=float, default=float("inf"))
    parser.add_argument("--max-step-m", type=float, default=float("inf"))
    parser.add_argument("--enable-semantic-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-turn-class-mismatch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-semantic-final-heading-error-rad", type=float, default=0.75)
    parser.add_argument("--max-semantic-path-angle-error-rad", type=float, default=0.75)
    parser.add_argument("--max-semantic-endpoint-lateral-error-m", type=float, default=4.0)
    parser.add_argument("--keep-gt-by-default", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gt-keep-min-reward", type=float, default=0.85)
    parser.add_argument("--gt-replace-margin", type=float, default=0.05)
    parser.add_argument("--include-il-anchor", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--high-pdms-threshold", type=float, default=0.95)
    parser.add_argument("--top-pdms-count", type=int, default=3)
    parser.add_argument("--pareto-count", type=int, default=5)
    parser.add_argument("--score-diversity-weight", type=float, default=0.5)
    parser.add_argument("--min-trajectory-diversity-score", type=float, default=0.0)
    parser.add_argument("--v4-max-gt-ade-m", type=float, default=1.5)
    parser.add_argument("--v4-max-gt-fde-m", type=float, default=4.0)
    parser.add_argument("--v4-mode-distance-threshold", type=float, default=0.25)
    parser.add_argument("--v4-reward-gain-cap", type=float, default=0.05)
    parser.add_argument("--v4-max-gt-reward-drop", type=float, default=0.05)
    parser.add_argument("--v4-pareto-eps", type=float, default=0.01)
    parser.add_argument("--v4-exclude-derived-external", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--v4-require-policy-reachability",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--v4-max-policy-snsad", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = {
        "support_top_m": int(args.support_top_m),
        "support_archive_version": int(args.archive_version),
        "fpv3_ddc_min_absolute": float(args.ddc_min_absolute),
        "fpv3_ddc_drop_tolerance": float(args.ddc_drop_tolerance),
        "fpv3_feas_cost_max": float(args.feas_cost_max),
        "fpv3_comfort_min": float(args.comfort_min),
        "support_selection_strategy": str(args.selection_strategy),
        "support_quality_enable": bool(args.enable_train_quality_gate),
        "support_ddc_gate_mode": str(args.ddc_gate_mode),
        "support_relax_ddc_when_ref_below_min": bool(args.relax_ddc_when_ref_below_min),
        "support_feas_gate_mode": str(args.feas_gate_mode),
        "support_feas_cost_tolerance": float(args.feas_cost_tolerance),
        "support_semantic_enable": bool(args.enable_semantic_gate),
        "support_allow_turn_class_mismatch": bool(args.allow_turn_class_mismatch),
        "support_max_semantic_final_heading_error_rad": float(args.max_semantic_final_heading_error_rad),
        "support_max_semantic_path_angle_error_rad": float(args.max_semantic_path_angle_error_rad),
        "support_max_semantic_endpoint_lateral_error_m": float(args.max_semantic_endpoint_lateral_error_m),
        "support_min_non_gt_reward": float(args.min_non_gt_pdms),
        "support_reward_gate_mode": str(args.reward_gate_mode),
        "support_min_non_gt_improver_reward": float(args.min_non_gt_improver_pdms),
        "support_gt_improver_margin": float(args.gt_improver_margin),
        "support_gt_improver_ref_max_reward": float(args.gt_improver_ref_max_pdms),
        "support_comfort_gate_mode": str(args.comfort_gate_mode),
        "support_comfort_drop_tolerance": float(args.comfort_drop_tolerance),
        "support_max_first_xy_error_m": float(args.max_first_xy_error_m),
        "support_max_first_heading_error_rad": float(args.max_first_heading_error_rad),
        "support_max_xy_turn_rad": float(args.max_xy_turn_rad),
        "support_max_early_xy_turn_rad": float(args.max_early_xy_turn_rad),
        "support_max_step_m": float(args.max_step_m),
        "support_keep_gt_by_default": bool(args.keep_gt_by_default),
        "support_gt_keep_min_reward": float(args.gt_keep_min_reward),
        "support_gt_replace_margin": float(args.gt_replace_margin),
        "support_include_il_anchor": bool(args.include_il_anchor),
        "support_high_pdms_threshold": float(args.high_pdms_threshold),
        "support_top_pdms_count": int(args.top_pdms_count),
        "support_pareto_count": int(args.pareto_count),
        "support_score_diversity_weight": float(args.score_diversity_weight),
        "support_min_trajectory_diversity_score": float(args.min_trajectory_diversity_score),
        "support_v4_max_gt_ade_m": float(args.v4_max_gt_ade_m),
        "support_v4_max_gt_fde_m": float(args.v4_max_gt_fde_m),
        "support_v4_mode_distance_threshold": float(args.v4_mode_distance_threshold),
        "support_v4_reward_gain_cap": float(args.v4_reward_gain_cap),
        "support_v4_max_gt_reward_drop": float(args.v4_max_gt_reward_drop),
        "support_v4_pareto_eps": float(args.v4_pareto_eps),
        "support_v4_exclude_derived_external": bool(args.v4_exclude_derived_external),
        "support_build_metadata": {
            "migration": "shadow_reselect_without_rescoring",
            "raw_internal_candidates": False,
            "expand_external_candidates": True,
            "source_archive_path": str(args.support_archive_path),
        },
    }
    if args.v4_require_policy_reachability is not None:
        cfg["support_v4_require_policy_reachability"] = bool(args.v4_require_policy_reachability)
    if args.v4_max_policy_snsad is not None:
        cfg["support_v4_max_policy_snsad"] = float(args.v4_max_policy_snsad)
    root = Path(args.support_archive_path)
    output_root = Path(args.output_support_archive_path) if args.output_support_archive_path else None
    if output_root is not None and not output_root.suffix:
        output_root.mkdir(parents=True, exist_ok=True)
    totals = Counter()
    examples: list[dict[str, Any]] = []
    processed = 0
    for idx, path in enumerate(_iter_paths(root)):
        if idx % max(int(args.world_size), 1) != int(args.rank):
            continue
        if int(args.max_records) > 0 and processed >= int(args.max_records):
            break
        processed += 1
        try:
            record = _load(path)
            rebuilt, stats = reselect_record(record, cfg)
        except Exception as exc:
            totals["errors"] += 1
            if len(examples) < 16:
                examples.append({"path": str(path), "error": str(exc)})
            continue
        totals.update(stats)
        if not args.dry_run:
            out_path = _output_path(root, output_root, path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _save_atomic(out_path, rebuilt)

    summary = {
        "support_archive_path": str(root),
        "output_support_archive_path": str(output_root) if output_root is not None else "",
        "dry_run": bool(args.dry_run),
        "config": cfg,
        "rank": int(args.rank),
        "world_size": int(args.world_size),
        "stats": dict(totals),
        "examples": examples,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
