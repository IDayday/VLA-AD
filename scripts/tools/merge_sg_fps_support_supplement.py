#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import lzma
import os
import pickle
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.pareto_support import CandidateRecord, build_archive_record, select_feasible_pareto_support


def _iter_paths(root: Path):
    if root.is_file():
        yield root
        return
    yield from sorted(root.glob("*.pkl.xz"))
    yield from sorted(root.glob("*.pkl"))


def _load(path: Path) -> dict[str, Any]:
    if path.name.endswith(".pkl.xz"):
        with lzma.open(path, "rb") as f:
            return pickle.load(f)
    with path.open("rb") as f:
        return pickle.load(f)


def _save(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with lzma.open(tmp, "wb") as f:
            pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _fingerprint(traj: np.ndarray) -> bytes:
    return np.round(np.asarray(traj, dtype=np.float32), 3).tobytes()


def _candidate_records(record: dict[str, Any]) -> list[CandidateRecord]:
    trajectories = np.asarray(record.get("candidates", []), dtype=np.float32)
    rewards = np.asarray(record.get("rewards", np.zeros(trajectories.shape[:1])), dtype=np.float32)
    selection = np.asarray(record.get("selection_score", rewards), dtype=np.float32)
    sources = [str(source) for source in record.get("sources", ["unknown"] * len(rewards))]
    components = dict(record.get("components", {}))
    feasibility = dict(record.get("feasibility", {}))
    token = str(record.get("token", ""))
    out: list[CandidateRecord] = []
    if trajectories.ndim != 3 or trajectories.shape[-1] != 3:
        return out
    for idx in range(trajectories.shape[0]):
        comp = {
            str(key): float(np.asarray(value, dtype=np.float32)[idx])
            for key, value in components.items()
            if np.asarray(value).shape[:1] == (trajectories.shape[0],)
        }
        feas = {
            str(key): float(np.asarray(value, dtype=np.float32)[idx])
            for key, value in feasibility.items()
            if np.asarray(value).shape[:1] == (trajectories.shape[0],)
        }
        out.append(
            CandidateRecord(
                trajectory=trajectories[idx],
                source=sources[idx] if idx < len(sources) else "unknown",
                token=token,
                components=comp,
                reward=float(rewards[idx]) if idx < rewards.shape[0] else 0.0,
                feas=feas,
                selection_score=float(selection[idx]) if idx < selection.shape[0] else 0.0,
            )
        )
    return out


def _reference_components(candidates: list[CandidateRecord]) -> dict[str, float]:
    gt = next((candidate for candidate in candidates if candidate.source.lower() == "gt"), None)
    return dict((gt or candidates[0]).components) if candidates else {}


def _dedupe(records: list[CandidateRecord]) -> list[CandidateRecord]:
    best: dict[bytes, CandidateRecord] = {}
    for record in records:
        key = _fingerprint(record.trajectory)
        prev = best.get(key)
        if prev is None or (float(record.reward), float(record.selection_score)) > (
            float(prev.reward),
            float(prev.selection_score),
        ):
            best[key] = record
    return list(best.values())


def _build_cfg(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "support_top_m": int(args.support_top_m),
        "support_selection_strategy": "quality_pareto",
        "fpv3_ddc_min_absolute": float(args.ddc_min_absolute),
        "fpv3_ddc_drop_tolerance": float(args.ddc_drop_tolerance),
        "support_ddc_gate_mode": str(args.ddc_gate_mode),
        "support_relax_ddc_when_ref_below_min": bool(args.relax_ddc_when_ref_below_min),
        "fpv3_feas_cost_max": float(args.feas_cost_max),
        "support_feas_gate_mode": str(args.feas_gate_mode),
        "support_feas_cost_tolerance": float(args.feas_cost_tolerance),
        "fpv3_comfort_min": float(args.comfort_min),
        "support_comfort_gate_mode": str(args.comfort_gate_mode),
        "support_comfort_drop_tolerance": float(args.comfort_drop_tolerance),
        "support_quality_enable": True,
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
        "support_max_first_xy_error_m": float(args.max_first_xy_error_m),
        "support_max_first_heading_error_rad": float(args.max_first_heading_error_rad),
        "support_max_xy_turn_rad": float(args.max_xy_turn_rad),
        "support_max_early_xy_turn_rad": float(args.max_early_xy_turn_rad),
        "support_max_step_m": float(args.max_step_m),
        "support_keep_gt_by_default": True,
        "support_gt_keep_min_reward": float(args.gt_keep_min_reward),
        "support_gt_replace_margin": float(args.gt_replace_margin),
        "support_include_il_anchor": False,
        "support_high_pdms_threshold": float(args.high_pdms_threshold),
        "support_top_pdms_count": int(args.top_pdms_count),
        "support_pareto_count": int(args.pareto_count),
        "support_score_diversity_weight": float(args.score_diversity_weight),
        "support_min_trajectory_diversity_score": float(args.min_trajectory_diversity_score),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge supplemental SG-FPS support records into a clean base archive.")
    parser.add_argument("--base-support-archive-path", required=True)
    parser.add_argument("--supplement-support-archive-path", action="append", required=True)
    parser.add_argument("--output-support-archive-path", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--support-top-m", type=int, default=12)
    parser.add_argument("--ddc-min-absolute", type=float, default=0.95)
    parser.add_argument("--ddc-drop-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--ddc-gate-mode",
        choices=["absolute_and_ref", "absolute", "ref_relative", "relax_ref_below_min"],
        default="ref_relative",
    )
    parser.add_argument("--relax-ddc-when-ref-below-min", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feas-cost-max", type=float, default=0.10)
    parser.add_argument(
        "--feas-gate-mode",
        choices=["absolute", "ref_relative", "relax_ref_above_max", "disabled"],
        default="relax_ref_above_max",
    )
    parser.add_argument("--feas-cost-tolerance", type=float, default=0.03)
    parser.add_argument("--comfort-min", type=float, default=0.95)
    parser.add_argument(
        "--comfort-gate-mode",
        choices=["absolute", "ref_relative", "relax_ref_below_min", "disabled"],
        default="ref_relative",
    )
    parser.add_argument("--comfort-drop-tolerance", type=float, default=0.05)
    parser.add_argument("--min-non-gt-pdms", type=float, default=0.90)
    parser.add_argument(
        "--reward-gate-mode",
        choices=["absolute", "absolute_or_gt_improver", "gt_relative"],
        default="absolute",
    )
    parser.add_argument("--min-non-gt-improver-pdms", type=float, default=0.70)
    parser.add_argument("--gt-improver-margin", type=float, default=0.05)
    parser.add_argument("--gt-improver-ref-max-pdms", type=float, default=0.90)
    parser.add_argument("--max-first-xy-error-m", type=float, default=1.0)
    parser.add_argument("--max-first-heading-error-rad", type=float, default=0.8)
    parser.add_argument("--max-xy-turn-rad", type=float, default=1.2)
    parser.add_argument("--max-early-xy-turn-rad", type=float, default=1.0)
    parser.add_argument("--max-step-m", type=float, default=12.0)
    parser.add_argument("--enable-semantic-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-turn-class-mismatch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-semantic-final-heading-error-rad", type=float, default=0.75)
    parser.add_argument("--max-semantic-path-angle-error-rad", type=float, default=0.75)
    parser.add_argument("--max-semantic-endpoint-lateral-error-m", type=float, default=4.0)
    parser.add_argument("--gt-keep-min-reward", type=float, default=0.85)
    parser.add_argument("--gt-replace-margin", type=float, default=0.05)
    parser.add_argument("--high-pdms-threshold", type=float, default=0.95)
    parser.add_argument("--top-pdms-count", type=int, default=3)
    parser.add_argument("--pareto-count", type=int, default=5)
    parser.add_argument("--score-diversity-weight", type=float, default=0.8)
    parser.add_argument("--min-trajectory-diversity-score", type=float, default=0.05)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--no-link-unchanged", action="store_true")
    args = parser.parse_args()

    base_root = Path(args.base_support_archive_path)
    out_root = Path(args.output_support_archive_path)
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = _build_cfg(args)
    supplement: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for root_raw in args.supplement_support_archive_path:
        root = Path(root_raw)
        for path in _iter_paths(root):
            try:
                record = _load(path)
            except Exception:
                continue
            token = str(record.get("token", ""))
            if token:
                supplement[token].append(record)

    base_paths = list(_iter_paths(base_root))
    rank = int(args.rank)
    world_size = max(int(args.world_size), 1)
    stats = Counter()
    examples: list[dict[str, Any]] = []
    for idx, base_path in enumerate(base_paths):
        if idx % world_size != rank:
            continue
        out_path = out_root / base_path.name
        try:
            base = _load(base_path)
            token = str(base.get("token", ""))
            extras = supplement.get(token, [])
            if not extras:
                stats["linked_or_copied_unchanged"] += 1
                if args.no_link_unchanged:
                    shutil.copy2(base_path, out_path)
                else:
                    try:
                        if out_path.exists():
                            out_path.unlink()
                        os.link(base_path, out_path)
                    except OSError:
                        shutil.copy2(base_path, out_path)
                continue
            candidates = _candidate_records(base)
            before_count = len(candidates)
            for extra in extras:
                candidates.extend(_candidate_records(extra))
            candidates = _dedupe(candidates)
            selected = select_feasible_pareto_support(candidates, _reference_components(candidates), cfg)
            rebuilt = build_archive_record(token, candidates, selected, ref=_reference_components(candidates), cfg=cfg)
            _save(out_path, rebuilt)
            stats["merged_records"] += 1
            stats["base_candidate_count"] += before_count
            stats["merged_candidate_count"] += len(candidates)
            stats["selected_count"] += sum(1 for tag in rebuilt.get("support_tags", []) if tag)
        except Exception as exc:
            stats["errors"] += 1
            if len(examples) < 16:
                examples.append({"path": str(base_path), "error": repr(exc)})

    summary = {
        "base_support_archive_path": str(base_root),
        "supplement_support_archive_path": args.supplement_support_archive_path,
        "output_support_archive_path": str(out_root),
        "rank": rank,
        "world_size": world_size,
        "stats": dict(stats),
        "examples": examples,
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
