#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import lzma
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.pareto_support import CandidateRecord
from navsim.agents.recogdrive.pareto_support.pareto_archive import support_quality_metrics, support_reward_pass


def _load_record(path: Path) -> dict[str, Any]:
    if path.name.endswith(".pkl.xz"):
        with lzma.open(path, "rb") as f:
            return pickle.load(f)
    with path.open("rb") as f:
        return pickle.load(f)


def _iter_paths(root: Path):
    if root.is_file():
        yield root
        return
    yield from sorted(root.glob("*.pkl.xz"))
    yield from sorted(root.glob("*.pkl"))


def _is_gt(source: str) -> bool:
    lower = str(source).lower()
    return lower == "gt" or lower.startswith("gt:")


def _row_for_record(
    path: Path,
    record: dict[str, Any],
    *,
    min_support_count: int,
    min_non_gt_reward: float,
    poor_gt_reward_threshold: float,
    min_candidate_count: int,
    max_first_xy_error_m: float,
    max_xy_turn_rad: float,
    max_early_xy_turn_rad: float,
    reward_gate_mode: str,
    min_non_gt_improver_reward: float,
    gt_improver_margin: float,
    gt_improver_ref_max_reward: float,
) -> dict[str, Any]:
    token = str(record.get("token", path.stem))
    candidates = np.asarray(record.get("candidates", []), dtype=np.float32)
    rewards = np.asarray(record.get("rewards", []), dtype=np.float32)
    sources = [str(item) for item in record.get("sources", [])]
    valid_mask = np.asarray(record.get("valid_mask", np.zeros(len(sources), dtype=np.bool_)), dtype=np.bool_)
    support_tags = [str(item) for item in record.get("support_tags", [""] * len(sources))]
    selected_mask = np.asarray([bool(tag) for tag in support_tags], dtype=np.bool_)
    if rewards.shape[:1] != (len(sources),) or valid_mask.shape[:1] != (len(sources),):
        return {
            "token": token,
            "path": str(path),
            "reason": "shape_error",
            "candidate_count": len(sources),
            "support_count": 0,
        }

    gt_indices = [idx for idx, source in enumerate(sources) if _is_gt(source)]
    gt_idx = gt_indices[0] if gt_indices else -1
    gt_reward = float(rewards[gt_idx]) if gt_idx >= 0 and gt_idx < rewards.shape[0] else float(record.get("gt_reward", 0.0))
    ref_traj = candidates[gt_idx] if gt_idx >= 0 and candidates.ndim == 3 and gt_idx < candidates.shape[0] else None
    non_gt_selected = [idx for idx in np.flatnonzero(selected_mask) if not _is_gt(sources[idx])]
    non_gt_quality_bad = 0
    reward_cfg = {
        "support_quality_exempt_gt": True,
        "support_min_non_gt_reward": float(min_non_gt_reward),
        "support_reward_gate_mode": str(reward_gate_mode),
        "support_min_non_gt_improver_reward": float(min_non_gt_improver_reward),
        "support_gt_improver_margin": float(gt_improver_margin),
        "support_gt_improver_ref_max_reward": float(gt_improver_ref_max_reward),
    }
    for idx in non_gt_selected:
        metrics = support_quality_metrics(candidates[idx], ref_traj)
        reward_ok = support_reward_pass(
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
        if (
            not reward_ok
            or metrics["first_xy_error_m"] > max_first_xy_error_m
            or metrics["max_xy_turn_rad"] > max_xy_turn_rad
            or metrics["early_xy_turn_rad"] > max_early_xy_turn_rad
        ):
            non_gt_quality_bad += 1

    valid_count = int(valid_mask.sum())
    support_count = int(selected_mask.sum())
    selected_valid_count = int((selected_mask & valid_mask).sum())
    best_valid_reward = float(rewards[valid_mask].max()) if valid_count else 0.0
    best_selected_reward = float(rewards[selected_mask].max()) if support_count else 0.0
    reasons: list[str] = []
    if valid_count == 0:
        reasons.append("no_evaluator_valid_candidate")
    if support_count == 0:
        reasons.append("no_usable_support_after_quality_gate")
    elif support_count < min_support_count:
        reasons.append("low_support_count")
    if support_count > 0 and selected_valid_count == 0:
        reasons.append("no_valid_selected_support")
    if non_gt_quality_bad:
        reasons.append("selected_non_gt_quality_bad")
    if gt_reward < poor_gt_reward_threshold and len(sources) < min_candidate_count:
        reasons.append("poor_gt_few_candidates")

    return {
        "token": token,
        "path": str(path),
        "reason": "|".join(reasons),
        "candidate_count": len(sources),
        "valid_count": valid_count,
        "support_count": support_count,
        "selected_valid_count": selected_valid_count,
        "gt_reward": gt_reward,
        "best_valid_reward": best_valid_reward,
        "best_selected_reward": best_selected_reward,
        "non_gt_selected_quality_bad_count": int(non_gt_quality_bad),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Find SG-FPS support archive scenes that need targeted supplement mining.")
    parser.add_argument("--support-archive-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-support-count", type=int, default=4)
    parser.add_argument("--min-non-gt-pdms", type=float, default=0.90)
    parser.add_argument(
        "--reward-gate-mode",
        choices=["absolute", "absolute_or_gt_improver", "gt_relative"],
        default="absolute",
    )
    parser.add_argument("--min-non-gt-improver-pdms", type=float, default=0.70)
    parser.add_argument("--gt-improver-margin", type=float, default=0.05)
    parser.add_argument("--gt-improver-ref-max-pdms", type=float, default=0.90)
    parser.add_argument("--poor-gt-reward-threshold", type=float, default=0.85)
    parser.add_argument("--min-candidate-count", type=int, default=24)
    parser.add_argument("--max-first-xy-error-m", type=float, default=1.0)
    parser.add_argument("--max-xy-turn-rad", type=float, default=1.2)
    parser.add_argument("--max-early-xy-turn-rad", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()

    root = Path(args.support_archive_path)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_paths = list(_iter_paths(root))
    world_size = max(int(args.world_size), 1)
    paths = [path for idx, path in enumerate(all_paths) if idx % world_size == int(args.rank)]
    rows: list[dict[str, Any]] = []
    counters = Counter()
    for path in paths:
        try:
            record = _load_record(path)
            row = _row_for_record(
                path,
                record,
                min_support_count=int(args.min_support_count),
                min_non_gt_reward=float(args.min_non_gt_pdms),
                poor_gt_reward_threshold=float(args.poor_gt_reward_threshold),
                min_candidate_count=int(args.min_candidate_count),
                max_first_xy_error_m=float(args.max_first_xy_error_m),
                max_xy_turn_rad=float(args.max_xy_turn_rad),
                max_early_xy_turn_rad=float(args.max_early_xy_turn_rad),
                reward_gate_mode=str(args.reward_gate_mode),
                min_non_gt_improver_reward=float(args.min_non_gt_improver_pdms),
                gt_improver_margin=float(args.gt_improver_margin),
                gt_improver_ref_max_reward=float(args.gt_improver_ref_max_pdms),
            )
        except Exception as exc:
            row = {
                "token": path.stem,
                "path": str(path),
                "reason": "load_error",
                "error": repr(exc),
                "candidate_count": 0,
                "valid_count": 0,
                "support_count": 0,
                "selected_valid_count": 0,
                "gt_reward": 0.0,
                "best_valid_reward": 0.0,
                "best_selected_reward": 0.0,
                "non_gt_selected_quality_bad_count": 0,
            }
        counters["record_count"] += 1
        if row.get("reason"):
            rows.append(row)
            for reason in str(row["reason"]).split("|"):
                if reason:
                    counters[reason] += 1

    rows = sorted(rows, key=lambda item: (str(item.get("reason", "")), str(item.get("token", ""))))
    token_path = output / f"gap_tokens_rank{args.rank}.txt"
    token_path.write_text("\n".join(str(row["token"]) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    csv_path = output / f"gap_scenes_rank{args.rank}.csv"
    fieldnames = [
        "token",
        "reason",
        "candidate_count",
        "valid_count",
        "support_count",
        "selected_valid_count",
        "gt_reward",
        "best_valid_reward",
        "best_selected_reward",
        "non_gt_selected_quality_bad_count",
        "path",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    summary = {
        "support_archive_path": str(root),
        "rank": int(args.rank),
        "world_size": int(world_size),
        "gap_count": len(rows),
        "counts": dict(counters),
        "gap_tokens_path": str(token_path),
        "gap_csv_path": str(csv_path),
        "examples": rows[:16],
    }
    json_path = output / f"gap_summary_rank{args.rank}.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
