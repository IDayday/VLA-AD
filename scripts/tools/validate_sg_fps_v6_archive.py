#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import lzma
import pickle
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.pareto_support import (  # noqa: E402
    FullTrainingV6Config,
    V6_SELECTION_STRATEGY,
    V6_TEACHER_CONTRACT,
    full_training_v6_selection,
    source_family,
    trajectory_snsad_distance,
)


def _load(path: Path) -> dict[str, Any]:
    with lzma.open(path, "rb") as stream:
        record = pickle.load(stream)
    if not isinstance(record, dict):
        raise TypeError(f"record must be dict, got {type(record).__name__}")
    return record


def _same(actual: Any, expected: Any) -> bool:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if actual_array.shape != expected_array.shape:
        return False
    if actual_array.dtype.kind in "fc" or expected_array.dtype.kind in "fc":
        return bool(np.allclose(actual_array, expected_array, rtol=1e-5, atol=2e-6))
    return bool(np.array_equal(actual_array, expected_array))


def _chunk(paths: Sequence[Path], cfg: FullTrainingV6Config) -> dict[str, Any]:
    counts = Counter()
    tokens: list[str] = []
    errors: list[dict[str, str]] = []
    values = {key: [] for key in ("reward_delta", "pairwise_snsad", "confidence", "difficulty")}
    for path in paths:
        try:
            record = _load(path)
            if int(record.get("version", 0)) != 6:
                raise ValueError(f"version={record.get('version')!r}, expected 6")
            token = str(record.get("token", ""))
            if not token:
                raise ValueError("empty token")
            tokens.append(token)
            expected = full_training_v6_selection(record, cfg)
            for key in (
                "support_indices",
                "support_tags",
                "mode_ids",
                "teacher_eligible_mask",
                "source_conditioned_mask",
                "teacher_tier",
                "teacher_confidence",
                "learning_frontier_difficulty",
                "pareto_objective_novelty",
                "pareto_front_mask",
                "full_training_objectives",
                "gt_mode_distance_snsad",
            ):
                if key not in record or not _same(record[key], expected[key]):
                    raise ValueError(f"{key} disagrees with independent v6 reselection")
            if dict(record.get("candidate_funnel", {}) or {}) != expected["candidate_funnel"]:
                raise ValueError("candidate_funnel disagrees with independent v6 reselection")
            metadata = dict(record.get("build_metadata", {}) or {})
            if metadata.get("selection_strategy") != V6_SELECTION_STRATEGY:
                raise ValueError("selection_strategy contract mismatch")
            if metadata.get("teacher_contract") != V6_TEACHER_CONTRACT:
                raise ValueError("teacher_contract mismatch")
            if not bool(metadata.get("policy_candidates_excluded", False)):
                raise ValueError("policy_candidates_excluded must be true")
            if bool(metadata.get("policy_fields_used_for_admission", True)):
                raise ValueError("policy_fields_used_for_admission must be false")
            if bool(metadata.get("policy_fields_used_for_ranking", True)):
                raise ValueError("policy_fields_used_for_ranking must be false")

            selected = np.asarray(record["support_indices"], dtype=np.int64)
            mode_ids = np.asarray(record["mode_ids"], dtype=np.int64)
            non_gt = selected[mode_ids[selected] > 0]
            sources = [str(source) for source in record["sources"]]
            if any(source_family(sources[int(index)]) == "policy" for index in non_gt.tolist()):
                raise ValueError("selected previous-policy candidate")
            counts["records"] += 1
            counts["selected_non_gt"] += int(non_gt.size)
            counts["scenes_with_non_gt"] += int(non_gt.size > 0)
            counts["scenes_with_multiple_non_gt"] += int(non_gt.size > 1)
            counts["eligible_non_gt"] += int(
                (
                    np.asarray(record["teacher_eligible_mask"], dtype=np.bool_)
                    & (np.arange(len(sources)) != selected[0])
                ).sum()
            )
            for index in non_gt.tolist():
                family = source_family(sources[int(index)])
                counts[f"source:{family}"] += 1
                values["reward_delta"].append(
                    float(np.asarray(record["rewards"])[int(index)] - float(record["gt_reward"]))
                )
                values["confidence"].append(
                    float(np.asarray(record["teacher_confidence"])[int(index)])
                )
                values["difficulty"].append(
                    float(np.asarray(record["learning_frontier_difficulty"])[int(index)])
                )
            if non_gt.size:
                candidates = np.asarray(record["candidates"], dtype=np.float32)
                pairs = [
                    trajectory_snsad_distance(candidates[int(lhs)], candidates[int(rhs)])
                    for left_pos, lhs in enumerate(selected.tolist())
                    for rhs in selected.tolist()[left_pos + 1 :]
                ]
                values["pairwise_snsad"].append(float(min(pairs)))
        except Exception as exc:
            counts["errors"] += 1
            if len(errors) < 16:
                errors.append({"path": str(path), "error": str(exc)})
    return {"counts": dict(counts), "tokens": tokens, "errors": errors, "values": values}


def _statistics(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the SG-FPS v6 full-training archive.")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--expected-record-count", type=int, default=103288)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--min-scene-coverage", type=float, default=0.90)
    parser.add_argument("--min-multiple-mode-scene-ratio", type=float, default=0.35)
    args = parser.parse_args()

    root = Path(args.archive).expanduser()
    paths = sorted(root.glob("*.pkl.xz"))
    if args.max_records > 0:
        paths = paths[: args.max_records]
    workers = max(1, int(args.workers))
    chunks = [paths[index:: workers * 4] for index in range(workers * 4)]
    cfg = FullTrainingV6Config()
    counts = Counter()
    tokens: list[str] = []
    errors: list[dict[str, str]] = []
    values = {key: [] for key in ("reward_delta", "pairwise_snsad", "confidence", "difficulty")}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(_chunk, chunks, [cfg] * len(chunks)):
            counts.update(result["counts"])
            tokens.extend(result["tokens"])
            if len(errors) < 32:
                errors.extend(result["errors"][: 32 - len(errors)])
            for key, chunk_values in result["values"].items():
                values[key].extend(chunk_values)

    record_count = int(counts["records"])
    expected_count = int(args.expected_record_count)
    unique_count = len(set(tokens))
    duplicate_count = len(tokens) - unique_count
    scene_coverage = counts["scenes_with_non_gt"] / max(record_count, 1)
    multiple_ratio = counts["scenes_with_multiple_non_gt"] / max(record_count, 1)
    selected_policy_count = int(counts.get("source:policy", 0))
    source_counts = {
        key.split(":", 1)[1]: int(value)
        for key, value in counts.items()
        if key.startswith("source:")
    }
    full_count_required = args.max_records <= 0 and expected_count > 0
    contract_pass = counts["errors"] == 0 and duplicate_count == 0
    coverage_pass = (
        scene_coverage >= float(args.min_scene_coverage)
        and multiple_ratio >= float(args.min_multiple_mode_scene_ratio)
        and selected_policy_count == 0
        and all(source_counts.get(family, 0) > 0 for family in ("ddv2", "driveor", "progress", "lateral", "timing"))
    )
    count_pass = not full_count_required or (record_count == expected_count and unique_count == expected_count)
    report = {
        "archive": str(root),
        "record_count": record_count,
        "unique_token_count": unique_count,
        "duplicate_token_count": duplicate_count,
        "expected_record_count": expected_count,
        "contract_error_count": int(counts["errors"]),
        "errors": errors,
        "selected_non_gt_count": int(counts["selected_non_gt"]),
        "eligible_non_gt_count": int(counts["eligible_non_gt"]),
        "scene_coverage": float(scene_coverage),
        "multiple_non_gt_scene_ratio": float(multiple_ratio),
        "source_counts": source_counts,
        "selected_policy_count": selected_policy_count,
        "statistics": {key: _statistics(item) for key, item in values.items()},
        "contract_pass": bool(contract_pass),
        "count_pass": bool(count_pass),
        "coverage_pass": bool(coverage_pass),
        "static_full_training_gate_pass": bool(contract_pass and count_pass and coverage_pass),
        "dynamic_random_init_gate_required": True,
    }
    if args.output_json:
        output = Path(args.output_json).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["static_full_training_gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
