#!/usr/bin/env python3
from __future__ import annotations

import argparse
import lzma
import pickle
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator


CORE_COMPONENT_MAP: Tuple[Tuple[str, str], ...] = (
    ("pdms", "score"),
    ("no_at_fault_collisions", "no_at_fault_collisions"),
    ("drivable_area_compliance", "drivable_area_compliance"),
    ("ego_progress", "ego_progress"),
    ("time_to_collision_within_bound", "time_to_collision_within_bound"),
    ("history_comfort", "comfort"),
    ("driving_direction_compliance", "driving_direction_compliance"),
)


def _iter_record_paths(buffer_dir: Path) -> Iterable[Path]:
    for path in buffer_dir.iterdir():
        if path.name.endswith(".pkl.xz") and path.is_file():
            yield path


def _reservoir_sample(paths: Iterable[Path], sample_size: int, seed: int) -> Tuple[List[Path], int]:
    rng = random.Random(seed)
    sample: List[Path] = []
    seen = 0
    for path in paths:
        seen += 1
        if len(sample) < sample_size:
            sample.append(path)
            continue
        index = rng.randrange(seen)
        if index < sample_size:
            sample[index] = path
    return sample, seen


def _load_pickle_xz(path: Path) -> Any:
    with lzma.open(path, "rb") as f:
        return pickle.load(f)


def _saved_dtype_exact(saved: Any, reference: float) -> bool:
    saved_array = np.asarray(saved)
    if np.issubdtype(saved_array.dtype, np.floating):
        return bool(saved_array == np.asarray(reference, dtype=saved_array.dtype))
    return bool(float(saved) == float(reference))


def _check_record(
    record: Dict[str, Any],
    metric_cache: Any,
    *,
    simulator: PDMSimulator,
    proposal_sampling: TrajectorySampling,
    scorer_config: PDMScorerConfig,
    max_candidates_per_record: int,
) -> Tuple[int, float, Dict[str, float], Tuple[str, int, str, float, float, float] | None, int]:
    candidates = np.asarray(record["candidates"])
    if max_candidates_per_record > 0:
        candidates = candidates[:max_candidates_per_record]

    max_abs_diff = 0.0
    max_by_key: Dict[str, float] = defaultdict(float)
    max_case = None
    saved_dtype_mismatches = 0
    token = str(record["token"])

    for index, trajectory in enumerate(candidates):
        scalar_row = asdict(
            pdm_score(
                metric_cache=metric_cache,
                model_trajectory=Trajectory(trajectory),
                future_sampling=proposal_sampling,
                simulator=simulator,
                scorer=PDMScorer(proposal_sampling, scorer_config),
            )
        )
        checks = [("rewards", record["rewards"][index], float(scalar_row["score"]))]
        checks.extend(
            (
                buffer_key,
                record["components"][buffer_key][index],
                float(scalar_row[scalar_key]),
            )
            for buffer_key, scalar_key in CORE_COMPONENT_MAP
        )
        for key, saved, reference in checks:
            saved_float = float(saved)
            diff = abs(saved_float - reference)
            max_by_key[key] = max(max_by_key[key], diff)
            if diff > max_abs_diff:
                max_abs_diff = diff
                max_case = (token, index, key, saved_float, reference, diff)
            if not _saved_dtype_exact(saved, reference):
                saved_dtype_mismatches += 1

    return len(candidates), max_abs_diff, dict(max_by_key), max_case, saved_dtype_mismatches


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Spot-check AWAC elite buffer PDMS fields against the original scalar pdm_score implementation. "
            "This script is read-only and does not use any fast/batched scorer path."
        )
    )
    parser.add_argument("--buffer-dir", required=True, type=Path)
    parser.add_argument("--metric-cache-dir", required=True, type=Path)
    parser.add_argument("--sample-records", type=int, default=32)
    parser.add_argument("--max-candidates-per-record", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--max-abs-diff", type=float, default=1e-6)
    parser.add_argument(
        "--require-saved-dtype-exact",
        action="store_true",
        help="Also require saved float32 values to equal scalar pdm_score values rounded to the saved dtype.",
    )
    args = parser.parse_args()

    if args.sample_records <= 0:
        raise ValueError("--sample-records must be positive.")
    if args.max_candidates_per_record < 0:
        raise ValueError("--max-candidates-per-record must be non-negative.")
    if args.max_abs_diff < 0:
        raise ValueError("--max-abs-diff must be non-negative.")

    sample, total_records = _reservoir_sample(_iter_record_paths(args.buffer_dir), args.sample_records, args.seed)
    if not sample:
        raise FileNotFoundError(f"No *.pkl.xz records found under {args.buffer_dir}")

    metric_loader = MetricCacheLoader(args.metric_cache_dir)
    proposal_sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)
    simulator = PDMSimulator(proposal_sampling)
    scorer_config = PDMScorerConfig(progress_weight=10.0, ttc_weight=5.0, comfortable_weight=2.0)

    checked_candidates = 0
    max_abs_diff = 0.0
    max_by_key: Dict[str, float] = defaultdict(float)
    max_case = None
    saved_dtype_mismatches = 0

    for record_path in sample:
        record = _load_pickle_xz(record_path)
        token = str(record["token"])
        if token not in metric_loader.metric_cache_paths:
            raise KeyError(f"Metric cache missing token={token!r}")
        metric_cache = _load_pickle_xz(metric_loader.metric_cache_paths[token])
        count, record_max, record_by_key, record_case, record_dtype_mismatches = _check_record(
            record,
            metric_cache,
            simulator=simulator,
            proposal_sampling=proposal_sampling,
            scorer_config=scorer_config,
            max_candidates_per_record=args.max_candidates_per_record,
        )
        checked_candidates += count
        saved_dtype_mismatches += record_dtype_mismatches
        if record_max > max_abs_diff:
            max_abs_diff = record_max
            max_case = record_case
        for key, value in record_by_key.items():
            max_by_key[key] = max(max_by_key[key], value)

    print(f"record_files_seen={total_records}")
    print(f"records_sampled={len(sample)}")
    print(f"candidates_checked={checked_candidates}")
    print(f"max_abs_diff={max_abs_diff}")
    print(f"max_case={max_case}")
    print("max_by_key=" + ", ".join(f"{key}:{value:.8g}" for key, value in sorted(max_by_key.items())))
    print(f"saved_dtype_mismatches={saved_dtype_mismatches}")

    if max_abs_diff > args.max_abs_diff:
        raise SystemExit(
            f"PDMS exactness check failed: max_abs_diff={max_abs_diff} > --max-abs-diff={args.max_abs_diff}"
        )
    if args.require_saved_dtype_exact and saved_dtype_mismatches:
        raise SystemExit(
            "PDMS exactness check failed: saved values do not match scalar pdm_score rounded to saved dtype "
            f"for {saved_dtype_mismatches} fields."
        )


if __name__ == "__main__":
    main()
