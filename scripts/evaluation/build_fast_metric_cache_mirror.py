#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import lzma
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.common.dataloader import MetricCacheLoader  # noqa: E402
from navsim.common.dataclasses import Trajectory  # noqa: E402
from navsim.evaluate.pdm_score import pdm_score  # noqa: E402
from navsim.planning.metric_caching.fast_metric_cache_loader import (  # noqa: E402
    load_metric_cache_auto,
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (  # noqa: E402
    PDMScorer,
    PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator  # noqa: E402
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # noqa: E402


PDM_KEYS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a plain-pickle mirror of NAVSIM metric cache for opt-in faster PDMS evaluation I/O."
    )
    parser.add_argument("--source-cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-samples", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _read_original(cache_path: Path) -> Any:
    with lzma.open(cache_path, "rb") as f:
        return pickle.load(f)


def _token_to_rel_path(token: str) -> Path:
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
    return Path(digest[:2]) / digest[2:4] / token / "metric_cache.pkl"


def _iter_shard_items(loader: MetricCacheLoader, args: argparse.Namespace) -> List[Tuple[str, Path]]:
    if args.shard_count <= 0:
        raise ValueError("--shard-count must be positive.")
    if not (0 <= args.shard_index < args.shard_count):
        raise ValueError("--shard-index must be in [0, shard_count).")
    items = sorted((str(token), Path(path)) for token, path in loader.metric_cache_paths.items())
    if args.max_tokens is not None:
        items = items[: args.max_tokens]
    return [item for idx, item in enumerate(items) if idx % args.shard_count == args.shard_index]


def _write_metadata(out_dir: Path, rows: List[Dict[str, str]], shard_index: int) -> Path:
    metadata_dir = out_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f"fast_metric_cache_metadata_shard_{shard_index}.csv"
    tmp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    fields = ("token", "file_name", "source_file_name")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(metadata_path)
    return metadata_path


def _build_zero_trajectory() -> Trajectory:
    return Trajectory(np.zeros((8, 3), dtype=np.float32))


def _score_zero(metric_cache: Any) -> Dict[str, float]:
    proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    result = pdm_score(
        metric_cache=metric_cache,
        model_trajectory=_build_zero_trajectory(),
        future_sampling=proposal_sampling,
        simulator=PDMSimulator(proposal_sampling),
        scorer=PDMScorer(proposal_sampling, PDMScorerConfig()),
    )
    return {key: float(value) for key, value in asdict(result).items()}


def _validate_exact(rows: List[Dict[str, str]], validate_samples: int) -> Dict[str, Any]:
    if validate_samples <= 0 or not rows:
        return {"checked": 0, "max_abs_diff": 0.0, "passed": True}
    checked_rows = rows[: min(validate_samples, len(rows))]
    max_abs_diff = 0.0
    failures: List[Dict[str, Any]] = []
    for row in checked_rows:
        token = row["token"]
        source_cache = _read_original(Path(row["source_file_name"]))
        mirror_cache = load_metric_cache_auto(Path(row["file_name"]))
        source_scores = _score_zero(source_cache)
        mirror_scores = _score_zero(mirror_cache)
        for key in PDM_KEYS:
            diff = abs(float(source_scores[key]) - float(mirror_scores[key]))
            max_abs_diff = max(max_abs_diff, diff)
            if diff != 0.0:
                failures.append(
                    {
                        "token": token,
                        "key": key,
                        "source": source_scores[key],
                        "mirror": mirror_scores[key],
                        "diff": diff,
                    }
                )
    return {
        "checked": len(checked_rows),
        "max_abs_diff": max_abs_diff,
        "passed": not failures,
        "failures": failures[:10],
    }


def build(args: argparse.Namespace) -> Dict[str, Any]:
    loader = MetricCacheLoader(args.source_cache_dir)
    out_dir = args.out_dir
    items = _iter_shard_items(loader, args)
    rows: List[Dict[str, str]] = []
    written = 0
    skipped = 0

    for idx, (token, source_path) in enumerate(items, start=1):
        rel_path = _token_to_rel_path(token)
        out_path = out_dir / rel_path
        rows.append(
            {
                "token": token,
                "file_name": str(out_path),
                "source_file_name": str(source_path),
            }
        )
        if args.dry_run:
            continue
        if out_path.exists() and args.skip_existing and not args.overwrite:
            skipped += 1
        else:
            metric_cache = _read_original(source_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = out_path.with_suffix(out_path.suffix + f".tmp.{args.shard_index}")
            with tmp_path.open("wb") as f:
                pickle.dump(metric_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(out_path)
            written += 1
        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "state": "progress",
                        "shard_index": args.shard_index,
                        "processed": idx,
                        "total": len(items),
                        "written": written,
                        "skipped": skipped,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    metadata_path = None
    validation: Dict[str, Any] = {"checked": 0, "max_abs_diff": 0.0, "passed": True}
    if not args.dry_run:
        metadata_path = _write_metadata(out_dir, rows, args.shard_index)
        validation = _validate_exact(rows, int(args.validate_samples))
        if not validation["passed"]:
            raise AssertionError(json.dumps(validation, indent=2, sort_keys=True))

    summary = {
        "source_cache_dir": str(args.source_cache_dir),
        "out_dir": str(args.out_dir),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "num_items": len(items),
        "written": written,
        "skipped": skipped,
        "metadata_path": str(metadata_path) if metadata_path is not None else None,
        "validation": validation,
        "dry_run": bool(args.dry_run),
    }
    return summary


def main() -> int:
    args = parse_args()
    summary = build(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
