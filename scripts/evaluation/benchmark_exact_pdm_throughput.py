#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.common.dataloader import MetricCacheLoader  # noqa: E402
from navsim.common.dataclasses import Trajectory  # noqa: E402
from navsim.evaluate.pdm_score import pdm_score  # noqa: E402
from navsim.planning.metric_caching.fast_metric_cache_loader import (  # noqa: E402
    FastMetricCacheLoader,
    load_metric_cache_auto,
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (  # noqa: E402
    PDMScorer,
    PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator  # noqa: E402


PDM_KEYS: Tuple[str, ...] = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
)

_THREAD_LOCAL = threading.local()
_PROCESS_TOOLS: Tuple[TrajectorySampling, PDMSimulator, PDMScorer] | None = None


def _make_tools() -> Tuple[TrajectorySampling, PDMSimulator, PDMScorer]:
    proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    scorer_config = PDMScorerConfig(
        progress_weight=5.0,
        ttc_weight=5.0,
        comfortable_weight=2.0,
        driving_direction_weight=0.0,
    )
    return proposal_sampling, PDMSimulator(proposal_sampling), PDMScorer(proposal_sampling, scorer_config)


def _get_thread_tools() -> Tuple[TrajectorySampling, PDMSimulator, PDMScorer]:
    tools = getattr(_THREAD_LOCAL, "tools", None)
    if tools is None:
        tools = _make_tools()
        _THREAD_LOCAL.tools = tools
    return tools


def _init_process_tools() -> None:
    global _PROCESS_TOOLS
    _PROCESS_TOOLS = _make_tools()


def _get_process_tools() -> Tuple[TrajectorySampling, PDMSimulator, PDMScorer]:
    global _PROCESS_TOOLS
    if _PROCESS_TOOLS is None:
        _PROCESS_TOOLS = _make_tools()
    return _PROCESS_TOOLS


def _load_items(cache_dir: Path, loader_kind: str) -> List[Tuple[str, Path]]:
    if loader_kind == "original":
        loader = MetricCacheLoader(cache_dir)
    elif loader_kind == "fast":
        loader = FastMetricCacheLoader(cache_dir)
    else:
        raise ValueError(f"Unsupported loader kind: {loader_kind}")
    return sorted((str(token), Path(path)) for token, path in loader.metric_cache_paths.items())


def _select_items(
    items: List[Tuple[str, Path]],
    *,
    max_tokens: int,
    seed: int,
    shuffle: bool,
    shard_count: int,
    shard_index: int,
) -> List[Tuple[str, Path]]:
    if shard_count <= 0:
        raise ValueError("--shard-count must be positive.")
    if not (0 <= shard_index < shard_count):
        raise ValueError("--shard-index must be in [0, shard_count).")
    selected = list(items)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(selected)
    if max_tokens > 0:
        selected = selected[:max_tokens]
    return [item for index, item in enumerate(selected) if index % shard_count == shard_index]


def _score_one(task: Tuple[str, str, np.ndarray, str]) -> Dict[str, Any]:
    token, path_text, poses, tools_mode = task
    row: Dict[str, Any] = {"token": token, "valid": True}
    try:
        t0 = time.perf_counter()
        metric_cache = load_metric_cache_auto(Path(path_text))
        t1 = time.perf_counter()
        if tools_mode == "process":
            proposal_sampling, simulator, scorer = _get_process_tools()
        else:
            proposal_sampling, simulator, scorer = _get_thread_tools()
        result = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=Trajectory(np.asarray(poses, dtype=np.float32)),
            future_sampling=proposal_sampling,
            simulator=simulator,
            scorer=scorer,
        )
        t2 = time.perf_counter()
        row.update(asdict(result))
        row["load_s"] = t1 - t0
        row["score_s"] = t2 - t1
        row["total_s"] = t2 - t0
    except Exception as exc:  # pragma: no cover - diagnostics script
        row["valid"] = False
        row["error"] = repr(exc)
        row["traceback"] = traceback.format_exc()
    return row


def _zero_trajectory() -> np.ndarray:
    return np.zeros((8, 3), dtype=np.float32)


def _run_serial(tasks: List[Tuple[str, str, np.ndarray, str]]) -> List[Dict[str, Any]]:
    return [_score_one((token, path, poses, "thread")) for token, path, poses, _ in tasks]


def _score_chunk(tasks: List[Tuple[str, str, np.ndarray, str]]) -> List[Dict[str, Any]]:
    return [_score_one(task) for task in tasks]


def _chunk_tasks(
    tasks: List[Tuple[str, str, np.ndarray, str]],
    task_chunk_size: int,
) -> List[List[Tuple[str, str, np.ndarray, str]]]:
    if task_chunk_size <= 1:
        return [[task] for task in tasks]
    return [tasks[start : start + task_chunk_size] for start in range(0, len(tasks), task_chunk_size)]


def _run_parallel(
    tasks: List[Tuple[str, str, np.ndarray, str]],
    *,
    backend: str,
    workers: int,
    task_chunk_size: int,
    progress_every: int,
) -> List[Dict[str, Any]]:
    if backend == "serial" or workers <= 1:
        return _run_serial(tasks)

    executor_cls = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
    kwargs: Dict[str, Any] = {"max_workers": workers}
    if backend == "process":
        kwargs["initializer"] = _init_process_tools

    task_chunks = _chunk_tasks(tasks, task_chunk_size)
    rows: List[Dict[str, Any]] = []
    with executor_cls(**kwargs) as executor:
        futures = [executor.submit(_score_chunk, chunk) for chunk in task_chunks]
        for done_count, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            completed = min(done_count * max(1, task_chunk_size), len(tasks))
            if progress_every > 0 and completed % progress_every == 0:
                print(
                    json.dumps(
                        {"state": "progress", "completed": completed, "total": len(tasks)},
                        sort_keys=True,
                    ),
                    flush=True,
                )
    rows.sort(key=lambda row: row["token"])
    return rows


def _metric_diff(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, float]:
    return {key: abs(float(left[key]) - float(right[key])) for key in PDM_KEYS if key in left and key in right}


def _exactness_check(
    items: List[Tuple[str, Path]],
    rows_by_token: Dict[str, Dict[str, Any]],
    *,
    exactness_tokens: int,
    max_abs_diff: float,
) -> Dict[str, Any]:
    if exactness_tokens <= 0:
        return {"checked": 0, "max_abs_diff": 0.0, "passed": True}

    reference_tasks = [
        (token, str(path), _zero_trajectory(), "thread")
        for token, path in items[: min(exactness_tokens, len(items))]
    ]
    reference_rows = _run_serial(reference_tasks)
    max_diff = 0.0
    failures: List[Dict[str, Any]] = []
    for ref in reference_rows:
        token = ref["token"]
        row = rows_by_token.get(token)
        if row is None:
            failures.append({"token": token, "error": "missing benchmark row"})
            continue
        if bool(ref.get("valid")) != bool(row.get("valid")):
            failures.append({"token": token, "error": "valid mismatch", "ref": ref.get("valid"), "row": row.get("valid")})
            continue
        if not ref.get("valid"):
            continue
        diffs = _metric_diff(ref, row)
        row_max = max(diffs.values()) if diffs else 0.0
        max_diff = max(max_diff, row_max)
        if row_max > max_abs_diff:
            failures.append({"token": token, "max_abs_diff": row_max, "diffs": diffs})
    return {
        "checked": len(reference_rows),
        "max_abs_diff": max_diff,
        "passed": not failures,
        "failures": failures[:10],
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys() if key != "traceback"})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def _summarize(rows: List[Dict[str, Any]], elapsed_s: float) -> Dict[str, Any]:
    valid_rows = [row for row in rows if row.get("valid")]

    def mean_field(name: str) -> float | None:
        values = [float(row[name]) for row in valid_rows if name in row]
        return float(np.mean(values)) if values else None

    return {
        "num_rows": len(rows),
        "num_valid": len(valid_rows),
        "num_failed": len(rows) - len(valid_rows),
        "elapsed_s": elapsed_s,
        "tokens_per_s": (len(rows) / elapsed_s) if elapsed_s > 0 else None,
        "valid_tokens_per_s": (len(valid_rows) / elapsed_s) if elapsed_s > 0 else None,
        "mean_load_s": mean_field("load_s"),
        "mean_score_s": mean_field("score_s"),
        "mean_total_s": mean_field("total_s"),
        "mean_pdms": mean_field("score"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark exact scalar PDM throughput without changing PDMS semantics. "
            "Every scored token calls navsim.evaluate.pdm_score.pdm_score()."
        )
    )
    parser.add_argument("--metric-cache-dir", type=Path, required=True)
    parser.add_argument("--loader", choices=("original", "fast"), default="original")
    parser.add_argument("--backend", choices=("serial", "thread", "process"), default="thread")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--task-chunk-size",
        type=int,
        default=1,
        help=(
            "Group this many scalar pdm_score calls into one executor Future. "
            "This changes scheduling overhead only; every token still calls the original scalar scorer."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--exactness-tokens", type=int, default=16)
    parser.add_argument("--max-abs-diff", type=float, default=0.0)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--rows-csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if args.task_chunk_size <= 0:
        raise ValueError("--task-chunk-size must be positive.")
    if args.max_tokens < 0:
        raise ValueError("--max-tokens must be non-negative.")
    if args.exactness_tokens < 0:
        raise ValueError("--exactness-tokens must be non-negative.")
    if args.max_abs_diff < 0:
        raise ValueError("--max-abs-diff must be non-negative.")

    all_items = _load_items(args.metric_cache_dir, args.loader)
    items = _select_items(
        all_items,
        max_tokens=args.max_tokens,
        seed=args.seed,
        shuffle=args.shuffle,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    if not items:
        raise FileNotFoundError(f"No metric-cache items selected from {args.metric_cache_dir}")

    tools_mode = "process" if args.backend == "process" else "thread"
    tasks = [(token, str(path), _zero_trajectory(), tools_mode) for token, path in items]

    t0 = time.perf_counter()
    rows = _run_parallel(
        tasks,
        backend=args.backend,
        workers=args.workers,
        task_chunk_size=args.task_chunk_size,
        progress_every=args.progress_every,
    )
    elapsed_s = time.perf_counter() - t0

    rows_by_token = {str(row["token"]): row for row in rows}
    exactness = _exactness_check(items, rows_by_token, exactness_tokens=args.exactness_tokens, max_abs_diff=args.max_abs_diff)
    summary = {
        "metric_cache_dir": str(args.metric_cache_dir),
        "loader": args.loader,
        "backend": args.backend,
        "workers": args.workers,
        "task_chunk_size": args.task_chunk_size,
        "max_tokens": args.max_tokens,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "selected_tokens": len(items),
        "benchmark": _summarize(rows, elapsed_s),
        "exactness": exactness,
    }

    if args.rows_csv is not None:
        _write_csv(args.rows_csv, rows)
    payload = json.dumps(summary, indent=2, sort_keys=True)
    print(payload)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(payload + "\n", encoding="utf-8")
    if not exactness.get("passed", False):
        raise SystemExit("Exactness check failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
