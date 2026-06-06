from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


METRIC_KEYS = (
    "score",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "comfort",
)


def as_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(values: List[float], q: float) -> Optional[float]:
    return float(np.percentile(np.asarray(values, dtype=np.float32), q)) if values else None


def mean(values: List[float]) -> Optional[float]:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else None


def median(values: List[float]) -> Optional[float]:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else None


def zero_count(values: List[float]) -> int:
    return int(sum(1 for value in values if value <= 1e-9))


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in rows if bool(row.get("valid"))]
    values = {key: [as_float(row.get(key)) for row in valid] for key in METRIC_KEYS}
    clean = {key: [value for value in vals if value is not None] for key, vals in values.items()}
    scores = clean["score"]
    n = len(scores)
    hist_counts: List[int] = []
    hist_bins: List[float] = []
    if scores:
        counts, bins = np.histogram(np.asarray(scores, dtype=np.float32), bins=20, range=(0.0, 1.0))
        hist_counts = [int(item) for item in counts.tolist()]
        hist_bins = [float(item) for item in bins.tolist()]
    return {
        "num_samples": len(rows),
        "num_pdm_valid": len(valid),
        "mean_pdms": mean(scores),
        "median_pdms": median(scores),
        "p1_pdms": percentile(scores, 1),
        "p5_pdms": percentile(scores, 5),
        "p10_pdms": percentile(scores, 10),
        "zero_score_count": zero_count(scores),
        "zero_score_rate": zero_count(scores) / n if n else None,
        "no_at_fault_collision_zero_count": zero_count(clean["no_at_fault_collisions"]),
        "no_at_fault_collision_zero_rate": zero_count(clean["no_at_fault_collisions"]) / n if n else None,
        "drivable_area_compliance_zero_count": zero_count(clean["drivable_area_compliance"]),
        "drivable_area_compliance_zero_rate": zero_count(clean["drivable_area_compliance"]) / n if n else None,
        "time_to_collision_zero_count": zero_count(clean["time_to_collision_within_bound"]),
        "time_to_collision_zero_rate": zero_count(clean["time_to_collision_within_bound"]) / n if n else None,
        "ego_progress_mean": mean(clean["ego_progress"]),
        "ego_progress_median": median(clean["ego_progress"]),
        "comfort_mean": mean(clean["comfort"]),
        "comfort_median": median(clean["comfort"]),
        "histogram": {"bins": hist_bins, "counts": hist_counts},
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "sample_token",
        "scene_token",
        "chunk",
        "valid",
        "score",
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "time_to_collision_within_bound",
        "ego_progress",
        "comfort",
        "trajectory_l1",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Merge sharded eval_bit_drive_pdm.py outputs.")
    parser.add_argument("--sharded-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-pattern", default="shard_[0-9][0-9]")
    parser.add_argument("--require-all", type=int, default=None)
    args = parser.parse_args(argv)

    shards = sorted(path for path in args.sharded_output_dir.glob(args.shard_pattern) if path.is_dir())
    if args.require_all is not None and len([p for p in shards if (p / "aggregate_metrics.json").is_file()]) < int(args.require_all):
        raise RuntimeError(f"Only {sum((p / 'aggregate_metrics.json').is_file() for p in shards)} completed shards; require {args.require_all}")
    rows: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []
    completed = []
    missing = []
    for shard in shards:
        metrics_path = shard / "per_sample_metrics.jsonl"
        pred_path = shard / "predictions.jsonl"
        if not metrics_path.is_file():
            missing.append(shard.name)
            continue
        completed.append(shard.name)
        rows.extend(read_jsonl(metrics_path))
        predictions.extend(read_jsonl(pred_path))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    write_csv(args.output_dir / "per_sample_metrics.csv", rows)
    summary = aggregate(rows)
    summary["source_shards"] = len(completed)
    summary["missing_shards"] = missing
    (args.output_dir / "aggregate_metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "histogram.json").write_text(json.dumps(summary["histogram"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Sharded PDM Eval Report",
        "",
        f"Source: `{args.sharded_output_dir}`",
        f"Completed shards: `{len(completed)}`",
        f"Samples: `{summary['num_samples']}`",
        f"PDM valid: `{summary['num_pdm_valid']}`",
        f"Mean PDMS: `{summary['mean_pdms']}`",
        f"P1/P5/P10 PDMS: `{summary['p1_pdms']}` / `{summary['p5_pdms']}` / `{summary['p10_pdms']}`",
        f"Zero-score count: `{summary['zero_score_count']}`",
        f"DAC0/NC0/TTC0: `{summary['drivable_area_compliance_zero_count']}` / `{summary['no_at_fault_collision_zero_count']}` / `{summary['time_to_collision_zero_count']}`",
    ]
    if missing:
        lines.append(f"Missing shards: `{missing}`")
    (args.output_dir / "aggregate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
