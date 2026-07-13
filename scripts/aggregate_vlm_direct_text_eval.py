#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


PDM_KEYS = [
    "score",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "comfort",
    "ego_progress",
    "driving_direction_compliance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate sharded direct-text VLM PDM eval outputs.")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--run", action="append", default=[])
    return parser.parse_args()


def to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def aggregate_run(run_dir: Path) -> Dict[str, Any]:
    rows: List[Dict[str, str]] = []
    shard_metrics: List[Dict[str, Any]] = []
    for shard_dir in sorted(run_dir.glob("shard_*")):
        metrics_path = shard_dir / "metrics.json"
        if metrics_path.is_file():
            shard_metrics.append(json.loads(metrics_path.read_text(encoding="utf-8")))
        csv_path = shard_dir / "pdm_results.csv"
        if csv_path.is_file():
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                rows.extend(list(csv.DictReader(f)))

    metric_values = {key: [] for key in PDM_KEYS}
    parse_ok = 0
    valid = 0
    for row in rows:
        if to_bool(row.get("parse_ok")):
            parse_ok += 1
        if to_bool(row.get("valid")):
            valid += 1
            for key in PDM_KEYS:
                value = to_float(row.get(key))
                if value is not None:
                    metric_values[key].append(value)

    first_metrics = shard_metrics[0] if shard_metrics else {}
    summary = {
        "run": run_dir.name,
        "trajectory_output_key": "direct_text_pt",
        "model_path": first_metrics.get("model_path"),
        "lora_adapter_dir": first_metrics.get("lora_adapter_dir"),
        "prompt_type": first_metrics.get("prompt_type"),
        "cam_type": first_metrics.get("cam_type"),
        "precision": first_metrics.get("precision"),
        "num_shards": len(shard_metrics),
        "num_samples": len(rows),
        "num_parse_ok": parse_ok,
        "parse_success_rate": parse_ok / len(rows) if rows else None,
        "num_pdm_valid": valid,
        "PDMS": mean(metric_values["score"]),
        "pdm_score": mean(metric_values["score"]),
        "NC": mean(metric_values["no_at_fault_collisions"]),
        "DAC": mean(metric_values["drivable_area_compliance"]),
        "TTC": mean(metric_values["time_to_collision_within_bound"]),
        "comfort": mean(metric_values["comfort"]),
        "EP": mean(metric_values["ego_progress"]),
        "DDC": mean(metric_values["driving_direction_compliance"]),
        "shard_metrics": [str(run_dir / f"shard_{idx:02d}" / "metrics.json") for idx in range(len(shard_metrics))],
    }
    (run_dir / "aggregate_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    runs = args.run or [path.name for path in sorted(args.out_root.iterdir()) if path.is_dir()]
    summaries = []
    for run in runs:
        run_dir = args.out_root / run
        if run_dir.is_dir():
            summaries.append(aggregate_run(run_dir))

    fields = [
        "run",
        "trajectory_output_key",
        "model_path",
        "lora_adapter_dir",
        "prompt_type",
        "cam_type",
        "precision",
        "num_shards",
        "num_samples",
        "num_parse_ok",
        "parse_success_rate",
        "num_pdm_valid",
        "PDMS",
        "NC",
        "DAC",
        "TTC",
        "comfort",
        "EP",
        "DDC",
    ]
    with (args.out_root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key) for key in fields})
    print(json.dumps(summaries, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
