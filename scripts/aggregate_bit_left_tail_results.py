#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


METRIC_KEYS = [
    "mean_pdms",
    "median_pdms",
    "p5_pdms",
    "p10_pdms",
    "zero_score_count",
    "zero_score_rate",
    "drivable_area_compliance_zero_count",
    "no_at_fault_collision_zero_count",
    "time_to_collision_zero_count",
    "ego_progress_mean",
    "ego_progress_median",
    "comfort_mean",
    "comfort_median",
]


RUN_ORDER = {
    "eval_a0_base": 0,
    "eval_step1": 1,
    "eval_step2": 2,
    "eval_step3": 3,
    "eval_B1": 11,
    "eval_B2": 12,
    "eval_B3": 13,
    "eval_B4": 14,
    "eval_B5": 15,
    "eval_B6": 16,
    "eval_B7": 17,
}


def discover_eval_dirs(input_root: Path) -> List[Path]:
    dirs = [path.parent for path in input_root.rglob("aggregate_metrics.json") if path.is_file()]
    return sorted(dirs, key=lambda path: (RUN_ORDER.get(path.name, 1000), str(path)))


def load_metrics(eval_dir: Path) -> Dict[str, Any]:
    path = eval_dir / "aggregate_metrics.json"
    if not path.is_file():
        path = eval_dir / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"No aggregate_metrics.json or metrics.json in {eval_dir}")
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("name", eval_dir.name)
    data.setdefault("eval_dir", str(eval_dir))
    return data


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = ["name", "eval_dir", "num_samples", "num_pdm_valid", *METRIC_KEYS]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def delta(current: Any, baseline: Any) -> Any:
    if current is None or baseline is None:
        return None
    try:
        return float(current) - float(baseline)
    except (TypeError, ValueError):
        return None


def write_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    baseline = rows[0] if rows else {}
    lines = [
        "# BiT-Drive Left-Tail Aggregate Results",
        "",
        "| Run | Samples | Mean | Median | P5 | P10 | Zero | DAC0 | NC0 | TTC0 | Ego Progress |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('name')} | {row.get('num_samples')} | {row.get('mean_pdms')} | "
            f"{row.get('median_pdms')} | {row.get('p5_pdms')} | {row.get('p10_pdms')} | "
            f"{row.get('zero_score_count')} | {row.get('drivable_area_compliance_zero_count')} | "
            f"{row.get('no_at_fault_collision_zero_count')} | {row.get('time_to_collision_zero_count')} | "
            f"{row.get('ego_progress_mean')} |"
        )
    if baseline:
        lines.extend(["", "## Delta vs First Row", "", "| Run | Zero Delta | DAC0 Delta | NC0 Delta | P5 Delta | Mean Delta |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for row in rows[1:]:
            lines.append(
                f"| {row.get('name')} | {delta(row.get('zero_score_count'), baseline.get('zero_score_count'))} | "
                f"{delta(row.get('drivable_area_compliance_zero_count'), baseline.get('drivable_area_compliance_zero_count'))} | "
                f"{delta(row.get('no_at_fault_collision_zero_count'), baseline.get('no_at_fault_collision_zero_count'))} | "
                f"{delta(row.get('p5_pdms'), baseline.get('p5_pdms'))} | {delta(row.get('mean_pdms'), baseline.get('mean_pdms'))} |"
            )
    lines.extend([
        "",
        "Judge BiT by left-tail changes first: zero-score, DAC0, NC0, TTC0, and P5/P10 PDMS. Mean PDMS alone is insufficient.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate BiT-Drive v1/v2 left-tail evaluation directories.")
    parser.add_argument("--input-root", type=Path, default=Path("/mnt/project/VLA-AD/experiments/bit_drive"))
    parser.add_argument("--eval-dir", type=Path, action="append", default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()
    eval_dirs = args.eval_dir or discover_eval_dirs(args.input_root)
    if not eval_dirs:
        raise RuntimeError(f"No evaluation metrics found under {args.input_root}")
    rows = [load_metrics(path) for path in eval_dirs]
    output_csv = args.output_csv or args.input_root / "bit_left_tail_results.csv"
    output_md = args.output_md or args.input_root / "bit_left_tail_results.md"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_csv, rows)
    write_md(output_md, rows)
    print(json.dumps({"rows": len(rows), "output_csv": str(output_csv), "output_md": str(output_md)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
