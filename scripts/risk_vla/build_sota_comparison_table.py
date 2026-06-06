from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build same-protocol SOTA comparison table for RISK-VLA v2.")
    parser.add_argument("--run", action="append", default=[], help="name=...,aggregate=...")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in args.run:
        parts = dict(part.split("=", 1) for part in spec.split(",") if "=" in part)
        name = parts.get("name")
        aggregate = Path(parts.get("aggregate", ""))
        metrics = json.loads(aggregate.read_text(encoding="utf-8")) if aggregate.is_file() else {}
        rows.append(
            {
                "name": name,
                "aggregate": str(aggregate),
                "same_protocol": bool(metrics),
                "mean_pdms": metrics.get("mean_pdms"),
                "p10_pdms": metrics.get("p10_pdms"),
                "p5_pdms": metrics.get("p5_pdms"),
                "zero_score_count": metrics.get("zero_score_count"),
                "dac0": metrics.get("drivable_area_compliance_zero_count"),
                "nc0": metrics.get("no_at_fault_collision_zero_count"),
                "ttc0": metrics.get("time_to_collision_zero_count"),
            }
        )
    csv_path = args.output_dir / "sota_comparison_table.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir / "sota_comparison_table.json").write_text(json.dumps({"runs": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "sota_comparison_table.md").write_text(
        "# SOTA Comparison Table\n\nDo not claim SOTA unless benchmark, split, and protocol match exactly.\n",
        encoding="utf-8",
    )
    print(json.dumps({"runs": len(rows), "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
