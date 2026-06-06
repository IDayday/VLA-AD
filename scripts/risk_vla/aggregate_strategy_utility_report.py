from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, Sequence


def read_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def aggregate(path: Path) -> Dict[str, Any]:
    by_strategy: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    rows = list(read_rows(path))
    for row in rows:
        by_strategy[str(row.get("strategy_name"))].append(row)
    summary_rows = []
    for strategy, group in sorted(by_strategy.items()):
        summary_rows.append(
            {
                "strategy_name": strategy,
                "rows": len(group),
                "mean_pdms": mean(float(row.get("pdm_score") or 0.0) for row in group),
                "mean_delta_score": mean(float(row.get("delta_score") or 0.0) for row in group),
                "repairs_path": sum(bool(row.get("repairs_path")) for row in group),
                "regresses_path": sum(bool(row.get("regresses_path")) for row in group),
                "repairs_nc": sum(bool(row.get("repairs_nc")) for row in group),
                "regresses_nc": sum(bool(row.get("regresses_nc")) for row in group),
                "repairs_ttc": sum(bool(row.get("repairs_ttc")) for row in group),
                "regresses_ttc": sum(bool(row.get("regresses_ttc")) for row in group),
                "tail_risk_rows": sum(bool(row.get("tail_risk_label")) for row in group),
                "positive_anchor_rows": sum(str(row.get("candidate_id")) == str(row.get("positive_anchor")) for row in group),
                "negative_anchor_rows": sum(str(row.get("candidate_id")) == str(row.get("negative_anchor")) for row in group),
            }
        )
    return {"labels_jsonl": str(path), "num_rows": len(rows), "strategies": summary_rows}


def write_report(summary: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "strategy_utility_summary.csv"
    rows = summary["strategies"]
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Strategy Utility Aggregate Report",
        "",
        f"Rows: `{summary['num_rows']}`",
        "",
        "| Strategy | Rows | Mean PDMS | Mean Delta | Path Repair | NC Reg | TTC Reg | Positive Anchor | Negative Anchor |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['strategy_name']} | {row['rows']} | {row['mean_pdms']} | {row['mean_delta_score']} | "
            f"{row['repairs_path']} | {row['regresses_nc']} | {row['regresses_ttc']} | "
            f"{row['positive_anchor_rows']} | {row['negative_anchor_rows']} |"
        )
    (output_dir / "strategy_utility_aggregate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate RISK-VLA v2 strategy utility labels.")
    parser.add_argument("--labels-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = aggregate(args.labels_jsonl)
    write_report(summary, args.output_dir)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
