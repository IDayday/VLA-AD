from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from scripts.risk_vla.aggregate_strategy_utility_report import aggregate as aggregate_report
from scripts.risk_vla.aggregate_strategy_utility_report import read_rows, write_report


def aggregate(labels_jsonl: Path) -> Dict[str, Any]:
    summary = aggregate_report(labels_jsonl)
    rows = list(read_rows(labels_jsonl))
    summary["selected_candidate_distribution"] = dict(
        Counter(str(row.get("constrained_best_candidate", row.get("constrained_best_strategy"))) for row in rows)
    )
    summary["positive_anchor_distribution"] = dict(Counter(str(row.get("positive_anchor")) for row in rows))
    summary["negative_anchor_distribution"] = dict(Counter(str(row.get("negative_anchor")) for row in rows))
    summary["transition_counts"] = {
        "path_repair": sum(bool(row.get("repairs_path")) for row in rows),
        "path_regression": sum(bool(row.get("regresses_path")) for row in rows),
        "nc_repair": sum(bool(row.get("repairs_nc")) for row in rows),
        "nc_regression": sum(bool(row.get("regresses_nc")) for row in rows),
        "ttc_repair": sum(bool(row.get("repairs_ttc")) for row in rows),
        "ttc_regression": sum(bool(row.get("regresses_ttc")) for row in rows),
        "unsafe_any": sum(bool(row.get("unsafe_any")) for row in rows),
    }
    return summary


def write_outputs(summary: Dict[str, Any], output_dir: Path) -> None:
    write_report(summary, output_dir)
    lines = [
        "# Strategy Utility Label Aggregate",
        "",
        f"Rows: `{summary['num_rows']}`",
        "",
        "## Transition Counts",
        "",
    ]
    for key, value in sorted(summary["transition_counts"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Candidate Selection Distribution", ""])
    for key, value in sorted(summary["selected_candidate_distribution"].items()):
        lines.append(f"- candidate `{key}`: `{value}`")
    (output_dir / "strategy_utility_label_aggregate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate RISK-VLA v2 strategy utility labels.")
    parser.add_argument("--labels-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = aggregate(args.labels_jsonl)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(summary, args.output_dir)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
