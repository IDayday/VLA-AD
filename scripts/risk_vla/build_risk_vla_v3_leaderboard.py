#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def extract_row(path: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    metrics = summary.get("best_val") or summary.get("final_val") or summary.get("metrics") or {}
    selected = metrics.get("selected_metrics", {}) if isinstance(metrics, dict) else {}
    return {
        "path": str(path),
        "method": path.parent.name,
        "num_samples": summary.get("num_samples") or summary.get("num_train") or metrics.get("num_samples"),
        "pdms": selected.get("pdms"),
        "zero_score": selected.get("zero_score"),
        "dac0": selected.get("dac0"),
        "nc0": selected.get("nc0"),
        "ttc0": selected.get("ttc0"),
        "progress": selected.get("progress"),
        "comfort": selected.get("comfort"),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build a lightweight leaderboard from RISK-VLA v3 summary.json files.")
    parser.add_argument("--exp-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    rows = []
    if args.exp_root.exists():
        for path in sorted(args.exp_root.rglob("summary.json")):
            try:
                rows.append(extract_row(path, json.loads(path.read_text(encoding="utf-8"))))
            except Exception:
                continue
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "risk_vla_v3_leaderboard.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "num_samples", "pdms", "zero_score", "dac0", "nc0", "ttc0", "progress", "comfort", "path"])
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# RISK-VLA v3 Leaderboard", "", f"Rows: `{len(rows)}`", "", "| Method | Samples | PDMS | Zero | DAC0 | NC0 | TTC0 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['num_samples']} | {row['pdms']} | {row['zero_score']} | {row['dac0']} | {row['nc0']} | {row['ttc0']} |")
    (args.output_dir / "risk_vla_v3_leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"leaderboard": str(csv_path), "rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
