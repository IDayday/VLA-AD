#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


A0_BASELINE_PDMS = 0.864891


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Last-VLA v2 decoupled high-cap no-risk eval results.")
    parser.add_argument("--out-root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def iter_metrics(root: Path) -> Iterable[tuple[str, Path, Dict[str, Any]]]:
    for path in sorted(root.glob("decoupled_highcap_no_risk_eval/*/*/metrics.json")):
        payload = read_json(path)
        if payload:
            yield path.parts[-3], path, payload


def metric_pdms(payload: Dict[str, Any]) -> Optional[float]:
    for key in ("PDMS", "score", "mean_score"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    pdm = payload.get("pdm_averages")
    if isinstance(pdm, dict) and isinstance(pdm.get("score"), (int, float)):
        return float(pdm["score"])
    return None


def corruption_pdms(root: Path, mode: str) -> Optional[float]:
    payload = read_json(root / "decoupled_highcap_no_risk_corruption" / mode / "metrics.json")
    return metric_pdms(payload) if payload else None


def main() -> int:
    args = parse_args()
    rows: List[Dict[str, Any]] = []
    best_by_line: Dict[str, Dict[str, Any]] = {}
    for line, path, payload in iter_metrics(args.out_root):
        pdms = metric_pdms(payload)
        row = {
            "line": line,
            "metrics_path": str(path),
            "PDMS": pdms,
            "delta_vs_A0": (pdms - A0_BASELINE_PDMS) if pdms is not None else None,
            "NC": payload.get("NC"),
            "DAC": payload.get("DAC"),
            "TTC": payload.get("TTC"),
            "comfort": payload.get("comfort"),
            "EP": payload.get("EP"),
            "DDC": payload.get("DDC"),
            "trajectory_l1": payload.get("trajectory_l1"),
        }
        rows.append(row)
        if pdms is not None and (line not in best_by_line or pdms > best_by_line[line]["PDMS"]):
            best_by_line[line] = row

    args.out_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_root / "decoupled_highcap_no_risk_summary.csv"
    fieldnames = ["line", "metrics_path", "PDMS", "delta_vs_A0", "NC", "DAC", "TTC", "comfort", "EP", "DDC", "trajectory_l1"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    normal = corruption_pdms(args.out_root, "normal")
    raw_only = corruption_pdms(args.out_root, "raw_vlm_only")
    zero_all = corruption_pdms(args.out_root, "zero_all_cot")
    md_lines = [
        "# Last-VLA v2 Decoupled HighCap NoRisk Summary",
        "",
        f"- A0-official-aligned baseline PDMS: `{A0_BASELINE_PDMS}`",
        f"- Corruption normal PDMS: `{normal}`",
        f"- Corruption raw_vlm_only PDMS: `{raw_only}`",
        f"- Corruption zero_all_cot PDMS: `{zero_all}`",
        "- No performance is claimed unless full PDM eval artifacts are present.",
        "",
        "## Best By Line",
    ]
    for line, row in sorted(best_by_line.items()):
        pdms = row["PDMS"]
        md_lines.append(f"- `{line}` best PDMS `{pdms}`, delta vs A0 `{pdms - A0_BASELINE_PDMS}`")
    (args.out_root / "decoupled_highcap_no_risk_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "summary_csv": str(csv_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
