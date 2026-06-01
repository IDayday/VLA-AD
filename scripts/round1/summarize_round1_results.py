#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize LaST-RD Round1 eval and corruption results.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-pdms", type=float, default=0.864891)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 6) -> str:
    number = as_float(value)
    if number is None:
        return "NA"
    return f"{number:.{digits}f}"


def eval_rows(root: Path, baseline: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metrics_path in sorted((root / "wave3_eval").glob("*/*/metrics.json")):
        metrics = load_json(metrics_path)
        run_name = metrics_path.parents[1].name
        checkpoint_name = metrics_path.parent.name
        pdms = as_float(metrics.get("PDMS", metrics.get("pdm_score")))
        rows.append({
            "run_name": run_name,
            "checkpoint_name": checkpoint_name,
            "checkpoint_path": metrics.get("checkpoint", ""),
            "PDMS": "" if pdms is None else pdms,
            "delta_vs_a0_official": "" if pdms is None else pdms - baseline,
            "trajectory_l1": metrics.get("trajectory_l1", ""),
            "NC": metrics.get("NC", ""),
            "DAC": metrics.get("DAC", ""),
            "TTC": metrics.get("TTC", ""),
            "comfort": metrics.get("comfort", ""),
            "EP": metrics.get("EP", ""),
            "DDC": metrics.get("DDC", ""),
            "eval_num_samples": metrics.get("num_samples", ""),
            "eval_num_pdm_valid": metrics.get("num_pdm_valid", ""),
            "is_full_eval": bool(int(metrics.get("num_samples") or 0) >= 12146),
            "is_best_for_run": False,
            "config_type": "hybrid" if run_name == "hybrid" else "lastrd_only",
            "output_dir": str(metrics_path.parent),
        })
    best_by_run: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        pdms = as_float(row["PDMS"])
        if pdms is None:
            continue
        current = best_by_run.get(row["run_name"])
        if current is None or pdms > float(current["PDMS"]):
            best_by_run[row["run_name"]] = row
    for row in rows:
        best = best_by_run.get(row["run_name"])
        row["is_best_for_run"] = bool(best is row)
    return rows


def corruption_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metrics_path in sorted((root / "wave4_corruption").glob("*/*/metrics.json")):
        metrics = load_json(metrics_path)
        rows.append({
            "run_name": metrics_path.parents[1].name,
            "mode": metrics_path.parent.name,
            "PDMS": as_float(metrics.get("PDMS", metrics.get("pdm_score"))),
            "trajectory_l1": metrics.get("trajectory_l1"),
            "num_samples": metrics.get("num_samples"),
            "num_pdm_valid": metrics.get("num_pdm_valid"),
            "output_dir": str(metrics_path.parent),
        })
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    columns = [
        "run_name",
        "checkpoint_name",
        "checkpoint_path",
        "PDMS",
        "delta_vs_a0_official",
        "trajectory_l1",
        "NC",
        "DAC",
        "TTC",
        "comfort",
        "EP",
        "DDC",
        "eval_num_samples",
        "eval_num_pdm_valid",
        "is_full_eval",
        "is_best_for_run",
        "config_type",
        "output_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def markdown(root: Path, rows: List[Dict[str, Any]], corruptions: List[Dict[str, Any]], baseline: float) -> str:
    best = {row["run_name"]: row for row in rows if str(row.get("is_best_for_run")).lower() == "true"}
    lines = [
        "# LaST-RD Round1 Summary",
        "",
        "## Baseline",
        "",
        f"- A0-official-aligned PDMS: `{baseline:.6f}`",
        "- Main success requires best full navtest PDMS above this value.",
        "",
        "## Best Per Run",
        "",
        "| run | checkpoint | PDMS | delta vs A0 | full eval |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for run in ("hybrid", "lastrd_only"):
        row = best.get(run)
        if row is None:
            lines.append(f"| {run} | NA | NA | NA | NA |")
        else:
            lines.append(
                f"| {run} | {row['checkpoint_name']} | {fmt(row['PDMS'])} | "
                f"{fmt(row['delta_vs_a0_official'])} | {row['is_full_eval']} |"
            )
    hybrid = as_float(best.get("hybrid", {}).get("PDMS"))
    lastrd = as_float(best.get("lastrd_only", {}).get("PDMS"))
    lines.extend(["", "## Comparison", ""])
    lines.append(f"- hybrid - baseline: {fmt(None if hybrid is None else hybrid - baseline)}")
    lines.append(f"- lastrd_only - baseline: {fmt(None if lastrd is None else lastrd - baseline)}")
    lines.append(f"- hybrid - lastrd_only: {fmt(None if hybrid is None or lastrd is None else hybrid - lastrd)}")
    lines.extend([
        "",
        "## Submetrics",
        "",
        "| run | NC | DAC | TTC | EP | comfort | DDC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for run in ("hybrid", "lastrd_only"):
        row = best.get(run, {})
        lines.append(
            f"| {run} | {fmt(row.get('NC'))} | {fmt(row.get('DAC'))} | {fmt(row.get('TTC'))} | "
            f"{fmt(row.get('EP'))} | {fmt(row.get('comfort'))} | {fmt(row.get('DDC'))} |"
        )
    lines.extend(["", "## Corruption Summary", ""])
    by_run: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in corruptions:
        by_run.setdefault(row["run_name"], {})[row["mode"]] = row
    for run in ("hybrid", "lastrd_only"):
        modes = by_run.get(run, {})
        normal = modes.get("none") or modes.get("normal")
        zero_all = modes.get("zero-all-last-rd") or modes.get("zero_all_last_rd")
        normal_pdms = as_float(normal.get("PDMS")) if normal else None
        zero_pdms = as_float(zero_all.get("PDMS")) if zero_all else None
        lines.append(f"### {run}")
        lines.append("")
        lines.append(f"- normal: {fmt(normal_pdms)}")
        lines.append(f"- zero_all_last_rd: {fmt(zero_pdms)}")
        lines.append(f"- delta normal-zero_all: {fmt(None if normal_pdms is None or zero_pdms is None else normal_pdms - zero_pdms)}")
        for mode in ("zero-jepa-dynamic", "shuffle-jepa-dynamic", "zero-vggt-geometry", "shuffle-vggt-geometry", "zero-ego-tokens"):
            pdms = as_float(modes.get(mode, {}).get("PDMS"))
            delta = None if normal_pdms is None or pdms is None else normal_pdms - pdms
            lines.append(f"- {mode}: PDMS={fmt(pdms)}, delta={fmt(delta)}")
        lines.append("")
    lines.extend([
        "## Interpretation Guide",
        "",
        "- If best > baseline and corruption drop > 0.003: strong evidence LastRD is useful.",
        "- If best > baseline but corruption drop is near 0: performance gain is not attributable to LastRD tokens.",
        "- If hybrid > lastrd_only: legacy context helps.",
        "- If lastrd_only >= hybrid: LastRD tokens dominate.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    rows = eval_rows(args.root, args.baseline_pdms)
    corruptions = corruption_rows(args.root)
    write_csv(args.root / "round1_summary.csv", rows)
    (args.root / "round1_summary.md").write_text(
        markdown(args.root, rows, corruptions, args.baseline_pdms),
        encoding="utf-8",
    )
    print(f"Wrote {args.root / 'round1_summary.csv'}")
    print(f"Wrote {args.root / 'round1_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
