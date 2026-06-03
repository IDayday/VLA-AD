#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


RISK_ORDER = ["low_score", "path_dac", "interaction_nc", "ttc", "progress", "comfort"]
MVP_ORDER = [
    ("generic_risk_labels", "low_score"),
    ("drivable_risk_labels", "path_dac"),
    ("ttc_risk_labels", "interaction_ttc"),
    ("comfort_risk_labels", "comfort_progress"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check RISK-VLA label distribution statistics.")
    parser.add_argument("--labels-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def schema_type(row: Dict[str, Any]) -> str:
    has_mvp = any(key in row for key, _ in MVP_ORDER)
    has_extended = "risk_labels" in row
    if has_mvp and has_extended:
        return "both"
    if has_extended:
        return "extended"
    if has_mvp:
        return "mvp"
    return "unknown"


def values_from_row(row: Dict[str, Any]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    if "risk_labels" in row:
        arr = np.asarray(row["risk_labels"], dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= len(RISK_ORDER):
            for index, name in enumerate(RISK_ORDER):
                out[name] = arr[:, index].astype(float).tolist()
    for key, name in MVP_ORDER:
        if key in row:
            out[name] = np.asarray(row[key], dtype=float).reshape(-1).tolist()
    return out


def distribution(rows: List[Dict[str, Any]], threshold: float) -> tuple[pd.DataFrame, Dict[str, Any]]:
    schema_counts: Dict[str, int] = {}
    horizon_shapes: Dict[str, int] = {}
    values: Dict[str, List[float]] = {}
    missing_values: Dict[str, int] = {}
    for row in rows:
        kind = schema_type(row)
        schema_counts[kind] = schema_counts.get(kind, 0) + 1
        row_values = values_from_row(row)
        for name, vals in row_values.items():
            horizon_shapes[str((name, len(vals)))] = horizon_shapes.get(str((name, len(vals))), 0) + 1
            values.setdefault(name, []).extend(vals)
            missing_values[name] = missing_values.get(name, 0) + int(np.isnan(np.asarray(vals, dtype=float)).sum())
    records = []
    for name, vals in sorted(values.items()):
        arr = np.asarray(vals, dtype=float)
        clean = arr[~np.isnan(arr)]
        records.append(
            {
                "risk_class": name,
                "mean": float(clean.mean()) if clean.size else np.nan,
                "positive_rate_at_threshold": float((clean >= threshold).mean()) if clean.size else np.nan,
                "min": float(clean.min()) if clean.size else np.nan,
                "max": float(clean.max()) if clean.size else np.nan,
                "missing_values": int(missing_values.get(name, 0)),
                "value_count": int(arr.size),
            }
        )
    summary = {
        "num_tokens": len(rows),
        "schema_counts": schema_counts,
        "horizon_shape_counts": horizon_shapes,
        "threshold": threshold,
    }
    return pd.DataFrame(records), summary


def write_markdown(path: Path, stats: pd.DataFrame, summary: Dict[str, Any]) -> None:
    lines = [
        "# RISK-VLA Label Distribution",
        "",
        f"Tokens: `{summary['num_tokens']}`",
        f"Schema counts: `{summary['schema_counts']}`",
        f"Horizon shape counts: `{summary['horizon_shape_counts']}`",
        "",
        "| Risk class | Mean | Positive rate @ threshold | Min | Max | Missing |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in stats.to_dict(orient="records"):
        lines.append(
            f"| {row['risk_class']} | {row['mean']:.6f} | {row['positive_rate_at_threshold']:.6f} | "
            f"{row['min']:.6f} | {row['max']:.6f} | {row['missing_values']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = list(read_jsonl(args.labels_jsonl))
    stats, summary = distribution(rows, args.threshold)
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(args.output_csv, index=False)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_md, stats, summary)
    print(json.dumps({"summary": summary, "stats": stats.to_dict(orient="records")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
