#!/usr/bin/env python3
"""Summarize ReCogDrive policy diversity diagnostics CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = [
    "quality_min_ade",
    "quality_min_fde",
    "diversity_mean_pade",
    "diversity_mean_pfde",
    "perf_mean_pdms",
    "perf_best_pdms",
    "perf_min_pdms",
    "perf_std_pdms",
]


def _load_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "token" not in df.columns:
        raise ValueError(f"Missing token column in {path}")
    df = df[df["token"].astype(str) != "average"].copy()
    if "valid" in df.columns:
        valid = df["valid"].astype(str).str.lower().isin({"true", "1", "yes"})
        df = df[valid].copy()
    for metric in METRICS:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df


def _metric_columns(df: pd.DataFrame) -> list[str]:
    return [metric for metric in METRICS if metric in df.columns]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    df = _load_rows(args.input_csv)
    metrics = _metric_columns(df)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = []
    percentile_rows = []
    for metric in metrics:
        series = df[metric].dropna()
        overall_rows.append({"metric": metric, "mean": float(series.mean()) if len(series) else None})
        qs = series.quantile([0.05, 0.25, 0.50, 0.75, 0.95]) if len(series) else pd.Series(dtype=float)
        percentile_rows.append(
            {
                "metric": metric,
                "p5": float(qs.loc[0.05]) if len(series) else None,
                "p25": float(qs.loc[0.25]) if len(series) else None,
                "p50": float(qs.loc[0.50]) if len(series) else None,
                "p75": float(qs.loc[0.75]) if len(series) else None,
                "p95": float(qs.loc[0.95]) if len(series) else None,
            }
        )

    thresholds = [
        ("mean_pade_ge_paper_recogdrive_0p148", "diversity_mean_pade", ">=", 0.148),
        ("mean_pfde_ge_paper_recogdrive_0p325", "diversity_mean_pfde", ">=", 0.325),
        ("mean_pade_ge_paper_curious_0p641", "diversity_mean_pade", ">=", 0.641),
        ("mean_pfde_ge_paper_curious_1p415", "diversity_mean_pfde", ">=", 1.415),
        ("min_fde_le_paper_recogdrive_0p621", "quality_min_fde", "<=", 0.621),
        ("min_fde_le_paper_curious_0p547", "quality_min_fde", "<=", 0.547),
        ("mean_pdms_ge_0p90", "perf_mean_pdms", ">=", 0.90),
        ("mean_pdms_ge_0p95", "perf_mean_pdms", ">=", 0.95),
        ("mean_pdms_le_0p85", "perf_mean_pdms", "<=", 0.85),
    ]
    threshold_rows = []
    denominator = len(df)
    for name, metric, op, value in thresholds:
        if metric not in df.columns:
            continue
        series = df[metric].dropna()
        if op == ">=":
            count = int((series >= value).sum())
        else:
            count = int((series <= value).sum())
        threshold_rows.append(
            {
                "threshold": name,
                "n": count,
                "ratio": float(count / denominator) if denominator else None,
            }
        )

    _write_csv(args.output_dir / "overall_mean_summary.csv", overall_rows)
    _write_csv(args.output_dir / "percentile_summary.csv", percentile_rows)
    _write_csv(args.output_dir / "threshold_summary.csv", threshold_rows)

    summary = {
        "label": args.label,
        "input_csv": str(args.input_csv),
        "num_valid_rows": int(len(df)),
        "means": {row["metric"]: row["mean"] for row in overall_rows},
        "thresholds": {row["threshold"]: row["ratio"] for row in threshold_rows},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
