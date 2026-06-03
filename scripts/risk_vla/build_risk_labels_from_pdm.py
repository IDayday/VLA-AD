#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


TOKEN_CANDIDATES = ("token", "sample_token", "scene_token")
RISK_ORDER = ["low_score", "path_dac", "interaction_nc", "ttc", "progress", "comfort"]
MVP_KEYS = ("generic_risk_labels", "drivable_risk_labels", "ttc_risk_labels", "comfort_risk_labels")
METRIC_ALIASES = {
    "score": ("score", "pdm", "pdm_score"),
    "no_at_fault_collisions": ("no_at_fault_collisions", "no_at_fault_collision", "nc"),
    "drivable_area_compliance": ("drivable_area_compliance", "dac"),
    "ego_progress": ("ego_progress", "ego"),
    "time_to_collision_within_bound": ("time_to_collision_within_bound", "time_to_collision", "ttc"),
    "comfort": ("comfort",),
    "driving_direction_compliance": ("driving_direction_compliance", "ddc"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hard and soft RISK-VLA labels from PDM CSV metrics.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--schema", choices=("mvp", "extended", "both"), default="both")
    parser.add_argument("--progress-percentile", type=float, default=10.0)
    parser.add_argument("--comfort-percentile", type=float, default=10.0)
    return parser.parse_args()


def detect_token_column(df: pd.DataFrame) -> str:
    for column in TOKEN_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(f"No token column found; expected one of {TOKEN_CANDIDATES}")


def resolve_metric_column(df: pd.DataFrame, metric: str) -> Optional[str]:
    for column in METRIC_ALIASES[metric]:
        if column in df.columns:
            return column
    return None


def value(row: pd.Series, columns: Dict[str, Optional[str]], metric: str) -> Optional[float]:
    column = columns.get(metric)
    if column is None:
        return None
    raw = row.get(column)
    if pd.isna(raw):
        return None
    return float(raw)


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def percentile_threshold(series: pd.Series, percentile: float) -> Optional[float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(np.percentile(clean.to_numpy(dtype=float), percentile))


def low_value_score(metric_value: Optional[float], threshold: Optional[float]) -> float:
    if metric_value is None or threshold is None:
        return 0.0
    scale = max(abs(threshold), 1e-6)
    return clip01((threshold - metric_value) / scale)


def repeated(value_: float, horizon: int) -> List[float]:
    return [float(value_)] * int(horizon)


def extended_tensor(values: Dict[str, float], horizon: int) -> List[List[float]]:
    row = [float(values[name]) for name in RISK_ORDER]
    return [list(row) for _ in range(int(horizon))]


def build_rows(
    df: pd.DataFrame,
    *,
    horizon: int,
    schema: str,
    progress_percentile: float,
    comfort_percentile: float,
) -> tuple[List[Dict[str, Any]], pd.DataFrame, Dict[str, Any]]:
    token_col = detect_token_column(df)
    columns = {metric: resolve_metric_column(df, metric) for metric in METRIC_ALIASES}
    progress_threshold = percentile_threshold(df[columns["ego_progress"]], progress_percentile) if columns["ego_progress"] else None
    comfort_threshold = percentile_threshold(df[columns["comfort"]], comfort_percentile) if columns["comfort"] else None
    missing_metric_columns = [metric for metric, column in columns.items() if column is None]
    json_rows: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        token = str(row[token_col])
        score = value(row, columns, "score")
        nc = value(row, columns, "no_at_fault_collisions")
        dac = value(row, columns, "drivable_area_compliance")
        progress = value(row, columns, "ego_progress")
        ttc = value(row, columns, "time_to_collision_within_bound")
        comfort = value(row, columns, "comfort")

        hard = {
            "low_score": bool(score is not None and score == 0.0),
            "path_dac": bool(dac is not None and dac == 0.0),
            "interaction_nc": bool(nc is not None and nc == 0.0),
            "ttc": bool(ttc is not None and ttc == 0.0),
            "progress": bool(progress is not None and progress_threshold is not None and progress <= progress_threshold),
            "comfort": bool(comfort is not None and comfort_threshold is not None and comfort <= comfort_threshold),
        }
        progress_score = low_value_score(progress, progress_threshold)
        comfort_percentile_score = low_value_score(comfort, comfort_threshold)
        soft = {
            "low_score": clip01(1.0 - score) if score is not None else 0.0,
            "path_dac": clip01(1.0 - dac) if dac is not None else 0.0,
            "interaction_nc": clip01(1.0 - nc) if nc is not None else 0.0,
            "ttc": clip01(1.0 - ttc) if ttc is not None else 0.0,
            "progress": progress_score,
            "comfort": max(clip01(1.0 - comfort) if comfort is not None else 0.0, comfort_percentile_score),
        }
        scores = {name: max(float(hard[name]), soft[name]) for name in RISK_ORDER}
        mvp = {
            "generic_risk_labels": repeated(scores["low_score"], horizon),
            "drivable_risk_labels": repeated(scores["path_dac"], horizon),
            "ttc_risk_labels": repeated(max(scores["interaction_nc"], scores["ttc"]), horizon),
            "comfort_risk_labels": repeated(max(scores["progress"], scores["comfort"]), horizon),
        }
        output: Dict[str, Any] = {
            "token": token,
            "sample_token": token,
            "risk_order": RISK_ORDER,
            "risk_horizon": int(horizon),
            "hard_risk_flags": hard,
            "soft_risk_scores": soft,
            "source_metrics": {
                "score": score,
                "no_at_fault_collisions": nc,
                "drivable_area_compliance": dac,
                "ego_progress": progress,
                "time_to_collision_within_bound": ttc,
                "comfort": comfort,
            },
            "missing_metrics": [metric for metric, metric_value in {
                "score": score,
                "no_at_fault_collisions": nc,
                "drivable_area_compliance": dac,
                "ego_progress": progress,
                "time_to_collision_within_bound": ttc,
                "comfort": comfort,
            }.items() if metric_value is None],
        }
        if schema in {"mvp", "both"}:
            output.update(mvp)
        if schema in {"extended", "both"}:
            output["risk_labels"] = extended_tensor(scores, horizon)
        json_rows.append(output)
        csv_row: Dict[str, Any] = {"token": token}
        for name in RISK_ORDER:
            csv_row[f"hard_{name}"] = int(hard[name])
            csv_row[f"soft_{name}"] = soft[name]
            csv_row[f"label_{name}"] = scores[name]
        csv_rows.append(csv_row)

    summary = {
        "tokens": len(json_rows),
        "schema": schema,
        "horizon": int(horizon),
        "risk_order": RISK_ORDER,
        "progress_threshold": progress_threshold,
        "comfort_threshold": comfort_threshold,
        "missing_metric_columns": missing_metric_columns,
        "mvp_keys": list(MVP_KEYS) if schema in {"mvp", "both"} else [],
        "extended": schema in {"extended", "both"},
    }
    return json_rows, pd.DataFrame(csv_rows), summary


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def write_markdown(path: Path, summary: Dict[str, Any], csv_df: pd.DataFrame) -> None:
    lines = [
        "# RISK-VLA Label Build Report",
        "",
        f"Tokens: `{summary['tokens']}`",
        f"Schema: `{summary['schema']}`",
        f"Horizon: `{summary['horizon']}`",
        f"Risk order: `{summary['risk_order']}`",
        f"Progress threshold: `{summary['progress_threshold']}`",
        f"Comfort threshold: `{summary['comfort_threshold']}`",
        f"Missing metric columns: `{summary['missing_metric_columns']}`",
        "",
        "| Risk | Hard positives | Mean label |",
        "| --- | ---: | ---: |",
    ]
    for name in RISK_ORDER:
        lines.append(
            f"| {name} | {int(csv_df[f'hard_{name}'].sum())} | {float(csv_df[f'label_{name}'].mean()):.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    df = pd.read_csv(args.input_csv)
    rows, csv_df, summary = build_rows(
        df,
        horizon=args.horizon,
        schema=args.schema,
        progress_percentile=args.progress_percentile,
        comfort_percentile=args.comfort_percentile,
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_jsonl, rows)
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_df.to_csv(args.output_csv, index=False)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_md, summary, csv_df)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
