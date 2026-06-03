#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


TOKEN_CANDIDATES = ("token", "sample_token", "scene_token")
METRIC_ALIASES = {
    "score": ("score", "pdm", "pdm_score"),
    "nc": ("no_at_fault_collisions", "no_at_fault_collision", "nc"),
    "dac": ("drivable_area_compliance", "dac"),
    "progress": ("ego_progress", "ego"),
    "ttc": ("time_to_collision_within_bound", "time_to_collision", "ttc"),
    "comfort": ("comfort",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Base-vs-method RISK-VLA transition metrics.")
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--method-csv", type=Path, required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def detect_token_column(df: pd.DataFrame, path: Path) -> str:
    for column in TOKEN_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(f"No token column found in {path}; expected one of {TOKEN_CANDIDATES}")


def resolve_metric(df: pd.DataFrame, metric: str) -> Optional[str]:
    for column in METRIC_ALIASES[metric]:
        if column in df.columns:
            return column
    return None


def prepare(df: pd.DataFrame, path: Path, prefix: str) -> pd.DataFrame:
    token_col = detect_token_column(df, path)
    if df[token_col].duplicated().any():
        df = df.drop_duplicates(subset=[token_col], keep="first")
    columns = {metric: resolve_metric(df, metric) for metric in METRIC_ALIASES}
    out = pd.DataFrame({"token": df[token_col].astype(str)})
    for metric, column in columns.items():
        out[f"{prefix}_{metric}"] = pd.to_numeric(df[column], errors="coerce") if column else np.nan
    return out


def safe_gt0(series: pd.Series) -> pd.Series:
    return series.fillna(0.0) > 0.0


def zero(series: pd.Series) -> pd.Series:
    return series.fillna(np.nan) == 0.0


def rate(num: int, denom: int) -> float:
    return float(num) / float(denom) if denom > 0 else 0.0


def transition_metrics(base_df: pd.DataFrame, method_df: pd.DataFrame, method_name: str) -> Dict[str, Any]:
    merged = prepare(base_df, Path("base"), "base").merge(
        prepare(method_df, Path(method_name), "method"),
        on="token",
        how="inner",
        validate="one_to_one",
    )
    base_dac0 = zero(merged["base_dac"])
    method_dac0 = zero(merged["method_dac"])
    base_nc0 = zero(merged["base_nc"])
    method_nc0 = zero(merged["method_nc"])
    base_ttc0 = zero(merged["base_ttc"])
    method_ttc0 = zero(merged["method_ttc"])
    base_nc_ttc_safe = safe_gt0(merged["base_nc"]) & safe_gt0(merged["base_ttc"])
    method_nc_ttc_safe = safe_gt0(merged["method_nc"]) & safe_gt0(merged["method_ttc"])
    path_repair = base_dac0 & safe_gt0(merged["method_dac"])
    path_regression = safe_gt0(merged["base_dac"]) & method_dac0
    interaction_regression = base_nc_ttc_safe & (method_nc0 | method_ttc0)
    safe_path_repair = path_repair & method_nc_ttc_safe

    progress_threshold = float(np.nanpercentile(merged["base_progress"].to_numpy(dtype=float), 10)) if merged["base_progress"].notna().any() else np.nan
    comfort_threshold = float(np.nanpercentile(merged["base_comfort"].to_numpy(dtype=float), 10)) if merged["base_comfort"].notna().any() else np.nan
    base_low_progress = merged["base_progress"] <= progress_threshold if not np.isnan(progress_threshold) else pd.Series(False, index=merged.index)
    base_low_comfort = merged["base_comfort"] <= comfort_threshold if not np.isnan(comfort_threshold) else pd.Series(False, index=merged.index)
    progress_recovery = base_low_progress & (merged["method_progress"] > merged["base_progress"]) & method_nc_ttc_safe
    comfort_recovery = base_low_comfort & (merged["method_comfort"] > merged["base_comfort"]) & method_nc_ttc_safe

    counts = {
        "num_matched_tokens": int(len(merged)),
        "num_base_dac0": int(base_dac0.sum()),
        "num_method_dac0": int(method_dac0.sum()),
        "num_base_nc0": int(base_nc0.sum()),
        "num_method_nc0": int(method_nc0.sum()),
        "num_base_ttc0": int(base_ttc0.sum()),
        "num_method_ttc0": int(method_ttc0.sum()),
        "num_path_repair": int(path_repair.sum()),
        "num_path_regression": int(path_regression.sum()),
        "num_interaction_regression": int(interaction_regression.sum()),
        "num_safe_path_repair": int(safe_path_repair.sum()),
        "num_progress_recovery": int(progress_recovery.sum()),
        "num_comfort_recovery": int(comfort_recovery.sum()),
        "num_base_low_progress": int(base_low_progress.sum()),
        "num_base_low_comfort": int(base_low_comfort.sum()),
    }
    counts.update(
        {
            "method_name": method_name,
            "path_repair_rate": rate(counts["num_path_repair"], counts["num_base_dac0"]),
            "path_regression_rate": rate(counts["num_path_regression"], int(safe_gt0(merged["base_dac"]).sum())),
            "interaction_regression_rate": rate(counts["num_interaction_regression"], int(base_nc_ttc_safe.sum())),
            "safe_path_repair_rate": rate(counts["num_safe_path_repair"], counts["num_base_dac0"]),
            "progress_recovery_rate": rate(counts["num_progress_recovery"], counts["num_base_low_progress"]),
            "comfort_recovery_rate": rate(counts["num_comfort_recovery"], counts["num_base_low_comfort"]),
            "progress_low_threshold": progress_threshold,
            "comfort_low_threshold": comfort_threshold,
        }
    )
    return counts


def write_markdown(path: Path, row: Dict[str, Any]) -> None:
    lines = [
        "# RISK-VLA Transition Matrix",
        "",
        f"Method: `{row['method_name']}`",
        f"Matched tokens: `{row['num_matched_tokens']}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in (
        "path_repair_rate",
        "path_regression_rate",
        "interaction_regression_rate",
        "safe_path_repair_rate",
        "progress_recovery_rate",
        "comfort_recovery_rate",
        "num_path_repair",
        "num_path_regression",
        "num_interaction_regression",
        "num_safe_path_repair",
    ):
        lines.append(f"| {key} | {row[key]} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    row = transition_metrics(pd.read_csv(args.base_csv), pd.read_csv(args.method_csv), args.method_name)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(args.output_csv, index=False)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_md, row)
    print(json.dumps(row, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
