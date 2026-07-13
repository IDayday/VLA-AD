#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


METRIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "score": ("score", "pdm_score", "pdms", "PDMS"),
    "no_at_fault_collisions": ("no_at_fault_collisions", "nc", "NC"),
    "drivable_area_compliance": ("drivable_area_compliance", "dac", "DAC"),
    "time_to_collision_within_bound": ("time_to_collision_within_bound", "ttc", "TTC"),
    "ego_progress": ("ego_progress", "ep", "EP", "progress"),
    "comfort": ("comfort", "history_comfort"),
    "driving_direction_compliance": ("driving_direction_compliance", "ddc", "DDC"),
    "traffic_light_compliance": ("traffic_light_compliance", "tlc", "TLC"),
    "lane_keeping": ("lane_keeping", "lane_keeping_compliance"),
}


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    unnamed = [column for column in frame.columns if str(column).startswith("Unnamed:")]
    if unnamed:
        frame = frame.drop(columns=unnamed)
    if "token" not in frame.columns:
        raise KeyError(f"{path} does not contain a token column.")
    frame = frame[frame["token"].astype(str) != "average"].copy()
    frame["token"] = frame["token"].astype(str)
    if frame["token"].duplicated().any():
        duplicated = sorted(frame.loc[frame["token"].duplicated(), "token"].unique().tolist())
        raise ValueError(f"{path} contains duplicated tokens: {duplicated[:5]}")
    return frame.set_index("token", drop=False)


def _resolve_column(frame: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    lower_to_column = {str(column).lower(): str(column) for column in frame.columns}
    for alias in aliases:
        if alias in frame.columns:
            return alias
        column = lower_to_column.get(str(alias).lower())
        if column is not None:
            return column
    return None


def compare(left: pd.DataFrame, right: pd.DataFrame, tolerance: float) -> Dict[str, object]:
    left_tokens = set(left.index)
    right_tokens = set(right.index)
    missing_left = sorted(right_tokens - left_tokens)
    missing_right = sorted(left_tokens - right_tokens)
    if missing_left or missing_right:
        raise ValueError(
            "Token sets differ: "
            f"missing_from_left={missing_left[:10]} ({len(missing_left)}), "
            f"missing_from_right={missing_right[:10]} ({len(missing_right)})."
        )

    tokens = sorted(left_tokens)
    summary: Dict[str, object] = {
        "num_tokens": len(tokens),
        "tolerance": float(tolerance),
        "metrics": {},
    }

    if "valid" in left.columns and "valid" in right.columns:
        left_valid = left.loc[tokens, "valid"].astype(str).to_numpy()
        right_valid = right.loc[tokens, "valid"].astype(str).to_numpy()
        mismatch = np.flatnonzero(left_valid != right_valid)
        summary["valid_mismatch_count"] = int(mismatch.size)
        if mismatch.size:
            first = tokens[int(mismatch[0])]
            raise AssertionError(f"valid column differs for {mismatch.size} tokens; first token={first}")

    failures: List[str] = []
    for metric, aliases in METRIC_ALIASES.items():
        left_col = _resolve_column(left, aliases)
        right_col = _resolve_column(right, aliases)
        if left_col is None and right_col is None:
            continue
        if left_col is None or right_col is None:
            failures.append(f"{metric}: column missing on one side ({left_col}, {right_col})")
            continue
        left_values = pd.to_numeric(left.loc[tokens, left_col], errors="coerce").to_numpy(dtype=np.float64)
        right_values = pd.to_numeric(right.loc[tokens, right_col], errors="coerce").to_numpy(dtype=np.float64)
        both_nan = np.isnan(left_values) & np.isnan(right_values)
        one_nan = np.isnan(left_values) ^ np.isnan(right_values)
        diff = np.abs(left_values - right_values)
        diff[both_nan] = 0.0
        diff[one_nan] = np.inf
        max_abs_diff = float(np.max(diff)) if diff.size else 0.0
        mismatch = np.flatnonzero(diff > tolerance)
        summary["metrics"][metric] = {
            "left_column": left_col,
            "right_column": right_col,
            "max_abs_diff": max_abs_diff,
            "mismatch_count": int(mismatch.size),
        }
        if mismatch.size:
            first = tokens[int(mismatch[0])]
            failures.append(
                f"{metric}: {mismatch.size} mismatches, first token={first}, "
                f"left={left.loc[first, left_col]}, right={right.loc[first, right_col]}, "
                f"diff={float(diff[int(mismatch[0])])}"
            )

    summary["passed"] = not failures
    if failures:
        summary["failures"] = failures
        raise AssertionError(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two PDMS CSVs token-by-token without changing scoring.")
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args()

    if args.tolerance < 0:
        raise ValueError("--tolerance must be non-negative.")
    summary = compare(_read_csv(args.left), _read_csv(args.right), args.tolerance)
    payload = json.dumps(summary, indent=2, sort_keys=True)
    print(payload)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
