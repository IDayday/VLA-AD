#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


TOKEN_CANDIDATES = ("token", "sample_token", "scene_token")
EXPECTED_METRICS = (
    "score",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple PDM CSV files by token.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Inputs in name=path format.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def parse_input_specs(values: List[str]) -> List[Tuple[str, Path]]:
    specs: List[Tuple[str, Path]] = []
    seen = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Input must use name=path format, got {value!r}")
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Input name is empty in {value!r}")
        if name in seen:
            raise ValueError(f"Duplicate input name: {name}")
        seen.add(name)
        specs.append((name, Path(path)))
    return specs


def detect_token_column(df: pd.DataFrame, path: Path) -> str:
    for column in TOKEN_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(f"No token column found in {path}; expected one of {TOKEN_CANDIDATES}")


def load_prefixed(name: str, path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = pd.read_csv(path)
    token_col = detect_token_column(df, path)
    duplicate_mask = df[token_col].duplicated(keep=False)
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        warnings.warn(f"{name}: found {duplicate_count} duplicate token rows; keeping first occurrence.")
        df = df.drop_duplicates(subset=[token_col], keep="first")
    missing_metrics = [column for column in EXPECTED_METRICS if column not in df.columns]
    if missing_metrics:
        warnings.warn(f"{name}: missing optional metric columns: {missing_metrics}")
    rename = {token_col: "token"}
    for column in df.columns:
        if column != token_col:
            rename[column] = f"{name}_{column}"
    out = df.rename(columns=rename)
    report = {
        "name": name,
        "path": str(path),
        "token_column": token_col,
        "token_count": int(out["token"].nunique()),
        "row_count_after_dedup": int(len(out)),
        "duplicate_token_rows": duplicate_count,
        "missing_metric_columns": missing_metrics,
    }
    return out, report


def merge_inputs(specs: List[Tuple[str, Path]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    frames: List[pd.DataFrame] = []
    reports: List[Dict[str, Any]] = []
    token_sets = []
    for name, path in specs:
        frame, report = load_prefixed(name, path)
        frames.append(frame)
        reports.append(report)
        token_sets.append(set(frame["token"].astype(str)))
    matched_tokens = set.intersection(*token_sets) if token_sets else set()
    merged = pd.DataFrame({"token": sorted(matched_tokens)})
    for frame in frames:
        frame = frame.copy()
        frame["token"] = frame["token"].astype(str)
        merged = merged.merge(frame, on="token", how="left", validate="one_to_one")
    for report, tokens in zip(reports, token_sets):
        report["matched_token_count"] = len(matched_tokens)
        report["missing_token_count"] = int(len(tokens - matched_tokens))
    summary = {
        "inputs": reports,
        "matched_token_count": int(len(matched_tokens)),
        "input_count": len(specs),
    }
    return merged, summary


def write_markdown(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Matched PDM Result Report",
        "",
        f"Matched token count: `{summary['matched_token_count']}`",
        "",
        "| Input | Tokens | Matched | Missing | Duplicate rows | Missing metric columns |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in summary["inputs"]:
        lines.append(
            f"| {item['name']} | {item['token_count']} | {item['matched_token_count']} | "
            f"{item['missing_token_count']} | {item['duplicate_token_rows']} | "
            f"`{item['missing_metric_columns']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    specs = parse_input_specs(args.inputs)
    merged, summary = merge_inputs(specs)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_md, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
