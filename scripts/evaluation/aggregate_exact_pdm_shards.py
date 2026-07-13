#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _find_csvs(eval_root: Path) -> List[Path]:
    return sorted(path for path in eval_root.rglob("*.csv") if path.is_file())


def _read_shard_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    if "token" not in df.columns:
        raise KeyError(f"CSV is missing token column: {path}")
    return df[df["token"].astype(str) != "average"].copy()


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"1", "true", "yes"})


def aggregate(csv_paths: List[Path], output_csv: Path) -> Dict[str, Any]:
    if not csv_paths:
        raise FileNotFoundError("No shard CSVs were provided.")

    frames = [_read_shard_csv(path) for path in csv_paths]
    merged = pd.concat(frames, ignore_index=True)
    if merged.empty:
        raise ValueError("Shard CSVs contain no non-average rows.")

    duplicated = merged["token"].astype(str).duplicated(keep=False)
    if duplicated.any():
        sample = sorted(merged.loc[duplicated, "token"].astype(str).unique())[:10]
        raise ValueError(f"Duplicate tokens across shard CSVs: sample={sample}")

    valid = _bool_series(merged["valid"]) if "valid" in merged.columns else pd.Series(True, index=merged.index)
    numeric_for_average = merged.drop(columns=[col for col in ("token", "valid", "rank") if col in merged.columns])
    average_row = numeric_for_average.apply(pd.to_numeric, errors="coerce").mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = bool(valid.all())
    average_row["rank"] = "0"

    output = pd.concat([merged, pd.DataFrame([average_row])], ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv)

    summary = {
        "output_csv": str(output_csv),
        "num_shards": len(csv_paths),
        "num_rows": int(len(merged)),
        "num_valid": int(valid.sum()),
        "num_failed": int(len(merged) - valid.sum()),
        "mean_score": float(pd.to_numeric(merged.get("score"), errors="coerce").mean(skipna=True))
        if "score" in merged.columns
        else None,
        "shard_csvs": [str(path) for path in csv_paths],
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate exact PDM shard CSVs without recomputing PDMS. Per-token rows are copied "
            "as-is; only the aggregate average row is regenerated."
        )
    )
    parser.add_argument("--eval-root", type=Path, help="Directory to scan recursively for shard CSV files.")
    parser.add_argument("--shard-csv", type=Path, action="append", default=[], help="Explicit shard CSV path.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_paths: List[Path] = []
    if args.eval_root is not None:
        csv_paths.extend(_find_csvs(args.eval_root))
    csv_paths.extend(args.shard_csv)
    csv_paths = sorted({path.resolve() for path in csv_paths})

    summary = aggregate(csv_paths, args.output_csv)
    payload = json.dumps(summary, indent=2, sort_keys=True)
    print(payload)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
