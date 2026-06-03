#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


TOKEN_COLUMNS = ("token", "sample_token", "scene_token")
PDM_COLUMNS = [
    "score",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
]
SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    "cache",
    "hf_home",
    "modelscope_cache",
    "wandb",
    "node_modules",
}


def env_roots() -> List[Path]:
    roots: List[Path] = []
    for name in ("BIT_EXP_ROOT", "BIT_WORK_ROOT", "CHECKPOINT_ROOT"):
        value = os.getenv(name)
        if value:
            roots.append(Path(value))
    if not roots:
        roots.extend(
            [
                Path("/mnt/project/bit_drive_left_tail/experiments/risk_vla"),
                Path("/mnt/project/bit_drive_left_tail"),
                Path("/mnt/project/VLA-AD/checkpoints"),
            ]
        )
    return roots


def token_column(columns: Iterable[str]) -> Optional[str]:
    available = set(columns)
    for column in TOKEN_COLUMNS:
        if column in available:
            return column
    return None


def depth_from_root(path: Path, root: Path) -> int:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return 999
    return max(len(rel.parts) - 1, 0)


def iter_csvs(root: Path, *, max_depth: int, include_patterns: List[str]) -> Iterable[Path]:
    if not root.exists():
        return
    root = root.resolve()
    stack = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > max_depth:
            continue
        try:
            entries = sorted(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                if entry.name in SKIP_DIR_NAMES:
                    continue
                stack.append((entry, depth + 1))
            elif entry.suffix.lower() == ".csv":
                text = str(entry)
                if include_patterns and not any(fnmatch.fnmatch(text, pattern) or fnmatch.fnmatch(entry.name, pattern) for pattern in include_patterns):
                    continue
                yield entry


def infer_hint(path: Path, hints: List[str]) -> tuple[str, bool]:
    text = str(path).lower()
    for hint in hints:
        if hint.lower() in text:
            return hint, False
    return "unknown", True


def inspect_csv(path: Path, *, method_hints: List[str], split_hints: List[str]) -> Dict[str, object]:
    row: Dict[str, object] = {
        "path": str(path),
        "row_count": 0,
        "readable": False,
        "has_token": False,
        "token_column": "",
        "has_required_metrics": False,
        "has_required_pdm_columns": False,
        "missing_columns": "",
        "method_inference": "unknown",
        "method_inference_uncertain": True,
        "split_inference": "unknown",
        "split_inference_uncertain": True,
        "rank_score": 0.0,
        "warning": "",
    }
    try:
        header = pd.read_csv(path, nrows=0)
        columns = [str(col) for col in header.columns]
        token = token_column(columns)
        missing = []
        if token is None:
            missing.append("token")
        missing.extend(column for column in PDM_COLUMNS if column not in columns)
        try:
            row_count = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1
        except Exception:
            row_count = len(pd.read_csv(path))
        method, method_uncertain = infer_hint(path, method_hints)
        split, split_uncertain = infer_hint(path, split_hints)
        row.update(
            {
                "row_count": max(int(row_count), 0),
                "readable": True,
                "has_token": token is not None,
                "token_column": token or "",
                "has_required_metrics": not any(column in missing for column in PDM_COLUMNS),
                "has_required_pdm_columns": not missing,
                "missing_columns": ",".join(missing),
                "method_inference": method,
                "method_inference_uncertain": method_uncertain,
                "split_inference": split,
                "split_inference_uncertain": split_uncertain,
            }
        )
        score = 0.0
        score += 10.0 if not missing else max(0.0, 8.0 - float(len(missing)))
        score += min(float(row["row_count"]) / 1000.0, 5.0)
        score += 2.0 if not method_uncertain else 0.0
        score += 1.0 if not split_uncertain else 0.0
        row["rank_score"] = score
    except Exception as exc:
        row["warning"] = f"malformed or unreadable CSV: {exc}"
    return row


def find_candidates(
    *,
    search_roots: List[Path],
    max_depth: int,
    include_patterns: List[str],
    method_hints: List[str],
    split_hints: List[str],
) -> pd.DataFrame:
    seen: set[Path] = set()
    rows: List[Dict[str, object]] = []
    for root in search_roots:
        if not root.exists():
            continue
        root_depth = max_depth
        if root == Path(os.getenv("CHECKPOINT_ROOT", "/mnt/project/VLA-AD/checkpoints")):
            root_depth = min(root_depth, 2)
        for path in iter_csvs(root, max_depth=root_depth, include_patterns=include_patterns):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            rows.append(inspect_csv(resolved, method_hints=method_hints, split_hints=split_hints))
    if not rows:
        return pd.DataFrame(
            columns=[
                "path",
                "row_count",
                "readable",
                "has_required_pdm_columns",
                "method_inference",
                "split_inference",
                "rank_score",
                "warning",
            ]
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["has_required_pdm_columns", "rank_score", "row_count"], ascending=[False, False, False])


def write_markdown(df: pd.DataFrame, output_md: Path, roots: List[Path]) -> None:
    lines = ["# RISK-VLA PDM CSV Candidate Search", "", "## Search Roots", ""]
    lines.extend(f"- `{root}`" for root in roots)
    lines.extend(["", "## Candidates", ""])
    if df.empty:
        lines.append("No PDM CSV candidates were found. Provide A0/B3 and train/val PDM CSVs explicitly in `configs/risk_vla/round1_input_manifest.yaml`.")
    else:
        columns = ["rank_score", "path", "row_count", "has_required_pdm_columns", "method_inference", "split_inference", "missing_columns", "warning"]
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for _, row in df.iterrows():
            values = []
            for column in columns:
                value = row.get(column, "")
                values.append(f"{float(value):.3f}" if column == "rank_score" else str(value))
            lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find candidate PDM CSV files for RISK-VLA Round 1.")
    parser.add_argument("--search-root", action="append", type=Path, default=[])
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--include-pattern", action="append", default=[])
    parser.add_argument("--method-hint", action="append", default=["A0", "base", "B3", "bit", "direct_bit"])
    parser.add_argument("--split-hint", action="append", default=["navtrain", "navval", "navtest", "train", "val"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = args.search_root or env_roots()
    df = find_candidates(
        search_roots=roots,
        max_depth=max(int(args.max_depth), 0),
        include_patterns=list(args.include_pattern or []),
        method_hints=list(args.method_hint or []),
        split_hints=list(args.split_hint or []),
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    write_markdown(df, args.output_md, roots)
    print(f"Found {len(df)} candidate CSV files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
