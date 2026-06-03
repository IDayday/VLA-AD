#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


BLOCKED_SPLIT_MARKERS = ("test", "navtest", "challenge", "eval-only")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_name(row: Dict[str, Any]) -> str:
    return str(row.get("split") or row.get("split_name") or row.get("split_alias") or "").strip()


def check_leakage(
    rows: Iterable[Dict[str, Any]],
    *,
    allowed_splits: Iterable[str],
    for_training: bool,
    allow_missing_split: bool = False,
) -> Tuple[Dict[str, int], List[str]]:
    allowed = {item.lower() for item in allowed_splits}
    counts: Counter[str] = Counter()
    errors: List[str] = []
    missing = 0
    for index, row in enumerate(rows):
        split = split_name(row)
        if not split:
            missing += 1
            counts["<missing>"] += 1
            if for_training and not allow_missing_split:
                errors.append(f"row {index}: missing split; pass --allow-missing-split only for audited train/val labels.")
            continue
        split_lower = split.lower()
        counts[split_lower] += 1
        blocked = any(marker in split_lower for marker in BLOCKED_SPLIT_MARKERS)
        allowed_exact = split_lower in allowed
        if for_training and (blocked or not allowed_exact):
            errors.append(f"row {index}: split={split!r} is not allowed for training labels.")
    if missing and allow_missing_split:
        counts["<missing_allowed>"] = missing
    return dict(counts), errors


def write_report(path: Path, counts: Dict[str, int], errors: List[str]) -> None:
    lines = ["# Risk Label Leakage Check", "", "## Split Counts", ""]
    for split, count in sorted(counts.items()):
        lines.append(f"- `{split}`: {count}")
    lines.extend(["", "## Result", ""])
    if errors:
        lines.append("FAILED")
        lines.append("")
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.append("PASSED")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Reject train-time RISK-VLA labels that include navtest/test splits.")
    parser.add_argument("--labels-jsonl", type=Path, required=True)
    parser.add_argument("--allowed-splits", nargs="+", default=["navtrain", "navval", "train", "val"])
    parser.add_argument("--for-training", action="store_true")
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--allow-missing-split", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.labels_jsonl)
    counts, errors = check_leakage(
        rows,
        allowed_splits=args.allowed_splits,
        for_training=bool(args.for_training),
        allow_missing_split=bool(args.allow_missing_split),
    )
    if args.output_md is not None:
        write_report(args.output_md, counts, errors)
    print(json.dumps({"split_counts": counts, "errors": errors}, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
