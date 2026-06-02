#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def label_distribution(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = {0: 0, 1: 0, None: 0}
    reasons: Dict[str, int] = {}
    for row in rows:
        label = row.get("label_use_bit")
        labels[label] = labels.get(label, 0) + 1
        reason = str(row.get("label_reason"))
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "num_samples": len(rows),
        "label_0": labels.get(0, 0),
        "label_1": labels.get(1, 0),
        "label_missing": labels.get(None, 0),
        "positive_rate": labels.get(1, 0) / len(rows) if rows else None,
        "reasons": reasons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple BiT counterfactual datasets.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True, help="Input counterfactual_samples.jsonl files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="navtrain")
    parser.add_argument("--dedupe-key", choices=("sample_token", "scene_token", "none"), default="sample_token")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    merged: List[Dict[str, Any]] = []
    seen = set()
    duplicate_count = 0
    for input_path in args.inputs:
        for row in read_jsonl(input_path):
            key = None if args.dedupe_key == "none" else row.get(args.dedupe_key)
            if key is not None:
                if key in seen:
                    duplicate_count += 1
                    continue
                seen.add(key)
            row = dict(row)
            row["source_counterfactual_jsonl"] = str(input_path)
            merged.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "counterfactual_samples.jsonl", merged)
    write_jsonl(
        args.output_dir / "base_predictions.jsonl",
        (
            {
                "scene_token": row.get("scene_token"),
                "sample_token": row.get("sample_token"),
                "pred_traj": row.get("base_pred_traj"),
                "metrics": row.get("base_metrics"),
            }
            for row in merged
        ),
    )
    write_jsonl(
        args.output_dir / "bit_predictions.jsonl",
        (
            {
                "scene_token": row.get("scene_token"),
                "sample_token": row.get("sample_token"),
                "pred_traj": row.get("bit_pred_traj"),
                "metrics": row.get("bit_metrics"),
            }
            for row in merged
        ),
    )
    distribution = label_distribution(merged)
    (args.output_dir / "label_distribution.json").write_text(
        json.dumps(distribution, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    split_lower = args.split.lower()
    metadata = {
        "split": args.split,
        "training_allowed": "test" not in split_lower,
        "analysis_only": False,
        "input_jsonl": [str(path) for path in args.inputs],
        "duplicate_count": duplicate_count,
        "dedupe_key": args.dedupe_key,
        "label_missing_count": distribution["label_missing"],
        "all_labels_present": distribution["label_missing"] == 0,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Merged BiT Counterfactual Dataset",
        "",
        f"Split: `{args.split}`",
        f"Samples: {len(merged)}",
        f"Duplicates removed: {duplicate_count}",
        f"Training allowed: `{metadata['training_allowed']}`",
        "",
        "```json",
        json.dumps(distribution, indent=2, sort_keys=True),
        "```",
    ]
    (args.output_dir / "counterfactual_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "duplicates": duplicate_count, **distribution}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
