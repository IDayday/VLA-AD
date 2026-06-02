#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.bit_safety_selector import (  # noqa: E402
    SELECTOR_FEATURE_NAMES,
    trajectory_delta_features,
)


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
    parser = argparse.ArgumentParser(description="Recompute metric-free BiT selector features for a counterfactual JSONL.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copy-prediction-files", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.input_jsonl)
    updated = []
    missing_traj = 0
    for row in rows:
        row = dict(row)
        base_traj = row.get("base_pred_traj")
        bit_traj = row.get("bit_pred_traj")
        if base_traj is None or bit_traj is None:
            missing_traj += 1
        else:
            row["features"] = trajectory_delta_features(base_traj, bit_traj)
        updated.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "counterfactual_samples.jsonl", updated)
    distribution = label_distribution(updated)
    (args.output_dir / "label_distribution.json").write_text(
        json.dumps(distribution, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    source_meta_path = args.input_jsonl.parent / "metadata.json"
    metadata = json.loads(source_meta_path.read_text(encoding="utf-8")) if source_meta_path.is_file() else {}
    metadata.update(
        {
            "source_counterfactual_jsonl": str(args.input_jsonl),
            "feature_schema": "bit_select_v2_enriched_trajectory",
            "feature_names": SELECTOR_FEATURE_NAMES,
            "missing_traj_count": missing_traj,
        }
    )
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.copy_prediction_files:
        for name in ("base_predictions.jsonl", "bit_predictions.jsonl"):
            source = args.input_jsonl.parent / name
            if source.is_file():
                shutil.copy2(source, args.output_dir / name)

    lines = [
        "# Augmented BiT Counterfactual Features",
        "",
        f"Source: `{args.input_jsonl}`",
        f"Rows: {len(updated)}",
        f"Missing trajectories: {missing_traj}",
        f"Feature count: {len(SELECTOR_FEATURE_NAMES)}",
        "",
        "```json",
        json.dumps(distribution, indent=2, sort_keys=True),
        "```",
    ]
    (args.output_dir / "counterfactual_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "rows": len(updated),
                "missing_traj_count": missing_traj,
                "feature_count": len(SELECTOR_FEATURE_NAMES),
                **distribution,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
