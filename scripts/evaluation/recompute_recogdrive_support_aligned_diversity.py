#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.support_aligned_diversity import (  # noqa: E402
    SupportAlignedDiversityConfig,
    compute_support_aligned_diversity,
    load_support_reference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute support-aligned metrics from saved trajectory samples.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--density-bandwidth", type=float, default=0.40)
    parser.add_argument("--hard-match-threshold", type=float, default=0.50)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    prediction_paths = sorted(args.input_root.glob("shard_*/predictions.npz"))
    if not prediction_paths:
        raise FileNotFoundError(f"No shard predictions found under {args.input_root}")
    config = SupportAlignedDiversityConfig(
        density_bandwidth=args.density_bandwidth,
        hard_match_threshold=args.hard_match_threshold,
    )
    for prediction_path in prediction_paths:
        shard_name = prediction_path.parent.name
        output_dir = args.output_root / shard_name
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = np.load(prediction_path)
        tokens = [str(token) for token in payload["tokens"].tolist()]
        trajectories = np.asarray(payload["trajectories"], dtype=np.float64)
        if trajectories.shape[0] != len(tokens):
            raise ValueError(f"Prediction token/trajectory counts differ: {prediction_path}")
        old_metrics_path = prediction_path.parent / "scene_metrics.csv"
        old_rows = {}
        if old_metrics_path.is_file():
            old_frame = pd.read_csv(old_metrics_path)
            old_rows = {str(row["token"]): row.to_dict() for _, row in old_frame.iterrows()}
        rows: list[dict] = []
        for token, prediction in zip(tokens, trajectories):
            if args.max_samples > 0:
                prediction = prediction[: args.max_samples]
            reference = load_support_reference(args.support_root, token)
            carried = {
                key: value
                for key, value in old_rows.get(token, {}).items()
                if key.startswith(("mean_gt_", "best_gt_", "support_reward_", "support_has_", "archive_"))
            }
            rows.append(
                {
                    "token": token,
                    **carried,
                    **compute_support_aligned_diversity(prediction, reference.trajectories, config),
                }
            )
        _write_csv(output_dir / "scene_metrics.csv", rows)
    print(f"recomputed {len(prediction_paths)} shards under {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
