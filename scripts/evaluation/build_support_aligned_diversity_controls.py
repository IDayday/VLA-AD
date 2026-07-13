#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
from pathlib import Path
import pickle
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.support_aligned_diversity import (  # noqa: E402
    SupportAlignedDiversityConfig,
    compute_support_aligned_diversity,
    density_balanced_support_weights,
    load_support_reference,
    trajectory_distance_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic GT and support-oracle calibration controls.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples-per-scene", type=int, default=32)
    parser.add_argument("--seed", type=int, default=260711)
    return parser.parse_args()


def _records(path: Path) -> list[tuple[str, Path]]:
    with path.open("r", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        return [(str(row["token"]), Path(row["cache_path"])) for row in reader]


def _gt(cache_path: Path) -> np.ndarray:
    with gzip.open(cache_path / "trajectory_target.gz", "rb") as stream:
        payload = pickle.load(stream)
    trajectory = np.asarray(payload["trajectory"], dtype=np.float64)
    if trajectory.shape != (8, 3):
        raise ValueError(f"Invalid GT trajectory shape {trajectory.shape}: {cache_path}")
    return trajectory


def _rng(seed: int, token: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{token}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.samples_per_scene <= 0:
        raise ValueError("--samples-per-scene must be positive.")
    config = SupportAlignedDiversityConfig()
    gt_rows: list[dict] = []
    oracle_rows: list[dict] = []
    for token, cache_path in _records(args.manifest):
        reference = load_support_reference(args.support_root, token)
        gt = _gt(cache_path)
        gt_predictions = np.repeat(gt[None], args.samples_per_scene, axis=0)
        support_distance = trajectory_distance_matrix(reference.trajectories, reference.trajectories, config)
        probabilities = density_balanced_support_weights(support_distance, config.density_bandwidth)
        selected = _rng(args.seed, token).choice(
            reference.trajectories.shape[0],
            size=args.samples_per_scene,
            replace=True,
            p=probabilities,
        )
        oracle_predictions = reference.trajectories[selected]
        common = {
            "token": token,
            "archive_version": reference.archive_version,
            "support_reward_mean": float(reference.rewards.mean()),
            "support_reward_max": float(reference.rewards.max()),
            "support_has_gt_anchor": float(any(tag == "gt_anchor" for tag in reference.tags)),
        }
        gt_rows.append(
            {
                **common,
                **compute_support_aligned_diversity(gt_predictions, reference.trajectories, config),
                "mean_gt_ade_m": 0.0,
                "best_gt_ade_m": 0.0,
            }
        )
        oracle_xy = np.linalg.norm(oracle_predictions[..., :2] - gt[None, ..., :2], axis=-1)
        oracle_rows.append(
            {
                **common,
                **compute_support_aligned_diversity(oracle_predictions, reference.trajectories, config),
                "mean_gt_ade_m": float(oracle_xy.mean(axis=1).mean()),
                "best_gt_ade_m": float(oracle_xy.mean(axis=1).min()),
            }
        )
    _write(args.output_root / "gt_repeat" / "scene_metrics.csv", gt_rows)
    _write(args.output_root / "support_oracle" / "scene_metrics.csv", oracle_rows)
    print(f"wrote {len(gt_rows)} GT-repeat and support-oracle scenes under {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
