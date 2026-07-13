#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.support_aligned_diversity import (
    SupportAlignedDiversityConfig,
    _mean_off_diagonal,
    complete_linkage_mode_medoids,
    density_balanced_support_weights,
    load_support_reference,
    trajectory_distance_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build scene-level support capacity metadata for the LFP active curriculum."
    )
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mode-threshold", type=float, default=0.40)
    parser.add_argument("--density-bandwidth", type=float, default=0.40)
    return parser.parse_args()


def _process_one(args: tuple[str, Dict[str, Any]]) -> tuple[str, Dict[str, Any]]:
    path_text, config_dict = args
    path = Path(path_text)
    config = SupportAlignedDiversityConfig(**config_dict)
    support = load_support_reference(path)
    distance = trajectory_distance_matrix(support.trajectories, support.trajectories, config)
    weights = density_balanced_support_weights(distance, config.density_bandwidth)
    medoids = complete_linkage_mode_medoids(distance, config.mode_threshold)
    delta_xy = support.trajectories[:, None, :, :2] - support.trajectories[None, :, :, :2]
    pairwise_ade_m = np.linalg.norm(delta_xy, axis=-1).mean(axis=-1)
    rewards = np.asarray(support.rewards, dtype=np.float64)
    return support.token, {
        "support_dispersion": float(_mean_off_diagonal(distance, weights)),
        "support_pairwise_ade_m": float(_mean_off_diagonal(pairwise_ade_m, weights)),
        "reference_mode_count": int(len(medoids)),
        "num_supports": int(support.trajectories.shape[0]),
        "support_reward_mean": float(rewards.mean()) if rewards.size else 0.0,
        "support_reward_max": float(rewards.max()) if rewards.size else 0.0,
        "archive_version": int(support.archive_version),
    }


def _archive_fingerprint(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    support_root = args.support_root.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    if not support_root.is_dir():
        raise FileNotFoundError(support_root)
    if args.workers < 1:
        raise ValueError("--workers must be positive.")
    paths = sorted(support_root.glob("*.pkl.xz"))
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise RuntimeError(f"No support archives found under {support_root}.")

    config = SupportAlignedDiversityConfig(
        mode_threshold=float(args.mode_threshold),
        density_bandwidth=float(args.density_bandwidth),
    )
    records: Dict[str, Dict[str, Any]] = {}
    work = ((str(path), asdict(config)) for path in paths)
    with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        for index, (token, record) in enumerate(executor.map(_process_one, work, chunksize=64), start=1):
            if token in records:
                raise ValueError(f"Duplicate support token {token!r}.")
            records[token] = record
            if index % 5000 == 0 or index == len(paths):
                print(f"processed={index}/{len(paths)}", flush=True)

    metadata = {
        "version": 2,
        "coverage_representation": "density_balanced_xy_pairwise_ade_m",
        "mode_representation": "support_relative_trajectory_distance",
        "scene_balanced": True,
        "num_scenes": len(records),
        "support_root": str(support_root),
        "archive_fingerprint": _archive_fingerprint(paths, support_root),
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "metric_config": asdict(config),
    }
    payload = {"metadata": metadata, "records": records}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output_path)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
