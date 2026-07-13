#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import lzma
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.pareto_support.fs_norm import fit_fs_norm_stats, save_fs_norm_stats


def _selected_supports(path: Path) -> tuple[list[np.ndarray], str]:
    with lzma.open(path, "rb") as file:
        payload = pickle.load(file)
    support_set = getattr(payload, "support_set", None)
    if support_set is not None:
        return [np.asarray(candidate.trajectory, dtype=np.float32) for candidate in support_set], path.stem
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported support archive payload in {path}: {type(payload).__name__}.")
    scene_id = str(payload.get("token", payload.get("scene_token", path.stem)))
    if "candidates" in payload and "support_indices" in payload:
        candidates = np.asarray(payload["candidates"], dtype=np.float32)
        indices = np.asarray(payload["support_indices"], dtype=np.int64).reshape(-1)
        if candidates.ndim != 3 or candidates.shape[1:] != (8, 3):
            raise ValueError(f"Expected candidates [N, 8, 3] in {path}, got {candidates.shape}.")
        if np.any(indices < 0) or np.any(indices >= candidates.shape[0]):
            raise ValueError(f"support_indices are out of range in {path}.")
        return [candidates[index] for index in indices.tolist()], scene_id
    support_items = payload.get("support_set")
    if support_items is not None:
        trajectories = []
        for item in support_items:
            trajectory = item.get("trajectory") if isinstance(item, dict) else getattr(item, "trajectory", None)
            if trajectory is not None:
                trajectories.append(np.asarray(trajectory, dtype=np.float32))
        return trajectories, scene_id
    raise KeyError(f"Support archive {path} has no support_set or candidates/support_indices.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FS-Norm stats from evaluator-verified support archives.")
    parser.add_argument("--archive_dir", required=True)
    parser.add_argument("--output_stats", required=True)
    parser.add_argument("--robust", default="true", choices=["true", "false"])
    parser.add_argument("--lower_quantile", type=float, default=0.001)
    parser.add_argument("--upper_quantile", type=float, default=0.999)
    parser.add_argument("--max_scenes", type=int, default=0, help="Optional deterministic archive subset size.")
    args = parser.parse_args()

    archive_root = Path(args.archive_dir)
    if (archive_root / "full_archive").is_dir():
        archive_root = archive_root / "full_archive"
    trajectories = []
    scene_ids = []
    fingerprint = hashlib.sha256()
    paths = sorted(archive_root.glob("*.pkl.xz"))
    if args.max_scenes > 0:
        paths = paths[: args.max_scenes]
    for path in paths:
        pool, scene_id = _selected_supports(path)
        if not pool:
            continue
        trajectories.extend(pool)
        scene_ids.extend([scene_id] * len(pool))
        stat = path.stat()
        fingerprint.update(f"{path.relative_to(archive_root)}:{stat.st_size}:{stat.st_mtime_ns}\n".encode("utf-8"))
    if not trajectories:
        raise RuntimeError(f"No support trajectories found under {args.archive_dir}.")
    stats = fit_fs_norm_stats(
        np.stack(trajectories, axis=0),
        robust=args.robust == "true",
        scene_ids=np.asarray(scene_ids),
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
        archive_path=str(archive_root.resolve()),
        archive_fingerprint=fingerprint.hexdigest(),
    )
    output_path = Path(args.output_stats)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_fs_norm_stats(output_path, stats)
    print(
        {
            "version": stats.version,
            "num_scenes": stats.num_scenes,
            "num_supports": stats.num_supports,
            "output_stats": args.output_stats,
            "robust": args.robust,
            "archive_fingerprint": stats.archive_fingerprint,
        }
    )


if __name__ == "__main__":
    main()
