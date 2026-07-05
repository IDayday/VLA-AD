#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.pareto_support.fs_norm import fit_fs_norm_stats, save_fs_norm_stats
from navsim.agents.recogdrive.pareto_support.io import load_archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FS-Norm stats from evaluator-verified support archives.")
    parser.add_argument("--archive_dir", required=True)
    parser.add_argument("--output_stats", required=True)
    parser.add_argument("--robust", default="true", choices=["true", "false"])
    parser.add_argument("--include_hard_negatives", action="store_true")
    args = parser.parse_args()

    archive_root = Path(args.archive_dir)
    if (archive_root / "full_archive").is_dir():
        archive_root = archive_root / "full_archive"
    trajectories = []
    for path in sorted(archive_root.glob("*.pkl.xz")):
        archive = load_archive(path)
        pool = list(archive.support_set)
        if args.include_hard_negatives:
            pool.extend(archive.hard_negatives)
        trajectories.extend([candidate.trajectory for candidate in pool])
    if not trajectories:
        raise RuntimeError(f"No support trajectories found under {args.archive_dir}.")
    stats = fit_fs_norm_stats(np.stack(trajectories, axis=0), robust=args.robust == "true")
    save_fs_norm_stats(args.output_stats, stats)
    print({"trajectory_count": len(trajectories), "output_stats": args.output_stats, "robust": args.robust})


if __name__ == "__main__":
    main()
