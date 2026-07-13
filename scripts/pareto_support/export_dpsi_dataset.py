#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.pareto_support.io import load_archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Export support archive targets for DPSI training.")
    parser.add_argument("--archive_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_support_per_scene", type=int, default=12)
    args = parser.parse_args()

    output = Path(args.output_dir)
    samples_dir = output / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    archive_root = Path(args.archive_dir)
    if (archive_root / "full_archive").is_dir():
        archive_root = archive_root / "full_archive"
    for path in sorted(archive_root.glob("*.pkl.xz")):
        archive = load_archive(path)
        for idx, candidate in enumerate(archive.support_set[: args.max_support_per_scene]):
            sample_name = f"{archive.scene_token}_{idx:02d}.npz"
            np.savez_compressed(
                samples_dir / sample_name,
                scene_token=archive.scene_token,
                support_candidate_id=candidate.candidate_id,
                trajectory=candidate.trajectory.astype(np.float32),
                loss_weight=np.asarray(candidate.support_weight, dtype=np.float32),
                support_category=str(candidate.support_category or ""),
            )
            index_rows.append({"scene_token": archive.scene_token, "path": str(Path("samples") / sample_name), "weight": candidate.support_weight, "category": candidate.support_category})
    with open(output / "index.jsonl", "w", encoding="utf-8") as f:
        for row in index_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print({"exported_samples": len(index_rows), "output_dir": str(output)})


if __name__ == "__main__":
    main()
