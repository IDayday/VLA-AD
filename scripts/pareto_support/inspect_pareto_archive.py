#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.pareto_support.io import load_archive
from navsim.agents.recogdrive.pareto_support.visualization import archive_summary_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect SG-FPS support archives.")
    parser.add_argument("--archive_dir", required=True)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_csv", default="")
    args = parser.parse_args()

    rows = []
    tag_counts = Counter()
    archive_root = Path(args.archive_dir)
    if (archive_root / "full_archive").is_dir():
        archive_root = archive_root / "full_archive"
    for path in sorted(archive_root.glob("*.pkl.xz")):
        archive = load_archive(path)
        rows.append(archive_summary_row(archive))
        tag_counts.update(str(c.support_category) for c in archive.support_set)
    summary = {
        "scene_count": len(rows),
        "avg_support_count": sum(r["support_count"] for r in rows) / max(len(rows), 1),
        "avg_true_eval_count": sum(r["true_eval_count"] for r in rows) / max(len(rows), 1),
        "support_tag_counts": dict(tag_counts),
    }
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "rows": rows}, f, indent=2, sort_keys=True)
    if args.output_csv and rows:
        with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
