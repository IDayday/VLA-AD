#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.risk_vla.build_strategy_utility_labels import parse_candidate, read_table


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Merge evaluated candidate PDM CSV/JSONL files into a matched candidate PDM table.")
    parser.add_argument("--candidate", type=parse_candidate, action="append", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-real-samples", type=int, default=0)
    parser.add_argument("--debug-allow-small", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    tables = [(candidate, read_table(candidate.path)) for candidate in args.candidate]
    tokens = sorted(set.intersection(*(set(table) for _, table in tables))) if tables else []
    summary = {"split": args.split, "matched_tokens": len(tokens), "candidates": [item.name for item, _ in tables], "output_dir": str(args.output_dir)}
    if args.dry_run:
        print(json.dumps(summary, sort_keys=True))
        return 0
    if args.min_real_samples and len(tokens) < args.min_real_samples and not args.debug_allow_small:
        raise RuntimeError(f"Refusing formal candidate PDM merge with {len(tokens)} matched tokens; minimum is {args.min_real_samples}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[Dict[str, Any]] = []
    for token in tokens:
        for candidate_id, (candidate, table) in enumerate(tables):
            metrics = table[token]
            rows.append(
                {
                    "split": args.split,
                    "token": token,
                    "sample_token": token,
                    "candidate_id": candidate_id,
                    "strategy_name": candidate.strategy_name,
                    "source_method": candidate.source_method or candidate.name,
                    "source_checkpoint": candidate.source_checkpoint,
                    "score": metrics.get("score"),
                    "drivable_area_compliance": metrics.get("dac"),
                    "no_at_fault_collisions": metrics.get("nc"),
                    "time_to_collision_within_bound": metrics.get("ttc"),
                    "ego_progress": metrics.get("progress"),
                    "comfort": metrics.get("comfort"),
                }
            )
    with (args.output_dir / "matched_candidate_pdm.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["split", "token"])
        writer.writeheader()
        writer.writerows(rows)
    summary["rows"] = len(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
