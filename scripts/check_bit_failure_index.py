#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.bit_failure_sampler import (  # noqa: E402
    failure_index_metadata_path,
    load_failure_index_metadata,
    validate_failure_index_for_training,
)


def parse_tags(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                if "__metadata__" not in record:
                    records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a BiT failure index before training.")
    parser.add_argument("--failure-index-path", type=Path, required=True)
    parser.add_argument("--for-training", action="store_true")
    parser.add_argument("--expected-tags", default=None)
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--max-weight", type=float, default=10.0)
    args = parser.parse_args()

    if not args.failure_index_path.is_file():
        raise FileNotFoundError(args.failure_index_path)
    metadata = load_failure_index_metadata(args.failure_index_path)
    if not metadata:
        raise RuntimeError(f"Missing failure index metadata sidecar: {failure_index_metadata_path(args.failure_index_path)}")
    if "split" not in metadata or "allowed_for_training" not in metadata:
        raise RuntimeError("Failure index metadata must include split and allowed_for_training.")
    if args.for_training:
        validate_failure_index_for_training(args.failure_index_path)

    records = load_records(args.failure_index_path)
    if not records:
        raise RuntimeError(f"No records in {args.failure_index_path}")
    tag_counts: Counter[str] = Counter()
    bad_weights = []
    for record in records:
        tags = [str(tag) for tag in record.get("failure_tags", [])]
        tag_counts.update(tags)
        weight = record.get("sample_weight")
        try:
            weight_f = float(weight)
        except (TypeError, ValueError):
            bad_weights.append({"sample_token": record.get("sample_token"), "sample_weight": weight})
            continue
        if weight_f < args.min_weight or weight_f > args.max_weight:
            bad_weights.append({"sample_token": record.get("sample_token"), "sample_weight": weight_f})
    if bad_weights:
        raise RuntimeError(f"Found unreasonable sample weights: {bad_weights[:10]}")

    missing_tags = [tag for tag in parse_tags(args.expected_tags) if tag not in tag_counts]
    if missing_tags:
        raise RuntimeError(f"Expected failure tags not found: {missing_tags}")

    report = {
        "failure_index_path": str(args.failure_index_path),
        "metadata": metadata,
        "num_records": len(records),
        "tag_distribution": dict(sorted(tag_counts.items())),
        "for_training": bool(args.for_training),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
