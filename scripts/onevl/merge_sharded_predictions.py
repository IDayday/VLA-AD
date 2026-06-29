#!/usr/bin/env python3
"""Merge sharded inference prediction files into one JSON list."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SHARD_ID_RE = re.compile(r"_(\d+)\.jsonl?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--prediction-prefix", default="predict")
    parser.add_argument("--allow-missing-shards", action="store_true")
    return parser.parse_args()


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as src:
            for line in src:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    with path.open(encoding="utf-8") as src:
        data = json.load(src)
    if not isinstance(data, list):
        raise ValueError(f"Prediction file is not a list: {path}")
    return data


def shard_id(path: Path) -> int:
    match = SHARD_ID_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse shard id from {path}")
    return int(match.group(1))


def expected_prediction_paths(shard_dir: Path, prediction_prefix: str) -> list[Path]:
    manifest_path = shard_dir / "manifest.json"
    if not manifest_path.is_file():
        return sorted(shard_dir.glob(f"{prediction_prefix}_*.json"), key=shard_id)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = []
    for shard in manifest.get("shards", []):
        if int(shard.get("rows", 0)) <= 0:
            continue
        sid = int(shard["shard_id"])
        paths.append(shard_dir / f"{prediction_prefix}_{sid}.json")
    return paths


def main() -> None:
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    expected_paths = expected_prediction_paths(args.shard_dir, args.prediction_prefix)
    missing = [str(path) for path in expected_paths if not path.is_file()]
    if missing and not args.allow_missing_shards:
        raise SystemExit(f"Missing prediction shard(s): {missing[:10]}")

    merged: list[dict[str, Any]] = []
    used_paths = [path for path in expected_paths if path.is_file()]
    for path in sorted(used_paths, key=shard_id):
        merged.extend(load_json_or_jsonl(path))

    args.output_json.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "shard_dir": str(args.shard_dir),
        "output_json": str(args.output_json),
        "prediction_files": len(used_paths),
        "rows": len(merged),
        "missing_prediction_files": missing,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
