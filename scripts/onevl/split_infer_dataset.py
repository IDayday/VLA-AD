#!/usr/bin/env python3
"""Split a JSON or JSONL inference dataset into contiguous JSON shards."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--prefix", default="shard")
    return parser.parse_args()


def load_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as src:
            for line_index, line in enumerate(src):
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    row.setdefault("source_line_index", line_index)
                rows.append(row)
        return rows, "jsonl"

    with path.open(encoding="utf-8") as src:
        data = json.load(src)
    if isinstance(data, dict) and isinstance(data.get("predictions"), list):
        rows = data["predictions"]
        source_shape = "json.predictions"
    elif isinstance(data, list):
        rows = data
        source_shape = "json.list"
    else:
        raise ValueError(f"Unsupported JSON shape in {path}")
    return rows, source_shape


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be positive")

    rows, source_shape = load_rows(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_size = math.ceil(len(rows) / args.num_shards) if rows else 0

    manifest: dict[str, Any] = {
        "input": str(args.input),
        "source_shape": source_shape,
        "out_dir": str(args.out_dir),
        "rows": len(rows),
        "num_shards": args.num_shards,
        "shards": [],
    }
    for shard_id in range(args.num_shards):
        start = shard_id * shard_size if shard_size else 0
        end = min((shard_id + 1) * shard_size, len(rows)) if shard_size else 0
        shard_rows = rows[start:end]
        shard_path = args.out_dir / f"{args.prefix}_{shard_id}.json"
        with shard_path.open("w", encoding="utf-8") as dst:
            json.dump(shard_rows, dst, ensure_ascii=False, indent=2)
        manifest["shards"].append(
            {
                "shard_id": shard_id,
                "path": str(shard_path),
                "start": start,
                "end": end,
                "rows": len(shard_rows),
            }
        )

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
