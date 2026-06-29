#!/usr/bin/env python3
"""Split a JSONL dataset into contiguous shards for per-GPU inference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be positive")
    rows = []
    with args.input_jsonl.open(encoding="utf-8") as src:
        for line_index, line in enumerate(src):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("source_line_index", line_index)
            rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_size = math.ceil(len(rows) / args.num_shards)
    manifest = {
        "input_jsonl": str(args.input_jsonl),
        "out_dir": str(args.out_dir),
        "rows": len(rows),
        "num_shards": args.num_shards,
        "shards": [],
    }
    for shard_id in range(args.num_shards):
        start = shard_id * shard_size
        end = min((shard_id + 1) * shard_size, len(rows))
        shard_rows = rows[start:end]
        shard_path = args.out_dir / f"shard_{shard_id}.jsonl"
        with shard_path.open("w", encoding="utf-8") as dst:
            for row in shard_rows:
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["shards"].append(
            {"shard_id": shard_id, "path": str(shard_path), "start": start, "end": end, "rows": len(shard_rows)}
        )
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
