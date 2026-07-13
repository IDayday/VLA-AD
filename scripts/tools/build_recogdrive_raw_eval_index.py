#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build lightweight index.jsonl chunks for original ReCogDrive raw hidden caches."
    )
    parser.add_argument("--raw-cache-root", type=Path, required=True)
    parser.add_argument("--sample-token-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--chunk-prefix", default="val6000_chunk")
    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_tokens(path: Path) -> List[str]:
    tokens: List[str] = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            token = raw.strip()
            if not token or token.startswith("#") or token in seen:
                continue
            tokens.append(token)
            seen.add(token)
    if not tokens:
        raise RuntimeError(f"No tokens found in {path}")
    return tokens


def scan_raw_cache(root: Path, wanted: set[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for feature_path in root.glob("*/*/internvl_feature.gz"):
        token_dir = feature_path.parent
        token = token_dir.name
        if token in wanted and token not in mapping:
            mapping[token] = token_dir
            if len(mapping) == len(wanted):
                break
    return mapping


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True))
            f.write("\n")
            count += 1
    tmp.replace(path)
    return count


def main() -> int:
    args = parse_args()
    if args.num_chunks <= 0:
        raise ValueError("--num-chunks must be positive")
    if not args.raw_cache_root.is_dir():
        raise FileNotFoundError(args.raw_cache_root)
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_root} is not empty; pass --overwrite")
    args.output_root.mkdir(parents=True, exist_ok=True)

    tokens = read_tokens(args.sample_token_file)
    mapping = scan_raw_cache(args.raw_cache_root, set(tokens))
    missing = [token for token in tokens if token not in mapping]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} requested tokens under {args.raw_cache_root}; first={missing[:10]}")

    chunk_records: List[List[dict]] = [[] for _ in range(args.num_chunks)]
    for idx, token in enumerate(tokens):
        token_dir = mapping[token]
        chunk_records[idx % args.num_chunks].append(
            {
                "path": str(token_dir),
                "sample_token": token,
                "scene_token": token,
                "source": "recogdrive_raw_disk",
            }
        )

    counts = {}
    for chunk_idx, records in enumerate(chunk_records):
        chunk_dir = args.output_root / f"{args.chunk_prefix}_{chunk_idx:02d}"
        counts[chunk_dir.name] = write_jsonl(chunk_dir / "index.jsonl", records)

    summary = {
        "raw_cache_root": str(args.raw_cache_root),
        "sample_token_file": str(args.sample_token_file),
        "output_root": str(args.output_root),
        "num_tokens": len(tokens),
        "num_chunks": args.num_chunks,
        "chunk_counts": counts,
    }
    (args.output_root / "index_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
