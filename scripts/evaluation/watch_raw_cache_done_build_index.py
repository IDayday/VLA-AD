#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_manifest(path: Path) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            log_name, token = line.split()[:2]
            entries.append((log_name, token))
    if not entries:
        raise RuntimeError(f"empty manifest: {path}")
    return entries


def build_index(raw_cache_root: Path, manifest: Path, output_root: Path, chunk_prefix: str, num_chunks: int) -> None:
    entries = read_manifest(manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    counts = {}
    for chunk_idx in range(num_chunks):
        chunk_dir = output_root / f"{chunk_prefix}_{chunk_idx:02d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        tmp = chunk_dir / "index.jsonl.tmp"
        count = 0
        with tmp.open("w", encoding="utf-8") as f:
            for idx, (log_name, token) in enumerate(entries):
                if idx % num_chunks != chunk_idx:
                    continue
                token_dir = raw_cache_root / log_name / token
                if not (token_dir / "internvl_feature.gz").is_file():
                    raise FileNotFoundError(f"missing internvl_feature.gz for {token}: {token_dir}")
                f.write(
                    json.dumps(
                        {
                            "path": str(token_dir),
                            "sample_token": token,
                            "scene_token": token,
                            "source": "recogdrive_raw_disk",
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                count += 1
        tmp.replace(chunk_dir / "index.jsonl")
        counts[chunk_dir.name] = count
    summary = {
        "built_at": utc_now(),
        "raw_cache_root": str(raw_cache_root),
        "manifest": str(manifest),
        "output_root": str(output_root),
        "num_tokens": len(entries),
        "num_chunks": num_chunks,
        "chunk_counts": counts,
    }
    (output_root / "index_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for raw ReCogDrive cache completion, then build index chunks.")
    parser.add_argument("--done-marker", type=Path, required=True)
    parser.add_argument("--fail-marker", type=Path, default=None)
    parser.add_argument("--raw-cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--chunk-prefix", default="chunk")
    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_chunks <= 0:
        raise ValueError("--num-chunks must be positive")
    print(f"[{utc_now()}] watching done marker: {args.done_marker}", flush=True)
    while not args.done_marker.is_file():
        if args.fail_marker is not None and args.fail_marker.is_file():
            raise RuntimeError(f"cache generation failed: {args.fail_marker}")
        time.sleep(max(args.poll_seconds, 1.0))
    print(f"[{utc_now()}] done marker found; building index", flush=True)
    build_index(args.raw_cache_root, args.manifest, args.output_root, args.chunk_prefix, args.num_chunks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
