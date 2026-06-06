#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Last-VLA JEPA/VGGT overlay shard caches into one overlay root.")
    parser.add_argument("--sharded-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-num-shards", type=int, required=True)
    parser.add_argument("--overlay-type", choices=("jepa128", "geometry192"), required=True)
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def iter_shard_dirs(sharded_root: Path) -> List[Path]:
    shards_root = sharded_root / "shards"
    if not shards_root.is_dir():
        return []
    return sorted(path for path in shards_root.glob("shard_*") if path.is_dir() and (path / "index.jsonl").is_file())


def shard_index_from_dir(path: Path) -> int | None:
    try:
        return int(path.name.removeprefix("shard_"))
    except ValueError:
        return None


def resolve_record_path(shard_dir: Path, record: Dict[str, Any]) -> Path:
    raw = Path(record["path"])
    return raw if raw.is_absolute() else shard_dir / raw


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def token_summary(tokens: List[str]) -> Dict[str, Any]:
    digest = hashlib.sha256("\n".join(sorted(tokens)).encode("utf-8")).hexdigest()
    return {"count": len(tokens), "sha256": digest, "first": sorted(tokens)[:20]}


def load_shard_records(shard_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metadata_path = shard_dir / "metadata.json"
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    records: List[Dict[str, Any]] = []
    with (shard_dir / "index.jsonl").open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            token = str(record.get("sample_token") or Path(record["path"]).stem)
            if not token:
                raise ValueError(f"{shard_dir}/index.jsonl:{line_no}: missing sample_token")
            record["sample_token"] = token
            records.append(record)
    return records, metadata


def merge_shards(args: argparse.Namespace) -> Dict[str, Any]:
    if int(args.expected_num_shards) <= 0:
        raise ValueError("--expected-num-shards must be positive.")

    shard_dirs = iter_shard_dirs(args.sharded_root)
    found_indexes = sorted(idx for idx in (shard_index_from_dir(path) for path in shard_dirs) if idx is not None)
    expected_indexes = list(range(int(args.expected_num_shards)))
    missing_shards = [idx for idx in expected_indexes if idx not in found_indexes]
    extra_shards = [idx for idx in found_indexes if idx not in expected_indexes]

    output_samples = args.output_root / "samples"
    output_samples.mkdir(parents=True, exist_ok=True)
    merged_records: List[Dict[str, Any]] = []
    seen: Dict[str, str] = {}
    duplicates: List[str] = []
    missing_sample_files: List[str] = []
    per_shard: List[Dict[str, Any]] = []

    for shard_dir in shard_dirs:
        records, metadata = load_shard_records(shard_dir)
        metadata_overlay_type = metadata.get("overlay_type")
        if metadata_overlay_type is not None and str(metadata_overlay_type) != str(args.overlay_type):
            raise ValueError(f"{shard_dir}: overlay_type {metadata_overlay_type!r} does not match {args.overlay_type!r}.")
        copied = 0
        shard_idx = shard_index_from_dir(shard_dir)
        for record in records:
            token = str(record["sample_token"])
            if token in seen:
                duplicates.append(token)
                continue
            source = resolve_record_path(shard_dir, record)
            if not source.is_file():
                missing_sample_files.append(str(source))
                continue
            dest = output_samples / source.name
            copy_or_link(source, dest, str(args.copy_mode))
            out_record = dict(record)
            out_record["path"] = str(dest.relative_to(args.output_root))
            merged_records.append(out_record)
            seen[token] = str(shard_dir)
            copied += 1
        per_shard.append(
            {
                "shard_dir": str(shard_dir),
                "shard_index": shard_idx,
                "records": len(records),
                "copied": copied,
                "metadata_num_written": metadata.get("num_written"),
            }
        )

    summary = {
        "version": "last_vla_overlay_shard_merge_v1",
        "sharded_root": str(args.sharded_root),
        "output_root": str(args.output_root),
        "expected_num_shards": int(args.expected_num_shards),
        "found_num_shards": len(shard_dirs),
        "found_shard_indexes": found_indexes,
        "missing_shards": missing_shards,
        "extra_shards": extra_shards,
        "total_records": len(merged_records),
        "duplicate_count": len(set(duplicates)),
        "duplicate_sample_tokens": sorted(set(duplicates))[:50],
        "missing_sample_file_count": len(missing_sample_files),
        "missing_sample_files": missing_sample_files[:50],
        "overlay_type": str(args.overlay_type),
        "copy_mode": str(args.copy_mode),
        "sample_token_minimal_list_or_hash": token_summary(list(seen)),
        "per_shard": per_shard,
    }

    if args.strict:
        blockers: List[str] = []
        if summary["found_num_shards"] != int(args.expected_num_shards):
            blockers.append("found_num_shards_mismatch")
        if missing_shards:
            blockers.append("missing_shards")
        if extra_shards:
            blockers.append("extra_shards")
        if duplicates:
            blockers.append("duplicate_sample_tokens")
        if missing_sample_files:
            blockers.append("missing_sample_files")
        if blockers:
            summary["blockers"] = blockers
            write_json(args.output_root / "metadata.json", summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
            raise RuntimeError(f"Strict shard merge failed: {', '.join(blockers)}")

    with (args.output_root / "index.jsonl").open("w", encoding="utf-8") as f:
        for record in merged_records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    write_json(args.output_root / "metadata.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    args = parse_args()
    merge_shards(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
