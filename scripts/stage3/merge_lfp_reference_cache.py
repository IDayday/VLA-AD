#!/usr/bin/env python3
"""Merge validated LFP reference-cache shards into one coherent cache."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


_CONSISTENT_METADATA = (
    "version",
    "benchmark",
    "stage2_checkpoint_path",
    "stage2_checkpoint_sha256",
    "metric_cache_path",
    "metric_cache_fingerprint",
    "total_scene_count",
    "dataset_token_hash",
    "num_shards",
    "ddc_gt_tolerance",
    "v2_require_tlc",
)


def _load(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"Reference shard must be a mapping: {path}")
    return payload


def merge_reference_shards(shard_paths: Sequence[Path]) -> dict[str, Any]:
    paths = [Path(path).expanduser().resolve() for path in shard_paths]
    if not paths:
        raise ValueError("At least one reference shard is required.")
    payloads = [_load(path) for path in paths]
    first_metadata = dict(payloads[0].get("metadata", {}))
    num_shards = int(first_metadata.get("num_shards", 1))
    if len(payloads) != num_shards:
        raise ValueError(f"Expected {num_shards} shards, received {len(payloads)}.")

    shard_indices: set[int] = set()
    records: dict[str, Any] = {}
    raw_ddc_count = 0.0
    for path, payload in zip(paths, payloads):
        metadata = dict(payload.get("metadata", {}))
        for key in _CONSISTENT_METADATA:
            if metadata.get(key) != first_metadata.get(key):
                raise ValueError(f"Reference shard metadata mismatch for {key!r}: {path}")
        shard_index = int(metadata.get("shard_index", -1))
        if shard_index in shard_indices or not 0 <= shard_index < num_shards:
            raise ValueError(f"Invalid or duplicate shard_index={shard_index}: {path}")
        shard_indices.add(shard_index)
        shard_records = payload.get("records")
        if not isinstance(shard_records, Mapping):
            raise TypeError(f"Reference shard has no records mapping: {path}")
        overlap = records.keys() & shard_records.keys()
        if overlap:
            raise ValueError(f"Duplicate token across reference shards: {next(iter(overlap))!r}")
        records.update((str(token), record) for token, record in shard_records.items())
        raw_ddc_count += float(metadata.get("raw_ddc_availability_ratio", 0.0)) * len(shard_records)

    expected_indices = set(range(num_shards))
    if shard_indices != expected_indices:
        raise ValueError(f"Missing shard indices: {sorted(expected_indices - shard_indices)}")
    total_scene_count = int(first_metadata.get("total_scene_count", 0))
    if len(records) != total_scene_count:
        raise ValueError(
            f"Merged cache has {len(records)} unique records, expected {total_scene_count}."
        )

    metadata = dict(first_metadata)
    metadata.update(
        {
            "scene_count": len(records),
            "creation_timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_ddc_availability_ratio": raw_ddc_count / max(len(records), 1),
            "source_num_shards": num_shards,
            "merged_shards": [path.name for path in sorted(paths)],
        }
    )
    metadata.pop("shard_index", None)
    metadata.pop("num_shards", None)
    return {"metadata": metadata, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output_path", "--output-path", required=True, type=Path)
    args = parser.parse_args()
    payload = merge_reference_shards(args.shards)
    output_path = args.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"merged_scene_count={len(payload['records'])}")
    print(f"output_path={output_path}")


if __name__ == "__main__":
    main()
