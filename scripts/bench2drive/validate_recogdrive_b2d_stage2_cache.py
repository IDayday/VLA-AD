#!/usr/bin/env python3
"""Validate that a sharded Bench2Drive Stage2 cache is complete and trainable."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch


class CacheValidationError(RuntimeError):
    """Raised when a cache is incomplete or violates the Stage2 contract."""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--expected-clips", type=int, required=True)
    parser.add_argument("--expected-records", type=int, default=None)
    parser.add_argument("--expected-vlm-path", type=Path, required=True)
    parser.add_argument("--skip-sample-check", action="store_true")
    return parser.parse_args(argv)


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheValidationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CacheValidationError(f"expected a JSON object in {path}")
    return value


def _index_rows(path: Path) -> list[Dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheValidationError(f"cannot read {path}: {exc}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise CacheValidationError(f"index contains a non-object row: {path}")
    return rows


def _check_tensor(sample: Dict[str, Any], key: str, shape: tuple[int, ...]) -> None:
    value = sample.get(key)
    if not isinstance(value, torch.Tensor):
        raise CacheValidationError(f"sample field {key!r} is not a tensor")
    if tuple(value.shape) != shape:
        raise CacheValidationError(f"sample field {key!r} has shape {tuple(value.shape)}, expected {shape}")
    if not torch.isfinite(value.float()).all():
        raise CacheValidationError(f"sample field {key!r} contains non-finite values")


def _check_sample(shard: Path, row: Dict[str, Any]) -> Dict[str, Any]:
    relative_path = row.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        raise CacheValidationError(f"index row in {shard} has no sample path")
    sample_path = shard / relative_path
    if not sample_path.is_file():
        raise CacheValidationError(f"indexed sample does not exist: {sample_path}")
    try:
        sample = torch.load(sample_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CacheValidationError(f"cannot load sample {sample_path}: {exc}") from exc
    if not isinstance(sample, dict):
        raise CacheValidationError(f"sample is not a dictionary: {sample_path}")
    hidden = sample.get("last_hidden_state")
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 2 or hidden.shape[-1] != 1536:
        shape = tuple(hidden.shape) if isinstance(hidden, torch.Tensor) else None
        raise CacheValidationError(f"last_hidden_state has invalid shape {shape}: {sample_path}")
    if not torch.isfinite(hidden.float()).all():
        raise CacheValidationError(f"last_hidden_state contains non-finite values: {sample_path}")
    _check_tensor(sample, "history_trajectory", (4, 3))
    _check_tensor(sample, "trajectory", (8, 3))
    _check_tensor(sample, "high_command_one_hot", (3,))
    _check_tensor(sample, "status_feature", (8,))
    return {
        "path": str(sample_path),
        "sample_token": sample.get("sample_token"),
        "hidden_shape": list(hidden.shape),
        "hidden_dtype": str(hidden.dtype),
    }


def validate_cache(
    cache_root: Path,
    *,
    expected_shards: int,
    expected_clips: int,
    expected_records: Optional[int],
    expected_vlm_path: Path,
    check_samples: bool = True,
) -> Dict[str, Any]:
    if expected_shards <= 0 or expected_clips <= 0:
        raise ValueError("expected_shards and expected_clips must be positive")
    if expected_records is not None and expected_records <= 0:
        raise ValueError("expected_records must be positive when supplied")
    if not cache_root.is_dir():
        raise CacheValidationError(f"cache root does not exist: {cache_root}")

    shards = sorted(path for path in cache_root.glob("shard_*") if path.is_dir())
    if len(shards) != expected_shards:
        raise CacheValidationError(f"found {len(shards)} cache shards, expected {expected_shards}")

    resolved_vlm = expected_vlm_path.resolve()
    ranges: list[tuple[int, int, str]] = []
    total_records = 0
    total_selected_clips = 0
    total_skipped_windows = 0
    sample_checks: list[Dict[str, Any]] = []
    clip_lists: set[str] = set()

    for shard in shards:
        metadata_path = shard / "metadata.json"
        index_path = shard / "index.jsonl"
        if not metadata_path.is_file() or not index_path.is_file():
            missing = metadata_path if not metadata_path.is_file() else index_path
            raise CacheValidationError(f"shard is not complete; missing {missing}")
        metadata = _load_json(metadata_path)
        if metadata.get("is_dummy"):
            raise CacheValidationError(f"dummy cache shard is not trainable: {shard}")
        if metadata.get("contains_vlm_hidden") is not True:
            raise CacheValidationError(f"shard lacks VLM hidden states: {shard}")
        if metadata.get("hidden_source") != "recogdrive_vlm":
            raise CacheValidationError(f"unexpected hidden source in {shard}: {metadata.get('hidden_source')}")
        if metadata.get("system_prompt_profile") != "bench2drive":
            raise CacheValidationError(f"unexpected prompt profile in {shard}: {metadata.get('system_prompt_profile')}")
        source_vlm = metadata.get("recogdrive_vlm_path")
        if not source_vlm or Path(str(source_vlm)).resolve() != resolved_vlm:
            raise CacheValidationError(f"wrong source VLM in {shard}: {source_vlm!r}, expected {resolved_vlm}")

        rows = _index_rows(index_path)
        num_records = int(metadata.get("num_records", -1))
        if num_records <= 0 or len(rows) != num_records:
            raise CacheValidationError(
                f"record count mismatch in {shard}: metadata={num_records}, index={len(rows)}"
            )
        start = int(metadata.get("clip_start", -1))
        stop = int(metadata.get("clip_stop", -1))
        selected = int(metadata.get("num_selected_clips", -1))
        if start < 0 or stop <= start or selected != stop - start:
            raise CacheValidationError(
                f"invalid clip range in {shard}: start={start}, stop={stop}, selected={selected}"
            )
        ranges.append((start, stop, shard.name))
        total_records += num_records
        total_selected_clips += selected
        total_skipped_windows += int(metadata.get("num_skipped_windows", 0))
        clip_list = metadata.get("clip_list")
        if not clip_list:
            raise CacheValidationError(f"shard does not record its clip list: {shard}")
        clip_lists.add(str(Path(str(clip_list)).resolve()))
        if check_samples:
            sample_checks.append(_check_sample(shard, rows[0]))
            if len(rows) > 1:
                sample_checks.append(_check_sample(shard, rows[-1]))

    expected_start = 0
    for start, stop, shard_name in sorted(ranges):
        if start != expected_start:
            raise CacheValidationError(
                f"clip coverage has a gap or overlap before {shard_name}: expected start {expected_start}, got {start}"
            )
        expected_start = stop
    if expected_start != expected_clips or total_selected_clips != expected_clips:
        raise CacheValidationError(
            f"clip coverage mismatch: range_end={expected_start}, selected={total_selected_clips}, "
            f"expected={expected_clips}"
        )
    if len(clip_lists) != 1:
        raise CacheValidationError(f"shards use different clip lists: {sorted(clip_lists)}")
    if expected_records is not None and total_records != expected_records:
        raise CacheValidationError(f"record count is {total_records}, expected {expected_records}")

    return {
        "ready": True,
        "cache_root": str(cache_root.resolve()),
        "source_vlm_path": str(resolved_vlm),
        "num_shards": len(shards),
        "num_selected_clips": total_selected_clips,
        "num_records": total_records,
        "num_skipped_windows": total_skipped_windows,
        "clip_list": next(iter(clip_lists)),
        "sample_checks": sample_checks,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        summary = validate_cache(
            args.cache_root,
            expected_shards=args.expected_shards,
            expected_clips=args.expected_clips,
            expected_records=args.expected_records,
            expected_vlm_path=args.expected_vlm_path,
            check_samples=not args.skip_sample_check,
        )
    except (CacheValidationError, ValueError) as exc:
        print(f"cache not ready: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
