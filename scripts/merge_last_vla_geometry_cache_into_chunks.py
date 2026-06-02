#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import (  # noqa: E402
    CONTEXT_SCHEMA,
    atomic_torch_save,
    iter_index,
    load_sample,
    load_metadata,
    validate_token_tensor,
    write_json,
)


GEOMETRY_KEYS = (
    "vggt_geometry_tokens",
    "vggt_geometry_target_tokens",
    "vggt_depth_tokens",
    "vggt_pointmap_tokens",
    "vggt_camera_tokens",
    "vggt_depth_target_tokens",
    "vggt_pointmap_target_tokens",
)
CONTEXT_KEYS = ("vggt_context_tokens", "vggt_target_tokens")
MODE_KEY = "vggt_geometry_mode"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge strict Last-VLA/VGGT geometry overlay cache into chunk samples.")
    parser.add_argument("--base-chunk-root", type=Path, required=True)
    parser.add_argument("--geometry-cache-root", type=Path, required=True)
    parser.add_argument("--geometry-chunk-name-pattern", default="*")
    parser.add_argument("--output-chunk-root", type=Path, required=True)
    parser.add_argument("--chunk-name-pattern", default="train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*")
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--overwrite-context", action="store_true")
    parser.add_argument("--strict-coverage", action="store_true")
    parser.add_argument("--min-coverage", type=float, default=0.99)
    return parser.parse_args()


def chunk_dirs(root: Path, pattern: str) -> List[Path]:
    if (root / "index.jsonl").is_file():
        return [root]
    dirs: List[Path] = []
    seen = set()
    for item in str(pattern).split(","):
        item = item.strip()
        if not item:
            continue
        for path in sorted(root.glob(item)):
            if path.is_dir() and (path / "index.jsonl").is_file() and path not in seen:
                dirs.append(path)
                seen.add(path)
    if not dirs:
        raise FileNotFoundError(f"No chunk directories matching {pattern!r} under {root}")
    return dirs


def resolve_index_path(root: Path, record: Dict[str, Any]) -> Path:
    raw = Path(record["path"])
    if raw.is_absolute():
        return raw
    return root / raw


def overlay_index_paths(root: Path, pattern: str) -> List[Path]:
    index_path = root / "index.jsonl"
    if index_path.is_file():
        return [index_path]
    paths: List[Path] = []
    seen = set()
    for item in str(pattern).split(","):
        item = item.strip()
        if not item:
            continue
        for path in sorted(root.glob(item)):
            index = path / "index.jsonl"
            if path.is_dir() and index.is_file() and index not in seen:
                paths.append(index)
                seen.add(index)
    if not paths:
        raise FileNotFoundError(f"Geometry overlay index not found under {root} with pattern {pattern!r}.")
    return paths


def load_overlay_index(root: Path, pattern: str) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for index_path in overlay_index_paths(root, pattern):
        index_root = index_path.parent
        with index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                token = str(record.get("sample_token") or Path(record["path"]).stem)
                path = Path(record["path"])
                mapping[token] = path if path.is_absolute() else index_root / path
    return mapping


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


def _validate_geometry_value(key: str, payload: Dict[str, Any]) -> None:
    if key not in payload:
        return
    if key in CONTEXT_SCHEMA:
        validate_token_tensor(payload, key, CONTEXT_SCHEMA[key])
        return
    value = payload[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Geometry cache key '{key}' must be a tensor, got {type(value).__name__}.")
    if value.ndim != 2 or value.shape[-1] != 2048:
        raise ValueError(f"Geometry cache key '{key}' expected [K, 2048], got {tuple(value.shape)}.")
    if not torch.isfinite(value.float()).all():
        raise ValueError(f"Geometry cache key '{key}' contains non-finite values.")


def extract_geometry_payload(overlay: Dict[str, Any], *, overwrite_context: bool) -> Dict[str, Any]:
    keys: Iterable[str] = (*GEOMETRY_KEYS, *(CONTEXT_KEYS if overwrite_context else ()))
    output: Dict[str, Any] = {}
    for key in keys:
        if key in overlay:
            _validate_geometry_value(key, overlay)
            output[key] = overlay[key]
    if MODE_KEY in overlay:
        output[MODE_KEY] = str(overlay[MODE_KEY])
    if "vggt_geometry_tokens" not in output and "vggt_geometry_target_tokens" not in output:
        raise KeyError("Geometry overlay sample does not contain vggt_geometry_tokens or vggt_geometry_target_tokens.")
    return output


def write_chunk_metadata(base_chunk: Path, out_chunk: Path, num_records: int, merged: int, missing: int, *, overwrite_context: bool) -> None:
    try:
        metadata = dict(load_metadata(base_chunk))
    except FileNotFoundError:
        metadata = {}
    token_shapes = dict(metadata.get("token_shapes") or {})
    for key in GEOMETRY_KEYS:
        if key in CONTEXT_SCHEMA:
            token_shapes[key] = list(CONTEXT_SCHEMA[key])
        elif key.endswith("_tokens"):
            token_shapes[key] = [12, 2048]
    metadata.update(
        {
            "contains_vggt_geometry": True,
            "last_vla_geometry_overlay_merged": True,
            "last_vla_geometry_overlay_overwrite_context": bool(overwrite_context),
            "num_records": int(num_records),
            "num_geometry_merged": int(merged),
            "num_geometry_missing": int(missing),
            "geometry_coverage": float(merged / num_records) if num_records else 0.0,
            "token_shapes": token_shapes,
        }
    )
    write_json(out_chunk / "metadata.json", metadata)


def merge_cache(args: argparse.Namespace) -> Dict[str, Any]:
    if args.in_place:
        args.output_chunk_root = args.base_chunk_root
    elif args.output_chunk_root.resolve() == args.base_chunk_root.resolve():
        raise ValueError("Refusing in-place merge without --in-place.")

    overlay_index = load_overlay_index(args.geometry_cache_root, args.geometry_chunk_name_pattern)
    total_merged = 0
    total_missing = 0
    chunk_reports = []
    for chunk_dir in chunk_dirs(args.base_chunk_root, args.chunk_name_pattern):
        out_chunk = args.output_chunk_root / chunk_dir.name
        out_chunk.mkdir(parents=True, exist_ok=True)
        index_records = []
        chunk_merged = 0
        chunk_missing = 0
        for record in iter_index(chunk_dir):
            src_path = resolve_index_path(chunk_dir, record)
            token = str(record.get("sample_token") or src_path.stem)
            rel = Path("samples") / src_path.name
            dst_path = out_chunk / rel
            if token in overlay_index:
                sample = load_sample(src_path)
                overlay = load_sample(overlay_index[token])
                geometry_payload = extract_geometry_payload(overlay, overwrite_context=bool(args.overwrite_context))
                updated = dict(sample)
                updated.update(geometry_payload)
                atomic_torch_save(updated, dst_path)
                chunk_merged += 1
            else:
                copy_or_link(src_path, dst_path, args.copy_mode)
                chunk_missing += 1
            out_record = dict(record)
            out_record["path"] = str(rel)
            index_records.append(out_record)
        with (out_chunk / "index.jsonl").open("w", encoding="utf-8") as f:
            for item in index_records:
                f.write(json.dumps(item, sort_keys=True) + "\n")
        write_chunk_metadata(
            chunk_dir,
            out_chunk,
            len(index_records),
            chunk_merged,
            chunk_missing,
            overwrite_context=bool(args.overwrite_context),
        )
        total_merged += chunk_merged
        total_missing += chunk_missing
        chunk_reports.append(
            {
                "chunk": chunk_dir.name,
                "records": len(index_records),
                "merged": chunk_merged,
                "missing": chunk_missing,
                "coverage": float(chunk_merged / len(index_records)) if index_records else 0.0,
            }
        )

    total = total_merged + total_missing
    summary = {
        "base_chunk_root": str(args.base_chunk_root),
        "geometry_cache_root": str(args.geometry_cache_root),
        "output_chunk_root": str(args.output_chunk_root),
        "merged": int(total_merged),
        "missing": int(total_missing),
        "coverage": float(total_merged / total) if total else 0.0,
        "overwrite_context": bool(args.overwrite_context),
        "chunk_reports": chunk_reports,
    }
    args.output_chunk_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_chunk_root / "last_vla_geometry_merge_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    summary = merge_cache(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.strict_coverage and summary["coverage"] < float(args.min_coverage):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
