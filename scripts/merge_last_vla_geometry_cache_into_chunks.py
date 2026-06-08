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
CONTEXT_KEYS = ("vggt_context_tokens",)
LEGACY_VGGT_TARGET_KEY = "vggt_target_tokens"
MODE_KEY = "vggt_geometry_mode"
MODE_CODE_KEY = "vggt_geometry_mode_code"
GEOMETRY_META_KEYS = ("vggt_geometry_source", "vggt_geometry_tokenizer_metadata", MODE_CODE_KEY)
JEPA_KEYS = ("jepa_context_tokens", "jepa_target_tokens")
JEPA_META_KEYS = ("jepa_tokenizer_metadata", "jepa_num_tokens")


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
    parser.add_argument("--geometry-teacher-dim", type=int, default=512)
    parser.add_argument("--num-geometry-tokens", type=int, default=12)
    parser.add_argument("--geometry-grid-rows", type=int, default=3)
    parser.add_argument("--geometry-grid-cols", type=int, default=4)
    parser.add_argument("--jepa-cache-root", type=Path, default=None)
    parser.add_argument("--jepa-chunk-name-pattern", default="*")
    parser.add_argument("--expected-jepa-tokens", type=int, default=128)
    parser.add_argument("--jepa-dim", type=int, default=1024)
    parser.add_argument("--expected-vggt-context-tokens", type=int, default=None)
    parser.add_argument("--expected-vggt-tokens", type=int, default=None, help="Deprecated alias for --expected-vggt-context-tokens.")
    parser.add_argument("--vggt-dim", type=int, default=2048)
    parser.add_argument("--keep-legacy-vggt-target", action="store_true")
    parser.add_argument("--strict-jepa-coverage", action="store_true")
    parser.add_argument("--min-jepa-coverage", type=float, default=0.99)
    return parser.parse_args()


def expected_vggt_context_tokens(args: argparse.Namespace) -> int:
    if args.expected_vggt_context_tokens is not None:
        return int(args.expected_vggt_context_tokens)
    if args.expected_vggt_tokens is not None:
        return int(args.expected_vggt_tokens)
    return 12


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


def _validate_geometry_value(key: str, payload: Dict[str, Any], *, geometry_teacher_dim: int, num_geometry_tokens: int) -> None:
    if key not in payload:
        return
    if key in CONTEXT_SCHEMA and key in CONTEXT_KEYS:
        raise RuntimeError("Context keys require explicit VGGT shape validation.")
    value = payload[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Geometry cache key '{key}' must be a tensor, got {type(value).__name__}.")
    if value.ndim != 2 or tuple(value.shape) != (num_geometry_tokens, geometry_teacher_dim):
        raise ValueError(f"Geometry cache key '{key}' expected [{num_geometry_tokens}, {geometry_teacher_dim}], got {tuple(value.shape)}.")
    if not torch.isfinite(value.float()).all():
        raise ValueError(f"Geometry cache key '{key}' contains non-finite values.")


def extract_geometry_payload(
    overlay: Dict[str, Any],
    *,
    overwrite_context: bool,
    geometry_teacher_dim: int,
    num_geometry_tokens: int,
    expected_vggt_context_tokens: int,
    vggt_dim: int,
    keep_legacy_vggt_target: bool,
) -> Dict[str, Any]:
    context_keys = (*CONTEXT_KEYS, *( (LEGACY_VGGT_TARGET_KEY,) if keep_legacy_vggt_target else () ))
    keys: Iterable[str] = (*GEOMETRY_KEYS, *(context_keys if overwrite_context else ()))
    output: Dict[str, Any] = {}
    for key in keys:
        if key in overlay:
            if key in context_keys:
                validate_token_tensor(overlay, key, (int(expected_vggt_context_tokens), int(vggt_dim)))
            else:
                _validate_geometry_value(
                    key,
                    overlay,
                    geometry_teacher_dim=geometry_teacher_dim,
                    num_geometry_tokens=num_geometry_tokens,
                )
            output[key] = overlay[key]
    if MODE_KEY in overlay:
        output[MODE_KEY] = str(overlay[MODE_KEY])
    if MODE_CODE_KEY in overlay:
        code = overlay[MODE_CODE_KEY]
        output[MODE_CODE_KEY] = code.detach().cpu().long() if isinstance(code, torch.Tensor) else torch.tensor(int(code), dtype=torch.int64)
    for key in GEOMETRY_META_KEYS:
        if key in overlay and key not in output:
            output[key] = overlay[key]
    if "vggt_geometry_tokens" not in output and "vggt_geometry_target_tokens" not in output:
        raise KeyError("Geometry overlay sample does not contain vggt_geometry_tokens or vggt_geometry_target_tokens.")
    return output


def extract_jepa_payload(
    overlay: Dict[str, Any],
    *,
    expected_jepa_tokens: int,
    jepa_dim: int,
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key in JEPA_KEYS:
        if key not in overlay:
            raise KeyError(f"JEPA overlay sample is missing {key}.")
        value = overlay[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"JEPA cache key '{key}' must be a tensor, got {type(value).__name__}.")
        if value.ndim != 2 or tuple(value.shape) != (expected_jepa_tokens, jepa_dim):
            raise ValueError(f"JEPA cache key '{key}' expected [{expected_jepa_tokens}, {jepa_dim}], got {tuple(value.shape)}.")
        if not torch.isfinite(value.float()).all():
            raise ValueError(f"JEPA cache key '{key}' contains non-finite values.")
        output[key] = value
    for key in JEPA_META_KEYS:
        if key in overlay:
            output[key] = overlay[key]
    output["jepa_num_tokens"] = int(expected_jepa_tokens)
    return output


def write_chunk_metadata(
    base_chunk: Path,
    out_chunk: Path,
    num_records: int,
    merged: int,
    missing: int,
    *,
    overwrite_context: bool,
    geometry_teacher_dim: int,
    num_geometry_tokens: int,
    geometry_grid: tuple[int, int],
    contains_jepa_overlay: bool = False,
    expected_jepa_tokens: int = 128,
    jepa_dim: int = 1024,
    expected_vggt_context_tokens: int = 12,
    vggt_dim: int = 2048,
    keep_legacy_vggt_target: bool = False,
) -> None:
    try:
        metadata = dict(load_metadata(base_chunk))
    except FileNotFoundError:
        metadata = {}
    token_shapes = dict(metadata.get("token_shapes") or {})
    for key in GEOMETRY_KEYS:
        if key in CONTEXT_SCHEMA:
            token_shapes[key] = list(CONTEXT_SCHEMA[key])
        elif key.endswith("_tokens"):
            token_shapes[key] = [num_geometry_tokens, geometry_teacher_dim]
    if not keep_legacy_vggt_target:
        token_shapes.pop(LEGACY_VGGT_TARGET_KEY, None)
    if overwrite_context:
        for key in CONTEXT_KEYS:
            token_shapes[key] = [int(expected_vggt_context_tokens), int(vggt_dim)]
        if keep_legacy_vggt_target:
            token_shapes[LEGACY_VGGT_TARGET_KEY] = [int(expected_vggt_context_tokens), int(vggt_dim)]
    if contains_jepa_overlay:
        for key in JEPA_KEYS:
            token_shapes[key] = [int(expected_jepa_tokens), int(jepa_dim)]
    metadata.update(
        {
            "contains_vggt_geometry": True,
            "last_vla_geometry_overlay_merged": True,
            "last_vla_geometry_overlay_overwrite_context": bool(overwrite_context),
            "num_records": int(num_records),
            "num_geometry_merged": int(merged),
            "num_geometry_missing": int(missing),
            "geometry_coverage": float(merged / num_records) if num_records else 0.0,
            "geometry_teacher_dim": int(geometry_teacher_dim),
            "num_geometry_tokens": int(num_geometry_tokens),
            "geometry_grid": [int(geometry_grid[0]), int(geometry_grid[1])],
            "num_vggt_context_tokens": int(expected_vggt_context_tokens) if overwrite_context else metadata.get("num_vggt_context_tokens", metadata.get("num_vggt_tokens")),
            "num_vggt_tokens": int(expected_vggt_context_tokens) if overwrite_context else metadata.get("num_vggt_tokens"),
            "vggt_dim": int(vggt_dim) if overwrite_context else metadata.get("vggt_dim"),
            "legacy_vggt_target_tokens_kept": bool(keep_legacy_vggt_target),
            "legacy_vggt_target_tokens_removed": not bool(keep_legacy_vggt_target),
            "contains_highcap_jepa": bool(contains_jepa_overlay),
            "num_jepa_tokens": int(expected_jepa_tokens) if contains_jepa_overlay else metadata.get("num_jepa_tokens"),
            "jepa_dim": int(jepa_dim) if contains_jepa_overlay else metadata.get("jepa_dim"),
            "token_shapes": token_shapes,
        }
    )
    write_json(out_chunk / "metadata.json", metadata)


def merge_cache(args: argparse.Namespace) -> Dict[str, Any]:
    if int(args.num_geometry_tokens) != int(args.geometry_grid_rows) * int(args.geometry_grid_cols):
        raise ValueError("--num-geometry-tokens must equal --geometry-grid-rows * --geometry-grid-cols.")
    if args.in_place:
        args.output_chunk_root = args.base_chunk_root
    elif args.output_chunk_root.resolve() == args.base_chunk_root.resolve():
        raise ValueError("Refusing in-place merge without --in-place.")

    overlay_index = load_overlay_index(args.geometry_cache_root, args.geometry_chunk_name_pattern)
    total_merged = 0
    total_missing = 0
    mode_counts = {"full_geometry": 0, "patch_fallback": 0, "missing": 0}
    geometry_shape_counts: Dict[str, int] = {}
    jepa_index = load_overlay_index(args.jepa_cache_root, args.jepa_chunk_name_pattern) if args.jepa_cache_root is not None else {}
    expected_context_tokens = expected_vggt_context_tokens(args)
    total_jepa_merged = 0
    total_jepa_missing = 0
    jepa_shape_counts: Dict[str, int] = {}
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
            has_geometry = token in overlay_index
            has_jepa = token in jepa_index
            if has_geometry or has_jepa:
                sample = load_sample(src_path)
                updated = dict(sample)
                if not args.keep_legacy_vggt_target:
                    updated.pop(LEGACY_VGGT_TARGET_KEY, None)
                if has_geometry:
                    overlay = load_sample(overlay_index[token])
                    geometry_payload = extract_geometry_payload(
                        overlay,
                        overwrite_context=bool(args.overwrite_context),
                        geometry_teacher_dim=int(args.geometry_teacher_dim),
                        num_geometry_tokens=int(args.num_geometry_tokens),
                        expected_vggt_context_tokens=expected_context_tokens,
                        vggt_dim=int(args.vggt_dim),
                        keep_legacy_vggt_target=bool(args.keep_legacy_vggt_target),
                    )
                    updated.update(geometry_payload)
                    tokens = geometry_payload.get("vggt_geometry_tokens")
                    if isinstance(tokens, torch.Tensor):
                        shape_key = str(tuple(tokens.shape))
                        geometry_shape_counts[shape_key] = geometry_shape_counts.get(shape_key, 0) + 1
                    chunk_merged += 1
                    mode_counts[str(geometry_payload.get(MODE_KEY, "missing"))] = mode_counts.get(str(geometry_payload.get(MODE_KEY, "missing")), 0) + 1
                else:
                    chunk_missing += 1
                    mode_counts["missing"] += 1
                if has_jepa:
                    jepa_payload = extract_jepa_payload(
                        load_sample(jepa_index[token]),
                        expected_jepa_tokens=int(args.expected_jepa_tokens),
                        jepa_dim=int(args.jepa_dim),
                    )
                    updated.update(jepa_payload)
                    shape_key = str(tuple(jepa_payload["jepa_context_tokens"].shape))
                    jepa_shape_counts[shape_key] = jepa_shape_counts.get(shape_key, 0) + 1
                    total_jepa_merged += 1
                else:
                    total_jepa_missing += 1
                atomic_torch_save(updated, dst_path)
            else:
                if args.keep_legacy_vggt_target:
                    copy_or_link(src_path, dst_path, args.copy_mode)
                else:
                    sample = load_sample(src_path)
                    if LEGACY_VGGT_TARGET_KEY in sample:
                        updated = dict(sample)
                        updated.pop(LEGACY_VGGT_TARGET_KEY, None)
                        atomic_torch_save(updated, dst_path)
                    else:
                        copy_or_link(src_path, dst_path, args.copy_mode)
                chunk_missing += 1
                mode_counts["missing"] += 1
                if args.jepa_cache_root is not None:
                    total_jepa_missing += 1
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
            geometry_teacher_dim=int(args.geometry_teacher_dim),
            num_geometry_tokens=int(args.num_geometry_tokens),
            geometry_grid=(int(args.geometry_grid_rows), int(args.geometry_grid_cols)),
            contains_jepa_overlay=args.jepa_cache_root is not None,
            expected_jepa_tokens=int(args.expected_jepa_tokens),
            jepa_dim=int(args.jepa_dim),
            expected_vggt_context_tokens=expected_context_tokens,
            vggt_dim=int(args.vggt_dim),
            keep_legacy_vggt_target=bool(args.keep_legacy_vggt_target),
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
        "full_geometry": int(mode_counts.get("full_geometry", 0)),
        "patch_fallback": int(mode_counts.get("patch_fallback", 0)),
        "mode_counts": mode_counts,
        "geometry_teacher_dim": int(args.geometry_teacher_dim),
        "num_geometry_tokens": int(args.num_geometry_tokens),
        "geometry_grid": [int(args.geometry_grid_rows), int(args.geometry_grid_cols)],
        "geometry_token_shape_distribution": dict(sorted(geometry_shape_counts.items())),
        "jepa_cache_root": str(args.jepa_cache_root) if args.jepa_cache_root is not None else None,
        "jepa_merged": int(total_jepa_merged),
        "jepa_missing": int(total_jepa_missing),
        "jepa_coverage": float(total_jepa_merged / (total_jepa_merged + total_jepa_missing)) if (total_jepa_merged + total_jepa_missing) else 0.0,
        "jepa_token_shape_distribution": dict(sorted(jepa_shape_counts.items())),
        "expected_jepa_tokens": int(args.expected_jepa_tokens),
        "jepa_dim": int(args.jepa_dim),
        "expected_vggt_context_tokens": int(expected_context_tokens),
        "expected_vggt_tokens": int(expected_context_tokens),
        "vggt_dim": int(args.vggt_dim),
        "overwrite_context": bool(args.overwrite_context),
        "keep_legacy_vggt_target": bool(args.keep_legacy_vggt_target),
        "chunk_reports": chunk_reports,
    }
    args.output_chunk_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_chunk_root / "merge_summary.json", summary)
    write_json(args.output_chunk_root / "last_vla_geometry_merge_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    summary = merge_cache(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.strict_coverage and summary["coverage"] < float(args.min_coverage):
        return 2
    if args.strict_jepa_coverage and summary["jepa_coverage"] < float(args.min_jepa_coverage):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
