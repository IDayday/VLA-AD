#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import atomic_torch_save, iter_index, load_sample, write_json  # noqa: E402
from navsim.agents.recogdrive.expert_extractors.vggt_extractor import VGGTExtractor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strict Last-VLA full VGGT geometry overlay cache.")
    parser.add_argument("--chunk-cache-root", type=Path, required=True)
    parser.add_argument("--chunk-name-pattern", default="train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*")
    parser.add_argument("--output-cache-root", type=Path, required=True)
    parser.add_argument("--vggt-model-path", default="facebook/VGGT-1B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="fp32")
    parser.add_argument("--geometry-teacher-dim", type=int, default=512)
    parser.add_argument("--num-geometry-tokens", type=int, default=12)
    parser.add_argument("--geometry-grid-rows", type=int, default=3)
    parser.add_argument("--geometry-grid-cols", type=int, default=4)
    parser.add_argument("--require-full-geometry", action="store_true")
    parser.add_argument("--allow-patch-fallback", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--output-shard-subdir",
        action="store_true",
        help="When --num-shards > 1, write to output-cache-root/shards/shard_XXXXX instead of the root.",
    )
    parser.add_argument(
        "--write-shard-manifest",
        action="store_true",
        help="Write shard_manifest.json next to shard metadata for explicit merge auditing.",
    )
    return parser.parse_args()


def output_root_for_args(args: argparse.Namespace) -> Path:
    if int(args.num_shards) <= 1:
        return args.output_cache_root
    if not bool(args.output_shard_subdir):
        raise ValueError("--num-shards > 1 requires --output-shard-subdir to avoid root index overwrite.")
    return args.output_cache_root / "shards" / f"shard_{int(args.shard_index):05d}"


def sample_token_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    tokens = [str(record.get("sample_token") or "") for record in records]
    digest = hashlib.sha256("\n".join(sorted(tokens)).encode("utf-8")).hexdigest()
    return {"count": len(tokens), "sha256": digest, "first": sorted(tokens)[:20]}


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


def resolve_sample_path(chunk_dir: Path, record: Dict[str, Any]) -> Path:
    raw = Path(record["path"])
    if raw.is_file():
        return raw
    return raw if raw.is_absolute() else chunk_dir / raw


def iter_records(root: Path, pattern: str, max_samples: Optional[int]) -> Iterable[Tuple[Path, Path, Dict[str, Any]]]:
    count = 0
    for chunk_dir in chunk_dirs(root, pattern):
        for record in iter_index(chunk_dir):
            yield chunk_dir, resolve_sample_path(chunk_dir, record), record
            count += 1
            if max_samples is not None and count >= max_samples:
                return


def _decode_image_path_tensor(value: torch.Tensor) -> Optional[str]:
    try:
        chars = [chr(int(item)) for item in value.detach().cpu().view(-1).tolist()]
    except Exception:
        return None
    text = "".join(chars)
    return text if text else None


def current_frame_path(sample: Dict[str, Any], record: Dict[str, Any]) -> Path:
    for container in (record, sample):
        raw = container.get("history_cam_f0")
        if isinstance(raw, (list, tuple)) and raw:
            return Path(raw[-1])
        for key in ("current_cam_f0", "image_path", "cam_f0", "frame_path"):
            value = container.get(key)
            if isinstance(value, (str, Path)):
                return Path(value)
    image_tensor = sample.get("image_path_tensor")
    if isinstance(image_tensor, torch.Tensor):
        decoded = _decode_image_path_tensor(image_tensor)
        if decoded:
            return Path(decoded)
    token = sample.get("sample_token") or record.get("sample_token") or record.get("path")
    raise KeyError(f"Cannot resolve current frame image path for sample {token!r}.")


def validate_payload(payload: Dict[str, Any], *, num_tokens: int, teacher_dim: int, allow_patch_fallback: bool) -> None:
    tokens = payload.get("vggt_geometry_tokens")
    if not isinstance(tokens, torch.Tensor) or tuple(tokens.shape) != (num_tokens, teacher_dim):
        raise ValueError(f"vggt_geometry_tokens must have shape ({num_tokens}, {teacher_dim}), got {tuple(tokens.shape) if isinstance(tokens, torch.Tensor) else type(tokens).__name__}.")
    mode = str(payload.get("vggt_geometry_mode", "missing"))
    if mode == "patch_fallback" and not allow_patch_fallback:
        raise RuntimeError("Patch fallback geometry was produced while --allow-patch-fallback is false.")
    if mode not in {"full_geometry", "patch_fallback"}:
        raise RuntimeError(f"Unsupported geometry mode produced: {mode!r}")


def main() -> int:
    args = parse_args()
    if args.require_full_geometry and args.allow_patch_fallback:
        raise ValueError("--require-full-geometry and --allow-patch-fallback are mutually incompatible.")
    if int(args.num_geometry_tokens) != int(args.geometry_grid_rows) * int(args.geometry_grid_cols):
        raise ValueError("--num-geometry-tokens must equal --geometry-grid-rows * --geometry-grid-cols.")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards).")

    extractor = VGGTExtractor(
        args.vggt_model_path,
        device=args.device,
        precision=args.precision,
        require_geometry=bool(args.require_full_geometry),
        geometry_output_dim=int(args.geometry_teacher_dim),
        geometry_num_tokens=int(args.num_geometry_tokens),
        geometry_grid=(int(args.geometry_grid_rows), int(args.geometry_grid_cols)),
    )
    output_root = output_root_for_args(args)
    samples_dir = output_root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    index_records: List[Dict[str, Any]] = []
    counts = {
        "processed": 0,
        "written": 0,
        "full_geometry": 0,
        "patch_fallback": 0,
        "errors": 0,
        "num_skipped_by_shard": 0,
    }
    errors: List[str] = []

    for idx, (_, sample_path, record) in enumerate(iter_records(args.chunk_cache_root, args.chunk_name_pattern, args.max_samples)):
        if idx % int(args.num_shards) != int(args.shard_index):
            counts["num_skipped_by_shard"] += 1
            continue
        sample = load_sample(sample_path)
        sample_token = str(sample.get("sample_token") or record.get("sample_token") or sample_path.stem)
        scene_token = str(sample.get("scene_token") or record.get("scene_token") or sample_path.parent.name)
        counts["processed"] += 1
        try:
            frame = current_frame_path(sample, record)
            payload = extractor.extract_with_geometry(frame)
            validate_payload(
                payload,
                num_tokens=int(args.num_geometry_tokens),
                teacher_dim=int(args.geometry_teacher_dim),
                allow_patch_fallback=bool(args.allow_patch_fallback),
            )
            if args.require_full_geometry and payload.get("vggt_geometry_mode") != "full_geometry":
                raise RuntimeError("Extractor did not produce full_geometry in strict mode.")
        except Exception as exc:
            counts["errors"] += 1
            errors.append(f"{sample_token}:{type(exc).__name__}:{exc}")
            if args.require_full_geometry:
                raise
            continue
        mode = str(payload["vggt_geometry_mode"])
        counts[mode] += 1
        out_path = samples_dir / f"{sample_token}.pt"
        out_payload = {
            "sample_token": sample_token,
            "scene_token": scene_token,
            **payload,
        }
        atomic_torch_save(out_payload, out_path)
        index_records.append({"sample_token": sample_token, "scene_token": scene_token, "path": str(out_path.relative_to(output_root))})
        counts["written"] += 1

    with (output_root / "index.jsonl").open("w", encoding="utf-8") as f:
        for record in index_records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    metadata = {
        "version": "last_vla_full_geometry_overlay_v1",
        "overlay_type": "geometry192",
        "output_cache_root": str(args.output_cache_root),
        "output_root": str(output_root),
        "source_chunk_cache_root": str(args.chunk_cache_root),
        "vggt_model_path": str(args.vggt_model_path),
        "precision": args.precision,
        "geometry_teacher_dim": int(args.geometry_teacher_dim),
        "num_geometry_tokens": int(args.num_geometry_tokens),
        "geometry_grid": [int(args.geometry_grid_rows), int(args.geometry_grid_cols)],
        "require_full_geometry": bool(args.require_full_geometry),
        "allow_patch_fallback": bool(args.allow_patch_fallback),
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "num_written": int(counts["written"]),
        "sample_token_minimal_list_or_hash": sample_token_summary(index_records),
        **counts,
        "errors": errors[:100],
    }
    write_json(output_root / "metadata.json", metadata)
    if args.write_shard_manifest:
        write_json(output_root / "shard_manifest.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
