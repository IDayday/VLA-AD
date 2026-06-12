#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import atomic_torch_save, iter_index, load_sample, write_json  # noqa: E402


PACKER_VERSION = "two_expert_vggt_feature23_packer_v1"
PRODUCTION_EXTRACTOR_IMPLEMENTED = False


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
        raise FileNotFoundError(f"No indexed chunk dirs matching {pattern!r} under {root}")
    return dirs


def iter_samples(root: Path, pattern: str, max_samples: Optional[int]) -> Iterable[Tuple[Path, Path, Dict[str, Any]]]:
    count = 0
    for chunk_dir in chunk_dirs(root, pattern):
        for record in iter_index(chunk_dir):
            sample_path = Path(record["path"])
            yield chunk_dir, sample_path, record
            count += 1
            if max_samples is not None and count >= max_samples:
                return


def _as_feature_tensor(value: Any, key: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{key} must be a torch.Tensor, got {type(value).__name__}.")
    value = value.detach().float().cpu()
    if not torch.isfinite(value).all():
        raise ValueError(f"{key} contains non-finite values.")
    return value


def pack_to_12_tokens(tokens: torch.Tensor, key: str = "vggt_geometry_tokens") -> torch.Tensor:
    tokens = _as_feature_tensor(tokens, key)
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim > 2:
        tokens = tokens.reshape(-1, tokens.shape[-1])
    if tokens.ndim != 2:
        raise ValueError(f"{key} must be flattenable to [N,D], got {tuple(tokens.shape)}.")
    if tokens.shape[0] == 12:
        return tokens.contiguous()
    if tokens.shape[0] < 12:
        repeat = (12 + tokens.shape[0] - 1) // tokens.shape[0]
        tokens = tokens.repeat(repeat, 1)
    indices = torch.linspace(0, tokens.shape[0] - 1, steps=12).round().long()
    return tokens.index_select(0, indices).contiguous()


def discover_aggregator_layer(model: torch.nn.Module, layer_index: int = 23) -> Tuple[str, torch.nn.Module]:
    candidates = []
    for name, module in model.named_modules():
        lowered = name.lower()
        if "aggregator" in lowered or "agg" in lowered:
            candidates.append((name, module))
    if layer_index < 0 or layer_index >= len(candidates):
        raise LookupError(f"VGGT aggregator layer index {layer_index} not found; discovered {len(candidates)} candidates.")
    return candidates[layer_index]


def resolve_feature23_teacher(sample: Dict[str, Any], *, strict_teacher: bool) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if "vggt_feature23_tokens" in sample:
        tokens = _as_feature_tensor(sample["vggt_feature23_tokens"], "vggt_feature23_tokens")
        if tokens.ndim != 2 or tokens.shape[0] != 12:
            raise ValueError(f"vggt_feature23_tokens must have shape [12,D], got {tuple(tokens.shape)}.")
        return tokens.contiguous(), {
            "teacher_type": "vggt_feature23",
            "layer_index": 23,
            "layer_name": sample.get("vggt_feature23_layer_name", "provided_by_cache"),
            "original_shape": list(tokens.shape),
            "packed_shape": [12, int(tokens.shape[-1])],
            "feature_dim": int(tokens.shape[-1]),
            "strict_geometry_teacher": True,
            "packer_version": PACKER_VERSION,
        }
    if strict_teacher:
        raise KeyError("STRICT_TEACHER requires vggt_feature23_tokens; old vggt_geometry_tokens fallback is disabled.")
    if "vggt_geometry_tokens" not in sample:
        raise KeyError("No VGGT Feature(23) teacher found. Expected vggt_feature23_tokens or old vggt_geometry_tokens fallback.")
    original = _as_feature_tensor(sample["vggt_geometry_tokens"], "vggt_geometry_tokens")
    tokens = pack_to_12_tokens(original, "vggt_geometry_tokens")
    return tokens, {
        "teacher_type": "vggt_feature23",
        "layer_index": 23,
        "layer_name": "legacy_vggt_geometry_tokens_fallback",
        "original_shape": list(original.shape),
        "packed_shape": [12, int(tokens.shape[-1])],
        "feature_dim": int(tokens.shape[-1]),
        "strict_geometry_teacher": False,
        "packer_version": PACKER_VERSION,
    }


def output_dir_for_shard(output_root: Path, shard_index: int) -> Path:
    return output_root / "shards" / f"shard_{int(shard_index):05d}"


def build_shard(args: argparse.Namespace) -> Dict[str, Any]:
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards).")
    out_dir = output_dir_for_shard(args.output_root, args.shard_index)
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output shard exists: {out_dir}. Pass --overwrite to replace it.")
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    written = skipped = fallback_count = strict_count = 0
    feature_dims = set()
    for idx, (_, sample_path, record) in enumerate(iter_samples(args.input_chunk_root, args.chunk_name_pattern, args.max_samples)):
        if idx % int(args.num_shards) != int(args.shard_index):
            skipped += 1
            continue
        sample = load_sample(sample_path)
        token = str(sample.get("sample_token") or record.get("sample_token") or sample_path.stem)
        scene_token = str(sample.get("scene_token") or record.get("scene_token") or "")
        log_name = sample.get("log_name") or record.get("log_name")
        tokens, metadata = resolve_feature23_teacher(sample, strict_teacher=bool(args.strict_teacher))
        feature_dims.add(int(tokens.shape[-1]))
        strict_count += int(bool(metadata["strict_geometry_teacher"]))
        fallback_count += int(not bool(metadata["strict_geometry_teacher"]))
        payload: Dict[str, Any] = {
            "sample_token": token,
            "vggt_feature23_tokens": tokens,
            "vggt_feature23_metadata": metadata,
        }
        if scene_token:
            payload["scene_token"] = scene_token
        if log_name:
            payload["log_name"] = str(log_name)
        out_path = samples_dir / f"{token}.pt"
        atomic_torch_save(payload, out_path)
        row = {"sample_token": token, "path": str(out_path.relative_to(out_dir))}
        if scene_token:
            row["scene_token"] = scene_token
        if log_name:
            row["log_name"] = str(log_name)
        rows.append(row)
        written += 1
    with (out_dir / "index.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    metadata = {
        "version": "two_expert_vggt_feature23_teacher_cache_v1",
        "builder_mode": "existing_feature_repack",
        "production_extractor_implemented": PRODUCTION_EXTRACTOR_IMPLEMENTED,
        "split": str(args.split),
        "source_chunk_root": str(args.input_chunk_root),
        "teacher_type": "vggt_feature23",
        "layer_index": 23,
        "num_geo_tokens": 12,
        "feature_dims": sorted(feature_dims),
        "strict_teacher_requested": bool(args.strict_teacher),
        "num_samples": int(written),
        "num_strict_teacher": int(strict_count),
        "num_legacy_fallback": int(fallback_count),
        "num_skipped_by_shard": int(skipped),
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "packer_version": PACKER_VERSION,
    }
    write_json(out_dir / "metadata.json", metadata)
    return metadata


def merge_shards(output_root: Path, *, overwrite: bool = False) -> Dict[str, Any]:
    shards_root = output_root / "shards"
    if not shards_root.is_dir():
        raise FileNotFoundError(f"No shards directory found: {shards_root}")
    index_path = output_root / "index.jsonl"
    if index_path.exists() and not overwrite:
        raise FileExistsError(f"Merged index exists: {index_path}. Pass --overwrite to replace it.")
    rows: List[Dict[str, Any]] = []
    tokens = set()
    duplicate_tokens = []
    metadata_items = []
    feature_dims = set()
    for shard_dir in sorted(shards_root.glob("shard_*")):
        if not (shard_dir / "index.jsonl").is_file():
            continue
        if (shard_dir / "metadata.json").is_file():
            meta = json.loads((shard_dir / "metadata.json").read_text(encoding="utf-8"))
            metadata_items.append(meta)
            feature_dims.update(int(dim) for dim in meta.get("feature_dims", []))
        for row in iter_index(shard_dir):
            token = str(row.get("sample_token") or Path(row.get("path", "")).stem)
            if token in tokens:
                duplicate_tokens.append(token)
            tokens.add(token)
            path = Path(row["path"])
            if path.is_absolute():
                row["path"] = str(path.relative_to(output_root))
            else:
                row["path"] = str((shard_dir / path).relative_to(output_root))
            rows.append(row)
    if duplicate_tokens:
        raise ValueError(f"Duplicate sample_token values across shards: {duplicate_tokens[:5]}")
    output_root.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    metadata = {
        "version": "two_expert_vggt_feature23_teacher_cache_v1",
        "merged": True,
        "builder_mode": "existing_feature_repack",
        "production_extractor_implemented": PRODUCTION_EXTRACTOR_IMPLEMENTED,
        "num_samples": len(rows),
        "num_shards": len(metadata_items),
        "num_legacy_fallback": sum(int(item.get("num_legacy_fallback", 0)) for item in metadata_items),
        "num_strict_teacher": sum(int(item.get("num_strict_teacher", 0)) for item in metadata_items),
        "feature_dims": sorted(feature_dims),
        "packer_version": PACKER_VERSION,
    }
    write_json(output_root / "metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build [12,D] VGGT Feature(23) teacher cache for two_expert_slot.")
    parser.add_argument("--input-chunk-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vggt-model-path", type=Path, default=None)
    parser.add_argument("--split", default="navtrain")
    parser.add_argument("--chunk-name-pattern", default="train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*,navtest_full_chunk_*")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--strict-teacher", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--merge", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.vggt_model_path is not None:
        raise NotImplementedError(
            "build_vggt_feature23_cache.py does not extract VGGT Feature(23) from --vggt-model-path yet. "
            "It only validates/reuses existing vggt_feature23_tokens or repacks legacy vggt_geometry_tokens "
            "when strict mode is disabled. Use a strict cache with existing production tokens, or implement the model hook."
        )
    if args.merge:
        print(json.dumps(merge_shards(args.output_root, overwrite=args.overwrite), indent=2, sort_keys=True))
        return 0
    if os.getenv("RUN_CACHE", "0") != "1":
        payload = {
            "status": "blocked_by_RUN_CACHE_gate",
            "builder_mode": "existing_feature_repack",
            "production_extractor_implemented": PRODUCTION_EXTRACTOR_IMPLEMENTED,
            "message": "Set RUN_CACHE=1 to build VGGT Feature(23) teacher cache.",
            "output_root": str(args.output_root),
            "split": str(args.split),
        }
        args.output_root.mkdir(parents=True, exist_ok=True)
        write_json(args.output_root / "vggt_feature23_cache_dry_run.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(json.dumps(build_shard(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
