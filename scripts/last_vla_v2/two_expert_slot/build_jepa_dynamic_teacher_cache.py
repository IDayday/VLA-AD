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


PACKER_VERSION = "two_expert_jepa_dynamic_packer_v1"
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


def pack_legacy_jepa_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """Deterministically pack legacy [N,1024] JEPA tokens into [3,12,1024]."""
    tokens = _as_feature_tensor(tokens, "jepa_target_tokens")
    if tokens.ndim == 3 and tuple(tokens.shape[:2]) == (3, 12) and tokens.shape[-1] == 1024:
        return tokens.contiguous()
    if tokens.ndim != 2 or tokens.shape[-1] != 1024:
        raise ValueError(f"legacy JEPA tokens must have shape [N,1024], got {tuple(tokens.shape)}.")
    if tokens.shape[0] < 36:
        raise ValueError(f"Need at least 36 legacy JEPA tokens to pack [3,12,1024], got {tokens.shape[0]}.")
    indices = torch.linspace(0, tokens.shape[0] - 1, steps=36).round().long()
    return tokens.index_select(0, indices).reshape(3, 12, 1024).contiguous()


def resolve_dynamic_teacher(sample: Dict[str, Any], *, strict_teacher: bool) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if "jepa_dynamic_teacher_tokens" in sample:
        tokens = _as_feature_tensor(sample["jepa_dynamic_teacher_tokens"], "jepa_dynamic_teacher_tokens")
        if tuple(tokens.shape) != (3, 12, 1024):
            raise ValueError(f"jepa_dynamic_teacher_tokens shape {tuple(tokens.shape)} != (3, 12, 1024).")
        return tokens.contiguous(), {
            "teacher_type": "jepa_dynamic",
            "teacher_source": "multi_horizon_jepa",
            "strict_dynamic_teacher": True,
            "num_dyn_groups": 3,
            "tokens_per_group": 12,
            "feature_dim": 1024,
            "horizon_definition": "provided_by_cache",
            "packer_version": PACKER_VERSION,
        }
    for key in ("multi_horizon_jepa_tokens", "future_multi_horizon_jepa_tokens"):
        if key in sample:
            tokens = _as_feature_tensor(sample[key], key)
            if tuple(tokens.shape) != (3, 12, 1024):
                raise ValueError(f"{key} shape {tuple(tokens.shape)} != (3, 12, 1024).")
            return tokens.contiguous(), {
                "teacher_type": "jepa_dynamic",
                "teacher_source": "multi_horizon_jepa",
                "strict_dynamic_teacher": True,
                "num_dyn_groups": 3,
                "tokens_per_group": 12,
                "feature_dim": 1024,
                "horizon_definition": "short_mid_long",
                "packer_version": PACKER_VERSION,
            }
    if strict_teacher:
        raise KeyError(
            "STRICT_TEACHER requires jepa_dynamic_teacher_tokens or multi_horizon_jepa_tokens; "
            "legacy jepa_target_tokens fallback is disabled."
        )
    if "jepa_target_tokens" not in sample:
        raise KeyError("No JEPA dynamic teacher found. Expected jepa_dynamic_teacher_tokens or legacy jepa_target_tokens.")
    tokens = pack_legacy_jepa_tokens(sample["jepa_target_tokens"])
    return tokens, {
        "teacher_type": "jepa_dynamic",
        "teacher_source": "legacy_jepa_target_downsampled",
        "strict_dynamic_teacher": False,
        "num_dyn_groups": 3,
        "tokens_per_group": 12,
        "feature_dim": 1024,
        "horizon_definition": "deterministic_linspace_from_legacy_target_tokens",
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
    for idx, (_, sample_path, record) in enumerate(iter_samples(args.input_chunk_root, args.chunk_name_pattern, args.max_samples)):
        if idx % int(args.num_shards) != int(args.shard_index):
            skipped += 1
            continue
        sample = load_sample(sample_path)
        token = str(sample.get("sample_token") or record.get("sample_token") or sample_path.stem)
        scene_token = str(sample.get("scene_token") or record.get("scene_token") or "")
        log_name = sample.get("log_name") or record.get("log_name")
        tokens, metadata = resolve_dynamic_teacher(sample, strict_teacher=bool(args.strict_teacher))
        strict_count += int(bool(metadata["strict_dynamic_teacher"]))
        fallback_count += int(not bool(metadata["strict_dynamic_teacher"]))
        payload: Dict[str, Any] = {
            "sample_token": token,
            "jepa_dynamic_teacher_tokens": tokens,
            "jepa_dynamic_teacher_metadata": metadata,
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
        "version": "two_expert_jepa_dynamic_teacher_cache_v1",
        "builder_mode": "existing_feature_repack",
        "production_extractor_implemented": PRODUCTION_EXTRACTOR_IMPLEMENTED,
        "split": str(args.split),
        "source_chunk_root": str(args.input_chunk_root),
        "teacher_type": "jepa_dynamic",
        "num_dyn_groups": 3,
        "tokens_per_group": 12,
        "feature_dim": 1024,
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
    for shard_dir in sorted(shards_root.glob("shard_*")):
        if not (shard_dir / "index.jsonl").is_file():
            continue
        if (shard_dir / "metadata.json").is_file():
            metadata_items.append(json.loads((shard_dir / "metadata.json").read_text(encoding="utf-8")))
        for row in iter_index(shard_dir):
            token = str(row.get("sample_token") or Path(row.get("path", "")).stem)
            if token in tokens:
                duplicate_tokens.append(token)
            tokens.add(token)
            path = Path(row["path"])
            rel_path = path if not path.is_absolute() else path.relative_to(output_root)
            if path.is_absolute():
                row["path"] = str(rel_path)
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
        "version": "two_expert_jepa_dynamic_teacher_cache_v1",
        "merged": True,
        "builder_mode": "existing_feature_repack",
        "production_extractor_implemented": PRODUCTION_EXTRACTOR_IMPLEMENTED,
        "num_samples": len(rows),
        "num_shards": len(metadata_items),
        "num_legacy_fallback": sum(int(item.get("num_legacy_fallback", 0)) for item in metadata_items),
        "num_strict_teacher": sum(int(item.get("num_strict_teacher", 0)) for item in metadata_items),
        "packer_version": PACKER_VERSION,
    }
    write_json(output_root / "metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build [3,12,1024] JEPA dynamic teacher cache for two_expert_slot.")
    parser.add_argument("--input-chunk-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jepa-model-path", type=Path, default=None)
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
    if args.jepa_model_path is not None:
        raise NotImplementedError(
            "build_jepa_dynamic_teacher_cache.py does not extract JEPA features from --jepa-model-path yet. "
            "It only repacks existing jepa_dynamic_teacher_tokens/multi_horizon_jepa_tokens or legacy "
            "jepa_target_tokens. Use a strict cache with existing production tokens, or implement the model extractor."
        )
    if args.merge:
        print(json.dumps(merge_shards(args.output_root, overwrite=args.overwrite), indent=2, sort_keys=True))
        return 0
    if os.getenv("RUN_CACHE", "0") != "1":
        payload = {
            "status": "blocked_by_RUN_CACHE_gate",
            "builder_mode": "existing_feature_repack",
            "production_extractor_implemented": PRODUCTION_EXTRACTOR_IMPLEMENTED,
            "message": "Set RUN_CACHE=1 to build JEPA dynamic teacher cache.",
            "output_root": str(args.output_root),
            "split": str(args.split),
        }
        args.output_root.mkdir(parents=True, exist_ok=True)
        write_json(args.output_root / "jepa_dynamic_teacher_cache_dry_run.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(json.dumps(build_shard(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
