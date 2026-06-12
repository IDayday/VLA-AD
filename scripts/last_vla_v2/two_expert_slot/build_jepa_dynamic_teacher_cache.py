#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from transformers import AutoModel, AutoVideoProcessor

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import atomic_torch_save, iter_index, load_sample, write_json  # noqa: E402
from scripts.last_vla_v2.two_expert_slot.two_expert_cache_utils import (  # noqa: E402
    iter_indexed_records,
    normalize_merged_index_path,
)


PACKER_VERSION = "two_expert_jepa_dynamic_packer_v1"
PRODUCTION_EXTRACTOR_IMPLEMENTED = True


def decode_path_tensor(path_tensor: torch.Tensor) -> str:
    chars = []
    for item in path_tensor.detach().cpu().view(-1):
        value = int(item.item())
        if value:
            chars.append(chr(value))
    return "".join(chars)


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


def pack_vjepa_hidden_to_dynamic_tokens(
    hidden: torch.Tensor,
    *,
    num_dyn_groups: int = 3,
    tokens_per_group: int = 12,
    frames_per_clip: int = 64,
    tubelet_size: int = 2,
    spatial_tokens: int = 256,
) -> torch.Tensor:
    hidden = _as_feature_tensor(hidden, "vjepa2_last_hidden_state")
    if hidden.ndim == 3:
        if hidden.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 V-JEPA hidden state, got {tuple(hidden.shape)}.")
        hidden = hidden[0]
    if hidden.ndim != 2 or hidden.shape[-1] != 1024:
        raise ValueError(f"V-JEPA hidden state must have shape [N,1024], got {tuple(hidden.shape)}.")
    temporal_tokens = int(frames_per_clip) // int(tubelet_size)
    expected_tokens = temporal_tokens * int(spatial_tokens)
    if hidden.shape[0] < expected_tokens:
        raise ValueError(
            f"V-JEPA hidden state has too few tokens: {hidden.shape[0]} < expected {expected_tokens} "
            f"({temporal_tokens} temporal x {spatial_tokens} spatial)."
        )
    hidden = hidden[:expected_tokens].reshape(temporal_tokens, int(spatial_tokens), 1024)
    groups: List[torch.Tensor] = []
    for group_idx in range(int(num_dyn_groups)):
        t_start = int(round(group_idx * temporal_tokens / int(num_dyn_groups)))
        t_end = int(round((group_idx + 1) * temporal_tokens / int(num_dyn_groups)))
        t_end = max(t_start + 1, min(t_end, temporal_tokens))
        group_tokens = hidden[t_start:t_end].reshape(-1, 1024)
        indices = torch.linspace(0, group_tokens.shape[0] - 1, steps=int(tokens_per_group)).round().long()
        groups.append(group_tokens.index_select(0, indices))
    return torch.stack(groups, dim=0).contiguous()


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


def load_vjepa2(args: argparse.Namespace):
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[str(args.precision)]
    model = AutoModel.from_pretrained(
        args.jepa_model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).eval()
    model.to(torch.device(args.device))
    processor = AutoVideoProcessor.from_pretrained(
        args.jepa_model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    return model, processor


def load_rgb(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Image path does not exist: {path}")
    return Image.open(path).convert("RGB")


def build_image_path_index(args: argparse.Namespace) -> Dict[str, Any]:
    if args.image_path_index_jsonl is None:
        raise ValueError("--image-path-index-jsonl is required with --build-image-path-index.")
    if args.image_path_index_jsonl.exists() and not args.overwrite:
        raise FileExistsError(f"Image path index exists: {args.image_path_index_jsonl}. Pass --overwrite to replace it.")
    records = list(iter_indexed_records(args.input_chunk_root, pattern=args.chunk_name_pattern, max_records=args.max_samples))
    args.image_path_index_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.image_path_index_jsonl.open("w", encoding="utf-8") as f:
        for order_index, (_, sample_path, record) in enumerate(records):
            sample = load_sample(sample_path)
            if "image_path_tensor" not in sample:
                raise KeyError(f"Sample {sample_path} missing image_path_tensor required for V-JEPA2 extraction.")
            token = str(sample.get("sample_token") or record.get("sample_token") or sample_path.stem)
            row = {
                "order_index": int(order_index),
                "sample_token": token,
                "sample_path": str(sample_path),
                "image_path": decode_path_tensor(sample["image_path_tensor"]),
                "log_name": str(sample.get("log_name") or record.get("log_name") or ""),
                "scene_token": str(sample.get("scene_token") or record.get("scene_token") or ""),
            }
            f.write(json.dumps(row, sort_keys=True) + "\n")
            written += 1
    metadata = {
        "version": "two_expert_jepa_image_path_index_v1",
        "source_chunk_root": str(args.input_chunk_root),
        "chunk_name_pattern": str(args.chunk_name_pattern),
        "num_records": int(written),
        "path": str(args.image_path_index_jsonl),
    }
    write_json(args.image_path_index_jsonl.with_suffix(".metadata.json"), metadata)
    return metadata


def load_image_path_index(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            token = str(row.get("sample_token") or "")
            image_path = str(row.get("image_path") or "")
            if token and image_path:
                mapping[token] = image_path
    return mapping


def _sample_sequence_indices(length: int, out_len: int) -> List[int]:
    if length <= 0:
        raise ValueError("Cannot sample an empty image sequence.")
    if length == 1:
        return [0] * int(out_len)
    return torch.linspace(0, length - 1, steps=int(out_len)).round().long().tolist()


def _future_record_indices(
    records: Sequence[Tuple[Path, Path, Dict[str, Any]]],
    current_index: int,
    sequence_sample_count: int,
) -> List[int]:
    current = records[current_index][2]
    current_log = str(current.get("log_name") or "")
    indices = [current_index]
    cursor = current_index + 1
    while cursor < len(records) and len(indices) < int(sequence_sample_count):
        record = records[cursor][2]
        if current_log and str(record.get("log_name") or "") != current_log:
            break
        indices.append(cursor)
        cursor += 1
    while len(indices) < int(sequence_sample_count):
        indices.append(indices[-1])
    return indices


def _sample_image_path(sample_path: Path, record: Dict[str, Any], image_path_index: Dict[str, str]) -> Path:
    token = str(record.get("sample_token") or sample_path.stem)
    if token in image_path_index:
        return Path(image_path_index[token])
    sample = load_sample(sample_path)
    if "image_path_tensor" not in sample:
        raise KeyError(f"Sample {sample_path} missing image_path_tensor required for V-JEPA2 extraction.")
    return Path(decode_path_tensor(sample["image_path_tensor"]))


def extract_vjepa2_dynamic_teacher(
    model,
    processor,
    records: Sequence[Tuple[Path, Path, Dict[str, Any]]],
    current_index: int,
    args: argparse.Namespace,
    image_path_index: Dict[str, str],
) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
    source_indices = _future_record_indices(records, current_index, int(args.sequence_sample_count))
    source_image_paths = [
        _sample_image_path(records[index][1], records[index][2], image_path_index) for index in source_indices
    ]
    frame_indices = _sample_sequence_indices(len(source_image_paths), int(args.frames_per_clip))
    frames = [load_rgb(source_image_paths[index]) for index in frame_indices]
    inputs = processor(frames, return_tensors="pt")
    pixel_values = inputs["pixel_values_videos"].to(device=torch.device(args.device))
    if str(args.precision) == "bf16":
        pixel_values = pixel_values.to(dtype=torch.bfloat16)
    elif str(args.precision) == "fp16":
        pixel_values = pixel_values.to(dtype=torch.float16)
    else:
        pixel_values = pixel_values.to(dtype=torch.float32)
    with torch.no_grad():
        output = model(pixel_values_videos=pixel_values, skip_predictor=bool(args.skip_predictor))
    hidden = output.last_hidden_state.detach().float().cpu()
    tokens = pack_vjepa_hidden_to_dynamic_tokens(
        hidden,
        frames_per_clip=int(args.frames_per_clip),
        tubelet_size=int(args.tubelet_size),
        spatial_tokens=int(args.spatial_tokens),
    )
    metadata = {
        "teacher_type": "jepa_dynamic",
        "teacher_source": "vjepa2_index_order_future_sequence",
        "strict_dynamic_teacher": True,
        "num_dyn_groups": 3,
        "tokens_per_group": 12,
        "feature_dim": 1024,
        "horizon_definition": "short_mid_long_from_base_index_order_future_front_camera_frames",
        "vjepa_model_path": str(args.jepa_model_path),
        "frames_per_clip": int(args.frames_per_clip),
        "sequence_sample_count": int(args.sequence_sample_count),
        "tubelet_size": int(args.tubelet_size),
        "spatial_tokens": int(args.spatial_tokens),
        "packer_version": PACKER_VERSION,
    }
    diagnostics = {
        "source_image_paths": [str(path) for path in source_image_paths],
        "frame_indices": [int(item) for item in frame_indices],
        "vjepa_last_hidden_shape": list(hidden.shape),
    }
    return tokens, metadata, diagnostics


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
    records = list(iter_indexed_records(args.input_chunk_root, pattern=args.chunk_name_pattern, max_records=args.max_samples))
    image_path_index = load_image_path_index(args.image_path_index_jsonl)
    model = processor = None
    if args.jepa_model_path is not None:
        model, processor = load_vjepa2(args)
    rows: List[Dict[str, Any]] = []
    written = skipped = fallback_count = strict_count = 0
    for idx, (_, sample_path, record) in enumerate(records):
        if idx % int(args.num_shards) != int(args.shard_index):
            skipped += 1
            continue
        sample = {} if model is not None else load_sample(sample_path)
        token = str(sample.get("sample_token") or record.get("sample_token") or sample_path.stem)
        scene_token = str(sample.get("scene_token") or record.get("scene_token") or "")
        log_name = sample.get("log_name") or record.get("log_name")
        if model is not None and processor is not None:
            tokens, metadata, diagnostics = extract_vjepa2_dynamic_teacher(model, processor, records, idx, args, image_path_index)
        else:
            tokens, metadata = resolve_dynamic_teacher(sample, strict_teacher=bool(args.strict_teacher))
            diagnostics = {}
        strict_count += int(bool(metadata["strict_dynamic_teacher"]))
        fallback_count += int(not bool(metadata["strict_dynamic_teacher"]))
        payload: Dict[str, Any] = {
            "sample_token": token,
            "jepa_dynamic_teacher_tokens": tokens,
            "jepa_dynamic_teacher_metadata": metadata,
        }
        if diagnostics:
            payload["jepa_dynamic_teacher_diagnostics"] = diagnostics
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
        "builder_mode": "vjepa2_model_extraction" if args.jepa_model_path is not None else "existing_feature_repack",
        "production_extractor_implemented": PRODUCTION_EXTRACTOR_IMPLEMENTED,
        "route_status": "production_teacher_ok" if fallback_count == 0 else "dev_fallback_not_for_final",
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
        "jepa_model_path": str(args.jepa_model_path) if args.jepa_model_path is not None else None,
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
            rows.append(normalize_merged_index_path(output_root, shard_dir, row))
    if duplicate_tokens:
        raise ValueError(f"Duplicate sample_token values across shards: {duplicate_tokens[:5]}")
    output_root.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    builder_modes = sorted({str(item.get("builder_mode", "unknown")) for item in metadata_items})
    jepa_model_paths = sorted(
        {str(item.get("jepa_model_path")) for item in metadata_items if item.get("jepa_model_path") is not None}
    )
    metadata = {
        "version": "two_expert_jepa_dynamic_teacher_cache_v1",
        "merged": True,
        "builder_mode": builder_modes[0] if len(builder_modes) == 1 else "mixed",
        "builder_modes": builder_modes,
        "production_extractor_implemented": any(bool(item.get("production_extractor_implemented")) for item in metadata_items),
        "route_status": "production_teacher_ok"
        if sum(int(item.get("num_legacy_fallback", 0)) for item in metadata_items) == 0
        else "dev_fallback_not_for_final",
        "num_samples": len(rows),
        "num_shards": len(metadata_items),
        "num_legacy_fallback": sum(int(item.get("num_legacy_fallback", 0)) for item in metadata_items),
        "num_strict_teacher": sum(int(item.get("num_strict_teacher", 0)) for item in metadata_items),
        "jepa_model_paths": jepa_model_paths,
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--frames-per-clip", type=int, default=64)
    parser.add_argument("--sequence-sample-count", type=int, default=8)
    parser.add_argument("--tubelet-size", type=int, default=2)
    parser.add_argument("--spatial-tokens", type=int, default=256)
    parser.add_argument("--skip-predictor", action="store_true")
    parser.add_argument("--image-path-index-jsonl", type=Path, default=None)
    parser.add_argument("--build-image-path-index", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.build_image_path_index:
        print(json.dumps(build_image_path_index(args), indent=2, sort_keys=True))
        return 0
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
