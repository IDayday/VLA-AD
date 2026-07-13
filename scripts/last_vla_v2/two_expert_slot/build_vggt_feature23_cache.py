#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import atomic_torch_save, iter_index, load_sample, write_json  # noqa: E402
from scripts.last_vla_v2.two_expert_slot.two_expert_cache_utils import (  # noqa: E402
    iter_indexed_records,
    normalize_merged_index_path,
)


PACKER_VERSION = "two_expert_vggt_feature23_packer_v1"
PRODUCTION_EXTRACTOR_IMPLEMENTED = True


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


def decode_path_tensor(path_tensor: torch.Tensor) -> str:
    chars = []
    for item in path_tensor.detach().cpu().view(-1):
        value = int(item.item())
        if value:
            chars.append(chr(value))
    return "".join(chars)


def sample_image_path(sample: Dict[str, Any], args: argparse.Namespace) -> Path:
    key = str(args.image_key)
    if key in sample:
        value = sample[key]
    elif "image_path_tensor" in sample:
        value = sample["image_path_tensor"]
    else:
        raise KeyError(f"Sample missing image path key {key!r} and image_path_tensor.")
    if isinstance(value, torch.Tensor):
        return Path(decode_path_tensor(value))
    return Path(str(value))


def load_vggt_image(path: Path, image_size: int) -> torch.Tensor:
    from PIL import Image

    image = Image.open(path).convert("RGB").resize((int(image_size), int(image_size)))
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes())).view(image_size, image_size, 3)
    return data.permute(2, 0, 1).float().div(255.0)


def _import_symbol(path: str):
    module_name, _, symbol = path.replace(":", ".").rpartition(".")
    if not module_name or not symbol:
        raise ValueError(f"Invalid import path {path!r}. Use module.submodule:ClassName or module.submodule.ClassName.")
    module = importlib.import_module(module_name)
    return getattr(module, symbol)


def load_vggt_model(args: argparse.Namespace) -> torch.nn.Module:
    class_candidates = [args.vggt_model_class] if args.vggt_model_class else [
        "vggt.models.vggt:VGGT",
        "vggt.VGGT",
    ]
    last_error: Optional[BaseException] = None
    for class_path in class_candidates:
        try:
            cls = _import_symbol(class_path)
            break
        except BaseException as exc:  # pragma: no cover - error reported by smoke in real envs
            last_error = exc
    else:
        raise ImportError(
            "Could not import a VGGT model class. Set --vggt-model-class, for example "
            "'vggt.models.vggt:VGGT'."
        ) from last_error

    if hasattr(cls, "from_pretrained"):
        try:
            model = cls.from_pretrained(str(args.vggt_model_path))
        except Exception:
            model = cls()
            state = torch.load(args.vggt_model_path, map_location="cpu")
            if isinstance(state, dict):
                state = state.get("state_dict") or state.get("model") or state
            model.load_state_dict(state, strict=False)
    else:
        model = cls()
        state = torch.load(args.vggt_model_path, map_location="cpu")
        if isinstance(state, dict):
            state = state.get("state_dict") or state.get("model") or state
        model.load_state_dict(state, strict=False)
    model.to(args.device).eval()
    return model


def discover_aggregator_layer(model: torch.nn.Module, layer_index: int = 23) -> Tuple[str, torch.nn.Module]:
    aggregator = getattr(model, "aggregator", None)
    global_blocks = getattr(aggregator, "global_blocks", None)
    if global_blocks is not None and 0 <= int(layer_index) < len(global_blocks):
        return f"aggregator.global_blocks.{int(layer_index)}", global_blocks[int(layer_index)]
    candidates = []
    for name, module in model.named_modules():
        lowered = name.lower()
        if "aggregator" in lowered or "agg" in lowered:
            candidates.append((name, module))
    if layer_index < 0 or layer_index >= len(candidates):
        raise LookupError(f"VGGT aggregator layer index {layer_index} not found; discovered {len(candidates)} candidates.")
    return candidates[layer_index]


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for item in value.values():
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    raise TypeError(f"Hook output does not contain a tensor: {type(value).__name__}")


def extract_feature23_tokens(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    layer_index: int,
    pack_tokens: int,
    device: str,
    precision: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    layer_name, layer = discover_aggregator_layer(model, layer_index)
    captured: Dict[str, torch.Tensor] = {}

    def hook(_, __, output):
        captured["feature"] = _first_tensor(output).detach()

    handle = layer.register_forward_hook(hook)
    try:
        images = image.unsqueeze(0).unsqueeze(0).to(device)
        dtype = torch.float32
        if precision == "bf16":
            dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
        device_type = "cuda" if str(device).startswith("cuda") else "cpu"
        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=dtype, enabled=device_type == "cuda" and dtype != torch.float32):
            _ = model(images)
    finally:
        handle.remove()
    if "feature" not in captured:
        raise RuntimeError(f"VGGT hook on {layer_name} did not capture a feature tensor.")
    feature = captured["feature"].detach().float().cpu()
    packed = pack_to_12_tokens(feature, key=f"vggt_feature_layer_{layer_index}") if int(pack_tokens) == 12 else pack_to_12_tokens(feature, key=f"vggt_feature_layer_{layer_index}")[: int(pack_tokens)]
    if int(pack_tokens) != 12:
        raise ValueError("two_expert_slot currently requires --pack-tokens 12.")
    metadata = {
        "teacher_type": "vggt_feature23",
        "teacher_source": "vggt_model_hook",
        "layer_index": int(layer_index),
        "layer_name": layer_name,
        "original_shape": list(feature.shape),
        "packed_shape": [12, int(packed.shape[-1])],
        "feature_dim": int(packed.shape[-1]),
        "strict_geometry_teacher": True,
        "packer_version": PACKER_VERSION,
    }
    return packed.contiguous(), metadata


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
    extractor_model = load_vggt_model(args) if args.vggt_model_path is not None else None
    for idx, (_, sample_path, record) in enumerate(
        iter_indexed_records(args.input_chunk_root, pattern=args.chunk_name_pattern, max_records=args.max_samples)
    ):
        if idx % int(args.num_shards) != int(args.shard_index):
            skipped += 1
            continue
        sample = load_sample(sample_path)
        token = str(sample.get("sample_token") or record.get("sample_token") or sample_path.stem)
        scene_token = str(sample.get("scene_token") or record.get("scene_token") or "")
        log_name = sample.get("log_name") or record.get("log_name")
        if extractor_model is not None:
            image = load_vggt_image(sample_image_path(sample, args), int(args.image_size))
            tokens, metadata = extract_feature23_tokens(
                extractor_model,
                image,
                layer_index=int(args.feature_layer_index),
                pack_tokens=int(args.pack_tokens),
                device=str(args.device),
                precision=str(args.precision),
            )
        else:
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
        "builder_mode": "vggt_model_feature23_extraction" if extractor_model is not None else "existing_feature_repack",
        "production_extractor_implemented": bool(extractor_model is not None),
        "route_status": "production_teacher_ok" if fallback_count == 0 else "dev_fallback_not_for_final",
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
            rows.append(normalize_merged_index_path(output_root, shard_dir, row))
    if duplicate_tokens:
        raise ValueError(f"Duplicate sample_token values across shards: {duplicate_tokens[:5]}")
    output_root.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    metadata = {
        "version": "two_expert_vggt_feature23_teacher_cache_v1",
        "merged": True,
        "builder_mode": "mixed"
        if len({str(item.get("builder_mode", "existing_feature_repack")) for item in metadata_items}) > 1
        else (str(metadata_items[0].get("builder_mode", "existing_feature_repack")) if metadata_items else "unknown"),
        "production_extractor_implemented": any(
            bool(item.get("production_extractor_implemented", False)) for item in metadata_items
        ),
        "route_status": "production_teacher_ok"
        if sum(int(item.get("num_legacy_fallback", 0)) for item in metadata_items) == 0
        else "dev_fallback_not_for_final",
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
    parser.add_argument("--vggt-model-class", default=os.getenv("VGGT_MODEL_CLASS"))
    parser.add_argument("--image-key", default="image_path_tensor")
    parser.add_argument("--camera", default="front")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--feature-layer-index", type=int, default=23)
    parser.add_argument("--pack-tokens", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
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
