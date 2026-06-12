#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from navsim.agents.recogdrive.two_expert_slots import TwoExpertSlotConfig, TwoExpertSoftSlots  # noqa: E402
from scripts.last_vla_v2.two_expert_slot.two_expert_prompt_utils import (  # noqa: E402
    TWO_EXPERT_PROMPT_VERSION,
    build_two_expert_prompt,
)
from scripts.last_vla_v2.two_expert_slot.two_expert_cache_utils import (  # noqa: E402
    iter_indexed_records,
    normalize_merged_index_path,
)


PRESERVE_KEYS = (
    "history_trajectory",
    "high_command_one_hot",
    "status_feature",
    "trajectory",
    "image_path_tensor",
    "sample_token",
    "scene_token",
    "log_name",
)
TRAIN_TEACHER_KEYS = (
    "jepa_dynamic_teacher_tokens",
    "vggt_feature23_tokens",
)

def decode_path_tensor(path_tensor: torch.Tensor) -> str:
    if not isinstance(path_tensor, torch.Tensor):
        raise TypeError("image_path_tensor must be a torch.Tensor.")
    chars = []
    for item in path_tensor.detach().cpu().view(-1):
        value = int(item.item())
        if value == 0:
            continue
        chars.append(chr(value))
    return "".join(chars)


def _load_checkpoint_payload(path: Path) -> Any:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    return payload


def _state_dict_from_payload(payload: Any, path: Path) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "full_state_dict" in payload and isinstance(payload["full_state_dict"], dict):
        payload = payload["full_state_dict"]
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint {path} must contain a state_dict.")
    return {str(key): value for key, value in payload.items() if isinstance(value, torch.Tensor)}


def _load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    return _state_dict_from_payload(_load_checkpoint_payload(path), path)


def _metadata_from_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    metadata: Dict[str, Any] = {}
    for key in ("two_expert_metadata", "stage1_metadata", "metadata", "hyper_parameters"):
        value = payload.get(key)
        if isinstance(value, dict):
            metadata.update(value)
    cfg = payload.get("config")
    if isinstance(cfg, dict):
        metadata.update({f"config_{key}": value for key, value in cfg.items() if isinstance(key, str)})
    if "checkpoint_schema" in payload:
        metadata["checkpoint_schema"] = payload["checkpoint_schema"]
    return metadata


def _slot_key_from_checkpoint_key(key: str) -> Optional[str]:
    key = key.removeprefix("module.").removeprefix("model.").removeprefix("agent.")
    markers = ("two_expert_slots.", "slots.")
    for marker in markers:
        if marker in key:
            return key.split(marker, 1)[1]
    return key if key in {"dyn_slots", "geo_slots", "dyn_group_embeddings", "geo_type_embedding"} else None


def load_two_expert_slots_with_report(
    checkpoint: Optional[Path],
    config: TwoExpertSlotConfig,
) -> Tuple[TwoExpertSoftSlots, Dict[str, Any]]:
    slots = TwoExpertSoftSlots(config)
    if checkpoint is None:
        return slots, {"loaded_slot_keys": [], "skipped_slot_keys": [], "source_stage1_checkpoint": None}
    payload = _load_checkpoint_payload(checkpoint)
    if isinstance(payload, dict) and isinstance(payload.get("two_expert_slots"), dict):
        state = {str(key): value for key, value in payload["two_expert_slots"].items() if isinstance(value, torch.Tensor)}
    else:
        state = _state_dict_from_payload(payload, checkpoint)
    expected = slots.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for key, value in state.items():
        mapped = _slot_key_from_checkpoint_key(key)
        if mapped is not None and mapped in expected and tuple(expected[mapped].shape) == tuple(value.shape):
            filtered[mapped] = value
        elif mapped is not None:
            skipped.append(key)
    required = {"dyn_slots", "geo_slots"}
    missing = sorted(required.difference(filtered))
    if missing:
        raise RuntimeError(f"Stage1 checkpoint {checkpoint} is missing compatible two-expert slot tensors: {missing}")
    slots.load_state_dict(filtered, strict=False)
    slots.eval()
    if skipped:
        print(f"Skipped {len(skipped)} incompatible two-expert slot tensors.", flush=True)
    return slots, {
        "loaded_slot_keys": sorted(filtered),
        "skipped_slot_keys": skipped,
        "source_stage1_checkpoint": str(checkpoint),
    }


def load_two_expert_slots(checkpoint: Optional[Path], config: TwoExpertSlotConfig) -> TwoExpertSoftSlots:
    return load_two_expert_slots_with_report(checkpoint, config)[0]


def load_backbone(args: argparse.Namespace):
    from navsim.agents.recogdrive.recogdrive_backbone import RecogDriveBackbone

    backbone = RecogDriveBackbone(model_type=args.vlm_type, checkpoint_path=str(args.vlm_path), device=args.device)
    backbone.eval()
    return backbone


def _resolve_stage1_train_mode(payload_metadata: Dict[str, Any], args: argparse.Namespace) -> str:
    for key in ("stage1_train_mode", "train_mode", "two_expert_train_mode", "config_train_mode"):
        value = payload_metadata.get(key)
        if value in {"frozen", "lora", "top_layers", "full"}:
            return str(value)
    if args.stage1_train_mode:
        return str(args.stage1_train_mode)
    return str(args.train_vlm_mode)


def _backbone_key_from_checkpoint_key(key: str) -> Optional[str]:
    key = key.removeprefix("module.").removeprefix("model.").removeprefix("agent.")
    for marker in ("backbone.", "vlm_backbone."):
        if key.startswith(marker):
            return key.split(marker, 1)[1]
    if key.startswith("two_expert_slots.") or key.startswith("dynamic_adapter.") or key.startswith("geometry_adapter."):
        return None
    return key


def _load_compatible_backbone_state(backbone: Any, state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    expected = backbone.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        mapped = _backbone_key_from_checkpoint_key(key)
        if mapped is None or mapped not in expected:
            continue
        if tuple(expected[mapped].shape) != tuple(value.shape):
            continue
        filtered[mapped] = value
    if not filtered:
        return {"loaded_top_layer_keys": [], "missing_top_layer_keys": [], "unexpected_top_layer_keys": []}
    incompatible = backbone.load_state_dict(filtered, strict=False)
    return {
        "loaded_top_layer_keys": sorted(filtered),
        "missing_top_layer_keys": sorted(getattr(incompatible, "missing_keys", [])),
        "unexpected_top_layer_keys": sorted(getattr(incompatible, "unexpected_keys", [])),
    }


def _load_lora_from_checkpoint_state(backbone: Any, state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    lora_state = {
        key: value
        for key, value in state.items()
        if "lora_" in key or ".lora_A." in key or ".lora_B." in key
    }
    if not lora_state:
        return {"loaded_lora_adapter": False, "loaded_lora_keys": []}
    incompatible = backbone.load_state_dict(lora_state, strict=False)
    loaded = [
        key
        for key in lora_state
        if key not in set(getattr(incompatible, "unexpected_keys", []))
    ]
    return {
        "loaded_lora_adapter": bool(loaded),
        "loaded_lora_keys": sorted(loaded),
        "unexpected_lora_keys": sorted(getattr(incompatible, "unexpected_keys", [])),
    }


def _apply_lora_adapter_dir(backbone: Any, adapter_dir: Path) -> Dict[str, Any]:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"VLM LoRA adapter directory does not exist: {adapter_dir}")
    if hasattr(backbone.model, "load_adapter"):
        backbone.model.load_adapter(str(adapter_dir))
        return {"loaded_lora_adapter": True, "loaded_lora_adapter_dir": str(adapter_dir)}
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError(
            "Stage1 train_mode=lora requires --vlm-lora-adapter-dir or an embedded LoRA state. "
            "peft is not installed, so the adapter directory cannot be applied."
        ) from exc
    backbone.model = PeftModel.from_pretrained(backbone.model, str(adapter_dir), is_trainable=False)
    return {"loaded_lora_adapter": True, "loaded_lora_adapter_dir": str(adapter_dir)}


def load_stage1_state_for_hidden_cache(
    *,
    checkpoint: Optional[Path],
    config: TwoExpertSlotConfig,
    backbone: Optional[Any],
    args: argparse.Namespace,
) -> Tuple[TwoExpertSoftSlots, Dict[str, Any]]:
    payload = _load_checkpoint_payload(checkpoint) if checkpoint is not None else {}
    if checkpoint is not None:
        try:
            state = _state_dict_from_payload(payload, checkpoint)
        except TypeError:
            state = {}
    else:
        state = {}
    payload_metadata = _metadata_from_payload(payload)
    stage1_train_mode = _resolve_stage1_train_mode(payload_metadata, args)
    slots, slot_report = load_two_expert_slots_with_report(checkpoint, config)
    report: Dict[str, Any] = {
        **slot_report,
        "checkpoint_schema": payload_metadata.get("checkpoint_schema"),
        "stage1_train_mode": stage1_train_mode,
        "loaded_lora_adapter": False,
        "loaded_lora_keys": [],
        "loaded_top_layer_keys": [],
        "source_stage1_checkpoint": str(checkpoint) if checkpoint else None,
    }
    if backbone is None:
        return slots, report
    if stage1_train_mode == "lora":
        adapter_dir = args.vlm_lora_adapter_dir
        if adapter_dir is None and checkpoint is not None and payload_metadata.get("vlm_lora_adapter_dir"):
            adapter_dir = checkpoint.parent / str(payload_metadata["vlm_lora_adapter_dir"])
        if adapter_dir is not None:
            report.update(_apply_lora_adapter_dir(backbone, adapter_dir))
        else:
            report.update(_load_lora_from_checkpoint_state(backbone, state))
        if not report.get("loaded_lora_adapter"):
            raise RuntimeError(
                "Stage1 checkpoint metadata says train_mode=lora, but no VLM LoRA adapter was loaded. "
                "Pass --vlm-lora-adapter-dir or save LoRA weights inside the Stage1 checkpoint."
            )
    elif stage1_train_mode in {"top_layers", "full"}:
        trainable_state_path = None
        if checkpoint is not None and payload_metadata.get("vlm_trainable_state_path"):
            trainable_state_path = checkpoint.parent / str(payload_metadata["vlm_trainable_state_path"])
        if trainable_state_path is not None:
            if not trainable_state_path.is_file():
                raise FileNotFoundError(f"Stage1 VLM trainable state file not found: {trainable_state_path}")
            trainable_state = _load_state_dict(trainable_state_path)
            report["loaded_vlm_trainable_state_path"] = str(trainable_state_path)
        else:
            trainable_state = state
        report.update(_load_compatible_backbone_state(backbone, trainable_state))
        if not report.get("loaded_top_layer_keys"):
            raise RuntimeError(
                f"Stage1 checkpoint metadata says train_mode={stage1_train_mode}, "
                "but no compatible VLM backbone keys were loaded."
            )
    elif stage1_train_mode != "frozen":
        raise ValueError(f"Unsupported Stage1 train mode in checkpoint metadata: {stage1_train_mode!r}")
    return slots, report


def _precision_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def compute_batch(backbone: Any, slots: TwoExpertSoftSlots, batch: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, torch.Tensor]]:
    if not batch:
        return []
    from navsim.agents.recogdrive.utils.internvl_preprocess import load_image

    pixel_values_list = [load_image(item["image_path"], max_num=args.max_image_patches) for item in batch]
    pixel_values = torch.cat(pixel_values_list, dim=0).to(args.device)
    prompts = [item["prompt"] for item in batch]
    num_patches_list = [int(values.shape[0]) for values in pixel_values_list]
    with torch.no_grad():
        out = backbone.forward_with_two_expert_slots(
            pixel_values,
            {"questions": prompts, "num_patches_list": num_patches_list},
            slots,
            return_image_hidden=False,
            return_raw_hidden=True,
            train_vlm_mode=args.train_vlm_mode,
        )
    raw = out["raw_vlm_hidden"].detach().float().cpu()
    h_dyn = out["h_dyn"].detach().float().cpu()
    h_geo = out["h_geo"].detach().float().cpu()
    results = []
    for idx in range(len(batch)):
        results.append(
            {
                "last_hidden_state": raw[idx].contiguous(),
                "two_expert_h_dyn": h_dyn[idx].contiguous(),
                "two_expert_h_geo": h_geo[idx].contiguous(),
                "two_expert_slot_metadata": dict(out["slot_metadata"]),
            }
        )
    return results


def synthetic_outputs(sample: Dict[str, Any], config: TwoExpertSlotConfig) -> Dict[str, torch.Tensor]:
    hidden_dim = int(config.vlm_hidden_dim)
    return {
        "last_hidden_state": sample.get("last_hidden_state", torch.zeros(16, hidden_dim)).detach().float().cpu(),
        "two_expert_h_dyn": sample.get("two_expert_h_dyn", torch.zeros(3, 12, hidden_dim)).detach().float().cpu(),
        "two_expert_h_geo": sample.get("two_expert_h_geo", torch.zeros(12, hidden_dim)).detach().float().cpu(),
        "two_expert_slot_metadata": {
            "slot_mode": "vlm_soft_slots",
            "num_dyn_groups": 3,
            "num_dyn_tokens_per_group": 12,
            "num_geo_tokens": 12,
        },
    }


def validate_hidden_payload(payload: Dict[str, Any], config: TwoExpertSlotConfig) -> None:
    hidden_dim = int(config.vlm_hidden_dim)
    last_hidden = payload["last_hidden_state"]
    h_dyn = payload["two_expert_h_dyn"]
    h_geo = payload["two_expert_h_geo"]
    if not isinstance(last_hidden, torch.Tensor) or last_hidden.ndim != 2 or last_hidden.shape[-1] != hidden_dim:
        raise ValueError(f"last_hidden_state must have shape [N,{hidden_dim}], got {tuple(last_hidden.shape)}.")
    if tuple(h_dyn.shape) != (3, 12, hidden_dim):
        raise ValueError(f"two_expert_h_dyn shape {tuple(h_dyn.shape)} != (3, 12, {hidden_dim}).")
    if tuple(h_geo.shape) != (12, hidden_dim):
        raise ValueError(f"two_expert_h_geo shape {tuple(h_geo.shape)} != (12, {hidden_dim}).")
    for key in ("last_hidden_state", "two_expert_h_dyn", "two_expert_h_geo"):
        if not torch.isfinite(payload[key].float()).all():
            raise ValueError(f"{key} contains non-finite values.")


def output_dir_for_shard(output_root: Path, shard_index: int) -> Path:
    return output_root / "shards" / f"shard_{int(shard_index):05d}"


def strip_eval_teacher_targets(payload: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.split in {"eval", "navtest", "val"} and not args.allow_eval_teacher_targets:
        for key in TRAIN_TEACHER_KEYS:
            payload.pop(key, None)


def build_shard(args: argparse.Namespace) -> Dict[str, Any]:
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards).")
    if not args.synthetic_smoke and args.stage1_checkpoint is None:
        raise ValueError("Building a two-expert hidden cache requires --stage1-checkpoint outside --synthetic-smoke.")
    config = TwoExpertSlotConfig(
        vlm_hidden_dim=int(args.vlm_hidden_dim),
        planner_dim=384,
        num_dyn_groups=3,
        num_dyn_tokens_per_group=12,
        num_geo_tokens=12,
    )
    backbone = None if args.synthetic_smoke else load_backbone(args)
    slots, stage1_load_report = load_stage1_state_for_hidden_cache(
        checkpoint=None if args.synthetic_smoke else args.stage1_checkpoint,
        config=config,
        backbone=backbone,
        args=args,
    )
    slots = slots.to(args.device)
    out_dir = output_dir_for_shard(args.output_root, args.shard_index)
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output shard exists: {out_dir}. Pass --overwrite to replace it.")
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    batch: List[Dict[str, Any]] = []
    written = skipped = 0
    hidden_lengths: List[int] = []

    def write_item(item: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        nonlocal written
        payload = dict(item["preserved"])
        payload.update(outputs)
        payload["two_expert_metadata"] = {
            "schema": "two_expert_slot_hidden_cache_v1",
            "train_mode": str(args.train_vlm_mode),
            "stage1_train_mode": stage1_load_report.get("stage1_train_mode"),
            "loaded_lora_adapter": bool(stage1_load_report.get("loaded_lora_adapter", False)),
            "loaded_lora_key_count": len(stage1_load_report.get("loaded_lora_keys", [])),
            "loaded_top_layer_key_count": len(stage1_load_report.get("loaded_top_layer_keys", [])),
            "loaded_slot_keys": list(stage1_load_report.get("loaded_slot_keys", [])),
            "source_stage1_checkpoint": stage1_load_report.get("source_stage1_checkpoint"),
            "checkpoint_schema": stage1_load_report.get("checkpoint_schema"),
            "prompt_version": TWO_EXPERT_PROMPT_VERSION,
            "num_dyn_groups": 3,
            "tokens_per_group": 12,
            "num_geo_tokens": 12,
            "vlm_checkpoint": str(args.vlm_path),
            "slot_checkpoint": str(args.stage1_checkpoint) if args.stage1_checkpoint else None,
            "stage1_load_report": dict(stage1_load_report),
        }
        if args.include_teacher_targets:
            for key in TRAIN_TEACHER_KEYS:
                if key in item["sample"]:
                    payload[key] = item["sample"][key]
        strip_eval_teacher_targets(payload, args)
        validate_hidden_payload(payload, config)
        hidden_lengths.append(int(payload["last_hidden_state"].shape[0]))
        atomic_torch_save(payload, item["out_path"])
        rows.append(item["row"])
        written += 1

    def flush_batch() -> None:
        if not batch:
            return
        if args.synthetic_smoke:
            outputs = [synthetic_outputs(item["sample"], config) for item in batch]
        else:
            outputs = compute_batch(backbone, slots, batch, args)
        for item, out in zip(batch, outputs):
            write_item(item, out)
        batch.clear()

    for idx, (_, sample_path, record) in enumerate(
        iter_indexed_records(args.base_chunk_root, pattern=args.chunk_name_pattern, max_records=args.max_samples)
    ):
        if idx % int(args.num_shards) != int(args.shard_index):
            skipped += 1
            continue
        sample = load_sample(sample_path)
        token = str(sample.get("sample_token") or record.get("sample_token") or sample_path.stem)
        scene_token = str(sample.get("scene_token") or record.get("scene_token") or "")
        log_name = sample.get("log_name") or record.get("log_name")
        preserved = {key: sample[key] for key in PRESERVE_KEYS if key in sample}
        preserved["sample_token"] = token
        if scene_token:
            preserved["scene_token"] = scene_token
        if log_name:
            preserved["log_name"] = str(log_name)
        out_path = samples_dir / f"{token}.pt"
        row = {"sample_token": token, "path": str(out_path.relative_to(out_dir))}
        if scene_token:
            row["scene_token"] = scene_token
        if log_name:
            row["log_name"] = str(log_name)
        item = {"sample": sample, "preserved": preserved, "out_path": out_path, "row": row}
        if not args.synthetic_smoke:
            if "image_path_tensor" not in sample:
                raise KeyError("two_expert hidden cache generation requires image_path_tensor in the base chunk.")
            item["image_path"] = decode_path_tensor(sample["image_path_tensor"])
            item["prompt"] = build_two_expert_prompt(sample, allow_minimal_prompt=bool(args.allow_minimal_prompt))
        batch.append(item)
        if len(batch) >= int(args.batch_size):
            flush_batch()
    flush_batch()

    with (out_dir / "index.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    metadata = {
        "version": "two_expert_slot_hidden_cache_v1",
        "split": str(args.split),
        "source_base_cache": str(args.base_chunk_root),
        "vlm_checkpoint": str(args.vlm_path),
        "slot_checkpoint": str(args.stage1_checkpoint) if args.stage1_checkpoint else None,
        "num_samples": int(written),
        "num_skipped_by_shard": int(skipped),
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "teacher_targets_included": bool(args.include_teacher_targets),
        "eval_teacher_targets_allowed": bool(args.allow_eval_teacher_targets),
        "synthetic_smoke": bool(args.synthetic_smoke),
        "loaded_lora_adapter": bool(stage1_load_report.get("loaded_lora_adapter", False)),
        "loaded_lora_key_count": len(stage1_load_report.get("loaded_lora_keys", [])),
        "loaded_top_layer_key_count": len(stage1_load_report.get("loaded_top_layer_keys", [])),
        "loaded_slot_keys": list(stage1_load_report.get("loaded_slot_keys", [])),
        "source_stage1_checkpoint": stage1_load_report.get("source_stage1_checkpoint"),
        "stage1_load_report": dict(stage1_load_report),
        "hidden_token_length": {
            "min": min(hidden_lengths) if hidden_lengths else None,
            "max": max(hidden_lengths) if hidden_lengths else None,
            "mean": float(sum(hidden_lengths) / len(hidden_lengths)) if hidden_lengths else None,
        },
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
    metadata = {
        "version": "two_expert_slot_hidden_cache_v1",
        "merged": True,
        "num_samples": len(rows),
        "num_shards": len(metadata_items),
        "teacher_targets_included": any(bool(item.get("teacher_targets_included", False)) for item in metadata_items),
    }
    write_json(output_root / "metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build two_expert_slot hidden cache after Stage1 VLM SFT.")
    parser.add_argument("--base-chunk-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, default=None)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--vlm-type", default="internvl")
    parser.add_argument("--split", default="navtrain")
    parser.add_argument("--chunk-name-pattern", default="train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*,navtest_full_chunk_*")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--vlm-hidden-dim", type=int, default=1536)
    parser.add_argument("--train-vlm-mode", choices=("frozen", "lora", "top_layers"), default="frozen")
    parser.add_argument("--stage1-train-mode", choices=("frozen", "lora", "top_layers", "full"), default=None)
    parser.add_argument("--vlm-lora-adapter-dir", type=Path, default=None)
    parser.add_argument("--max-image-patches", type=int, default=12)
    parser.add_argument("--include-teacher-targets", action="store_true")
    parser.add_argument("--allow-eval-teacher-targets", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--allow-minimal-prompt", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.merge:
        print(json.dumps(merge_shards(args.output_root, overwrite=args.overwrite), indent=2, sort_keys=True))
        return 0
    if args.split in {"eval", "navtest", "val"} and args.include_teacher_targets and not args.allow_eval_teacher_targets:
        raise ValueError("Eval/navtest hidden cache must not include teacher targets unless --allow-eval-teacher-targets is set.")
    if os.getenv("RUN_CACHE", "0") != "1":
        payload = {
            "status": "blocked_by_RUN_CACHE_gate",
            "message": "Set RUN_CACHE=1 to build two-expert hidden cache.",
            "output_root": str(args.output_root),
            "split": str(args.split),
        }
        args.output_root.mkdir(parents=True, exist_ok=True)
        write_json(args.output_root / "two_expert_hidden_cache_dry_run.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(json.dumps(build_shard(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
