#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import load_sample, write_json  # noqa: E402
from navsim.agents.recogdrive.recogdrive_backbone import RecogDriveBackbone  # noqa: E402
from navsim.agents.recogdrive.two_expert_vlm_sft import TwoExpertVLMSFTConfig, TwoExpertVLMSFTModule  # noqa: E402
from navsim.agents.recogdrive.vlm_lora_utils import (  # noqa: E402
    audit_actual_trainable_lora_modules,
    audit_lora_target_modules,
    resolve_lora_target_modules,
    validate_lora_scope_audit,
)
from scripts.last_vla_v2.two_expert_slot.two_expert_cache_utils import iter_indexed_records, load_path_index  # noqa: E402


def decode_path_tensor(path_tensor: torch.Tensor) -> str:
    chars = []
    for item in path_tensor.detach().cpu().view(-1):
        value = int(item.item())
        if value:
            chars.append(chr(value))
    return "".join(chars)


def load_teacher_index(root: Path, key: str, max_samples: Optional[int] = None) -> Dict[str, Path]:
    index = load_path_index(root, max_records=max_samples)
    if not index:
        raise RuntimeError(f"Teacher cache {root} has no indexed samples for key {key}.")
    return index


def resolve_cache_roots(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.jepa_cache_root is not None and args.vggt_cache_root is not None:
        return args.jepa_cache_root, args.vggt_cache_root
    if args.teacher_cache_root is None:
        raise ValueError("Provide --teacher-cache-root or both --jepa-cache-root and --vggt-cache-root.")
    candidates = [
        (args.teacher_cache_root / "jepa_dynamic", args.teacher_cache_root / "vggt_feature23"),
        (args.teacher_cache_root / "jepa", args.teacher_cache_root / "vggt"),
        (args.teacher_cache_root, args.teacher_cache_root),
    ]
    for jepa_root, vggt_root in candidates:
        if (jepa_root / "index.jsonl").is_file() or (jepa_root / "shards").is_dir():
            if (vggt_root / "index.jsonl").is_file() or (vggt_root / "shards").is_dir():
                return jepa_root, vggt_root
    raise FileNotFoundError(f"Could not resolve JEPA/VGGT teacher caches under {args.teacher_cache_root}")


def _metadata_feature_dims(root: Path) -> List[int]:
    dims = set()
    paths = [root / "metadata.json"]
    paths.extend(sorted(root.glob("shards/shard_*/metadata.json")))
    for path in paths:
        if not path.is_file():
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if "feature_dims" in metadata:
            dims.update(int(dim) for dim in metadata["feature_dims"])
        if "feature_dim" in metadata:
            dims.add(int(metadata["feature_dim"]))
    return sorted(dims)


def _load_teacher_payload(path: Path, key: str) -> Dict[str, Any]:
    payload = load_sample(path)
    if key not in payload:
        raise KeyError(f"Teacher cache sample {path} missing {key}.")
    return payload


def resolve_vggt_feature_dim(vggt_root: Path, vggt_index: Dict[str, Path], override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    dims = set(_metadata_feature_dims(vggt_root))
    for path in list(vggt_index.values())[:32]:
        payload = _load_teacher_payload(path, "vggt_feature23_tokens")
        tokens = payload.get("vggt_feature23_tokens")
        if isinstance(tokens, torch.Tensor):
            dims.add(int(tokens.shape[-1]))
        metadata = payload.get("vggt_feature23_metadata")
        if isinstance(metadata, dict) and "feature_dim" in metadata:
            dims.add(int(metadata["feature_dim"]))
    if len(dims) != 1:
        raise ValueError(f"Expected exactly one VGGT Feature(23) dim, got {sorted(dims)}.")
    return int(next(iter(dims)))


def build_prompt(sample: Dict[str, Any]) -> str:
    command = sample.get("high_command_one_hot", torch.tensor([0.0, 1.0, 0.0])).float()
    command_names = ["turn left", "go straight", "turn right"]
    command_str = command_names[int(torch.argmax(command).item())]
    return (
        "<image>\nPredict the ego vehicle trajectory for the next 4 seconds. "
        f"Navigation command: {command_str}."
    )


class TwoExpertStage1Dataset(Dataset):
    def __init__(
        self,
        base_chunk_root: Path,
        jepa_index: Dict[str, Path],
        vggt_index: Dict[str, Path],
        *,
        max_samples: Optional[int] = None,
        teacher_lru_size: int = 0,
        require_strict_teachers: bool = True,
    ) -> None:
        self.items: List[Tuple[Path, str]] = []
        teacher_tokens = set(jepa_index).intersection(vggt_index)
        for _, sample_path, record in iter_indexed_records(base_chunk_root, max_records=max_samples):
            token = str(record.get("sample_token") or sample_path.stem)
            if token in teacher_tokens:
                self.items.append((sample_path, token))
        if not self.items:
            raise RuntimeError("No base samples intersect both JEPA and VGGT teacher caches.")
        self.jepa_index = jepa_index
        self.vggt_index = vggt_index
        self.teacher_lru_size = max(0, int(teacher_lru_size))
        self.require_strict_teachers = bool(require_strict_teachers)
        self._teacher_cache: OrderedDict[Tuple[str, str], Dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.items)

    def _load_teacher(self, kind: str, token: str, key: str) -> Dict[str, Any]:
        path = self.jepa_index[token] if kind == "jepa" else self.vggt_index[token]
        cache_key = (kind, token)
        if self.teacher_lru_size > 0 and cache_key in self._teacher_cache:
            payload = self._teacher_cache.pop(cache_key)
            self._teacher_cache[cache_key] = payload
            return payload
        payload = _load_teacher_payload(path, key)
        if self.teacher_lru_size > 0:
            self._teacher_cache[cache_key] = payload
            while len(self._teacher_cache) > self.teacher_lru_size:
                self._teacher_cache.popitem(last=False)
        return payload

    def _assert_strict_teacher(self, token: str, jepa_payload: Dict[str, Any], vggt_payload: Dict[str, Any]) -> None:
        if not self.require_strict_teachers:
            return
        jepa_meta = jepa_payload.get("jepa_dynamic_teacher_metadata")
        vggt_meta = vggt_payload.get("vggt_feature23_metadata")
        if not isinstance(jepa_meta, dict) or not bool(jepa_meta.get("strict_dynamic_teacher", False)):
            raise RuntimeError(
                f"sample_token={token} does not have strict jepa_dynamic_teacher_tokens. "
                "Full Stage1 training requires production [3,12,1024] JEPA dynamic teacher tokens; "
                "use --allow-dev-fallback-teachers only for smoke/dev."
            )
        if not isinstance(vggt_meta, dict) or not bool(vggt_meta.get("strict_geometry_teacher", False)):
            raise RuntimeError(
                f"sample_token={token} does not have strict vggt_feature23_tokens. "
                "Full Stage1 training requires production VGGT Feature(23) teachers; "
                "use --allow-dev-fallback-teachers only for smoke/dev."
            )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_path, token = self.items[index]
        sample = load_sample(sample_path)
        if "image_path_tensor" not in sample:
            raise KeyError(f"Base sample {sample_path} missing image_path_tensor.")
        jepa_payload = self._load_teacher("jepa", token, "jepa_dynamic_teacher_tokens")
        vggt_payload = self._load_teacher("vggt", token, "vggt_feature23_tokens")
        self._assert_strict_teacher(token, jepa_payload, vggt_payload)
        return {
            "sample_token": token,
            "image_path": decode_path_tensor(sample["image_path_tensor"]),
            "prompt": build_prompt(sample),
            "status_feature": sample["status_feature"].float(),
            "high_command_one_hot": sample["high_command_one_hot"].float(),
            "history_trajectory": sample["history_trajectory"].float(),
            "trajectory": sample.get("trajectory"),
            "trajectory_norm": sample.get("trajectory_norm"),
            "jepa_dynamic_teacher_tokens": jepa_payload["jepa_dynamic_teacher_tokens"].float(),
            "vggt_feature23_tokens": vggt_payload["vggt_feature23_tokens"].float(),
        }


def make_collate(max_image_patches: int):
    from navsim.agents.recogdrive.utils.internvl_preprocess import load_image

    def collate(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        pixel_values_list = [load_image(item["image_path"], max_num=max_image_patches) for item in items]
        batch: Dict[str, Any] = {
            "images": torch.cat(pixel_values_list, dim=0),
            "prompt_inputs": {
                "questions": [item["prompt"] for item in items],
                "num_patches_list": [int(values.shape[0]) for values in pixel_values_list],
            },
            "sample_token": [item["sample_token"] for item in items],
            "status_feature": torch.stack([item["status_feature"] for item in items]),
            "high_command_one_hot": torch.stack([item["high_command_one_hot"] for item in items]),
            "history_trajectory": torch.stack([item["history_trajectory"] for item in items]),
            "jepa_dynamic_teacher_tokens": torch.stack([item["jepa_dynamic_teacher_tokens"] for item in items]),
            "vggt_feature23_tokens": torch.stack([item["vggt_feature23_tokens"] for item in items]),
        }
        if all(isinstance(item.get("trajectory_norm"), torch.Tensor) for item in items):
            batch["trajectory_norm"] = torch.stack([item["trajectory_norm"].float() for item in items])
        elif all(isinstance(item.get("trajectory"), torch.Tensor) for item in items):
            batch["trajectory"] = torch.stack([item["trajectory"].float() for item in items])
        else:
            raise KeyError("Stage1 samples require trajectory or trajectory_norm.")
        return batch

    return collate


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def apply_lora(backbone: RecogDriveBackbone, args: argparse.Namespace) -> Dict[str, Any]:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("train_mode=lora requires peft to create real LoRA modules.") from exc
    target_modules = resolve_lora_target_modules(
        backbone.model,
        preset=args.lora_preset,
        scope=args.lora_scope,
        custom_target_modules=args.lora_target_modules,
        vision_last_n=args.lora_vision_last_n,
        allow_all_linear_global=False,
    )
    audit = audit_lora_target_modules(backbone.model, target_modules=target_modules, scope=args.lora_scope, preset=args.lora_preset)
    validate_lora_scope_audit(audit, allow_mixed_scope=args.lora_allow_mixed_scope)
    backbone.model = get_peft_model(
        backbone.model,
        LoraConfig(
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            target_modules=target_modules,
            bias="none",
        ),
    )
    actual = audit_actual_trainable_lora_modules(backbone.model, scope=args.lora_scope)
    if int(actual.get("actual_trainable_lora_param_count", 0)) <= 0:
        raise RuntimeError("PEFT LoRA injection produced zero trainable parameters.")
    return {"target_modules": target_modules, "intended_audit": audit, "actual_audit": actual}


def save_stage1_outputs(module: TwoExpertVLMSFTModule, output_dir: Path, metadata: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": module.state_dict(), "stage1_metadata": metadata}, output_dir / "stage1.ckpt")
    adapters_dir = output_dir / "adapters"
    adapters_dir.mkdir(exist_ok=True)
    torch.save(module.two_expert_slots.state_dict(), adapters_dir / "two_expert_slots.pt")
    torch.save(
        {
            "dynamic_adapter": module.dynamic_adapter.state_dict(),
            "geometry_adapter": module.geometry_adapter.state_dict(),
            "trajectory_probe": module.trajectory_probe.state_dict(),
        },
        adapters_dir / "adapters_and_probe.pt",
    )
    if metadata["train_mode"] == "lora":
        lora_dir = adapters_dir / "vlm_lora"
        if hasattr(module.backbone.model, "save_pretrained"):
            module.backbone.model.save_pretrained(str(lora_dir))
        else:
            raise RuntimeError("train_mode=lora but backbone.model cannot save_pretrained().")
    if metadata["train_mode"] in {"top_layers", "full"}:
        state = {
            name: value.detach().cpu()
            for name, value in module.named_parameters()
            if name.startswith("backbone.") and value.requires_grad
        }
        torch.save(state, adapters_dir / "vlm_trainable_state.pt")
        metadata["saved_vlm_trainable_key_count"] = len(state)
    write_json(output_dir / "stage1_metadata.json", metadata)
    write_json(output_dir / "trainable_parameter_report.json", module.trainable_parameter_report())


def build_optimizer(module: TwoExpertVLMSFTModule, args: argparse.Namespace) -> torch.optim.Optimizer:
    vlm_params = []
    route_params = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            vlm_params.append(parameter)
        else:
            route_params.append(parameter)
    if args.train_mode != "frozen" and not vlm_params:
        raise RuntimeError(f"train_mode={args.train_mode} requires nonzero trainable VLM parameters.")
    if not route_params:
        raise RuntimeError("Stage1 requires trainable slots/adapters/probe parameters.")
    groups = []
    if vlm_params:
        groups.append({"params": vlm_params, "lr": float(args.lr_vlm), "name": "vlm"})
    groups.append({"params": route_params, "lr": float(args.lr_slots_adapters), "name": "slots_adapters_probe"})
    return torch.optim.AdamW(groups, weight_decay=float(args.weight_decay))


def autocast_context(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision in {"bf16-mixed", "16-mixed", "fp16-mixed"}
    dtype = torch.bfloat16 if precision == "bf16-mixed" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train two_expert_slot Stage1 VLM SFT.")
    parser.add_argument("--base-chunk-root", type=Path, required=True)
    parser.add_argument("--teacher-cache-root", type=Path, default=None)
    parser.add_argument("--jepa-cache-root", type=Path, default=None)
    parser.add_argument("--vggt-cache-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--vlm-type", default="internvl")
    parser.add_argument("--train-mode", choices=("frozen", "lora", "top_layers", "full"), default="lora")
    parser.add_argument("--allow-full-vlm-sft", action="store_true")
    parser.add_argument("--top-layers", type=int, default=2)
    parser.add_argument("--vggt-feature-dim", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr-vlm", type=float, default=1e-5)
    parser.add_argument("--lr-slots-adapters", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "bf16-mixed", "16-mixed", "fp16-mixed"), default="bf16-mixed")
    parser.add_argument("--teacher-lru-size", type=int, default=0)
    parser.add_argument("--allow-dev-fallback-teachers", action="store_true")
    parser.add_argument("--max-image-patches", type=int, default=12)
    parser.add_argument("--device", default=None)
    parser.add_argument("--lora-preset", default="attention_mlp")
    parser.add_argument("--lora-scope", default="llm")
    parser.add_argument("--lora-target-modules", default="")
    parser.add_argument("--lora-vision-last-n", type=int, default=0)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-allow-mixed-scope", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if os.getenv("RUN_TRAIN", "0") != "1":
        payload = {
            "status": "dry_run",
            "entrypoint": "run_two_expert_vlm_sft.py",
            "train_mode": args.train_mode,
            "strict_teachers_required": not bool(args.allow_dev_fallback_teachers),
            "base_chunk_root": str(args.base_chunk_root),
            "teacher_cache_root": str(args.teacher_cache_root),
            "message": "Set RUN_TRAIN=1 to start Stage1 VLM SFT.",
        }
        write_json(args.output_dir / "stage1_dry_run.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    distributed = int(os.getenv("WORLD_SIZE", "1")) > 1
    if distributed:
        torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    device = torch.device(args.device or (f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)

    jepa_root, vggt_root = resolve_cache_roots(args)
    jepa_index = load_teacher_index(jepa_root, "jepa_dynamic_teacher_tokens", max_samples=args.max_samples)
    vggt_index = load_teacher_index(vggt_root, "vggt_feature23_tokens", max_samples=args.max_samples)
    vggt_dim = resolve_vggt_feature_dim(vggt_root, vggt_index, args.vggt_feature_dim)
    dataset = TwoExpertStage1Dataset(
        args.base_chunk_root,
        jepa_index,
        vggt_index,
        max_samples=args.max_samples,
        teacher_lru_size=int(args.teacher_lru_size),
        require_strict_teachers=not bool(args.allow_dev_fallback_teachers),
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) if distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=2,
        collate_fn=make_collate(args.max_image_patches),
    )
    backbone = RecogDriveBackbone(model_type=args.vlm_type, checkpoint_path=str(args.vlm_path), device=str(device))
    lora_report: Dict[str, Any] = {}
    if args.train_mode == "lora":
        lora_report = apply_lora(backbone, args)
    module = TwoExpertVLMSFTModule(
        backbone,
        TwoExpertVLMSFTConfig(
            train_mode=args.train_mode,
            allow_full_vlm_sft=bool(args.allow_full_vlm_sft),
            top_layers=int(args.top_layers),
            vggt_feature_dim=vggt_dim,
        ),
    ).to(device)
    if distributed:
        module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[local_rank] if device.type == "cuda" else None)
    train_module = module.module if hasattr(module, "module") else module
    optimizer = build_optimizer(train_module, args)
    optimizer.zero_grad(set_to_none=True)
    metrics = {"forward_steps": 0, "optimizer_steps": 0, "last_loss": None}
    for epoch in range(int(args.max_epochs)):
        if sampler is not None:
            sampler.set_epoch(epoch)
        pending_grads = 0
        for step, batch in enumerate(dataloader):
            with autocast_context(device, args.precision):
                out = module(move_batch(batch, device))
            loss = out["loss"] / int(args.grad_accum)
            loss.backward()
            pending_grads += 1
            if pending_grads >= int(args.grad_accum):
                torch.nn.utils.clip_grad_norm_(train_module.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                metrics["optimizer_steps"] += 1
                pending_grads = 0
            metrics["forward_steps"] += 1
            metrics["last_loss"] = float(out["loss"].detach().cpu())
        if pending_grads > 0:
            torch.nn.utils.clip_grad_norm_(train_module.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            metrics["optimizer_steps"] += 1
    if int(os.getenv("RANK", "0")) == 0:
        metadata = {
            "schema": "two_expert_slot_stage1_vlm_sft_v1",
            "train_mode": args.train_mode,
            "teacher_loading": "lazy_index_paths",
            "strict_teachers_required": not bool(args.allow_dev_fallback_teachers),
            "vggt_feature_dim": int(vggt_dim),
            "jepa_cache_root": str(jepa_root),
            "vggt_cache_root": str(vggt_root),
            "lr_vlm": float(args.lr_vlm),
            "lr_slots_adapters": float(args.lr_slots_adapters),
            "weight_decay": float(args.weight_decay),
            "base_chunk_root": str(args.base_chunk_root),
            "vlm_path": str(args.vlm_path),
            "lora_report": lora_report,
        }
        save_stage1_outputs(train_module, args.output_dir, metadata)
        write_json(args.output_dir / "metrics.json", metrics)
    if distributed:
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
