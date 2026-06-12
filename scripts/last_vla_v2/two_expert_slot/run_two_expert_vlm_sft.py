#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import iter_index, load_sample, write_json  # noqa: E402
from navsim.agents.recogdrive.recogdrive_backbone import RecogDriveBackbone  # noqa: E402
from navsim.agents.recogdrive.two_expert_vlm_sft import TwoExpertVLMSFTConfig, TwoExpertVLMSFTModule  # noqa: E402
from navsim.agents.recogdrive.vlm_lora_utils import (  # noqa: E402
    audit_actual_trainable_lora_modules,
    audit_lora_target_modules,
    resolve_lora_target_modules,
    validate_lora_scope_audit,
)


def chunk_dirs(root: Path) -> List[Path]:
    if (root / "index.jsonl").is_file():
        return [root]
    dirs = sorted(path for path in root.glob("**") if path.is_dir() and (path / "index.jsonl").is_file())
    if not dirs:
        raise FileNotFoundError(f"No index.jsonl found under {root}")
    return dirs


def iter_records(root: Path, max_samples: Optional[int] = None) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    count = 0
    for chunk_dir in chunk_dirs(root):
        for record in iter_index(chunk_dir):
            yield Path(record["path"]), record
            count += 1
            if max_samples is not None and count >= max_samples:
                return


def decode_path_tensor(path_tensor: torch.Tensor) -> str:
    chars = []
    for item in path_tensor.detach().cpu().view(-1):
        value = int(item.item())
        if value:
            chars.append(chr(value))
    return "".join(chars)


def load_teacher_map(root: Path, key: str, max_samples: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for path, record in iter_records(root, max_samples=max_samples):
        payload = load_sample(path)
        token = str(payload.get("sample_token") or record.get("sample_token") or path.stem)
        if key not in payload:
            raise KeyError(f"Teacher cache sample {path} missing {key}.")
        if token in mapping:
            raise ValueError(f"Duplicate sample_token in teacher cache {root}: {token}")
        mapping[token] = payload
    return mapping


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


def resolve_vggt_feature_dim(vggt_root: Path, vggt_map: Dict[str, Dict[str, Any]], override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    dims = set(_metadata_feature_dims(vggt_root))
    for payload in vggt_map.values():
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
        jepa_map: Dict[str, Dict[str, Any]],
        vggt_map: Dict[str, Dict[str, Any]],
        *,
        max_samples: Optional[int] = None,
    ) -> None:
        self.items: List[Tuple[Path, str]] = []
        teacher_tokens = set(jepa_map).intersection(vggt_map)
        for sample_path, record in iter_records(base_chunk_root, max_samples=max_samples):
            token = str(record.get("sample_token") or sample_path.stem)
            if token in teacher_tokens:
                self.items.append((sample_path, token))
        if not self.items:
            raise RuntimeError("No base samples intersect both JEPA and VGGT teacher caches.")
        self.jepa_map = jepa_map
        self.vggt_map = vggt_map

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_path, token = self.items[index]
        sample = load_sample(sample_path)
        if "image_path_tensor" not in sample:
            raise KeyError(f"Base sample {sample_path} missing image_path_tensor.")
        return {
            "sample_token": token,
            "image_path": decode_path_tensor(sample["image_path_tensor"]),
            "prompt": build_prompt(sample),
            "status_feature": sample["status_feature"].float(),
            "high_command_one_hot": sample["high_command_one_hot"].float(),
            "history_trajectory": sample["history_trajectory"].float(),
            "trajectory": sample.get("trajectory"),
            "trajectory_norm": sample.get("trajectory_norm"),
            "jepa_dynamic_teacher_tokens": self.jepa_map[token]["jepa_dynamic_teacher_tokens"].float(),
            "vggt_feature23_tokens": self.vggt_map[token]["vggt_feature23_tokens"].float(),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train two_expert_slot Stage1 VLM SFT.")
    parser.add_argument("--base-chunk-root", type=Path, required=True)
    parser.add_argument("--teacher-cache-root", type=Path, default=None)
    parser.add_argument("--jepa-cache-root", type=Path, default=None)
    parser.add_argument("--vggt-cache-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--vlm-type", default="internvl")
    parser.add_argument("--train-mode", choices=("frozen", "lora", "top_layers", "full"), default="top_layers")
    parser.add_argument("--allow-full-vlm-sft", action="store_true")
    parser.add_argument("--top-layers", type=int, default=2)
    parser.add_argument("--vggt-feature-dim", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
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
    jepa_map = load_teacher_map(jepa_root, "jepa_dynamic_teacher_tokens", max_samples=args.max_samples)
    vggt_map = load_teacher_map(vggt_root, "vggt_feature23_tokens", max_samples=args.max_samples)
    vggt_dim = resolve_vggt_feature_dim(vggt_root, vggt_map, args.vggt_feature_dim)
    dataset = TwoExpertStage1Dataset(args.base_chunk_root, jepa_map, vggt_map, max_samples=args.max_samples)
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
    optimizer = torch.optim.AdamW([p for p in train_module.parameters() if p.requires_grad], lr=float(args.lr), weight_decay=1e-4)
    metrics = {"steps": 0, "last_loss": None}
    for epoch in range(int(args.max_epochs)):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            out = module(move_batch(batch, device))
            loss = out["loss"] / int(args.grad_accum)
            loss.backward()
            if (step + 1) % int(args.grad_accum) == 0:
                torch.nn.utils.clip_grad_norm_(train_module.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            metrics["steps"] += 1
            metrics["last_loss"] = float(out["loss"].detach().cpu())
    if int(os.getenv("RANK", "0")) == 0:
        metadata = {
            "schema": "two_expert_slot_stage1_vlm_sft_v1",
            "train_mode": args.train_mode,
            "vggt_feature_dim": int(vggt_dim),
            "jepa_cache_root": str(jepa_root),
            "vggt_cache_root": str(vggt_root),
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
