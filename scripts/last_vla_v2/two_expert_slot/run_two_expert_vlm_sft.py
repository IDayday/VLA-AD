#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
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
from navsim.agents.recogdrive.trajectory_text_replay import parse_trajectory_answer  # noqa: E402
from navsim.agents.recogdrive.vlm_lora_utils import (  # noqa: E402
    audit_actual_trainable_lora_modules,
    audit_lora_target_modules,
    resolve_lora_target_modules,
    validate_lora_scope_audit,
)
from scripts.last_vla_v2.two_expert_slot.two_expert_prompt_utils import (  # noqa: E402
    TWO_EXPERT_PROMPT_VERSION,
    build_two_expert_prompt,
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


def load_replay_index(root: Optional[Path], max_samples: Optional[int] = None) -> Dict[str, Path]:
    if root is None:
        return {}
    index = load_path_index(root, max_records=max_samples)
    if not index:
        raise RuntimeError(f"Replay cache {root} has no indexed samples.")
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


class TwoExpertStage1Dataset(Dataset):
    def __init__(
        self,
        base_chunk_root: Path,
        jepa_index: Dict[str, Path],
        vggt_index: Dict[str, Path],
        *,
        max_samples: Optional[int] = None,
        chunk_name_pattern: Optional[str] = None,
        teacher_lru_size: int = 0,
        require_strict_teachers: bool = True,
        allow_minimal_prompt: bool = False,
        replay_index: Optional[Dict[str, Path]] = None,
        require_replay: bool = False,
        allow_replay_only_base: bool = False,
    ) -> None:
        self.items: List[Tuple[Optional[Path], str]] = []
        teacher_tokens = set(jepa_index).intersection(vggt_index)
        self.replay_index = replay_index or {}
        if self.replay_index:
            teacher_tokens = teacher_tokens.intersection(self.replay_index)
        self.allow_replay_only_base = bool(allow_replay_only_base)
        if base_chunk_root is not None and Path(base_chunk_root).exists():
            for _, sample_path, record in iter_indexed_records(
                base_chunk_root,
                pattern=chunk_name_pattern,
                max_records=max_samples,
            ):
                token = str(record.get("sample_token") or sample_path.stem)
                if token in teacher_tokens:
                    self.items.append((sample_path, token))
        if not self.items and self.allow_replay_only_base and self.replay_index:
            tokens = sorted(teacher_tokens)
            if max_samples is not None:
                tokens = tokens[: int(max_samples)]
            self.items = [(None, token) for token in tokens]
        if not self.items:
            raise RuntimeError("No base samples intersect both JEPA and VGGT teacher caches.")
        if require_replay and not self.replay_index:
            raise RuntimeError("Stage1 replay CE requires a non-empty replay cache index.")
        self.jepa_index = jepa_index
        self.vggt_index = vggt_index
        self.teacher_lru_size = max(0, int(teacher_lru_size))
        self.require_strict_teachers = bool(require_strict_teachers)
        self.allow_minimal_prompt = bool(allow_minimal_prompt)
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
        replay_fields = self._load_replay_fields(token)
        replay_payload = self._load_replay_payload(token) if token in self.replay_index else {}
        if sample_path is not None:
            sample = load_sample(sample_path)
            if "image_path_tensor" not in sample:
                raise KeyError(f"Base sample {sample_path} missing image_path_tensor.")
            image_path = decode_path_tensor(sample["image_path_tensor"])
            prompt = build_two_expert_prompt(sample, allow_minimal_prompt=self.allow_minimal_prompt)
            status_feature = sample["status_feature"].float()
            high_command_one_hot = sample["high_command_one_hot"].float()
            history_trajectory = sample["history_trajectory"].float()
            trajectory = sample.get("trajectory")
            trajectory_norm = sample.get("trajectory_norm")
        elif replay_payload:
            if replay_payload.get("image_path_tensor") is not None:
                image_path = decode_path_tensor(torch.as_tensor(replay_payload["image_path_tensor"]))
            else:
                image_path = str(replay_payload["image_path"])
            prompt = str(replay_payload["prompt"])
            history_trajectory = _tensor_or_prompt_history(replay_payload.get("history_trajectory"), prompt)
            high_command_one_hot = _tensor_or_prompt_command(replay_payload.get("high_command_one_hot"), prompt)
            status_feature = _tensor_or_status_feature(
                replay_payload.get("status_feature"),
                history_trajectory=history_trajectory,
                high_command_one_hot=high_command_one_hot,
            )
            trajectory = replay_payload.get("trajectory")
            if not isinstance(trajectory, torch.Tensor):
                trajectory = parse_trajectory_answer(str(replay_payload["answer_text"]))
            trajectory_norm = None
        else:
            raise RuntimeError(f"Replay-only base requested but no replay payload for token={token}.")
        jepa_payload = self._load_teacher("jepa", token, "jepa_dynamic_teacher_tokens")
        vggt_payload = self._load_teacher("vggt", token, "vggt_feature23_tokens")
        self._assert_strict_teacher(token, jepa_payload, vggt_payload)
        return {
            **replay_fields,
            "sample_token": token,
            "image_path": image_path,
            "prompt": prompt,
            "status_feature": status_feature,
            "high_command_one_hot": high_command_one_hot,
            "history_trajectory": history_trajectory,
            "trajectory": trajectory,
            "trajectory_norm": trajectory_norm,
            "jepa_dynamic_teacher_tokens": jepa_payload["jepa_dynamic_teacher_tokens"].float(),
            "vggt_feature23_tokens": vggt_payload["vggt_feature23_tokens"].float(),
        }

    def _load_replay_payload(self, token: str) -> Dict[str, Any]:
        if token not in self.replay_index:
            return {}
        return load_sample(self.replay_index[token])

    def _load_replay_fields(self, token: str) -> Dict[str, Any]:
        payload = self._load_replay_payload(token)
        if not payload:
            return {}
        prompt = payload.get("prompt")
        answer_text = payload.get("answer_text")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Replay sample for {token} has empty prompt.")
        if not isinstance(answer_text, str) or not answer_text.strip():
            raise ValueError(f"Replay sample for {token} has empty answer_text.")
        return {
            "replay_prompt": prompt,
            "replay_answer_text": answer_text,
            "replay_source": str(payload.get("replay_source", "unknown")),
            "replay_official_recogdrive_stage1": bool(payload.get("official_recogdrive_stage1", False)),
            "replay_parse_ok": bool(payload.get("parse_ok", True)),
        }


_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_TRIPLE_RE = re.compile(rf"\(\s*({_FLOAT_RE})\s*,\s*({_FLOAT_RE})\s*,\s*({_FLOAT_RE})\s*\)")


def _history_from_prompt(prompt: str) -> torch.Tensor:
    triples = [(float(x), float(y), float(h)) for x, y, h in _TRIPLE_RE.findall(prompt)]
    if len(triples) >= 4:
        return torch.tensor(triples[-4:], dtype=torch.float32)
    return torch.zeros(4, 3, dtype=torch.float32)


def _command_from_prompt(prompt: str) -> torch.Tensor:
    upper = str(prompt).upper()
    out = torch.zeros(3, dtype=torch.float32)
    if "TURN LEFT" in upper:
        out[0] = 1.0
    elif "TURN RIGHT" in upper:
        out[2] = 1.0
    else:
        out[1] = 1.0
    return out


def _float_tensor_or_none(value: Any, *, shape: Tuple[int, ...]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError):
        return None
    if tensor.numel() != int(torch.tensor(shape).prod().item()):
        return None
    tensor = tensor.reshape(shape)
    if not torch.isfinite(tensor).all():
        return None
    return tensor


def _tensor_or_prompt_history(value: Any, prompt: str) -> torch.Tensor:
    tensor = _float_tensor_or_none(value, shape=(4, 3))
    if tensor is not None:
        return tensor
    tensor = _history_from_prompt(prompt)
    return torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _tensor_or_prompt_command(value: Any, prompt: str) -> torch.Tensor:
    tensor = _float_tensor_or_none(value, shape=(3,))
    if tensor is not None and float(tensor.abs().sum().item()) > 0.0:
        out = torch.zeros(3, dtype=torch.float32)
        out[int(torch.argmax(tensor).item())] = 1.0
        return out
    return _command_from_prompt(prompt)


def _status_from_replay_context(history_trajectory: torch.Tensor, high_command_one_hot: torch.Tensor) -> torch.Tensor:
    history = torch.as_tensor(history_trajectory, dtype=torch.float32).reshape(-1, 3)
    command = torch.as_tensor(high_command_one_hot, dtype=torch.float32).reshape(-1)[:3]
    if command.numel() < 3:
        command = torch.nn.functional.pad(command, (0, 3 - command.numel()))
    if float(command.abs().sum().item()) <= 0.0:
        command = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    else:
        one_hot = torch.zeros(3, dtype=torch.float32)
        one_hot[int(torch.argmax(command).item())] = 1.0
        command = one_hot
    if history.shape[0] >= 2:
        velocity_xy = history[-1, :2] - history[-2, :2]
    else:
        velocity_xy = torch.zeros(2, dtype=torch.float32)
    if history.shape[0] >= 3:
        last_delta = history[-1] - history[-2]
        prev_delta = history[-2] - history[-3]
        acceleration = last_delta - prev_delta
    else:
        acceleration = torch.zeros(3, dtype=torch.float32)
    status = torch.cat([command, velocity_xy, acceleration[:3]], dim=0)
    return torch.nan_to_num(status.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _tensor_or_status_feature(
    value: Any,
    *,
    history_trajectory: torch.Tensor,
    high_command_one_hot: torch.Tensor,
) -> torch.Tensor:
    tensor = _float_tensor_or_none(value, shape=(8,))
    if tensor is not None:
        return tensor
    return _status_from_replay_context(history_trajectory, high_command_one_hot)


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
        if all(isinstance(item.get("replay_prompt"), str) and isinstance(item.get("replay_answer_text"), str) for item in items):
            batch["replay_prompt_inputs"] = {
                "prompts": [item["replay_prompt"] for item in items],
                "answers": [item["replay_answer_text"] for item in items],
                "num_patches_list": [int(values.shape[0]) for values in pixel_values_list],
            }
            batch["replay_parse_ok"] = torch.tensor([bool(item.get("replay_parse_ok", False)) for item in items], dtype=torch.float32)
            batch["replay_official_recogdrive_stage1"] = torch.tensor(
                [bool(item.get("replay_official_recogdrive_stage1", False)) for item in items],
                dtype=torch.float32,
            )
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


def _load_checkpoint_payload(path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Stage1 checkpoint must be a dict: {path}")
    return payload


def _stage1_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    metadata = payload.get("stage1_metadata")
    return metadata if isinstance(metadata, dict) else {}


def _resolve_init_lora_adapter_dir(args: argparse.Namespace, payload: Optional[Dict[str, Any]]) -> Optional[Path]:
    if args.init_vlm_lora_adapter_dir is not None:
        return args.init_vlm_lora_adapter_dir
    if args.init_stage1_checkpoint is None or payload is None:
        return None
    metadata = _stage1_metadata(payload)
    rel = metadata.get("vlm_lora_adapter_dir")
    if rel:
        return args.init_stage1_checkpoint.parent / str(rel)
    return None


def apply_lora_adapter_dir_for_training(backbone: RecogDriveBackbone, adapter_dir: Path) -> Dict[str, Any]:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Initial VLM LoRA adapter dir does not exist: {adapter_dir}")
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("Continuing Stage1 LoRA requires peft.") from exc
    backbone.model = PeftModel.from_pretrained(backbone.model, str(adapter_dir), is_trainable=True)
    actual = audit_actual_trainable_lora_modules(backbone.model, scope="llm")
    if int(actual.get("actual_trainable_lora_param_count", 0)) <= 0:
        for name, parameter in backbone.model.named_parameters():
            if "lora_" in name:
                parameter.requires_grad = True
        actual = audit_actual_trainable_lora_modules(backbone.model, scope="llm")
    if int(actual.get("actual_trainable_lora_param_count", 0)) <= 0:
        raise RuntimeError(f"Loaded LoRA adapter but no trainable LoRA parameters were found: {adapter_dir}")
    return {"loaded_initial_lora_adapter_dir": str(adapter_dir), "actual_audit": actual}


def _compat_probe_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    mapped = dict(state)
    for key, value in list(state.items()):
        if key.startswith("head."):
            mapped.setdefault("fused_head." + key[len("head.") :], value)
    return mapped


def load_stage1_initial_state(module: TwoExpertVLMSFTModule, checkpoint: Path) -> Dict[str, Any]:
    payload = _load_checkpoint_payload(checkpoint)
    report: Dict[str, Any] = {"init_stage1_checkpoint": str(checkpoint)}
    if isinstance(payload.get("two_expert_slots"), dict):
        incompatible = module.two_expert_slots.load_state_dict(payload["two_expert_slots"], strict=False)
        report["slots_missing"] = sorted(getattr(incompatible, "missing_keys", []))
        report["slots_unexpected"] = sorted(getattr(incompatible, "unexpected_keys", []))
    for attr, key in (
        ("dynamic_adapter", "dynamic_adapter"),
        ("geometry_adapter", "geometry_adapter"),
        ("trajectory_probe", "trajectory_probe"),
    ):
        state = payload.get(key)
        if not isinstance(state, dict):
            continue
        if key == "trajectory_probe":
            state = _compat_probe_state(state)
        incompatible = getattr(module, attr).load_state_dict(state, strict=False)
        report[f"{key}_missing"] = sorted(getattr(incompatible, "missing_keys", []))
        report[f"{key}_unexpected"] = sorted(getattr(incompatible, "unexpected_keys", []))
    return report


def save_stage1_outputs(
    module: TwoExpertVLMSFTModule,
    output_dir: Path,
    metadata: Dict[str, Any],
    *,
    save_full_stage1_state: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    adapters_dir = output_dir / "adapters"
    adapters_dir.mkdir(exist_ok=True)
    metadata["checkpoint_schema"] = "two_expert_stage1_compact_v1"
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
            metadata["vlm_lora_adapter_dir"] = str(lora_dir.relative_to(output_dir))
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
        metadata["vlm_trainable_state_path"] = str((adapters_dir / "vlm_trainable_state.pt").relative_to(output_dir))
    checkpoint = {
        "checkpoint_schema": metadata["checkpoint_schema"],
        "two_expert_slots": module.two_expert_slots.state_dict(),
        "dynamic_adapter": module.dynamic_adapter.state_dict(),
        "geometry_adapter": module.geometry_adapter.state_dict(),
        "trajectory_probe": module.trajectory_probe.state_dict(),
        "stage1_metadata": metadata,
    }
    if save_full_stage1_state:
        checkpoint["full_state_dict"] = module.state_dict()
        metadata["save_full_stage1_state"] = True
    else:
        metadata["save_full_stage1_state"] = False
    torch.save(checkpoint, output_dir / "stage1.ckpt")
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


def _metric_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().float()
        if value.numel() != 1:
            value = value.mean()
        return float(value.cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def append_progress_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")
        f.flush()


def autocast_context(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision in {"bf16-mixed", "16-mixed", "fp16-mixed"}
    dtype = torch.bfloat16 if precision == "bf16-mixed" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train two_expert_slot Stage1 VLM SFT.")
    parser.add_argument("--base-chunk-root", type=Path, default=None)
    parser.add_argument("--teacher-cache-root", type=Path, default=None)
    parser.add_argument("--jepa-cache-root", type=Path, default=None)
    parser.add_argument("--vggt-cache-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--vlm-type", default="internvl")
    parser.add_argument("--chunk-name-pattern", default=None)
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
    parser.add_argument("--recogdrive-replay-cache-root", type=Path, default=None)
    parser.add_argument("--recogdrive-replay-ce-loss-weight", type=float, default=0.0)
    parser.add_argument("--recogdrive-replay-every-n-steps", type=int, default=1)
    parser.add_argument("--recogdrive-replay-source", choices=("official", "navsim_generated", "mixed"), default="official")
    parser.add_argument("--recogdrive-replay-max-answer-tokens", type=int, default=256)
    parser.add_argument("--recogdrive-replay-train-lora-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recogdrive-replay-train-slots", action="store_true")
    parser.add_argument("--stage1-image-mask-ratio", type=float, default=None)
    parser.add_argument("--stage1-image-mask-ratio-start", type=float, default=None)
    parser.add_argument("--stage1-image-mask-warmup-fraction", type=float, default=0.25)
    parser.add_argument("--stage1-image-dropout-prob", type=float, default=0.0)
    parser.add_argument("--stage1-image-dropout-prob-start", type=float, default=0.0)
    parser.add_argument("--stage1-image-dropout-warmup-fraction", type=float, default=0.25)
    parser.add_argument("--slot-only-use-image-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--slot-only-dyn-loss-weight", type=float, default=0.0)
    parser.add_argument("--slot-only-geo-loss-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-dyn-loss-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-geo-loss-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-loss-warmup-fraction", type=float, default=0.30)
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--probe-fused-loss-weight", type=float, default=None)
    parser.add_argument("--probe-dyn-loss-weight", type=float, default=0.0)
    parser.add_argument("--probe-geo-loss-weight", type=float, default=0.0)
    parser.add_argument("--probe-heading-loss-weight", type=float, default=0.05)
    parser.add_argument("--probe-progress-loss-weight", type=float, default=0.05)
    parser.add_argument("--geo-lateral-profile-loss-weight", type=float, default=0.0)
    parser.add_argument("--geo-heading-profile-loss-weight", type=float, default=0.0)
    parser.add_argument("--hidden-anchor-weight", type=float, default=0.05)
    parser.add_argument("--hidden-anchor-every-n-steps", type=int, default=1)
    parser.add_argument("--replay-ce-loss-warmup-fraction", type=float, default=0.30)
    parser.add_argument("--init-stage1-checkpoint", type=Path, default=None)
    parser.add_argument("--init-vlm-lora-adapter-dir", type=Path, default=None)
    parser.add_argument("--allow-dev-fallback-teachers", action="store_true")
    parser.add_argument("--allow-minimal-prompt", action="store_true")
    parser.add_argument("--allow-replay-only-base", action="store_true")
    parser.add_argument("--save-full-stage1-state", action="store_true")
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
    parser.add_argument("--log-every-steps", type=int, default=50)
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
            "prompt_version": TWO_EXPERT_PROMPT_VERSION,
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
    replay_index = load_replay_index(args.recogdrive_replay_cache_root, max_samples=args.max_samples)
    if float(args.recogdrive_replay_ce_loss_weight) > 0.0 and not replay_index:
        raise ValueError("--recogdrive-replay-ce-loss-weight > 0 requires --recogdrive-replay-cache-root.")
    vggt_dim = resolve_vggt_feature_dim(vggt_root, vggt_index, args.vggt_feature_dim)
    dataset = TwoExpertStage1Dataset(
        args.base_chunk_root,
        jepa_index,
        vggt_index,
        max_samples=args.max_samples,
        chunk_name_pattern=args.chunk_name_pattern,
        teacher_lru_size=int(args.teacher_lru_size),
        require_strict_teachers=not bool(args.allow_dev_fallback_teachers),
        allow_minimal_prompt=bool(args.allow_minimal_prompt),
        replay_index=replay_index,
        require_replay=float(args.recogdrive_replay_ce_loss_weight) > 0.0,
        allow_replay_only_base=bool(args.allow_replay_only_base),
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
    init_payload = _load_checkpoint_payload(args.init_stage1_checkpoint) if args.init_stage1_checkpoint is not None else None
    if args.train_mode == "lora":
        init_lora_dir = _resolve_init_lora_adapter_dir(args, init_payload)
        if init_lora_dir is not None:
            lora_report = apply_lora_adapter_dir_for_training(backbone, init_lora_dir)
        else:
            lora_report = apply_lora(backbone, args)
    module = TwoExpertVLMSFTModule(
        backbone,
        TwoExpertVLMSFTConfig(
            train_mode=args.train_mode,
            allow_full_vlm_sft=bool(args.allow_full_vlm_sft),
            top_layers=int(args.top_layers),
            vggt_feature_dim=vggt_dim,
            slot_only_dyn_loss_weight=float(args.slot_only_dyn_loss_weight),
            slot_only_geo_loss_weight=float(args.slot_only_geo_loss_weight),
            contrastive_dyn_loss_weight=float(args.contrastive_dyn_loss_weight),
            contrastive_geo_loss_weight=float(args.contrastive_geo_loss_weight),
            contrastive_temperature=float(args.contrastive_temperature),
            probe_fused_loss_weight=args.probe_fused_loss_weight,
            probe_dyn_loss_weight=float(args.probe_dyn_loss_weight),
            probe_geo_loss_weight=float(args.probe_geo_loss_weight),
            probe_heading_loss_weight=float(args.probe_heading_loss_weight),
            probe_progress_loss_weight=float(args.probe_progress_loss_weight),
            geo_lateral_profile_loss_weight=float(args.geo_lateral_profile_loss_weight),
            geo_heading_profile_loss_weight=float(args.geo_heading_profile_loss_weight),
            hidden_anchor_weight=float(args.hidden_anchor_weight),
            hidden_anchor_every_n_steps=int(args.hidden_anchor_every_n_steps),
            stage1_image_mask_ratio=args.stage1_image_mask_ratio,
            stage1_image_mask_ratio_start=args.stage1_image_mask_ratio_start,
            stage1_image_mask_warmup_fraction=float(args.stage1_image_mask_warmup_fraction),
            stage1_image_dropout_prob=float(args.stage1_image_dropout_prob),
            stage1_image_dropout_prob_start=float(args.stage1_image_dropout_prob_start),
            stage1_image_dropout_warmup_fraction=float(args.stage1_image_dropout_warmup_fraction),
            slot_only_use_image_memory=bool(args.slot_only_use_image_memory),
            recogdrive_replay_ce_loss_weight=float(args.recogdrive_replay_ce_loss_weight),
            recogdrive_replay_every_n_steps=int(args.recogdrive_replay_every_n_steps),
            recogdrive_replay_source=str(args.recogdrive_replay_source),
            recogdrive_replay_max_answer_tokens=int(args.recogdrive_replay_max_answer_tokens),
            recogdrive_replay_train_lora_only=bool(args.recogdrive_replay_train_lora_only),
            recogdrive_replay_train_slots=bool(args.recogdrive_replay_train_slots),
            contrastive_loss_warmup_fraction=float(args.contrastive_loss_warmup_fraction),
            replay_ce_loss_warmup_fraction=float(args.replay_ce_loss_warmup_fraction),
        ),
    ).to(device)
    init_report: Dict[str, Any] = {}
    if args.init_stage1_checkpoint is not None:
        init_report = load_stage1_initial_state(module, args.init_stage1_checkpoint)
    if distributed:
        module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[local_rank] if device.type == "cuda" else None)
    train_module = module.module if hasattr(module, "module") else module
    optimizer = build_optimizer(train_module, args)
    optimizer.zero_grad(set_to_none=True)
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    is_rank0 = rank == 0
    progress_path = args.output_dir / "train_progress.jsonl"
    total_forward_steps_per_rank = len(dataloader) * int(args.max_epochs)
    metrics = {
        "forward_steps": 0,
        "optimizer_steps": 0,
        "last_loss": None,
        "total_forward_steps_per_rank": int(total_forward_steps_per_rank),
    }
    if is_rank0:
        append_progress_jsonl(
            progress_path,
            {
                "event": "stage1_start",
                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "batch_size_per_gpu": int(args.batch_size),
                "effective_batch_size": int(args.batch_size) * int(world_size) * int(args.grad_accum),
                "grad_accum": int(args.grad_accum),
                "log_every_steps": int(args.log_every_steps),
                "max_epochs": int(args.max_epochs),
                "precision": str(args.precision),
                "train_mode": str(args.train_mode),
                "total_forward_steps_per_rank": int(total_forward_steps_per_rank),
                "world_size": int(world_size),
            },
        )
    for epoch in range(int(args.max_epochs)):
        if sampler is not None:
            sampler.set_epoch(epoch)
        pending_grads = 0
        for step, batch in enumerate(dataloader):
            batch["global_step"] = torch.tensor(metrics["forward_steps"], dtype=torch.long)
            batch["total_forward_steps"] = torch.tensor(total_forward_steps_per_rank, dtype=torch.long)
            batch["train_progress_fraction"] = torch.tensor(
                float(metrics["forward_steps"]) / float(total_forward_steps_per_rank)
                if total_forward_steps_per_rank
                else 0.0,
                dtype=torch.float32,
            )
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
            log_every = max(1, int(args.log_every_steps))
            should_log = is_rank0 and (
                metrics["forward_steps"] == 1
                or metrics["forward_steps"] % log_every == 0
                or step + 1 == len(dataloader)
            )
            if should_log:
                progress = {
                    "event": "train_step",
                    "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "epoch": int(epoch),
                    "step_in_epoch": int(step),
                    "forward_steps": int(metrics["forward_steps"]),
                    "optimizer_steps": int(metrics["optimizer_steps"]),
                    "pending_grads": int(pending_grads),
                    "progress_fraction": float(metrics["forward_steps"]) / float(total_forward_steps_per_rank)
                    if total_forward_steps_per_rank
                    else None,
                    "loss": _metric_float(out.get("loss")),
                    "dyn_loss": _metric_float(out.get("dyn_loss")),
                    "geo_loss": _metric_float(out.get("geo_loss")),
                    "probe_loss": _metric_float(out.get("probe_loss")),
                    "probe_dyn_loss": _metric_float(out.get("probe_dyn_loss")),
                    "probe_geo_loss": _metric_float(out.get("probe_geo_loss")),
                    "probe_heading_loss": _metric_float(out.get("probe_heading_loss")),
                    "probe_progress_loss": _metric_float(out.get("probe_progress_loss")),
                    "slot_only_dyn_loss": _metric_float(out.get("slot_only_dyn_loss")),
                    "slot_only_geo_loss": _metric_float(out.get("slot_only_geo_loss")),
                    "contrastive_dyn_loss": _metric_float(out.get("contrastive_dyn_loss")),
                    "contrastive_geo_loss": _metric_float(out.get("contrastive_geo_loss")),
                    "recogdrive_replay_ce_loss": _metric_float(out.get("recogdrive_replay_ce_loss")),
                    "recogdrive_replay_token_count": _metric_float(out.get("recogdrive_replay_token_count")),
                    "recogdrive_replay_ce_loss_weight_effective": _metric_float(
                        out.get("recogdrive_replay_ce_loss_weight_effective")
                    ),
                    "hidden_anchor_loss": _metric_float(out.get("hidden_anchor_loss")),
                    "stage1_effective_image_mask_ratio": _metric_float(out.get("stage1_effective_image_mask_ratio")),
                    "stage1_effective_image_dropout_prob": _metric_float(out.get("stage1_effective_image_dropout_prob")),
                    "contrastive_dyn_loss_weight_effective": _metric_float(out.get("contrastive_dyn_loss_weight_effective")),
                    "contrastive_geo_loss_weight_effective": _metric_float(out.get("contrastive_geo_loss_weight_effective")),
                    "h_dyn_norm": _metric_float(out.get("h_dyn_norm")),
                    "h_geo_norm": _metric_float(out.get("h_geo_norm")),
                }
                append_progress_jsonl(progress_path, progress)
        if pending_grads > 0:
            torch.nn.utils.clip_grad_norm_(train_module.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            metrics["optimizer_steps"] += 1
            if is_rank0:
                append_progress_jsonl(
                    progress_path,
                    {
                        "event": "optimizer_flush",
                        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "epoch": int(epoch),
                        "forward_steps": int(metrics["forward_steps"]),
                        "optimizer_steps": int(metrics["optimizer_steps"]),
                    },
                )
    if int(os.getenv("RANK", "0")) == 0:
        metadata = {
            "schema": "two_expert_slot_stage1_vlm_sft_v1",
            "train_mode": args.train_mode,
            "teacher_loading": "lazy_index_paths",
            "strict_teachers_required": not bool(args.allow_dev_fallback_teachers),
            "prompt_version": TWO_EXPERT_PROMPT_VERSION,
            "vggt_feature_dim": int(vggt_dim),
            "jepa_cache_root": str(jepa_root),
            "vggt_cache_root": str(vggt_root),
            "lr_vlm": float(args.lr_vlm),
            "lr_slots_adapters": float(args.lr_slots_adapters),
            "weight_decay": float(args.weight_decay),
            "base_chunk_root": str(args.base_chunk_root),
            "allow_replay_only_base": bool(args.allow_replay_only_base),
            "vlm_path": str(args.vlm_path),
            "lora_report": lora_report,
            "init_report": init_report,
            "recogdrive_replay_cache_root": str(args.recogdrive_replay_cache_root) if args.recogdrive_replay_cache_root else None,
            "recogdrive_replay_ce_loss_weight": float(args.recogdrive_replay_ce_loss_weight),
            "recogdrive_replay_every_n_steps": int(args.recogdrive_replay_every_n_steps),
            "replay_ce_loss_warmup_fraction": float(args.replay_ce_loss_warmup_fraction),
            "stage1_image_mask_ratio": args.stage1_image_mask_ratio,
            "stage1_image_mask_ratio_start": args.stage1_image_mask_ratio_start,
            "stage1_image_mask_warmup_fraction": float(args.stage1_image_mask_warmup_fraction),
            "stage1_image_dropout_prob": float(args.stage1_image_dropout_prob),
            "stage1_image_dropout_prob_start": float(args.stage1_image_dropout_prob_start),
            "stage1_image_dropout_warmup_fraction": float(args.stage1_image_dropout_warmup_fraction),
            "slot_only_use_image_memory": bool(args.slot_only_use_image_memory),
            "slot_only_dyn_loss_weight": float(args.slot_only_dyn_loss_weight),
            "slot_only_geo_loss_weight": float(args.slot_only_geo_loss_weight),
            "contrastive_dyn_loss_weight": float(args.contrastive_dyn_loss_weight),
            "contrastive_geo_loss_weight": float(args.contrastive_geo_loss_weight),
            "contrastive_loss_warmup_fraction": float(args.contrastive_loss_warmup_fraction),
            "probe_fused_loss_weight": args.probe_fused_loss_weight,
            "probe_dyn_loss_weight": float(args.probe_dyn_loss_weight),
            "probe_geo_loss_weight": float(args.probe_geo_loss_weight),
        }
        save_stage1_outputs(
            train_module,
            args.output_dir,
            metadata,
            save_full_stage1_state=bool(args.save_full_stage1_state),
        )
        write_json(args.output_dir / "metrics.json", metrics)
        append_progress_jsonl(
            progress_path,
            {
                "event": "stage1_complete",
                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **metrics,
            },
        )
    if distributed:
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
