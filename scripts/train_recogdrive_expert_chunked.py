#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from transformers.feature_extraction_utils import BatchFeature  # noqa: E402
except ModuleNotFoundError:
    from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs  # noqa: E402
    _install_dependency_stubs()
    from transformers.feature_extraction_utils import BatchFeature  # noqa: E402

from navsim.agents.recogdrive.expert_cache import iter_index, load_sample, validate_sample_payload  # noqa: E402
from navsim.agents.recogdrive.recogdrive_diffusion_planner import (  # noqa: E402
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)

EXPERT_MARKERS = (
    "jepa_projector", "vggt_projector", "jepa_adapter", "vggt_adapter",
    "jepa_alignment_head", "vggt_alignment_head", "jepa_type_embedding", "vggt_type_embedding",
    "z_jepa_type_embedding", "z_vggt_type_embedding", "jepa_gate", "vggt_gate",
    "jepa_horizon_conditioner", "vggt_horizon_conditioner", "branch_logits",
)
GATE_MARKERS = ("jepa_gate", "vggt_gate", "branch_logits")
ACTION_HEAD_MARKERS = (
    "feature_encoder", "his_traj_encoder", "ego_status_encoder", "action_encoder",
    "fusion_projector", "model", "action_decoder", "position_embedding",
)


def resolve_index_path(chunk_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_file():
        return path
    if not path.is_absolute():
        candidate = chunk_dir / path
        if candidate.is_file():
            return candidate
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunked ReCogDrive expert-token training from cached Stage1-base features.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-il-checkpoint", type=Path, default=None, help="Deprecated alias for --init-policy-checkpoint.")
    parser.add_argument("--init-policy-checkpoint", type=Path, default=None, help="Optional policy/action-head checkpoint. Leave unset to train from config initialization.")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--resume-mode", choices=("weights-only", "full"), default="weights-only")
    parser.add_argument("--recogdrive-vlm-path", type=Path, default=None)
    parser.add_argument("--chunk-cache-root", type=Path, default=None)
    parser.add_argument("--chunk-cache-dir", type=Path, default=None)
    parser.add_argument("--chunk-name-pattern", default="chunk_*")
    parser.add_argument("--auto-build-next-chunk", action="store_true")
    parser.add_argument("--delete-old-chunk", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=None)
    parser.add_argument("--epochs-per-chunk", type=int, default=1)
    parser.add_argument(
        "--global-epochs",
        type=int,
        default=None,
        help="Epoch-major training over all chunks. Use this for Stage2-aligned full-cache training.",
    )
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--num-optimizer-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--flat-global-dataset",
        action="store_true",
        help="For --global-epochs, train from one flattened all-chunk dataset/DataLoader to avoid per-chunk worker restarts.",
    )
    parser.add_argument("--lr-scheduler", choices=("none", "official-cosine"), default="none")
    parser.add_argument("--lr-scheduler-epochs", type=int, default=200)
    parser.add_argument("--lr-scheduler-start-epoch", type=int, default=0)
    parser.add_argument("--lr-warmup-epochs", type=int, default=3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--final-check-precision", choices=("auto", "train", "fp32"), default="auto")
    parser.add_argument("--train-expert-only", action="store_true")
    parser.add_argument("--freeze-base-action-head", action="store_true")
    parser.add_argument("--freeze-expert", action="store_true")
    parser.add_argument("--freeze-vlm", action="store_true", default=True)
    parser.add_argument("--lr-expert", type=float, default=1e-4)
    parser.add_argument("--lr-expert-gate", type=float, default=None)
    parser.add_argument("--lr-action-head", type=float, default=2e-5)
    parser.add_argument("--jepa-align-weight", type=float, default=None)
    parser.add_argument("--vggt-align-weight", type=float, default=None)
    parser.add_argument("--diffusion-loss-weight", type=float, default=None)
    parser.add_argument("--expert-gate-init", type=float, default=None)
    parser.add_argument("--alignment-ramp-start-epoch", type=int, default=0)
    parser.add_argument("--alignment-ramp-end-epoch", type=int, default=0)
    parser.add_argument("--expert-context-ramp-start-epoch", type=int, default=0)
    parser.add_argument("--expert-context-ramp-end-epoch", type=int, default=0)
    parser.add_argument("--debug-overfit", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--true-bf16-weights", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-dummy-cache", action="store_true")
    return parser.parse_args()


def dtype_from_precision(precision: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed() -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA.")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
    return world_size > 1, world_size, rank, local_rank


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_rank0(rank: int) -> bool:
    return rank == 0


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def build_planner(cfg_dict: Dict[str, Any], args: argparse.Namespace) -> ReCogDriveDiffusionPlanner:
    cfg = ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg={
            "num_heads": 8,
            "head_dim": 48,
            "num_layers": int(cfg_dict.get("num_dit_layers", 16)),
            "output_dim": 512,
            "dropout": 0.0,
            "attention_bias": True,
            "norm_eps": 1e-5,
            "interleave_attention": True,
        },
        action_dim=int(cfg_dict.get("action_dim", 3)),
        action_horizon=int(cfg_dict.get("action_horizon", 8)),
        input_embedding_dim=int(cfg_dict.get("planner_dim", 384)),
        planner_dim=int(cfg_dict.get("planner_dim", 384)),
        hidden_size=1024,
        sampling_method=str(cfg_dict.get("sampling_method", "ddim")),
        num_inference_steps=int(cfg_dict.get("num_inference_steps", 5)),
        model_dtype={"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[args.precision],
    )
    for key, value in cfg_dict.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.vlm_size = "small"
    cfg.use_expert_features = as_bool(cfg_dict.get("use_expert_features"), True)
    cfg.planner_dim = 384
    if args.jepa_align_weight is not None:
        cfg.jepa_alignment_weight = args.jepa_align_weight
    if args.vggt_align_weight is not None:
        cfg.vggt_alignment_weight = args.vggt_align_weight
    if args.diffusion_loss_weight is not None:
        if args.diffusion_loss_weight < 0.0:
            raise ValueError("--diffusion-loss-weight must be non-negative.")
        cfg.diffusion_loss_weight = args.diffusion_loss_weight
    if args.expert_gate_init is not None:
        cfg.jepa_gate_init = args.expert_gate_init
        cfg.vggt_gate_init = args.expert_gate_init
    if as_bool(cfg_dict.get("use_alignment_loss"), True) is False:
        cfg.expert_alignment_weight = 0.0
        cfg.jepa_alignment_weight = 0.0
        cfg.vggt_alignment_weight = 0.0
    return ReCogDriveDiffusionPlanner(cfg)


def checkpoint_candidate_score(path: Path) -> tuple[int, int, str]:
    name = path.name.lower()
    size = path.stat().st_size if path.is_file() else 0
    score = 0
    if "il" in name:
        score += 1000
    if "model" in name:
        score += 500
    if path.suffix == ".safetensors":
        score += 100
    if path.suffix in {".ckpt", ".pth", ".pt"}:
        score += 75
    return score, size, str(path)


def find_weight_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    suffixes = (".ckpt", ".pth", ".pt", ".safetensors", ".bin")
    files: List[Path] = []
    for suffix in suffixes:
        files.extend(p for p in path.rglob(f"*{suffix}") if p.is_file())
    files = sorted(set(files), key=checkpoint_candidate_score, reverse=True)
    if not files:
        raise FileNotFoundError(f"No checkpoint/weight files found under {path}")
    print(f"Checkpoint candidates under {path}:")
    for idx, file in enumerate(files):
        marker = " <= selected first" if idx == 0 else ""
        print(f"  - {file} ({file.stat().st_size} bytes){marker}")
    return files


def load_state_file(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return dict(load_file(str(path), device="cpu"))
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint {path} did not contain a state dict.")
    return {key: value for key, value in obj.items() if isinstance(value, torch.Tensor)}


def is_expert_key(key: str) -> bool:
    return any(marker in key for marker in EXPERT_MARKERS)


def normalize_key(key: str) -> str:
    for prefix in ("agent.action_head.", "action_head."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def shape_safe_load(planner: ReCogDriveDiffusionPlanner, checkpoint_path: Path, *, strict_original: bool = True) -> Dict[str, Any]:
    model_state = planner.state_dict()
    loaded: Dict[str, torch.Tensor] = {}
    unexpected: List[str] = []
    mismatches: List[str] = []
    files = find_weight_files(checkpoint_path)
    for file in files:
        state = load_state_file(file)
        for raw_key, value in state.items():
            key = normalize_key(raw_key)
            if key not in model_state:
                unexpected.append(key)
                continue
            if tuple(value.shape) != tuple(model_state[key].shape):
                msg = f"{key}: checkpoint {tuple(value.shape)} vs model {tuple(model_state[key].shape)}"
                if is_expert_key(key):
                    continue
                mismatches.append(msg)
                continue
            loaded.setdefault(key, value)
    if strict_original and mismatches:
        raise RuntimeError("Base checkpoint shape mismatch for original ReCogDrive parameters:\n" + "\n".join(mismatches))
    incompatible = planner.load_state_dict(loaded, strict=False)
    missing_expert = [key for key in incompatible.missing_keys if is_expert_key(key)]
    missing_other = [key for key in incompatible.missing_keys if not is_expert_key(key)]
    print(f"Loaded {len(loaded)} matching keys from {checkpoint_path}")
    print(f"Expected missing expert keys: {len(missing_expert)}")
    if missing_other:
        print("Missing non-expert keys:")
        for key in missing_other[:50]:
            print(f"  - {key}")
    if unexpected:
        print(f"Unexpected checkpoint keys skipped: {len(unexpected)}")
    return {
        "loaded_key_count": len(loaded),
        "missing_expert_key_count": len(missing_expert),
        "missing_non_expert_key_count": len(missing_other),
        "unexpected_key_count": len(unexpected),
        "shape_mismatch_count": len(mismatches),
        "checkpoint_files": [str(file) for file in files],
    }


class ChunkDataset(Dataset):
    def __init__(
        self,
        chunk_dir: Path,
        *,
        max_samples: Optional[int] = None,
        allow_dummy_cache: bool = False,
        require_jepa: bool = True,
        require_vggt: bool = True,
        require_targets: bool = True,
    ) -> None:
        self.chunk_dir = chunk_dir
        metadata_path = chunk_dir / "metadata.json"
        self.metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        if self.metadata.get("is_dummy") and not allow_dummy_cache:
            raise RuntimeError("Refusing to train on dummy expert cache. Pass --allow-dummy-cache only for smoke/debug.")
        self.require_jepa = require_jepa
        self.require_vggt = require_vggt
        self.require_targets = require_targets
        self.records = list(iter_index(chunk_dir))
        if max_samples is not None:
            self.records = self.records[:max_samples]
        if not self.records:
            raise RuntimeError(f"No records found in {chunk_dir}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        path = resolve_index_path(self.chunk_dir, record["path"])
        sample = load_sample(path)
        validate_sample_payload(
            sample,
            require_jepa=self.require_jepa,
            require_vggt=self.require_vggt,
            require_targets=self.require_targets,
        )
        for key in ("last_hidden_state", "history_trajectory", "status_feature", "trajectory"):
            if key not in sample:
                raise KeyError(f"{path} missing required training key '{key}'. Rebuild chunk with --build-vlm-hidden.")
        return {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in sample.items()
        }


class FlatChunkDataset(Dataset):
    def __init__(
        self,
        chunk_paths: List[Path],
        *,
        max_samples: Optional[int] = None,
        allow_dummy_cache: bool = False,
        require_jepa: bool = True,
        require_vggt: bool = True,
        require_targets: bool = True,
    ) -> None:
        self.require_jepa = require_jepa
        self.require_vggt = require_vggt
        self.require_targets = require_targets
        self.records: List[Tuple[Path, Dict[str, Any]]] = []
        for chunk_dir in chunk_paths:
            metadata_path = chunk_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
            if metadata.get("is_dummy") and not allow_dummy_cache:
                raise RuntimeError("Refusing to train on dummy expert cache. Pass --allow-dummy-cache only for smoke/debug.")
            for record in iter_index(chunk_dir):
                self.records.append((chunk_dir, record))
                if max_samples is not None and len(self.records) >= max_samples:
                    break
            if max_samples is not None and len(self.records) >= max_samples:
                break
        if not self.records:
            raise RuntimeError("No records found in flattened chunk dataset.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        chunk_dir, record = self.records[idx]
        path = resolve_index_path(chunk_dir, record["path"])
        sample = load_sample(path)
        validate_sample_payload(
            sample,
            require_jepa=self.require_jepa,
            require_vggt=self.require_vggt,
            require_targets=self.require_targets,
        )
        for key in ("last_hidden_state", "history_trajectory", "status_feature", "trajectory"):
            if key not in sample:
                raise KeyError(f"{path} missing required training key '{key}'. Rebuild chunk with --build-vlm-hidden.")
        return {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in sample.items()
        }


def collate(samples: List[Dict[str, Any]]) -> Tuple[torch.Tensor, BatchFeature]:
    last_hidden_state = pad_sequence([sample["last_hidden_state"].float() for sample in samples], batch_first=True, padding_value=0.0)
    his = torch.stack([sample["history_trajectory"].float().view(-1) for sample in samples], dim=0)
    status = torch.stack([sample["status_feature"].float() for sample in samples], dim=0)
    action = torch.stack([sample["trajectory"].float() for sample in samples], dim=0)
    data: Dict[str, torch.Tensor] = {"his_traj": his, "status_feature": status, "action": action}
    for key in ("jepa_context_tokens", "jepa_target_tokens", "vggt_context_tokens", "vggt_target_tokens"):
        if key in samples[0]:
            data[key] = torch.stack([sample[key].float() for sample in samples], dim=0)
    return last_hidden_state, BatchFeature(data=data)


def set_trainable(planner: ReCogDriveDiffusionPlanner, args: argparse.Namespace) -> None:
    if args.freeze_expert and (args.train_expert_only or args.freeze_base_action_head):
        raise ValueError("--freeze-expert leaves no trainable parameters when combined with --train-expert-only or --freeze-base-action-head.")
    for name, parameter in planner.named_parameters():
        expert = is_expert_key(name)
        if args.train_expert_only or args.freeze_base_action_head:
            parameter.requires_grad = expert and not args.freeze_expert
        else:
            parameter.requires_grad = (expert and not args.freeze_expert) or any(marker in name for marker in ACTION_HEAD_MARKERS)


def optimizer_for(planner: ReCogDriveDiffusionPlanner, args: argparse.Namespace) -> torch.optim.Optimizer:
    expert_params = []
    action_params = []
    gate_params = []
    for name, param in planner.named_parameters():
        if not param.requires_grad:
            continue
        is_gate = any(marker in name for marker in GATE_MARKERS)
        if args.lr_expert_gate is not None and is_gate:
            gate_params.append(param)
        elif is_expert_key(name):
            expert_params.append(param)
        else:
            action_params.append(param)
    groups = []
    if expert_params and args.lr_expert > 0:
        groups.append({
            "params": expert_params,
            "lr": args.lr_expert,
            "base_lr": args.lr_expert,
            "weight_decay": 1e-4,
            "name": "expert",
        })
    if gate_params and args.lr_expert_gate is not None and args.lr_expert_gate > 0:
        groups.append({
            "params": gate_params,
            "lr": args.lr_expert_gate,
            "base_lr": args.lr_expert_gate,
            "weight_decay": 0.0,
            "name": "expert_gate",
        })
    if action_params and args.lr_action_head > 0:
        groups.append({
            "params": action_params,
            "lr": args.lr_action_head,
            "base_lr": args.lr_action_head,
            "weight_decay": 1e-4,
            "name": "action_head",
        })
    if not groups:
        raise RuntimeError("No trainable parameter groups. Check freeze flags and learning rates.")
    return torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))


def chunk_dirs(args: argparse.Namespace) -> List[Path]:
    if args.chunk_cache_dir is not None:
        return [args.chunk_cache_dir]
    if args.chunk_cache_root is None:
        raise ValueError("Set --chunk-cache-dir or --chunk-cache-root")
    chunks = sorted(path for path in args.chunk_cache_root.glob(args.chunk_name_pattern) if path.is_dir())
    if args.num_chunks is not None:
        chunks = chunks[:args.num_chunks]
    if not chunks:
        raise FileNotFoundError(f"No {args.chunk_name_pattern!r} directories found under {args.chunk_cache_root}")
    return chunks


def save_checkpoint(path: Path, planner: ReCogDriveDiffusionPlanner, optimizer: torch.optim.Optimizer, step: int, cfg: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": planner.state_dict(), "optimizer": optimizer.state_dict(), "global_step": step, "config": cfg, "metrics": metrics}, path)


def finite_scalar(value: torch.Tensor, name: str) -> float:
    if not torch.isfinite(value.detach()).all():
        raise RuntimeError(f"Non-finite {name}: {value}")
    return float(value.detach().float().item())


def expert_grad_present(planner: ReCogDriveDiffusionPlanner) -> bool:
    for name, param in planner.named_parameters():
        if param.requires_grad and is_expert_key(name) and param.grad is not None:
            grad = param.grad.detach()
            if torch.isfinite(grad).all() and grad.abs().sum().item() > 0:
                return True
    return False


def dtype_label(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.float32:
        return "fp32"
    return str(dtype).replace("torch.", "")


def count_tensor_dtypes(tensors: List[torch.Tensor]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for tensor in tensors:
        counts[dtype_label(tensor.dtype)] = counts.get(dtype_label(tensor.dtype), 0) + 1
    return dict(sorted(counts.items()))


def model_param_dtype_counts(planner: ReCogDriveDiffusionPlanner) -> Dict[str, int]:
    return count_tensor_dtypes([param.detach() for param in planner.parameters()])


def model_buffer_dtype_counts(planner: ReCogDriveDiffusionPlanner) -> Dict[str, int]:
    return count_tensor_dtypes([buffer.detach() for buffer in planner.buffers() if isinstance(buffer, torch.Tensor)])


def checkpoint_state_dtype_counts(checkpoint_path: Optional[Path]) -> Dict[str, int]:
    if checkpoint_path is None:
        return {}
    counts: Dict[str, int] = {}
    for file in find_weight_files(checkpoint_path):
        state = load_state_file(file)
        for value in state.values():
            label = dtype_label(value.dtype)
            counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def optimizer_group_metadata(optimizer: torch.optim.Optimizer) -> Tuple[List[str], List[float], List[float]]:
    names = [str(group.get("name", idx)) for idx, group in enumerate(optimizer.param_groups)]
    lrs_out = [float(group["lr"]) for group in optimizer.param_groups]
    weight_decays = [float(group.get("weight_decay", 0.0)) for group in optimizer.param_groups]
    return names, lrs_out, weight_decays


def final_pred_finite_check(
    planner: ReCogDriveDiffusionPlanner,
    last_batch: Tuple[torch.Tensor, BatchFeature],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    original_param_dtypes = {
        name: param.dtype
        for name, param in planner.named_parameters()
        if param.is_floating_point()
    }
    was_training = planner.training
    try:
        planner.to(device=device, dtype=dtype)
        planner.eval()
        vl_features, action_input = last_batch
        eval_input = BatchFeature(data={
            key: value.to(device=device, dtype=dtype) if isinstance(value, torch.Tensor) else value
            for key, value in action_input.items()
            if key not in {"action", "jepa_target_tokens", "vggt_target_tokens"}
        })
        with torch.no_grad():
            pred = planner.get_action(vl_features.to(device=device, dtype=dtype), eval_input, deterministic=True)["pred_traj"]
        finite = bool(torch.isfinite(pred).all().item())
        return {
            "finite": finite,
            "dtype": dtype_label(dtype),
            "shape": list(pred.shape),
            "max_abs": float(pred.detach().float().abs().max().item()) if pred.numel() else 0.0,
        }
    finally:
        with torch.no_grad():
            for name, param in planner.named_parameters():
                original_dtype = original_param_dtypes.get(name)
                if original_dtype is not None and param.dtype != original_dtype:
                    param.data = param.data.to(dtype=original_dtype)
        planner.train(was_training)


def current_gpu_memory(device: torch.device) -> Optional[int]:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def lrs(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    return {str(group.get("name", i)): float(group["lr"]) for i, group in enumerate(optimizer.param_groups)}


def scalar_or_zero(output: Dict[str, Any], key: str, device: torch.device) -> torch.Tensor:
    value = output.get(key)
    if isinstance(value, torch.Tensor):
        return value.detach().float().reshape(())
    return torch.tensor(0.0, device=device, dtype=torch.float32)


def reduce_metrics(output: Dict[str, Any], *, device: torch.device, distributed: bool, world_size: int) -> Dict[str, float]:
    keys = [
        "loss",
        "diffusion_loss",
        "jepa_alignment_loss",
        "vggt_alignment_loss",
        "jepa_gate_value",
        "vggt_gate_value",
        "branch_weight_vlm",
        "branch_weight_jepa",
        "branch_weight_vggt",
        "expert_context_scale",
        "expert_horizon_residual_scale",
    ]
    values = torch.stack([scalar_or_zero(output, key, device) for key in keys]).to(device=device)
    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= float(world_size)
    if not torch.isfinite(values).all():
        bad = {key: float(value.detach().cpu()) for key, value in zip(keys, values)}
        raise RuntimeError(f"Non-finite reduced metrics: {bad}")
    return {key: float(value.detach().cpu().item()) for key, value in zip(keys, values)}


def official_cosine_lr(args: argparse.Namespace, base_lr: float, epoch_index: int) -> float:
    warmup_epochs = max(int(args.lr_warmup_epochs), 0)
    total_epochs = max(int(args.lr_scheduler_epochs), 1)
    if warmup_epochs > 0 and epoch_index < warmup_epochs:
        return base_lr * float(epoch_index + 1) / float(warmup_epochs)
    progress_denominator = max(total_epochs - warmup_epochs, 1)
    progress = min(max(epoch_index - warmup_epochs, 0) / progress_denominator, 1.0)
    return args.min_lr + 0.5 * (base_lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))


def apply_epoch_lr(optimizer: torch.optim.Optimizer, args: argparse.Namespace, epoch_index: int) -> None:
    if args.lr_scheduler != "official-cosine":
        return
    scheduler_epoch = epoch_index + max(int(getattr(args, "lr_scheduler_start_epoch", 0)), 0)
    for group in optimizer.param_groups:
        base_lr = float(group.get("base_lr", group.get("initial_lr", group["lr"])))
        scale = float(group.get("lr_scale", 1.0))
        group["lr"] = official_cosine_lr(args, base_lr, scheduler_epoch) * scale


def ramp_factor(epoch_index: int, start_epoch: int, end_epoch: int) -> float:
    if end_epoch <= start_epoch:
        return 1.0
    if epoch_index < start_epoch:
        return 0.0
    if epoch_index >= end_epoch:
        return 1.0
    return float(epoch_index - start_epoch) / float(end_epoch - start_epoch)


def schedule_targets(planner: ReCogDriveDiffusionPlanner) -> Dict[str, float]:
    return {
        "jepa_alignment_weight": float(planner.config.jepa_alignment_weight),
        "vggt_alignment_weight": float(planner.config.vggt_alignment_weight),
        "expert_context_scale": float(getattr(planner.config, "expert_context_scale", 1.0)),
        "expert_horizon_residual_scale": float(getattr(planner.config, "expert_horizon_residual_scale", 0.0)),
    }


def apply_epoch_expert_schedules(
    planner: ReCogDriveDiffusionPlanner,
    args: argparse.Namespace,
    epoch_index: int,
    targets: Dict[str, float],
) -> None:
    align_factor = ramp_factor(epoch_index, args.alignment_ramp_start_epoch, args.alignment_ramp_end_epoch)
    context_factor = ramp_factor(epoch_index, args.expert_context_ramp_start_epoch, args.expert_context_ramp_end_epoch)
    planner.config.jepa_alignment_weight = targets["jepa_alignment_weight"] * align_factor
    planner.config.vggt_alignment_weight = targets["vggt_alignment_weight"] * align_factor
    planner.config.expert_context_scale = targets["expert_context_scale"] * context_factor
    planner.config.expert_horizon_residual_scale = targets["expert_horizon_residual_scale"] * context_factor


def optimizer_step(
    planner: ReCogDriveDiffusionPlanner,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
) -> Optional[float]:
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(planner.parameters(), 1.0)
    grad_norm_value = float(grad_norm.detach().float().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return grad_norm_value


def write_final_report(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# ReCogDrive Expert Chunked Training Report",
        "",
        f"Output dir: `{summary['output_dir']}`",
        f"Global steps: {summary['global_step']}",
        f"Optimizer steps: {summary.get('optimizer_step')}",
        f"Completed global epochs: {summary.get('completed_global_epochs')}",
        f"Effective batch size: {summary.get('effective_batch_size')}",
        f"World size: {summary.get('world_size')}",
        f"LR scheduler: {summary.get('lr_scheduler')}",
        f"Diffusion loss weight: {summary.get('diffusion_loss_weight')}",
        f"Freeze expert: {summary.get('freeze_expert')}",
        f"Best loss: {summary.get('best_loss')}",
        f"Last loss: {summary.get('last_loss')}",
        f"Expert gradients observed: {summary.get('expert_grad_observed')}",
        f"Prediction finite check: {summary.get('pred_traj_finite')}",
        f"Prediction finite check precision: {summary.get('pred_traj_finite_check_precision')}",
        f"Prediction finite checks: `{json.dumps(summary.get('pred_traj_finite_checks', {}), sort_keys=True)}`",
        f"Latest checkpoint: `{summary.get('latest_checkpoint')}`",
        f"Best checkpoint: `{summary.get('best_checkpoint')}`",
        "",
        "## Chunks",
        "",
    ]
    for chunk in summary.get("chunks", []):
        lines.append(f"- `{chunk}`")
    if summary.get("failure"):
        lines.extend(["", "## Failure", "", str(summary["failure"])])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    if args.resume_mode == "full":
        raise NotImplementedError("Full resume is not implemented; use weights-only for continuation ablations.")
    if args.true_bf16_weights and args.precision != "bf16":
        raise ValueError("--true-bf16-weights is only valid with --precision bf16.")
    if args.num_optimizer_steps is not None and args.num_optimizer_steps <= 0:
        raise ValueError("--num-optimizer-steps must be positive when set.")
    distributed, world_size, rank, local_rank = init_distributed()
    cfg_dict = load_yaml(args.config)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_precision(args.precision) if device.type == "cuda" else torch.float32
    planner = build_planner(cfg_dict, args).to(device)

    load_report: Dict[str, Any] = {
        "stage1_base_path": str(args.recogdrive_vlm_path) if args.recogdrive_vlm_path else None,
        "resume_mode": args.resume_mode,
        "optimizer_restored": False,
        "rng_restored": False,
        "scheduler_restored": False,
    }
    loaded_checkpoint_state_dtype_counts: Dict[str, int] = {}
    if args.resume_from:
        loaded_checkpoint_state_dtype_counts = checkpoint_state_dtype_counts(args.resume_from)
        load_report.update(shape_safe_load(planner, args.resume_from, strict_original=False))
        load_report.update({
            "init_mode": "resume_from",
            "init_policy_checkpoint": str(args.resume_from),
            "loaded_checkpoint_state_dtype_counts": loaded_checkpoint_state_dtype_counts,
        })
    else:
        init_policy_checkpoint = args.init_policy_checkpoint or args.base_il_checkpoint
        if init_policy_checkpoint:
            loaded_checkpoint_state_dtype_counts = checkpoint_state_dtype_counts(init_policy_checkpoint)
            load_report.update(shape_safe_load(planner, init_policy_checkpoint, strict_original=True))
            load_report.update({
                "init_mode": "init_policy_checkpoint" if args.init_policy_checkpoint else "base_il_checkpoint_compat",
                "init_policy_checkpoint": str(init_policy_checkpoint),
                "loaded_checkpoint_state_dtype_counts": loaded_checkpoint_state_dtype_counts,
            })
        else:
            load_report.update({
                "init_mode": "random_policy_from_config",
                "init_policy_checkpoint": None,
                "loaded_key_count": 0,
                "loaded_checkpoint_state_dtype_counts": loaded_checkpoint_state_dtype_counts,
            })

    if args.true_bf16_weights:
        planner = planner.to(dtype=torch.bfloat16)

    set_trainable(planner, args)
    optimizer = optimizer_for(planner, args)
    expert_schedule_targets = schedule_targets(planner)
    train_model = planner
    if distributed:
        train_model = DistributedDataParallel(
            planner,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            static_graph=True,
        )
    if is_rank0(rank):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        train_args_payload = vars(args).copy()
        train_args_payload.update({"distributed": distributed, "world_size": world_size})
        (args.output_dir / "train_args.json").write_text(json.dumps(train_args_payload, default=str, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (args.output_dir / "checkpoint_load.json").write_text(json.dumps(load_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        group_names, group_lrs, group_weight_decays = optimizer_group_metadata(optimizer)
        precision_report = {
            "requested_precision": args.precision,
            "true_bf16_weights": bool(args.true_bf16_weights),
            "model_param_dtype_counts": model_param_dtype_counts(planner),
            "model_buffer_dtype_counts": model_buffer_dtype_counts(planner),
            "loaded_checkpoint_state_dtype_counts": loaded_checkpoint_state_dtype_counts,
            "optimizer_group_names": group_names,
            "optimizer_group_lrs": group_lrs,
            "optimizer_group_weight_decay": group_weight_decays,
            "resume_mode": args.resume_mode,
            "optimizer_restored": False,
            "rng_restored": False,
            "scheduler_restored": False,
        }
        (args.output_dir / "precision_report.json").write_text(json.dumps(precision_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if distributed:
        dist.barrier()
    log_path = args.output_dir / "train_log.jsonl"
    log_fp = log_path.open("w", encoding="utf-8") if is_rank0(rank) else None

    global_step = 0
    optimizer_step_count = 0
    pending_grad_steps = 0
    completed_global_epochs = 0
    best_loss = math.inf
    last_loss: Optional[float] = None
    last_batch: Optional[Tuple[torch.Tensor, BatchFeature]] = None
    expert_grad_observed = False
    planner.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(args.precision == "fp16" and device.type == "cuda"))
    chunks = chunk_dirs(args)
    optimizer.zero_grad(set_to_none=True)
    stop = False
    require_jepa = bool(planner.config.use_expert_features and planner.config.use_jepa)
    require_vggt = bool(planner.config.use_expert_features and planner.config.use_vggt)
    require_targets = bool(planner.config.use_expert_features and (planner.config.jepa_alignment_weight > 0 or planner.config.vggt_alignment_weight > 0))
    summary: Dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "chunks": [str(chunk) for chunk in chunks],
        "global_step": 0,
        "optimizer_step": 0,
        "completed_global_epochs": 0,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps * world_size,
        "world_size": world_size,
        "lr_scheduler": args.lr_scheduler,
        "diffusion_loss_weight": float(planner.config.diffusion_loss_weight),
        "freeze_expert": bool(args.freeze_expert),
        "expert_grad_observed": False,
        "pred_traj_finite": None,
        "latest_checkpoint": str(args.output_dir / "latest.ckpt"),
        "best_checkpoint": str(args.output_dir / "best.ckpt"),
    }
    pred_finite: Optional[bool] = None
    pred_check_precision: Optional[str] = None
    pred_checks: Dict[str, Any] = {}

    def make_loader(dataset: Dataset, sampler: Optional[DistributedSampler]) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        loader_kwargs: Dict[str, Any] = {
            "batch_size": args.batch_size,
            "shuffle": sampler is None,
            "sampler": sampler,
            "collate_fn": collate,
            "num_workers": args.num_workers,
            "pin_memory": device.type == "cuda",
            "generator": generator,
        }
        if args.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = args.prefetch_factor
        return DataLoader(dataset, **loader_kwargs)

    def training_passes():
        if args.global_epochs is not None:
            if args.global_epochs <= 0:
                raise ValueError("--global-epochs must be positive when set.")
            if args.flat_global_dataset:
                flat_dataset = FlatChunkDataset(
                    chunks,
                    max_samples=args.max_samples,
                    allow_dummy_cache=args.allow_dummy_cache,
                    require_jepa=require_jepa,
                    require_vggt=require_vggt,
                    require_targets=require_targets,
                )
                flat_sampler = DistributedSampler(
                    flat_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    drop_last=False,
                ) if distributed else None
                flat_loader = make_loader(flat_dataset, flat_sampler)
                for global_epoch in range(args.global_epochs):
                    apply_epoch_lr(optimizer, args, global_epoch)
                    apply_epoch_expert_schedules(planner, args, global_epoch, expert_schedule_targets)
                    if flat_sampler is not None:
                        flat_sampler.set_epoch(global_epoch)
                    yield global_epoch, -1, None, 0, flat_loader
                return
            for global_epoch in range(args.global_epochs):
                apply_epoch_lr(optimizer, args, global_epoch)
                apply_epoch_expert_schedules(planner, args, global_epoch, expert_schedule_targets)
                for chunk_idx, chunk in enumerate(chunks):
                    dataset = ChunkDataset(
                        chunk,
                        max_samples=args.max_samples,
                        allow_dummy_cache=args.allow_dummy_cache,
                        require_jepa=require_jepa,
                        require_vggt=require_vggt,
                        require_targets=require_targets,
                    )
                    sampler = DistributedSampler(
                        dataset,
                        num_replicas=world_size,
                        rank=rank,
                        shuffle=True,
                        drop_last=False,
                    ) if distributed else None
                    if sampler is not None:
                        sampler.set_epoch(global_epoch * len(chunks) + chunk_idx)
                    yield global_epoch, chunk_idx, chunk, 0, make_loader(dataset, sampler)
            return
        chunk_cycle = 0
        while True:
            for chunk_idx, chunk in enumerate(chunks):
                for local_epoch in range(args.epochs_per_chunk):
                    apply_epoch_expert_schedules(planner, args, chunk_cycle, expert_schedule_targets)
                    dataset = ChunkDataset(
                        chunk,
                        max_samples=args.max_samples,
                        allow_dummy_cache=args.allow_dummy_cache,
                        require_jepa=require_jepa,
                        require_vggt=require_vggt,
                        require_targets=require_targets,
                    )
                    sampler = DistributedSampler(
                        dataset,
                        num_replicas=world_size,
                        rank=rank,
                        shuffle=True,
                        drop_last=False,
                    ) if distributed else None
                    if sampler is not None:
                        sampler.set_epoch(chunk_cycle * len(chunks) + chunk_idx)
                    yield chunk_cycle, chunk_idx, chunk, local_epoch, make_loader(dataset, sampler)
            if args.num_steps is None and args.num_optimizer_steps is None:
                return
            chunk_cycle += 1

    try:
        made_progress = False
        for global_epoch, chunk_idx, chunk, local_epoch, loader in training_passes():
            last_step_end = time.time()
            for batch_idx, (vl_features, action_input) in enumerate(loader):
                made_progress = True
                batch_ready = time.time()
                data_wait_sec = batch_ready - last_step_end
                step_start = time.time()
                transfer_start = time.time()
                vl_features = vl_features.to(device=device, dtype=dtype, non_blocking=True)
                for key, value in list(action_input.items()):
                    if isinstance(value, torch.Tensor):
                        action_input[key] = value.to(device=device, dtype=dtype, non_blocking=True)
                transfer_sec = time.time() - transfer_start
                last_batch = (vl_features.detach(), BatchFeature(data={k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in action_input.items()}))
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda" and dtype != torch.float32)):
                    output = train_model(vl_features, action_input)
                    loss = output["loss"] / args.gradient_accumulation_steps
                reduced = reduce_metrics(output, device=device, distributed=distributed, world_size=world_size)
                total_loss = reduced["loss"]
                diffusion_loss = reduced["diffusion_loss"]
                jepa_loss = reduced["jepa_alignment_loss"]
                vggt_loss = reduced["vggt_alignment_loss"]
                scaler.scale(loss).backward()
                pending_grad_steps += 1
                expert_grad_observed = expert_grad_observed or expert_grad_present(planner)
                grad_norm_value: Optional[float] = None
                if pending_grad_steps >= args.gradient_accumulation_steps:
                    grad_norm_value = optimizer_step(planner, optimizer, scaler)
                    optimizer_step_count += 1
                    pending_grad_steps = 0
                global_step += 1
                last_loss = total_loss
                record = {
                    "step": global_step,
                    "optimizer_step": optimizer_step_count,
                    "chunk": chunk.name if chunk is not None else "flat_all_chunks",
                    "chunk_index": chunk_idx,
                    "global_epoch": global_epoch,
                    "epoch": global_epoch if args.global_epochs is not None else local_epoch,
                    "local_epoch": local_epoch,
                    "batch_index": batch_idx,
                    "effective_batch_size": args.batch_size * args.gradient_accumulation_steps * world_size,
                    "world_size": world_size,
                    "total_loss": total_loss,
                    "diffusion_loss": diffusion_loss,
                    "jepa_alignment_loss": jepa_loss,
                    "vggt_alignment_loss": vggt_loss,
                    "jepa_gate": reduced["jepa_gate_value"] if "jepa_gate_value" in output else None,
                    "vggt_gate": reduced["vggt_gate_value"] if "vggt_gate_value" in output else None,
                    "branch_weight_vlm": reduced["branch_weight_vlm"] if "branch_weight_vlm" in output else None,
                    "branch_weight_jepa": reduced["branch_weight_jepa"] if "branch_weight_jepa" in output else None,
                    "branch_weight_vggt": reduced["branch_weight_vggt"] if "branch_weight_vggt" in output else None,
                    "expert_context_scale": reduced["expert_context_scale"] if "expert_context_scale" in output else None,
                    "expert_horizon_residual_scale": reduced["expert_horizon_residual_scale"] if "expert_horizon_residual_scale" in output else None,
                    "jepa_alignment_weight": float(planner.config.jepa_alignment_weight),
                    "vggt_alignment_weight": float(planner.config.vggt_alignment_weight),
                    "diffusion_loss_weight": float(planner.config.diffusion_loss_weight),
                    "grad_norm": grad_norm_value,
                    "learning_rate": lrs(optimizer),
                    "data_wait_sec": round(data_wait_sec, 4),
                    "input_transfer_sec": round(transfer_sec, 4),
                    "step_time_sec": round(time.time() - step_start, 4),
                    "gpu_memory_allocated_bytes": current_gpu_memory(device),
                    "expert_grad_observed": expert_grad_observed,
                }
                last_step_end = time.time()
                if log_fp is not None:
                    log_fp.write(json.dumps(record, sort_keys=True) + "\n")
                    log_fp.flush()
                if is_rank0(rank) and args.log_every and global_step % args.log_every == 0:
                    print(
                        f"step={global_step} opt_step={optimizer_step_count} chunk={record['chunk']} "
                        f"epoch={record['epoch']} loss={total_loss:.6f} diff={diffusion_loss:.6f} "
                        f"jepa={jepa_loss:.6f} vggt={vggt_loss:.6f} grad={grad_norm_value}"
                    )
                checkpoint_metrics = {
                    "step": global_step,
                    "optimizer_step": optimizer_step_count,
                    "global_epoch": global_epoch,
                    "loss": total_loss,
                    "best_loss": best_loss,
                }
                if total_loss < best_loss:
                    best_loss = total_loss
                    checkpoint_metrics["best_loss"] = best_loss
                    if is_rank0(rank):
                        save_checkpoint(args.output_dir / "best.ckpt", planner, optimizer, global_step, cfg_dict, checkpoint_metrics)
                if is_rank0(rank) and args.save_every and global_step % args.save_every == 0:
                    save_checkpoint(args.output_dir / f"step_{global_step:08d}.ckpt", planner, optimizer, global_step, cfg_dict, checkpoint_metrics)
                    save_checkpoint(args.output_dir / "latest.ckpt", planner, optimizer, global_step, cfg_dict, checkpoint_metrics)
                if args.num_steps is not None and global_step >= args.num_steps:
                    stop = True
                    break
                if args.num_optimizer_steps is not None and optimizer_step_count >= args.num_optimizer_steps:
                    stop = True
                    break
            if args.delete_old_chunk and chunk is not None:
                print(f"delete-old-chunk requested; not deleting {chunk} from this script to avoid accidental data loss.")
            if not stop and args.global_epochs is not None and (args.flat_global_dataset or chunk_idx == len(chunks) - 1):
                completed_global_epochs = max(completed_global_epochs, global_epoch + 1)
            if stop:
                break
        if not made_progress:
            raise RuntimeError("No training batches were produced.")
        if pending_grad_steps > 0:
            optimizer_step(planner, optimizer, scaler)
            optimizer_step_count += 1
            pending_grad_steps = 0
        final_metrics = {"step": global_step, "optimizer_step": optimizer_step_count, "loss": last_loss, "best_loss": best_loss}
        if is_rank0(rank):
            save_checkpoint(args.output_dir / "latest.ckpt", planner, optimizer, global_step, cfg_dict, final_metrics)
        if is_rank0(rank) and not (args.output_dir / "best.ckpt").is_file():
            save_checkpoint(args.output_dir / "best.ckpt", planner, optimizer, global_step, cfg_dict, final_metrics)
        if is_rank0(rank) and last_batch is not None:
            check_dtypes: List[torch.dtype] = []
            if args.final_check_precision in {"auto", "train"}:
                check_dtypes.append(dtype)
            if args.final_check_precision in {"auto", "fp32"} and torch.float32 not in check_dtypes:
                check_dtypes.append(torch.float32)
            for check_dtype in check_dtypes:
                label = dtype_label(check_dtype)
                try:
                    result = final_pred_finite_check(planner, last_batch, device=device, dtype=check_dtype)
                except Exception as check_exc:  # keep diagnostic detail in the final report
                    result = {"finite": False, "dtype": label, "error": repr(check_exc)}
                pred_checks[label] = result
                if result.get("finite"):
                    pred_finite = True
                    pred_check_precision = label
                    break
            if pred_finite is not True:
                pred_finite = False
                raise RuntimeError(f"Final pred_traj finite check failed: {json.dumps(pred_checks, sort_keys=True)}")
        summary.update({
            "global_step": global_step,
            "optimizer_step": optimizer_step_count,
            "completed_global_epochs": completed_global_epochs,
            "best_loss": best_loss,
            "last_loss": last_loss,
            "expert_grad_observed": expert_grad_observed,
            "pred_traj_finite": pred_finite,
            "pred_traj_finite_check_precision": pred_check_precision,
            "pred_traj_finite_checks": pred_checks,
        })
        if planner.config.use_expert_features and not args.freeze_expert and not expert_grad_observed:
            raise RuntimeError("No expert parameter gradients were observed during training.")
    except Exception as exc:
        summary.update({
            "global_step": global_step,
            "optimizer_step": optimizer_step_count,
            "completed_global_epochs": completed_global_epochs,
            "best_loss": best_loss,
            "last_loss": last_loss,
            "expert_grad_observed": expert_grad_observed,
            "pred_traj_finite": pred_finite,
            "pred_traj_finite_check_precision": pred_check_precision,
            "pred_traj_finite_checks": pred_checks,
            "failure": repr(exc),
        })
        if is_rank0(rank):
            write_final_report(args.output_dir / "final_report.md", summary)
        raise
    finally:
        if log_fp is not None:
            log_fp.close()
        if distributed:
            dist.barrier()
            cleanup_distributed(distributed)
    if is_rank0(rank):
        write_final_report(args.output_dir / "final_report.md", summary)
        print(f"Training finished at step={global_step}. Latest checkpoint: {args.output_dir / 'latest.ckpt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
