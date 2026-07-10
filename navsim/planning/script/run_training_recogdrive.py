from typing import Tuple
from pathlib import Path
import logging
import os
import json
import sys
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.distributed as dist
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.agents.recogdrive.recogdrive_features import (
    assert_real_expert_cache_for_training,
    stack_optional_expert_features,
)
from navsim.agents.recogdrive.expert_cache import iter_index, load_sample
import torch
import torch.nn.utils.rnn as rnn_utils
from typing import List, Dict, Any, Optional, Set

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


KEY_CHECKPOINT_STEPS = (50000, 60000, 80000, 100000, 120000, 140000, 160000)
GEOMETRY_MODE_TO_CODE = {"missing": -1, "no_geometry": 0, "patch_fallback": 1, "full_geometry": 2}


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "fp32"
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float64:
        return "fp64"
    if dtype == torch.int64:
        return "int64"
    return str(dtype).replace("torch.", "")


def _tensor_dtype_counts(tensors) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            continue
        key = _dtype_name(tensor.dtype)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _configured_optimizer_groups(agent_cfg: DictConfig) -> Tuple[List[str], List[float], List[float]]:
    base_lr = float(agent_cfg.get("lr", 1e-4))
    if bool(agent_cfg.get("last_vla_train_vlm_lora", False)):
        lr_lora = float(agent_cfg.get("lr_vlm_lora", 1e-5))
        lr_cot = float(agent_cfg.get("lr_last_vla_cot", agent_cfg.get("lr_action_head", base_lr)))
        return (
            ["last_vla_cot", "vlm_lora"],
            [lr_cot, lr_lora],
            [
                float(agent_cfg.get("weight_decay_last_vla_cot", 1e-4)),
                float(agent_cfg.get("weight_decay_vlm_lora", 0.0)),
            ],
        )
    lr_action_head = agent_cfg.get("lr_action_head", None)
    lr_expert = agent_cfg.get("lr_expert", None)
    lr_expert_gate = agent_cfg.get("lr_expert_gate", None)
    train_expert_only = bool(agent_cfg.get("train_expert_only", False))
    freeze_base_action_head = bool(agent_cfg.get("freeze_base_action_head", False))
    freeze_expert = bool(agent_cfg.get("freeze_expert", False))
    grouping = (
        lr_action_head is not None
        or lr_expert is not None
        or lr_expert_gate is not None
        or train_expert_only
        or freeze_base_action_head
        or freeze_expert
    )
    if not grouping:
        return ["default"], [base_lr], [1e-4]

    names: List[str] = []
    lrs: List[float] = []
    weight_decays: List[float] = []
    action_lr = base_lr if lr_action_head is None else float(lr_action_head)
    expert_lr = base_lr if lr_expert is None else float(lr_expert)
    if not freeze_expert and expert_lr > 0.0:
        names.append("expert")
        lrs.append(expert_lr)
        weight_decays.append(1e-4)
    if not freeze_expert and lr_expert_gate is not None and float(lr_expert_gate) > 0.0:
        names.append("expert_gate")
        lrs.append(float(lr_expert_gate))
        weight_decays.append(0.0)
    if not (train_expert_only or freeze_base_action_head) and action_lr > 0.0:
        names.append("action_head")
        lrs.append(action_lr)
        weight_decays.append(1e-4)
    return names, lrs, weight_decays


def _parse_key_steps() -> List[int]:
    raw = os.getenv("RECOGDRIVE_KEY_STEPS", os.getenv("A0_STAGE2_KEY_STEPS", ""))
    if not raw:
        return list(KEY_CHECKPOINT_STEPS)
    steps: List[int] = []
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if item:
            steps.append(int(item))
    return sorted(set(steps))


def _parse_key_epochs() -> List[int]:
    raw = os.getenv("RECOGDRIVE_KEY_EPOCHS", os.getenv("LAST_VLA_KEY_EPOCHS", ""))
    if not raw:
        return []
    epochs: List[int] = []
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if item:
            epoch = int(item)
            if epoch <= 0:
                raise ValueError(f"RECOGDRIVE_KEY_EPOCHS must contain 1-based positive epochs, got {epoch}")
            epochs.append(epoch)
    return sorted(set(epochs))


def _parse_key_epoch_interval() -> int:
    raw = os.getenv("RECOGDRIVE_KEY_EPOCH_INTERVAL", os.getenv("LAST_VLA_KEY_EPOCH_INTERVAL", "0")).strip()
    if not raw:
        return 0
    interval = int(raw)
    if interval < 0:
        raise ValueError(f"RECOGDRIVE_KEY_EPOCH_INTERVAL must be non-negative, got {interval}")
    return interval


class StepCheckpointCallback(pl.Callback):
    """Save exact global-step checkpoints for staged PDM evaluation."""

    def __init__(self, dirpath: Path, steps: List[int]) -> None:
        self.dirpath = dirpath
        self.steps = set(int(step) for step in steps)
        self.saved_steps: set[int] = set()

    def _save(self, trainer: pl.Trainer, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Lightning requires save_checkpoint to be called on every DDP rank.
        # The DDP strategy writes only on global zero and synchronizes internally.
        trainer.save_checkpoint(str(path))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = int(trainer.global_step)
        if step in self.steps and step not in self.saved_steps:
            self._save(trainer, self.dirpath / f"step_{step:08d}.ckpt")
            self.saved_steps.add(step)

    def on_train_end(self, trainer, pl_module) -> None:
        self._save(trainer, self.dirpath / "latest.ckpt")


class EpochCheckpointCallback(pl.Callback):
    """Save exact completed-epoch checkpoints independent of validation."""

    def __init__(self, dirpath: Path, epochs: List[int], every_n_epochs: int = 0) -> None:
        self.dirpath = dirpath
        self.epochs = set(int(epoch) for epoch in epochs)
        self.every_n_epochs = int(every_n_epochs)
        self.saved_epochs: set[int] = set()

    def _save(self, trainer: pl.Trainer, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(path))

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        completed_epoch = int(trainer.current_epoch) + 1
        should_save = completed_epoch in self.epochs
        if self.every_n_epochs > 0 and completed_epoch % self.every_n_epochs == 0:
            should_save = True
        if should_save and completed_epoch not in self.saved_epochs:
            self._save(trainer, self.dirpath / f"epoch_{completed_epoch:03d}.ckpt")
            self.saved_epochs.add(completed_epoch)


class ReCogDriveTrainingProgressCallback(pl.Callback):
    """Propagates epoch progress into schedulable planner components such as LaST-RD."""

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        agent = getattr(pl_module, "agent", None)
        if agent is not None and hasattr(agent, "set_training_progress"):
            agent.set_training_progress(int(trainer.current_epoch), int(trainer.max_epochs))


class ChunkCacheDataset(torch.utils.data.Dataset):
    """Local chunk-cache adapter for the official Lightning training loop."""

    REQUIRED_FEATURE_KEYS = ("history_trajectory", "high_command_one_hot", "last_hidden_state", "status_feature")
    REQUIRED_TARGET_KEYS = ("trajectory",)
    EXPERT_CONTEXT_KEYS = ("jepa_context_tokens", "vggt_context_tokens")
    EXPERT_GEOMETRY_KEYS = ("vggt_geometry_tokens", "vggt_depth_tokens", "vggt_pointmap_tokens", "vggt_camera_tokens")
    EXPERT_GEOMETRY_MODE_KEYS = ("vggt_geometry_mode_code",)
    EXPERT_TARGET_KEYS = (
        "jepa_target_tokens",
        "vggt_target_tokens",
        "vggt_geometry_target_tokens",
        "vggt_depth_target_tokens",
        "vggt_pointmap_target_tokens",
    )
    RISK_KEYS = ("risk_labels", "generic_risk_labels", "drivable_risk_labels", "ttc_risk_labels", "comfort_risk_labels")
    LAST_VLA_TEACHER_KEYS = (
        "teacher_trajectory",
        "teacher_trajectory_norm",
        "teacher_score",
        "gt_score",
        "oracle_best_of_k_score",
        "candidate_count",
    )
    LAST_VLA_TEXT_ANCHOR_KEYS = (
        "vlm_text_trajectory",
        "vlm_text_trajectory_norm",
        "vlm_text_parse_ok",
    )
    TWO_EXPERT_HIDDEN_KEYS = ("two_expert_h_dyn", "two_expert_h_geo")

    def __init__(
        self,
        cache_path: str,
        *,
        log_names: Optional[List[str]] = None,
        split_name: str = "all",
        include_expert_features: bool = False,
        include_expert_targets: bool = False,
        use_jepa: bool = True,
        use_vggt: bool = True,
        num_jepa_tokens: int = 12,
        num_vggt_tokens: int = 12,
        num_geometry_tokens: int = 12,
        jepa_dim: int = 1024,
        vggt_dim: int = 2048,
        use_last_rd: bool = False,
        require_vggt_geometry: bool = False,
        allow_patch_geometry_fallback: bool = True,
        future_jepa_loss_weight: float = 0.0,
        vggt_geometry_loss_weight: float = 0.0,
        use_last_vla: bool = False,
        last_vla_stage: str = "disabled",
        last_vla_teacher_traj_mode: str = "none",
        last_vla_use_residual_diffusion: bool = False,
        last_vla_residual_anchor_source: str = "vlm_text_traj",
        last_vla_require_residual_anchor: bool = True,
        last_vla_residual_anchor_cache_dir: Optional[str] = None,
        last_vla_require_full_geometry: bool = False,
        last_vla_allow_patch_geometry_fallback: bool = False,
        last_vla_geometry_teacher_dim: int = 512,
        last_vla_geometry_loss_weight: float = 0.0,
        use_two_expert_slots: bool = False,
        two_expert_num_dyn_groups: int = 3,
        two_expert_dyn_tokens_per_group: int = 12,
        two_expert_num_geo_tokens: int = 12,
        two_expert_vlm_hidden_dim: int = 1536,
        stage2_target_source: str = "gt",
        stage2_elite_target_index_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.cache_path = Path(cache_path)
        self.split_name = split_name
        self.include_expert_features = bool(include_expert_features)
        self.include_expert_targets = bool(include_expert_targets)
        self.use_jepa = bool(use_jepa)
        self.use_vggt = bool(use_vggt)
        self.num_jepa_tokens = int(num_jepa_tokens)
        self.num_vggt_tokens = int(num_vggt_tokens)
        self.num_geometry_tokens = int(num_geometry_tokens)
        self.jepa_dim = int(jepa_dim)
        self.vggt_dim = int(vggt_dim)
        self.use_last_rd = bool(use_last_rd)
        self.require_vggt_geometry = bool(require_vggt_geometry)
        self.allow_patch_geometry_fallback = bool(allow_patch_geometry_fallback)
        self.future_jepa_loss_weight = float(future_jepa_loss_weight)
        self.vggt_geometry_loss_weight = float(vggt_geometry_loss_weight)
        self.use_last_vla = bool(use_last_vla)
        self.last_vla_stage = str(last_vla_stage)
        self.last_vla_teacher_traj_mode = str(last_vla_teacher_traj_mode)
        self.last_vla_use_residual_diffusion = bool(last_vla_use_residual_diffusion)
        self.last_vla_residual_anchor_source = str(last_vla_residual_anchor_source)
        self.last_vla_require_residual_anchor = bool(last_vla_require_residual_anchor)
        self.last_vla_residual_anchor_cache_dir = Path(last_vla_residual_anchor_cache_dir) if last_vla_residual_anchor_cache_dir else None
        self.last_vla_residual_anchor_index: Dict[str, Path] = {}
        if self.last_vla_residual_anchor_cache_dir is not None:
            self.last_vla_residual_anchor_index = self._load_residual_anchor_index(self.last_vla_residual_anchor_cache_dir)
        self.last_vla_require_full_geometry = bool(last_vla_require_full_geometry)
        self.last_vla_allow_patch_geometry_fallback = bool(last_vla_allow_patch_geometry_fallback)
        self.last_vla_geometry_teacher_dim = int(last_vla_geometry_teacher_dim)
        self.last_vla_geometry_loss_weight = float(last_vla_geometry_loss_weight)
        self.use_two_expert_slots = bool(use_two_expert_slots)
        self.two_expert_num_dyn_groups = int(two_expert_num_dyn_groups)
        self.two_expert_dyn_tokens_per_group = int(two_expert_dyn_tokens_per_group)
        self.two_expert_num_geo_tokens = int(two_expert_num_geo_tokens)
        self.two_expert_vlm_hidden_dim = int(two_expert_vlm_hidden_dim)
        self.stage2_target_source = str(stage2_target_source or "gt")
        self.stage2_elite_target_index_path = (
            Path(stage2_elite_target_index_path) if stage2_elite_target_index_path else None
        )
        self.stage2_elite_target_index: Optional[Dict[str, int]] = None
        self.stage2_elite_target_trajectories: Optional[torch.Tensor] = None
        self.stage2_elite_target_metadata: Dict[str, Any] = {}
        if self.stage2_target_source not in {"gt", "awac_elite_best_valid_above_gt_or_gt"}:
            raise ValueError(
                "stage2_target_source must be one of {'gt', 'awac_elite_best_valid_above_gt_or_gt'}, "
                f"got {self.stage2_target_source!r}."
            )
        if self.stage2_target_source != "gt":
            self._load_stage2_elite_target_index()
        if self.include_expert_targets and not self.include_expert_features:
            raise ValueError("include_expert_targets=True requires include_expert_features=True.")
        self.log_name_filter: Optional[Set[str]] = set(str(item) for item in log_names) if log_names is not None else None
        self.skipped_by_log_name = 0
        if not self.cache_path.is_dir():
            raise FileNotFoundError(f"Chunk cache path {self.cache_path} does not exist.")
        self.records: List[tuple[Path, Path, Dict[str, Any]]] = []
        for chunk_dir in self._chunk_dirs(self.cache_path):
            for record in iter_index(chunk_dir):
                log_name = record.get("log_name")
                if self.log_name_filter is not None and str(log_name) not in self.log_name_filter:
                    self.skipped_by_log_name += 1
                    continue
                sample_path = Path(record["path"])
                if not sample_path.is_file() and not sample_path.is_absolute():
                    chunk_relative = chunk_dir / sample_path
                    sample_path = chunk_relative if chunk_relative.is_file() else sample_path
                self.records.append((chunk_dir, sample_path, record))
        if not self.records:
            raise FileNotFoundError(
                f"No chunk cache records found for split={self.split_name} under {self.cache_path}. "
                f"Skipped by log filter: {self.skipped_by_log_name}."
            )

    def _load_stage2_elite_target_index(self) -> None:
        if self.stage2_elite_target_index_path is None:
            raise FileNotFoundError(
                "stage2_target_source='awac_elite_best_valid_above_gt_or_gt' requires "
                "stage2_elite_target_index_path. Build it with "
                "scripts/last_vla_v2/two_expert_slot/build_stage2_elite_target_index.py."
            )
        if not self.stage2_elite_target_index_path.is_file():
            raise FileNotFoundError(f"Stage2 elite target index not found: {self.stage2_elite_target_index_path}")
        payload = torch.load(self.stage2_elite_target_index_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(
                f"Stage2 elite target index must be a dict, got {type(payload).__name__}: "
                f"{self.stage2_elite_target_index_path}"
            )
        tokens = payload.get("tokens")
        trajectories = payload.get("trajectories")
        if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
            raise TypeError("Stage2 elite target index field 'tokens' must be a list[str].")
        if not isinstance(trajectories, torch.Tensor) or trajectories.ndim != 3 or tuple(trajectories.shape[1:]) != (8, 3):
            raise ValueError(
                "Stage2 elite target index field 'trajectories' must be a tensor with shape [N, 8, 3]."
            )
        if len(tokens) != int(trajectories.shape[0]):
            raise ValueError(
                f"Stage2 elite target index token count {len(tokens)} != trajectories count {trajectories.shape[0]}."
            )
        self.stage2_elite_target_index = {str(token): idx for idx, token in enumerate(tokens)}
        if len(self.stage2_elite_target_index) != len(tokens):
            raise ValueError("Stage2 elite target index contains duplicate tokens.")
        self.stage2_elite_target_trajectories = trajectories.detach().float().contiguous()
        summary = payload.get("summary", {})
        self.stage2_elite_target_metadata = {
            "summary": summary if isinstance(summary, dict) else {},
            "selected_target_count": len(tokens),
        }

    def _stage2_training_target(self, token: str, gt_trajectory: torch.Tensor) -> torch.Tensor:
        if self.stage2_target_source == "gt":
            return gt_trajectory
        if self.stage2_elite_target_index is None or self.stage2_elite_target_trajectories is None:
            raise RuntimeError("Stage2 elite target index was not loaded.")
        idx = self.stage2_elite_target_index.get(str(token))
        if idx is None:
            return gt_trajectory
        target = self.stage2_elite_target_trajectories[idx].to(device=gt_trajectory.device, dtype=gt_trajectory.dtype)
        if tuple(target.shape) != tuple(gt_trajectory.shape):
            raise ValueError(
                f"Stage2 elite target for token={token!r} has shape {tuple(target.shape)}, "
                f"expected {tuple(gt_trajectory.shape)}."
            )
        return target

    @staticmethod
    def _chunk_dirs(cache_path: Path) -> List[Path]:
        if (cache_path / "index.jsonl").is_file():
            return [cache_path]
        chunk_dirs = [
            child for child in cache_path.iterdir()
            if child.is_dir()
            and (child / "index.jsonl").is_file()
            and not child.name.startswith("navtest")
        ]
        shard_root = cache_path / "shards"
        if shard_root.is_dir():
            chunk_dirs.extend(
                child for child in shard_root.iterdir()
                if child.is_dir()
                and (child / "index.jsonl").is_file()
                and not child.name.startswith("navtest")
            )
        return sorted(chunk_dirs)

    @staticmethod
    def looks_like(cache_path: str) -> bool:
        path = Path(cache_path)
        if not path.is_dir():
            return False

        def indexed_samples_exist(index_dir: Path) -> bool:
            index_path = index_dir / "index.jsonl"
            if not index_path.is_file():
                return False
            if (index_dir / "samples").is_dir():
                return True
            try:
                with index_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        sample_path = Path(str(record.get("path", "")))
                        if not str(sample_path):
                            return False
                        sample_path = sample_path if sample_path.is_absolute() else index_dir / sample_path
                        return sample_path.is_file()
            except (OSError, json.JSONDecodeError, TypeError):
                return False
            return False

        if indexed_samples_exist(path):
            return True
        if any(child.is_dir() and indexed_samples_exist(child) for child in path.iterdir()):
            return True
        shard_root = path / "shards"
        return shard_root.is_dir() and any(
            child.is_dir() and indexed_samples_exist(child) for child in shard_root.iterdir()
        )

    @staticmethod
    def _load_residual_anchor_index(cache_dir: Path) -> Dict[str, Path]:
        if not cache_dir.is_dir():
            raise FileNotFoundError(f"Last-VLA residual anchor cache dir does not exist: {cache_dir}")
        chunk_dirs = [cache_dir] if (cache_dir / "index.jsonl").is_file() else [
            child for child in sorted(cache_dir.iterdir()) if child.is_dir() and (child / "index.jsonl").is_file()
        ]
        if not chunk_dirs:
            raise FileNotFoundError(f"No indexed residual anchor cache found under {cache_dir}")
        index: Dict[str, Path] = {}
        for chunk_dir in chunk_dirs:
            for record in iter_index(chunk_dir):
                token = str(record.get("sample_token") or Path(record.get("path", "")).stem)
                path = Path(record.get("path", ""))
                if not path:
                    continue
                index[token] = path if path.is_absolute() else chunk_dir / path
        if not index:
            raise FileNotFoundError(f"Residual anchor cache index is empty: {cache_dir}")
        return index

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _require_tensor(sample: Dict[str, Any], key: str, sample_path: Path) -> torch.Tensor:
        if key not in sample:
            raise KeyError(f"Chunk sample {sample_path} is missing required key '{key}'.")
        value = sample[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Chunk sample {sample_path} key '{key}' must be a torch.Tensor, got {type(value).__name__}.")
        # Chunk cache tensors are supervision/features, never graph-carrying values.
        # Some historical caches saved VLM hidden states with requires_grad=True;
        # detach here so DataLoader workers never serialize autograd tensors.
        value = value.detach()
        finite_check = os.getenv("LAST_RD_RUNTIME_FINITE_CHECK", "1").strip().lower()
        if finite_check not in {"0", "false", "no", "off"} and (torch.is_floating_point(value) or torch.is_complex(value)):
            max_elements = int(os.getenv("LAST_RD_RUNTIME_FINITE_CHECK_MAX_ELEMENTS", "-1"))
            if max_elements < 0 or value.numel() <= max_elements:
                if not torch.isfinite(value).all():
                    raise ValueError(f"Chunk sample {sample_path} key '{key}' contains non-finite values.")
        return value

    def _required_expert_context_keys(self) -> List[str]:
        if not self.include_expert_features:
            return []
        keys: List[str] = []
        if self.use_jepa:
            keys.append("jepa_context_tokens")
        if self.use_vggt or (self.use_last_rd and self.vggt_geometry_loss_weight > 0.0):
            if not (
                self.use_last_vla
                and self.last_vla_require_full_geometry
                and not self.last_vla_allow_patch_geometry_fallback
            ):
                keys.append("vggt_context_tokens")
        if self.use_last_rd and self.require_vggt_geometry:
            keys.append("vggt_geometry_tokens")
        if self.use_last_vla and self.last_vla_require_full_geometry:
            keys.extend(["vggt_geometry_tokens", "vggt_geometry_mode_code"])
        return keys

    def _required_expert_target_keys(self) -> List[str]:
        if not self.include_expert_targets:
            return []
        keys: List[str] = []
        if self.use_jepa and (not self.use_last_rd or self.future_jepa_loss_weight > 0.0):
            keys.append("jepa_target_tokens")
        if self.use_vggt and not (
            self.use_last_vla
            and self.last_vla_require_full_geometry
            and not self.last_vla_allow_patch_geometry_fallback
        ):
            keys.append("vggt_target_tokens")
        if self.use_last_rd and self.require_vggt_geometry:
            keys.append("vggt_geometry_target_tokens")
        if (
            self.use_last_vla
            and self.last_vla_stage == "teacher_traj_sft"
            and self.last_vla_teacher_traj_mode != "none"
        ):
            keys.append("teacher_trajectory_norm_or_teacher_trajectory")
        return keys

    def _has_teacher_trajectory(self, sample: Dict[str, Any]) -> bool:
        return isinstance(sample.get("teacher_trajectory_norm"), torch.Tensor) or isinstance(sample.get("teacher_trajectory"), torch.Tensor)

    def _required_last_vla_anchor_keys(self) -> List[str]:
        if not (
            self.last_vla_use_residual_diffusion
            and self.last_vla_residual_anchor_source == "vlm_text_traj"
            and self.last_vla_require_residual_anchor
        ):
            return []
        return ["vlm_text_trajectory_norm_or_vlm_text_trajectory"]

    def _required_two_expert_hidden_keys(self) -> List[str]:
        return list(self.TWO_EXPERT_HIDDEN_KEYS) if self.use_two_expert_slots else []

    def _has_vlm_text_anchor(self, sample: Dict[str, Any], token: Optional[str] = None) -> bool:
        if token is not None and token in self.last_vla_residual_anchor_index:
            return True
        return isinstance(sample.get("vlm_text_trajectory_norm"), torch.Tensor) or isinstance(sample.get("vlm_text_trajectory"), torch.Tensor)

    def _load_vlm_text_anchor_sample(self, token: str, sample_path: Path) -> Dict[str, Any]:
        path = self.last_vla_residual_anchor_index.get(str(token))
        if path is None:
            return {}
        if not path.is_file():
            raise FileNotFoundError(f"Residual anchor cache file for token {token} not found: {path}")
        anchor = load_sample(path)
        if not self._has_vlm_text_anchor(anchor):
            raise KeyError(f"Residual anchor cache file {path} for {sample_path} lacks vlm_text_trajectory(_norm).")
        return anchor

    def _optional_expert_keys(self, sample: Dict[str, Any], required_keys: List[str]) -> List[str]:
        candidates: List[str] = []
        if self.include_expert_features:
            if self.use_jepa:
                candidates.append("jepa_context_tokens")
            if self.use_vggt:
                candidates.extend(
                    [
                        "vggt_context_tokens",
                        "vggt_geometry_tokens",
                        "vggt_depth_tokens",
                        "vggt_pointmap_tokens",
                        "vggt_camera_tokens",
                    ]
                )
        if self.include_expert_targets:
            if self.use_jepa:
                candidates.append("jepa_target_tokens")
            if self.use_vggt:
                candidates.extend(
                    [
                        "vggt_target_tokens",
                        "vggt_geometry_target_tokens",
                        "vggt_depth_target_tokens",
                        "vggt_pointmap_target_tokens",
                    ]
                )
        if self.use_last_rd:
            candidates.extend(self.RISK_KEYS)
        if self.use_last_vla:
            candidates.extend(self.LAST_VLA_TEACHER_KEYS)
            candidates.extend(self.EXPERT_GEOMETRY_MODE_KEYS)
            candidates.extend(self.RISK_KEYS)
        if self.last_vla_use_residual_diffusion and self.last_vla_residual_anchor_source == "vlm_text_traj":
            candidates.extend(self.LAST_VLA_TEXT_ANCHOR_KEYS)
        required = set(required_keys)
        composite_keys = {
            "teacher_trajectory_norm_or_teacher_trajectory",
            "vlm_text_trajectory_norm_or_vlm_text_trajectory",
        }
        return [key for key in candidates if key in sample and key not in required and key not in composite_keys]

    def _expected_shape(self, key: str) -> Optional[Tuple[int, ...]]:
        expert_shapes = {
            "jepa_context_tokens": (self.num_jepa_tokens, self.jepa_dim),
            "jepa_target_tokens": (self.num_jepa_tokens, self.jepa_dim),
            "vggt_context_tokens": (self.num_vggt_tokens, self.vggt_dim),
            "vggt_target_tokens": (self.num_vggt_tokens, self.vggt_dim),
            "vggt_geometry_tokens": (self.num_geometry_tokens, self.last_vla_geometry_teacher_dim if self.use_last_vla else self.vggt_dim),
            "vggt_geometry_target_tokens": (self.num_geometry_tokens, self.last_vla_geometry_teacher_dim if self.use_last_vla else self.vggt_dim),
            "vggt_depth_tokens": (self.num_geometry_tokens, self.last_vla_geometry_teacher_dim if self.use_last_vla else self.vggt_dim),
            "vggt_pointmap_tokens": (self.num_geometry_tokens, self.last_vla_geometry_teacher_dim if self.use_last_vla else self.vggt_dim),
            "vggt_camera_tokens": (self.num_geometry_tokens, self.last_vla_geometry_teacher_dim if self.use_last_vla else self.vggt_dim),
        }
        if key in expert_shapes:
            return expert_shapes[key]
        expected = {
            "history_trajectory": (4, 3),
            "high_command_one_hot": (3,),
            "status_feature": (8,),
            "trajectory": (8, 3),
            "teacher_trajectory": (8, 3),
            "teacher_trajectory_norm": (8, 3),
            "vlm_text_trajectory": (8, 3),
            "vlm_text_trajectory_norm": (8, 3),
        }.get(key)
        if key == "two_expert_h_dyn":
            return (
                self.two_expert_num_dyn_groups,
                self.two_expert_dyn_tokens_per_group,
                self.two_expert_vlm_hidden_dim,
            )
        if key == "two_expert_h_geo":
            return (self.two_expert_num_geo_tokens, self.two_expert_vlm_hidden_dim)
        return expected

    def _check_shape(self, tensor: torch.Tensor, key: str, sample_path: Path) -> None:
        expected = self._expected_shape(key)
        if expected is not None and tuple(tensor.shape) != expected:
            raise ValueError(f"Chunk sample {sample_path} key '{key}' shape {tuple(tensor.shape)} != {expected}.")
        if key == "last_hidden_state" and (tensor.ndim != 2 or tensor.shape[-1] != 1536):
            raise ValueError(
                f"Chunk sample {sample_path} key 'last_hidden_state' must have shape [N, 1536], got {tuple(tensor.shape)}."
            )

    @staticmethod
    def _normalize_high_command_one_hot(tensor: torch.Tensor, sample_path: Path) -> torch.Tensor:
        if tuple(tensor.shape) == (3,):
            return tensor
        if tuple(tensor.shape) == (4,) and torch.isclose(tensor[-1].float(), torch.tensor(0.0, device=tensor.device)):
            return tensor[:3].contiguous()
        raise ValueError(
            f"Chunk sample {sample_path} key 'high_command_one_hot' shape/value {tuple(tensor.shape)} is not supported. "
            "Expected [3] left/straight/right, or legacy [4] with unused fourth slot exactly zero. "
            "Do not silently pad or truncate this field."
        )

    def sample_tokens(self) -> List[str]:
        return [
            str(record.get("sample_token") or sample_path.stem)
            for _, sample_path, record in self.records
        ]

    def _report_records(self) -> List[tuple[Path, Path, Dict[str, Any]]]:
        limit = int(os.getenv("LAST_RD_DATA_REPORT_MAX_SAMPLES", "256"))
        if limit <= 0:
            return self.records
        return self.records[:limit]

    def report(self) -> Dict[str, Any]:
        tokens = self.sample_tokens()
        token_counts = Counter(tokens)
        log_counts = Counter(str(record.get("log_name") or "missing") for _, _, record in self.records)
        chunk_counts = Counter(str(chunk_dir.name) for chunk_dir, _, _ in self.records)
        report_sample_count = len(self._report_records())
        high_command_shape_distribution = self._shape_distribution("high_command_one_hot")
        high_command_normalized_distribution = self._high_command_normalized_distribution()
        return {
            "split_name": self.split_name,
            "cache_path": str(self.cache_path),
            "num_records": len(self.records),
            "unique_sample_tokens": len(token_counts),
            "duplicate_sample_tokens": sum(count - 1 for count in token_counts.values() if count > 1),
            "num_log_names": len(log_counts),
            "skipped_by_log_name": self.skipped_by_log_name,
            "include_expert_features": self.include_expert_features,
            "include_expert_targets": self.include_expert_targets,
            "use_jepa": self.use_jepa,
            "use_vggt": self.use_vggt,
            "num_jepa_tokens": self.num_jepa_tokens,
            "num_vggt_tokens": self.num_vggt_tokens,
            "num_geometry_tokens": self.num_geometry_tokens,
            "jepa_dim": self.jepa_dim,
            "vggt_dim": self.vggt_dim,
            "required_expert_context_keys": self._required_expert_context_keys(),
            "required_expert_target_keys": self._required_expert_target_keys(),
            "use_last_rd": self.use_last_rd,
            "use_last_vla": self.use_last_vla,
            "last_vla_stage": self.last_vla_stage,
            "last_vla_teacher_traj_mode": self.last_vla_teacher_traj_mode,
            "last_vla_use_residual_diffusion": self.last_vla_use_residual_diffusion,
            "last_vla_residual_anchor_source": self.last_vla_residual_anchor_source,
            "last_vla_require_residual_anchor": self.last_vla_require_residual_anchor,
            "last_vla_residual_anchor_cache_dir": str(self.last_vla_residual_anchor_cache_dir) if self.last_vla_residual_anchor_cache_dir is not None else None,
            "last_vla_residual_anchor_cache_records": len(self.last_vla_residual_anchor_index),
            "required_last_vla_anchor_keys": self._required_last_vla_anchor_keys(),
            "last_vla_require_full_geometry": self.last_vla_require_full_geometry,
            "last_vla_allow_patch_geometry_fallback": self.last_vla_allow_patch_geometry_fallback,
            "last_vla_geometry_teacher_dim": self.last_vla_geometry_teacher_dim,
            "last_vla_geometry_loss_weight": self.last_vla_geometry_loss_weight,
            "require_vggt_geometry": self.require_vggt_geometry,
            "allow_patch_geometry_fallback": self.allow_patch_geometry_fallback,
            "future_jepa_loss_weight": self.future_jepa_loss_weight,
            "vggt_geometry_loss_weight": self.vggt_geometry_loss_weight,
            "use_two_expert_slots": self.use_two_expert_slots,
            "two_expert_num_dyn_groups": self.two_expert_num_dyn_groups,
            "two_expert_dyn_tokens_per_group": self.two_expert_dyn_tokens_per_group,
            "two_expert_num_geo_tokens": self.two_expert_num_geo_tokens,
            "two_expert_vlm_hidden_dim": self.two_expert_vlm_hidden_dim,
            "required_two_expert_hidden_keys": self._required_two_expert_hidden_keys(),
            "stage2_target_source": self.stage2_target_source,
            "stage2_elite_target_index_path": (
                str(self.stage2_elite_target_index_path) if self.stage2_elite_target_index_path else None
            ),
            "stage2_elite_target_index_records": (
                len(self.stage2_elite_target_index) if self.stage2_elite_target_index is not None else 0
            ),
            "stage2_elite_target_metadata": self.stage2_elite_target_metadata,
            "shape_distribution_sample_count": report_sample_count,
            "shape_distribution_sample_limit": int(os.getenv("LAST_RD_DATA_REPORT_MAX_SAMPLES", "256")),
            "high_command_one_hot_shape_distribution": high_command_shape_distribution,
            "high_command_one_hot_normalized_shape_distribution": high_command_normalized_distribution,
            "top_log_names": log_counts.most_common(10),
            "chunk_counts": dict(sorted(chunk_counts.items())),
        }

    def _shape_distribution(self, key: str) -> Dict[str, int]:
        distribution: Counter[str] = Counter()
        for _, sample_path, _ in self._report_records():
            try:
                sample = load_sample(sample_path)
                value = sample.get(key)
                shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
            except Exception as exc:
                shape = f"error:{type(exc).__name__}"
            distribution[str(shape)] += 1
        return dict(sorted(distribution.items()))

    def _high_command_normalized_distribution(self) -> Dict[str, int]:
        distribution: Counter[str] = Counter()
        for _, sample_path, _ in self._report_records():
            try:
                sample = load_sample(sample_path)
                value = sample.get("high_command_one_hot")
                if not isinstance(value, torch.Tensor):
                    shape = type(value).__name__
                else:
                    normalized = self._normalize_high_command_one_hot(value.float(), sample_path)
                    shape = tuple(normalized.shape)
            except Exception as exc:
                shape = f"error:{type(exc).__name__}"
            distribution[str(shape)] += 1
        return dict(sorted(distribution.items()))

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]:
        _, sample_path, record = self.records[idx]
        sample = load_sample(sample_path)
        token = str(sample.get("sample_token") or record.get("sample_token") or sample_path.stem)
        anchor_sample = self._load_vlm_text_anchor_sample(token, sample_path) if self.last_vla_residual_anchor_index else {}
        if anchor_sample:
            sample = {**sample, **{key: anchor_sample[key] for key in self.LAST_VLA_TEXT_ANCHOR_KEYS if key in anchor_sample}}
        mode_code = self._geometry_mode_code_from_sample(sample, sample_path)
        if mode_code is not None:
            sample["vggt_geometry_mode_code"] = mode_code
        required_values: Dict[str, torch.Tensor] = {}
        for key in self.REQUIRED_FEATURE_KEYS + self.REQUIRED_TARGET_KEYS:
            value = self._require_tensor(sample, key, sample_path)
            if key == "high_command_one_hot":
                value = self._normalize_high_command_one_hot(value.float(), sample_path)
            self._check_shape(value, key, sample_path)
            required_values[key] = value
        expert_keys = (
            self._required_expert_context_keys()
            + self._required_expert_target_keys()
            + self._required_last_vla_anchor_keys()
        )
        for key in expert_keys:
            if key == "teacher_trajectory_norm_or_teacher_trajectory":
                if not self._has_teacher_trajectory(sample):
                    raise KeyError(
                        f"Chunk sample {sample_path} requires teacher_trajectory_norm or teacher_trajectory "
                        "for Last-VLA teacher_traj_sft."
                    )
                continue
            if key == "vlm_text_trajectory_norm_or_vlm_text_trajectory":
                if not self._has_vlm_text_anchor(sample, token):
                    raise KeyError(
                        f"Chunk sample {sample_path} requires vlm_text_trajectory_norm or vlm_text_trajectory "
                        "for Last-VLA residual anchor_source='vlm_text_traj'."
                    )
                continue
            self._check_shape(self._require_tensor(sample, key, sample_path), key, sample_path)
        optional_expert_keys = self._optional_expert_keys(sample, expert_keys)
        for key in optional_expert_keys:
            value = self._require_tensor(sample, key, sample_path)
            self._check_shape(value, key, sample_path)
        two_expert_hidden_keys = self._required_two_expert_hidden_keys()
        for key in two_expert_hidden_keys:
            self._check_shape(self._require_tensor(sample, key, sample_path), key, sample_path)
        features = {
            "history_trajectory": required_values["history_trajectory"].float(),
            "high_command_one_hot": required_values["high_command_one_hot"].float(),
            "last_hidden_state": required_values["last_hidden_state"].float(),
            "status_feature": required_values["status_feature"].float(),
        }
        for key in two_expert_hidden_keys:
            features[key] = self._require_tensor(sample, key, sample_path).float()
        for key in [*expert_keys, *optional_expert_keys]:
            if key == "teacher_trajectory_norm_or_teacher_trajectory":
                for teacher_key in ("teacher_trajectory_norm", "teacher_trajectory"):
                    if teacher_key in sample:
                        features[teacher_key] = self._require_tensor(sample, teacher_key, sample_path).float()
                continue
            if key == "vlm_text_trajectory_norm_or_vlm_text_trajectory":
                for anchor_key in ("vlm_text_trajectory_norm", "vlm_text_trajectory", "vlm_text_parse_ok"):
                    if anchor_key in sample:
                        features[anchor_key] = self._require_tensor(sample, anchor_key, sample_path).float()
                continue
            value = self._require_tensor(sample, key, sample_path)
            features[key] = value.long() if key == "vggt_geometry_mode_code" else value.float()
        trajectory = self._stage2_training_target(token, required_values["trajectory"].float())
        targets = {"trajectory": trajectory.float()}
        return features, targets, token

    def _geometry_mode_code_from_sample(self, sample: Dict[str, Any], sample_path: Path) -> Optional[torch.Tensor]:
        if not self.use_last_vla:
            return None

        raw_code = sample.get("vggt_geometry_mode_code")
        if isinstance(raw_code, torch.Tensor):
            if raw_code.numel() != 1:
                raise ValueError(f"Chunk sample {sample_path} key 'vggt_geometry_mode_code' must be scalar or [1].")
            code = int(raw_code.detach().cpu().view(-1)[0].item())
            return torch.tensor(code, dtype=torch.int64)
        if raw_code is not None:
            return torch.tensor(int(raw_code), dtype=torch.int64)

        raw_mode = sample.get("vggt_geometry_mode")
        if isinstance(raw_mode, bytes):
            raw_mode = raw_mode.decode("utf-8", errors="replace")
        if isinstance(raw_mode, str):
            if raw_mode not in GEOMETRY_MODE_TO_CODE:
                raise ValueError(f"Chunk sample {sample_path} has unsupported vggt_geometry_mode={raw_mode!r}.")
            return torch.tensor(GEOMETRY_MODE_TO_CODE[raw_mode], dtype=torch.int64)

        has_geometry_tokens = any(key in sample for key in self.EXPERT_GEOMETRY_KEYS)
        has_context_tokens = "vggt_context_tokens" in sample
        if has_geometry_tokens and self.last_vla_require_full_geometry:
            raise KeyError(
                f"Chunk sample {sample_path} has geometry tokens but no explicit full_geometry mode; "
                "strict Last-VLA geometry requires vggt_geometry_mode/full_geometry or vggt_geometry_mode_code=2."
            )
        if has_geometry_tokens:
            return torch.tensor(GEOMETRY_MODE_TO_CODE["patch_fallback"], dtype=torch.int64)
        if has_context_tokens and self.last_vla_allow_patch_geometry_fallback:
            return torch.tensor(GEOMETRY_MODE_TO_CODE["patch_fallback"], dtype=torch.int64)
        if self.use_last_vla and self.last_vla_geometry_loss_weight > 0.0:
            return torch.tensor(GEOMETRY_MODE_TO_CODE["missing"], dtype=torch.int64)
        return None


def write_run_reports(
    cfg: DictConfig,
    agent: AbstractAgent,
    loader_mode: str,
    key_steps: List[int],
    key_epochs: List[int],
    key_epoch_interval: int,
    data_report: Optional[Dict[str, Any]] = None,
) -> None:
    if int(os.getenv("RANK", "0")) != 0:
        return
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    group_names, group_lrs, group_wds = _configured_optimizer_groups(cfg.agent)
    trainable_counts = (
        agent.count_trainable_parameters_by_group()
        if hasattr(agent, "count_trainable_parameters_by_group")
        else None
    )
    lora_target_report = agent.get_lora_target_report() if hasattr(agent, "get_lora_target_report") else None
    lora_training_config = agent.get_lora_training_config() if hasattr(agent, "get_lora_training_config") else None
    optimizer_group_report = agent.get_optimizer_group_report() if hasattr(agent, "get_optimizer_group_report") else []
    report = {
        "requested_precision": str(cfg.trainer.params.get("precision", "")),
        "true_bf16_weights": False,
        "model_param_dtype_counts": _tensor_dtype_counts(agent.parameters()) if hasattr(agent, "parameters") else {},
        "model_buffer_dtype_counts": _tensor_dtype_counts(agent.buffers()) if hasattr(agent, "buffers") else {},
        "optimizer_group_names": group_names,
        "optimizer_group_lrs": group_lrs,
        "optimizer_group_weight_decay": group_wds,
        "optimizer_group_report": optimizer_group_report,
        "trainable_parameter_counts": trainable_counts,
        "lora_target_report": lora_target_report,
        "lora_training_config": lora_training_config,
        "loader_mode": loader_mode,
        "key_checkpoint_steps": key_steps,
        "key_checkpoint_epoch_interval": key_epoch_interval,
        "data_report": data_report,
        "agent_expert_config": {
            "use_expert_features": bool(cfg.agent.get("use_expert_features", False)),
            "expert_feature_source": cfg.agent.get("expert_feature_source", "none"),
            "use_jepa": bool(cfg.agent.get("use_jepa", False)),
            "use_vggt": bool(cfg.agent.get("use_vggt", False)),
            "allow_expert_target_features": bool(cfg.agent.get("allow_expert_target_features", False)),
            "num_jepa_tokens": cfg.agent.get("num_jepa_tokens", None),
            "num_vggt_tokens": cfg.agent.get("num_vggt_tokens", None),
            "jepa_dim": cfg.agent.get("jepa_dim", None),
            "vggt_dim": cfg.agent.get("vggt_dim", None),
            "diffusion_loss_weight": cfg.agent.get("diffusion_loss_weight", 1.0),
            "use_alignment_loss": cfg.agent.get("use_alignment_loss", True),
            "jepa_alignment_weight": cfg.agent.get("jepa_alignment_weight", 0.0),
            "vggt_alignment_weight": cfg.agent.get("vggt_alignment_weight", 0.0),
            "train_expert_only": cfg.agent.get("train_expert_only", False),
            "freeze_base_action_head": cfg.agent.get("freeze_base_action_head", False),
            "freeze_expert": cfg.agent.get("freeze_expert", False),
            "use_last_rd": bool(cfg.agent.get("use_last_rd", False)),
            "last_rd_stage": cfg.agent.get("last_rd_stage", "disabled"),
            "last_rd_adapter_checkpoint": cfg.agent.get("last_rd_adapter_checkpoint", None),
            "use_last_vla": bool(cfg.agent.get("use_last_vla", False)),
            "last_vla_stage": cfg.agent.get("last_vla_stage", "disabled"),
            "last_vla_teacher_traj_mode": cfg.agent.get("last_vla_teacher_traj_mode", "none"),
            "last_vla_adapter_checkpoint": cfg.agent.get("last_vla_adapter_checkpoint", None),
            "reference_a0_checkpoint": cfg.agent.get("reference_a0_checkpoint", None),
            "future_jepa_loss_weight": cfg.agent.get("future_jepa_loss_weight", 0.0),
            "vggt_geometry_loss_weight": cfg.agent.get("vggt_geometry_loss_weight", 0.0),
            "coarse_traj_loss_weight": cfg.agent.get("coarse_traj_loss_weight", 0.0),
            "risk_loss_weight": cfg.agent.get("risk_loss_weight", 0.0),
        },
    }
    (output_dir / "precision_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if trainable_counts is not None:
        (output_dir / "trainable_parameter_counts.json").write_text(
            json.dumps(trainable_counts, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    if lora_target_report is not None:
        (output_dir / "lora_target_report.json").write_text(
            json.dumps(lora_target_report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    if lora_training_config is not None:
        (output_dir / "lora_training_config.json").write_text(
            json.dumps(lora_training_config, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        dataloader_cfg = cfg.get("dataloader", {})
        dataloader_params = dataloader_cfg.get("params", {}) if dataloader_cfg is not None else {}
        trainer_cfg = cfg.get("trainer", {})
        trainer_params = trainer_cfg.get("params", {}) if trainer_cfg is not None else {}
        actual_lora_audit = (lora_target_report or {}).get("actual_trainable_audit", {})
        runtime_report = {
            "effective_batch_size": dataloader_params.get("batch_size", None),
            "precision": str(trainer_params.get("precision", "")),
            "preset": lora_training_config.get("preset"),
            "scope": lora_training_config.get("scope"),
            "r": lora_training_config.get("r"),
            "alpha": lora_training_config.get("alpha"),
            "dropout": lora_training_config.get("dropout"),
            "use_rslora": lora_training_config.get("use_rslora"),
            "use_dora": lora_training_config.get("use_dora"),
            "lr_vlm_lora": cfg.agent.get("lr_vlm_lora", None),
            "lr_last_vla_cot": cfg.agent.get("lr_last_vla_cot", None),
            "hidden_anchor_weight": cfg.agent.get("last_vla_hidden_anchor_weight", 0.0),
            "hidden_anchor_mode": cfg.agent.get("last_vla_hidden_anchor_mode", "none"),
            "hidden_anchor_every_n_steps": cfg.agent.get("last_vla_hidden_anchor_every_n_steps", None),
            "matched_module_counts": (lora_target_report or {}).get("matched_by_category", {}),
            "matched_total": (lora_target_report or {}).get("matched_total", 0),
            "actual_trainable_lora_audit": actual_lora_audit,
            "actual_trainable_lora_param_count": actual_lora_audit.get("actual_trainable_lora_param_count", 0),
            "actual_trainable_lora_module_count": actual_lora_audit.get("actual_trainable_lora_module_count", 0),
            "trainable_parameter_counts": trainable_counts,
            "trainable_ratio": (lora_target_report or {}).get("trainable_ratio", 0.0),
            "backbone_non_lora_trainable": bool(
                trainable_counts
                and trainable_counts.get("backbone_non_lora", {}).get("trainable", 0) > 0
            ),
        }
        (output_dir / "lora_runtime_report.json").write_text(
            json.dumps(runtime_report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    if data_report is not None:
        (output_dir / "data_report.json").write_text(json.dumps(data_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload = OmegaConf.to_container(cfg, resolve=False)
    if isinstance(payload, dict):
        payload["loader_mode"] = loader_mode
        payload["key_checkpoint_steps"] = key_steps
        payload["key_checkpoint_epochs"] = key_epochs
        payload["key_checkpoint_epoch_interval"] = key_epoch_interval
        payload["data_report"] = data_report
    (output_dir / "train_args.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")




def custom_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Any]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], List[Any]]:
    if not batch:
        raise ValueError("custom_collate_fn received an empty batch.")

    sample_size = len(batch[0])
    if sample_size == 3:
        features_list, targets_list, tokens_list = zip(*batch)
    elif sample_size == 2:
        features_list, targets_list = zip(*batch)
        tokens_list = [None] * len(batch)
    else:
        raise ValueError(f"Expected batch samples with 2 or 3 fields, got {sample_size}.")

    history_trajectory = torch.stack([features['history_trajectory'].detach() for features in features_list], dim=0).cpu()
    high_command_one_hot = torch.stack([features['high_command_one_hot'].detach() for features in features_list], dim=0).cpu()
    status_feature = torch.stack([features['status_feature'].detach() for features in features_list], dim=0).cpu()

    trajectory = torch.stack([targets['trajectory'].detach().float() for targets in targets_list], dim=0).cpu()

    features = {
        'history_trajectory': history_trajectory,
        'high_command_one_hot': high_command_one_hot,
        'status_feature': status_feature
    }
    first_features = features_list[0]
    if "last_hidden_state" in first_features:
        features["last_hidden_state"] = rnn_utils.pad_sequence(
            [sample_features["last_hidden_state"].detach() for sample_features in features_list],
            batch_first=True,
            padding_value=0.0,
        ).detach()
    elif "image_path_tensor" in first_features:
        features["image_path_tensor"] = rnn_utils.pad_sequence(
            [sample_features["image_path_tensor"].detach() for sample_features in features_list],
            batch_first=True,
            padding_value=0,
        ).cpu()
    else:
        raise KeyError(
            "features must contain either 'last_hidden_state' or 'image_path_tensor'. "
            f"Got keys: {list(first_features.keys())}"
        )
    for key in ("two_expert_h_dyn", "two_expert_h_geo"):
        if key in first_features:
            features[key] = torch.stack(
                [sample_features[key].detach() for sample_features in features_list],
                dim=0,
            ).detach()
    stack_optional_expert_features(features, list(features_list))

    targets = {
        'trajectory': trajectory
    }

    return features, targets, list(tokens_list)

def _expert_cache_index_tokens(expert_cache_dir: Optional[str]) -> tuple[Set[str], Set[str]]:
    """Returns sample/log names present in a chunk-style expert cache index."""
    if not expert_cache_dir:
        return set(), set()

    cache_root = Path(expert_cache_dir)
    if not cache_root.is_dir():
        return set(), set()

    if (cache_root / "index.jsonl").is_file():
        chunk_dirs = [cache_root]
    else:
        chunk_dirs = sorted(
            child for child in cache_root.iterdir()
            if child.is_dir() and (child / "index.jsonl").is_file()
        )

    sample_tokens: Set[str] = set()
    log_names: Set[str] = set()
    for chunk_dir in chunk_dirs:
        for record in iter_index(chunk_dir):
            token = record.get("sample_token")
            if token is not None:
                sample_tokens.add(str(token))
            log_name = record.get("log_name")
            if log_name is not None:
                log_names.add(str(log_name))
    return sample_tokens, log_names


def _maybe_restrict_scene_filter_to_expert_cache(
    scene_filter: SceneFilter,
    cfg: DictConfig,
    *,
    split_name: str,
) -> None:
    """Align online SceneLoader sampling with strict teacher-cache coverage."""
    agent_cfg = cfg.agent
    use_online_dataset = not bool(cfg.get("use_cache_without_dataset", False))
    cache_path = cfg.get("cache_path", None)
    uses_chunk_expert = (
        bool(agent_cfg.get("use_expert_features", False))
        and str(agent_cfg.get("expert_feature_source", "none")) in {"chunk", "disk"}
        and bool(agent_cfg.get("expert_cache_dir", None))
    )
    if not (use_online_dataset and cache_path is None and uses_chunk_expert):
        return

    sample_tokens, log_names = _expert_cache_index_tokens(str(agent_cfg.get("expert_cache_dir")))
    if not sample_tokens:
        raise FileNotFoundError(
            "Online expert-cache training requested but no sample tokens were found in "
            f"agent.expert_cache_dir={agent_cfg.get('expert_cache_dir')!r}."
        )

    if scene_filter.tokens is None:
        scene_filter.tokens = sorted(sample_tokens)
    else:
        scene_filter.tokens = sorted(set(str(token) for token in scene_filter.tokens) & sample_tokens)
    if not scene_filter.tokens:
        raise FileNotFoundError(
            f"{split_name} SceneLoader has no token overlap with expert cache "
            f"{agent_cfg.get('expert_cache_dir')!r}."
        )

    if scene_filter.log_names is not None and log_names:
        scene_filter.log_names = sorted(set(str(log_name) for log_name in scene_filter.log_names) & log_names)

    logger.info(
        "Restricted %s SceneLoader to %d expert-cache sample tokens from %s.",
        split_name,
        len(scene_filter.tokens),
        agent_cfg.get("expert_cache_dir"),
    )


def _normalized_dataloader_params(params_cfg: DictConfig) -> Dict[str, Any]:
    params = OmegaConf.to_container(params_cfg, resolve=True)
    if not isinstance(params, dict):
        raise TypeError(f"dataloader.params must resolve to a dict, got {type(params).__name__}.")
    num_workers = int(params.get("num_workers", 0) or 0)
    if num_workers <= 0:
        if params.get("prefetch_factor", None) is not None:
            params["prefetch_factor"] = None
        if params.get("persistent_workers", None) is not None:
            params["persistent_workers"] = False
    return params


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    _maybe_restrict_scene_filter_to_expert_cache(train_scene_filter, cfg, split_name="train")
    _maybe_restrict_scene_filter_to_expert_cache(val_scene_filter, cfg, split_name="val")

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True,
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True,
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', 0))

    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(local_rank)
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    assert_real_expert_cache_for_training(
        cfg.agent.get("expert_cache_dir", None),
        use_expert_features=cfg.agent.get("use_expert_features", False),
        allow_dummy_expert_cache=cfg.get(
            "allow_dummy_expert_cache",
            cfg.agent.get("allow_dummy_expert_cache", False),
        ),
        expert_feature_source=cfg.agent.get("expert_feature_source", "none"),
    )

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    if agent.__class__.__name__ == "ReCogDriveAgent":
        agent.initialize()

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        if ChunkCacheDataset.looks_like(cfg.cache_path):
            logger.warning("Using official-aligned-local-loader for chunk cache path: %s", cfg.cache_path)
            use_last_rd = bool(cfg.agent.get("use_last_rd", False))
            use_last_vla = bool(cfg.agent.get("use_last_vla", False))
            include_expert_features = bool(cfg.agent.get("use_expert_features", False) or use_last_rd or use_last_vla)
            include_expert_targets = bool(cfg.agent.get("allow_expert_target_features", False) or use_last_rd or use_last_vla)
            use_jepa = bool(cfg.agent.get("use_jepa", True))
            use_vggt = bool(cfg.agent.get("use_vggt", True))
            use_two_expert_slots = bool(cfg.agent.get("use_two_expert_slots", False))
            cache_train_all_records = bool(
                cfg.get("cache_train_all_records", cfg.get("train_all_cache_records", False))
            )
            train_log_names = None if cache_train_all_records else list(cfg.train_logs)
            stage2_target_source = str(cfg.get("stage2_target_source", "gt"))
            stage2_elite_target_index_path = cfg.get("stage2_elite_target_index_path", None)
            train_data = ChunkCacheDataset(
                cfg.cache_path,
                log_names=train_log_names,
                split_name="train_all_cache" if cache_train_all_records else "train",
                include_expert_features=include_expert_features,
                include_expert_targets=include_expert_targets,
                use_jepa=use_jepa,
                use_vggt=use_vggt,
                num_jepa_tokens=int(cfg.agent.get("num_jepa_tokens", 12)),
                num_vggt_tokens=int(cfg.agent.get("num_vggt_tokens", 12)),
                num_geometry_tokens=int(cfg.agent.get("num_geometry_tokens", cfg.agent.get("num_vggt_tokens", 12))),
                jepa_dim=int(cfg.agent.get("jepa_dim", 1024)),
                vggt_dim=int(cfg.agent.get("vggt_dim", 2048)),
                use_last_rd=use_last_rd,
                require_vggt_geometry=bool(cfg.agent.get("require_vggt_geometry", False)),
                allow_patch_geometry_fallback=bool(cfg.agent.get("allow_patch_geometry_fallback", True)),
                future_jepa_loss_weight=float(cfg.agent.get("future_jepa_loss_weight", 0.0)),
                vggt_geometry_loss_weight=float(cfg.agent.get("vggt_geometry_loss_weight", 0.0)),
                use_last_vla=use_last_vla,
                last_vla_stage=str(cfg.agent.get("last_vla_stage", "disabled")),
                last_vla_teacher_traj_mode=str(cfg.agent.get("last_vla_teacher_traj_mode", "none")),
                last_vla_use_residual_diffusion=bool(cfg.agent.get("last_vla_use_residual_diffusion", False)),
                last_vla_residual_anchor_source=str(cfg.agent.get("last_vla_residual_anchor_source", "vlm_text_traj")),
                last_vla_require_residual_anchor=bool(cfg.agent.get("last_vla_require_residual_anchor", True)),
                last_vla_residual_anchor_cache_dir=cfg.agent.get("last_vla_residual_anchor_cache_dir", None),
                last_vla_require_full_geometry=bool(cfg.agent.get("last_vla_require_full_geometry", False)),
                last_vla_allow_patch_geometry_fallback=bool(cfg.agent.get("last_vla_allow_patch_geometry_fallback", False)),
                last_vla_geometry_teacher_dim=int(cfg.agent.get("last_vla_geometry_teacher_dim", 512)),
                last_vla_geometry_loss_weight=float(cfg.agent.get("last_vla_geometry_loss_weight", 0.0)),
                use_two_expert_slots=use_two_expert_slots,
                two_expert_num_dyn_groups=int(cfg.agent.get("two_expert_num_dyn_groups", 3)),
                two_expert_dyn_tokens_per_group=int(cfg.agent.get("two_expert_dyn_tokens_per_group", 12)),
                two_expert_num_geo_tokens=int(cfg.agent.get("two_expert_num_geo_tokens", 12)),
                two_expert_vlm_hidden_dim=int(cfg.agent.get("two_expert_vlm_hidden_dim", cfg.agent.get("vlm_hidden_dim", 1536))),
                stage2_target_source=stage2_target_source,
                stage2_elite_target_index_path=stage2_elite_target_index_path,
            )
            val_data = ChunkCacheDataset(
                cfg.cache_path,
                log_names=list(cfg.val_logs),
                split_name="val",
                include_expert_features=include_expert_features,
                include_expert_targets=use_last_rd or use_last_vla,
                use_jepa=use_jepa,
                use_vggt=use_vggt,
                num_jepa_tokens=int(cfg.agent.get("num_jepa_tokens", 12)),
                num_vggt_tokens=int(cfg.agent.get("num_vggt_tokens", 12)),
                num_geometry_tokens=int(cfg.agent.get("num_geometry_tokens", cfg.agent.get("num_vggt_tokens", 12))),
                jepa_dim=int(cfg.agent.get("jepa_dim", 1024)),
                vggt_dim=int(cfg.agent.get("vggt_dim", 2048)),
                use_last_rd=use_last_rd,
                require_vggt_geometry=bool(cfg.agent.get("require_vggt_geometry", False)),
                allow_patch_geometry_fallback=bool(cfg.agent.get("allow_patch_geometry_fallback", True)),
                future_jepa_loss_weight=float(cfg.agent.get("future_jepa_loss_weight", 0.0)),
                vggt_geometry_loss_weight=float(cfg.agent.get("vggt_geometry_loss_weight", 0.0)),
                use_last_vla=use_last_vla,
                last_vla_stage=str(cfg.agent.get("last_vla_stage", "disabled")),
                last_vla_teacher_traj_mode=str(cfg.agent.get("last_vla_teacher_traj_mode", "none")),
                last_vla_use_residual_diffusion=bool(cfg.agent.get("last_vla_use_residual_diffusion", False)),
                last_vla_residual_anchor_source=str(cfg.agent.get("last_vla_residual_anchor_source", "vlm_text_traj")),
                last_vla_require_residual_anchor=bool(cfg.agent.get("last_vla_require_residual_anchor", True)),
                last_vla_residual_anchor_cache_dir=cfg.agent.get("last_vla_residual_anchor_cache_dir", None),
                last_vla_require_full_geometry=bool(cfg.agent.get("last_vla_require_full_geometry", False)),
                last_vla_allow_patch_geometry_fallback=bool(cfg.agent.get("last_vla_allow_patch_geometry_fallback", False)),
                last_vla_geometry_teacher_dim=int(cfg.agent.get("last_vla_geometry_teacher_dim", 512)),
                last_vla_geometry_loss_weight=float(cfg.agent.get("last_vla_geometry_loss_weight", 0.0)),
                use_two_expert_slots=use_two_expert_slots,
                two_expert_num_dyn_groups=int(cfg.agent.get("two_expert_num_dyn_groups", 3)),
                two_expert_dyn_tokens_per_group=int(cfg.agent.get("two_expert_dyn_tokens_per_group", 12)),
                two_expert_num_geo_tokens=int(cfg.agent.get("two_expert_num_geo_tokens", 12)),
                two_expert_vlm_hidden_dim=int(cfg.agent.get("two_expert_vlm_hidden_dim", cfg.agent.get("vlm_hidden_dim", 1536))),
                stage2_target_source=stage2_target_source,
                stage2_elite_target_index_path=stage2_elite_target_index_path,
            )
            train_tokens = set(train_data.sample_tokens())
            val_tokens = set(val_data.sample_tokens())
            overlap_count = len(train_tokens & val_tokens)
            if overlap_count and not cache_train_all_records:
                raise RuntimeError(f"Local chunk train/val split overlap is not allowed; overlap_count={overlap_count}")
            loader_mode = (
                "official-aligned-local-loader-all-cache-train-log-val"
                if cache_train_all_records
                else "official-aligned-local-loader-log-split"
            )
            if int(os.getenv("RANK", "0")) == 0:
                data_report = {
                    "loader_mode": loader_mode,
                    "cache_train_all_records": cache_train_all_records,
                    "train": train_data.report(),
                    "val": val_data.report(),
                    "train_val_overlap_count": overlap_count,
                    "train_unique_sample_token_count": len(train_tokens),
                    "val_unique_sample_token_count": len(val_tokens),
                    "a0_strict_feature_whitelist": not include_expert_features,
                }
            else:
                data_report = None
        else:
            cache_train_all_records = bool(
                cfg.get("cache_train_all_records", cfg.get("train_all_cache_records", False))
            )
            train_log_names = None if cache_train_all_records else cfg.train_logs
            if cache_train_all_records:
                logger.warning(
                    "cache_train_all_records=true: CacheOnlyDataset train split will load every valid cache record from %s.",
                    cfg.cache_path,
                )
            train_data = CacheOnlyDataset(
                cache_path=cfg.cache_path,
                feature_builders=agent.get_feature_builders(),
                target_builders=agent.get_target_builders(),
                log_names=train_log_names,
            )
            val_data = CacheOnlyDataset(
                cache_path=cfg.cache_path,
                feature_builders=agent.get_feature_builders(),
                target_builders=agent.get_target_builders(),
                log_names=cfg.val_logs,
            )
            loader_mode = "official-cache-loader-all-cache-train-log-val" if cache_train_all_records else "official-cache-loader"
            data_report = None
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)
        loader_mode = "official-scene-loader"
        data_report = None

    logger.info("Building Datasets")
    dataloader_params = _normalized_dataloader_params(cfg.dataloader.params)
    train_dataloader = DataLoader(train_data, collate_fn=custom_collate_fn,  **dataloader_params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, collate_fn=custom_collate_fn, **dataloader_params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    key_steps = _parse_key_steps()
    key_epochs = _parse_key_epochs()
    key_epoch_interval = _parse_key_epoch_interval()
    trainer_params = cfg.trainer.params
    limit_val_batches = trainer_params.get("limit_val_batches", 1.0)
    has_validation = not (
        limit_val_batches == 0
        or limit_val_batches == 0.0
        or str(limit_val_batches).strip().lower() in {"0", "0.0", "false", "none"}
    )
    if has_validation:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val/loss_epoch",
            mode='min',
            save_top_k=5,
            every_n_epochs=1,
            save_last=True,
        )
    else:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=None,
            save_top_k=-1,
            every_n_epochs=1,
            save_last=True,
        )
    callbacks = [
        checkpoint_callback,
        StepCheckpointCallback(Path(cfg.output_dir), key_steps),
        ReCogDriveTrainingProgressCallback(),
    ]
    if key_epochs or key_epoch_interval > 0:
        callbacks.append(EpochCheckpointCallback(Path(cfg.output_dir), key_epochs, every_n_epochs=key_epoch_interval))
    write_run_reports(cfg, agent, loader_mode, key_steps, key_epochs, key_epoch_interval, data_report)
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=callbacks)

    resume_ckpt_path = cfg.get("resume_ckpt_path", None)
    if resume_ckpt_path:
        resume_ckpt_path = str(resume_ckpt_path)
        logger.info("Resuming Training from checkpoint: %s", resume_ckpt_path)
    else:
        resume_ckpt_path = None

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=resume_ckpt_path,
    )


if __name__ == "__main__":
    main()
