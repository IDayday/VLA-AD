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
    EXPERT_TARGET_KEYS = (
        "jepa_target_tokens",
        "vggt_target_tokens",
        "vggt_geometry_target_tokens",
        "vggt_depth_target_tokens",
        "vggt_pointmap_target_tokens",
    )
    RISK_KEYS = ("risk_labels", "generic_risk_labels", "drivable_risk_labels", "ttc_risk_labels", "comfort_risk_labels")

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
        jepa_dim: int = 1024,
        vggt_dim: int = 2048,
        use_last_rd: bool = False,
        require_vggt_geometry: bool = False,
        allow_patch_geometry_fallback: bool = True,
        future_jepa_loss_weight: float = 0.0,
        vggt_geometry_loss_weight: float = 0.0,
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
        self.jepa_dim = int(jepa_dim)
        self.vggt_dim = int(vggt_dim)
        self.use_last_rd = bool(use_last_rd)
        self.require_vggt_geometry = bool(require_vggt_geometry)
        self.allow_patch_geometry_fallback = bool(allow_patch_geometry_fallback)
        self.future_jepa_loss_weight = float(future_jepa_loss_weight)
        self.vggt_geometry_loss_weight = float(vggt_geometry_loss_weight)
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

    @staticmethod
    def _chunk_dirs(cache_path: Path) -> List[Path]:
        if (cache_path / "index.jsonl").is_file():
            return [cache_path]
        return sorted(
            child for child in cache_path.iterdir()
            if child.is_dir()
            and (child / "index.jsonl").is_file()
            and not child.name.startswith("navtest")
        )

    @staticmethod
    def looks_like(cache_path: str) -> bool:
        path = Path(cache_path)
        if not path.is_dir():
            return False
        if (path / "index.jsonl").is_file() and (path / "samples").is_dir():
            return True
        return any(
            child.is_dir() and (child / "index.jsonl").is_file() and (child / "samples").is_dir()
            for child in path.iterdir()
        )

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _require_tensor(sample: Dict[str, Any], key: str, sample_path: Path) -> torch.Tensor:
        if key not in sample:
            raise KeyError(f"Chunk sample {sample_path} is missing required key '{key}'.")
        value = sample[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Chunk sample {sample_path} key '{key}' must be a torch.Tensor, got {type(value).__name__}.")
        if not torch.isfinite(value.float()).all():
            raise ValueError(f"Chunk sample {sample_path} key '{key}' contains non-finite values.")
        return value

    def _required_expert_context_keys(self) -> List[str]:
        if not self.include_expert_features:
            return []
        keys: List[str] = []
        if self.use_jepa:
            keys.append("jepa_context_tokens")
        if self.use_vggt or (self.use_last_rd and self.vggt_geometry_loss_weight > 0.0):
            keys.append("vggt_context_tokens")
        if self.use_last_rd and self.require_vggt_geometry:
            keys.append("vggt_geometry_tokens")
        return keys

    def _required_expert_target_keys(self) -> List[str]:
        if not self.include_expert_targets:
            return []
        keys: List[str] = []
        if self.use_jepa and (not self.use_last_rd or self.future_jepa_loss_weight > 0.0):
            keys.append("jepa_target_tokens")
        if self.use_vggt:
            keys.append("vggt_target_tokens")
        if self.use_last_rd and self.require_vggt_geometry:
            keys.append("vggt_geometry_target_tokens")
        return keys

    def _expected_shape(self, key: str) -> Optional[Tuple[int, ...]]:
        expert_shapes = {
            "jepa_context_tokens": (self.num_jepa_tokens, self.jepa_dim),
            "jepa_target_tokens": (self.num_jepa_tokens, self.jepa_dim),
            "vggt_context_tokens": (self.num_vggt_tokens, self.vggt_dim),
            "vggt_target_tokens": (self.num_vggt_tokens, self.vggt_dim),
            "vggt_geometry_tokens": (self.num_vggt_tokens, self.vggt_dim),
            "vggt_geometry_target_tokens": (self.num_vggt_tokens, self.vggt_dim),
            "vggt_depth_tokens": (self.num_vggt_tokens, self.vggt_dim),
            "vggt_pointmap_tokens": (self.num_vggt_tokens, self.vggt_dim),
            "vggt_camera_tokens": (self.num_vggt_tokens, self.vggt_dim),
        }
        if key in expert_shapes:
            return expert_shapes[key]
        expected = {
            "history_trajectory": (4, 3),
            "high_command_one_hot": (4,),
            "status_feature": (8,),
            "trajectory": (8, 3),
        }.get(key)
        return expected

    def _check_shape(self, tensor: torch.Tensor, key: str, sample_path: Path) -> None:
        expected = self._expected_shape(key)
        if expected is not None and tuple(tensor.shape) != expected:
            raise ValueError(f"Chunk sample {sample_path} key '{key}' shape {tuple(tensor.shape)} != {expected}.")
        if key == "last_hidden_state" and (tensor.ndim != 2 or tensor.shape[-1] != 1536):
            raise ValueError(
                f"Chunk sample {sample_path} key 'last_hidden_state' must have shape [N, 1536], got {tuple(tensor.shape)}."
            )

    def sample_tokens(self) -> List[str]:
        return [
            str(record.get("sample_token") or sample_path.stem)
            for _, sample_path, record in self.records
        ]

    def report(self) -> Dict[str, Any]:
        tokens = self.sample_tokens()
        token_counts = Counter(tokens)
        log_counts = Counter(str(record.get("log_name") or "missing") for _, _, record in self.records)
        chunk_counts = Counter(str(chunk_dir.name) for chunk_dir, _, _ in self.records)
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
            "jepa_dim": self.jepa_dim,
            "vggt_dim": self.vggt_dim,
            "required_expert_context_keys": self._required_expert_context_keys(),
            "required_expert_target_keys": self._required_expert_target_keys(),
            "use_last_rd": self.use_last_rd,
            "require_vggt_geometry": self.require_vggt_geometry,
            "allow_patch_geometry_fallback": self.allow_patch_geometry_fallback,
            "future_jepa_loss_weight": self.future_jepa_loss_weight,
            "vggt_geometry_loss_weight": self.vggt_geometry_loss_weight,
            "top_log_names": log_counts.most_common(10),
            "chunk_counts": dict(sorted(chunk_counts.items())),
        }

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]:
        _, sample_path, record = self.records[idx]
        sample = load_sample(sample_path)
        for key in self.REQUIRED_FEATURE_KEYS + self.REQUIRED_TARGET_KEYS:
            self._check_shape(self._require_tensor(sample, key, sample_path), key, sample_path)
        expert_keys = self._required_expert_context_keys() + self._required_expert_target_keys()
        for key in expert_keys:
            self._check_shape(self._require_tensor(sample, key, sample_path), key, sample_path)
        optional_expert_keys = [
            key for key in self.EXPERT_CONTEXT_KEYS + self.EXPERT_GEOMETRY_KEYS + self.EXPERT_TARGET_KEYS + self.RISK_KEYS
            if key in sample and key not in expert_keys
        ]
        for key in optional_expert_keys:
            value = self._require_tensor(sample, key, sample_path)
            self._check_shape(value, key, sample_path)
        features = {
            "history_trajectory": sample["history_trajectory"].float(),
            "high_command_one_hot": sample["high_command_one_hot"].float(),
            "last_hidden_state": sample["last_hidden_state"].float(),
            "status_feature": sample["status_feature"].float(),
        }
        for key in [*expert_keys, *optional_expert_keys]:
            features[key] = sample[key].float()
        targets = {"trajectory": sample["trajectory"].float()}
        token = str(sample.get("sample_token") or record.get("sample_token") or sample_path.stem)
        return features, targets, token


def write_run_reports(
    cfg: DictConfig,
    agent: AbstractAgent,
    loader_mode: str,
    key_steps: List[int],
    data_report: Optional[Dict[str, Any]] = None,
) -> None:
    if int(os.getenv("RANK", "0")) != 0:
        return
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    group_names, group_lrs, group_wds = _configured_optimizer_groups(cfg.agent)
    report = {
        "requested_precision": str(cfg.trainer.params.get("precision", "")),
        "true_bf16_weights": False,
        "model_param_dtype_counts": _tensor_dtype_counts(agent.parameters()) if hasattr(agent, "parameters") else {},
        "model_buffer_dtype_counts": _tensor_dtype_counts(agent.buffers()) if hasattr(agent, "buffers") else {},
        "optimizer_group_names": group_names,
        "optimizer_group_lrs": group_lrs,
        "optimizer_group_weight_decay": group_wds,
        "loader_mode": loader_mode,
        "key_checkpoint_steps": key_steps,
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
            "future_jepa_loss_weight": cfg.agent.get("future_jepa_loss_weight", 0.0),
            "vggt_geometry_loss_weight": cfg.agent.get("vggt_geometry_loss_weight", 0.0),
            "coarse_traj_loss_weight": cfg.agent.get("coarse_traj_loss_weight", 0.0),
            "risk_loss_weight": cfg.agent.get("risk_loss_weight", 0.0),
        },
    }
    (output_dir / "precision_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if data_report is not None:
        (output_dir / "data_report.json").write_text(json.dumps(data_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload = OmegaConf.to_container(cfg, resolve=False)
    if isinstance(payload, dict):
        payload["loader_mode"] = loader_mode
        payload["key_checkpoint_steps"] = key_steps
        payload["data_report"] = data_report
    (output_dir / "train_args.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")




def custom_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    features_list, targets_list, tokens_list = zip(*batch)

    history_trajectory = torch.stack([features['history_trajectory'] for features in features_list], dim=0).cpu()
    high_command_one_hot = torch.stack([features['high_command_one_hot'] for features in features_list], dim=0).cpu()
    status_feature = torch.stack([features['status_feature'] for features in features_list], dim=0).cpu()

    last_hidden_state = rnn_utils.pad_sequence(
        [features['last_hidden_state'] for features in features_list],
        batch_first=True,
        padding_value=0.0
    ).clone().detach()

    trajectory = torch.stack([targets['trajectory'].float() for targets in targets_list], dim=0).cpu()

    features = {
        'history_trajectory': history_trajectory,
        'high_command_one_hot': high_command_one_hot,
        'last_hidden_state': last_hidden_state,
        'status_feature': status_feature
    }
    stack_optional_expert_features(features, list(features_list))

    targets = {
        'trajectory': trajectory
    }

    return features, targets, tokens_list

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

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
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
            include_expert_features = bool(cfg.agent.get("use_expert_features", False) or use_last_rd)
            include_expert_targets = bool(cfg.agent.get("allow_expert_target_features", False) or use_last_rd)
            use_jepa = bool(cfg.agent.get("use_jepa", True))
            use_vggt = bool(cfg.agent.get("use_vggt", True))
            train_data = ChunkCacheDataset(
                cfg.cache_path,
                log_names=list(cfg.train_logs),
                split_name="train",
                include_expert_features=include_expert_features,
                include_expert_targets=include_expert_targets,
                use_jepa=use_jepa,
                use_vggt=use_vggt,
                num_jepa_tokens=int(cfg.agent.get("num_jepa_tokens", 12)),
                num_vggt_tokens=int(cfg.agent.get("num_vggt_tokens", 12)),
                jepa_dim=int(cfg.agent.get("jepa_dim", 1024)),
                vggt_dim=int(cfg.agent.get("vggt_dim", 2048)),
                use_last_rd=use_last_rd,
                require_vggt_geometry=bool(cfg.agent.get("require_vggt_geometry", False)),
                allow_patch_geometry_fallback=bool(cfg.agent.get("allow_patch_geometry_fallback", True)),
                future_jepa_loss_weight=float(cfg.agent.get("future_jepa_loss_weight", 0.0)),
                vggt_geometry_loss_weight=float(cfg.agent.get("vggt_geometry_loss_weight", 0.0)),
            )
            val_data = ChunkCacheDataset(
                cfg.cache_path,
                log_names=list(cfg.val_logs),
                split_name="val",
                include_expert_features=include_expert_features,
                include_expert_targets=use_last_rd,
                use_jepa=use_jepa,
                use_vggt=use_vggt,
                num_jepa_tokens=int(cfg.agent.get("num_jepa_tokens", 12)),
                num_vggt_tokens=int(cfg.agent.get("num_vggt_tokens", 12)),
                jepa_dim=int(cfg.agent.get("jepa_dim", 1024)),
                vggt_dim=int(cfg.agent.get("vggt_dim", 2048)),
                use_last_rd=use_last_rd,
                require_vggt_geometry=bool(cfg.agent.get("require_vggt_geometry", False)),
                allow_patch_geometry_fallback=bool(cfg.agent.get("allow_patch_geometry_fallback", True)),
                future_jepa_loss_weight=float(cfg.agent.get("future_jepa_loss_weight", 0.0)),
                vggt_geometry_loss_weight=float(cfg.agent.get("vggt_geometry_loss_weight", 0.0)),
            )
            train_tokens = set(train_data.sample_tokens())
            val_tokens = set(val_data.sample_tokens())
            overlap_count = len(train_tokens & val_tokens)
            if overlap_count:
                raise RuntimeError(f"Local chunk train/val split overlap is not allowed; overlap_count={overlap_count}")
            loader_mode = "official-aligned-local-loader-log-split"
            data_report = {
                "loader_mode": loader_mode,
                "train": train_data.report(),
                "val": val_data.report(),
                "train_val_overlap_count": overlap_count,
                "train_unique_sample_token_count": len(train_tokens),
                "val_unique_sample_token_count": len(val_tokens),
                "a0_strict_feature_whitelist": not include_expert_features,
            }
        else:
            train_data = CacheOnlyDataset(
                cache_path=cfg.cache_path,
                feature_builders=agent.get_feature_builders(),
                target_builders=agent.get_target_builders(),
                log_names=cfg.train_logs,
            )
            val_data = CacheOnlyDataset(
                cache_path=cfg.cache_path,
                feature_builders=agent.get_feature_builders(),
                target_builders=agent.get_target_builders(),
                log_names=cfg.val_logs,
            )
            loader_mode = "official-cache-loader"
            data_report = None
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)
        loader_mode = "official-scene-loader"
        data_report = None

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, collate_fn=custom_collate_fn,  **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, collate_fn=custom_collate_fn, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    key_steps = _parse_key_steps()
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="val/loss_epoch",
            mode='min',
            save_top_k=5,
            every_n_epochs=1,
            save_last=True,
        ),
        StepCheckpointCallback(Path(cfg.output_dir), key_steps),
        ReCogDriveTrainingProgressCallback(),
    ]
    write_run_reports(cfg, agent, loader_mode, key_steps, data_report)
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=callbacks)

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
