from typing import Any, Optional, Tuple
from pathlib import Path
import json
import logging
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.distributed as dist
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule, AgentLightningDiT
from navsim.agents.recogdrive.recogdrive_features import (
    assert_real_expert_cache_for_training,
    stack_optional_expert_features,
)
import torch
import torch.nn.utils.rnn as rnn_utils
from typing import List, Dict

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


class ReCogDriveTrainingProgressCallback(pl.Callback):
    """Propagates epoch progress into schedulable planner components."""

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        agent = getattr(pl_module, "agent", None)
        if agent is not None and hasattr(agent, "set_training_progress"):
            agent.set_training_progress(
                int(trainer.current_epoch),
                int(trainer.max_epochs),
                int(getattr(trainer, "global_step", 0)),
            )


class TokenizedDataset(torch.utils.data.Dataset):
    """Adds NAVSIM sample tokens to online Dataset items for GRPO rewards."""

    def __init__(self, dataset: Dataset):
        self._dataset = dataset
        scene_loader = getattr(dataset, "_scene_loader", None)
        tokens = getattr(scene_loader, "tokens", None)
        if tokens is None:
            raise AttributeError("TokenizedDataset requires an online Dataset with _scene_loader.tokens.")
        self._tokens = tokens

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]:
        sample = self._dataset[idx]
        if len(sample) == 3:
            return sample
        features, targets = sample
        return features, targets, self._tokens[idx]


class IndexedPtCacheDataset(torch.utils.data.Dataset):
    """Reads merged chunk-cache samples written as indexed .pt payloads."""

    FEATURE_KEYS = (
        "history_trajectory",
        "high_command_one_hot",
        "status_feature",
        "last_hidden_state",
        "image_path_tensor",
        "two_expert_h_dyn",
        "two_expert_h_geo",
    )

    def __init__(self, cache_path: str, log_names: Optional[List[str]] = None) -> None:
        super().__init__()
        self.cache_path = Path(cache_path)
        if not self.cache_path.is_dir():
            raise FileNotFoundError(f"Indexed cache path does not exist: {self.cache_path}")
        self.log_name_filter = set(str(log_name) for log_name in log_names) if log_names else None
        self.records: List[tuple[Path, Dict[str, Any]]] = []
        for index_dir in self._index_dirs(self.cache_path):
            with (index_dir / "index.jsonl").open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    log_name = record.get("log_name")
                    if self.log_name_filter is not None and str(log_name) not in self.log_name_filter:
                        continue
                    sample_path = Path(str(record.get("path", "")))
                    if not str(sample_path):
                        continue
                    sample_path = sample_path if sample_path.is_absolute() else index_dir / sample_path
                    if sample_path.is_file():
                        self.records.append((sample_path, record))
        if not self.records:
            raise FileNotFoundError(f"No indexed .pt cache records found under {self.cache_path}.")
        self.tokens = [
            str(record.get("sample_token") or sample_path.stem)
            for sample_path, record in self.records
        ]

    @staticmethod
    def _index_dirs(cache_path: Path) -> List[Path]:
        if (cache_path / "index.jsonl").is_file():
            return [cache_path]
        index_dirs = [
            child for child in cache_path.iterdir()
            if child.is_dir() and (child / "index.jsonl").is_file()
        ]
        shard_root = cache_path / "shards"
        if shard_root.is_dir():
            index_dirs.extend(
                child for child in shard_root.iterdir()
                if child.is_dir() and (child / "index.jsonl").is_file()
            )
        return sorted(index_dirs)

    @classmethod
    def looks_like(cls, cache_path: str) -> bool:
        path = Path(cache_path)
        if not path.is_dir():
            return False
        return bool(cls._index_dirs(path))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]:
        sample_path, record = self.records[idx]
        payload = torch.load(sample_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"Indexed cache payload must be a dict: {sample_path}")
        features: Dict[str, torch.Tensor] = {}
        for key in self.FEATURE_KEYS:
            value = payload.get(key)
            if isinstance(value, torch.Tensor):
                features[key] = value.long() if key == "image_path_tensor" else value.float()
        if "trajectory" not in payload or not isinstance(payload["trajectory"], torch.Tensor):
            raise KeyError(f"Indexed cache payload missing tensor trajectory: {sample_path}")
        targets = {"trajectory": payload["trajectory"].float()}
        token = str(payload.get("sample_token") or record.get("sample_token") or sample_path.stem)
        return features, targets, token


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

    history_trajectory = torch.stack([features['history_trajectory'] for features in features_list], dim=0).cpu()
    high_command_one_hot = torch.stack([features['high_command_one_hot'] for features in features_list], dim=0).cpu()
    status_feature = torch.stack([features['status_feature'] for features in features_list], dim=0).cpu()

    trajectory = torch.stack([targets['trajectory'] for targets in targets_list], dim=0).cpu()


    features = {
        'history_trajectory': history_trajectory,
        'high_command_one_hot': high_command_one_hot,
        'status_feature': status_feature,
    }
    first_features = features_list[0]
    if "last_hidden_state" in first_features:
        features["last_hidden_state"] = rnn_utils.pad_sequence(
            [features['last_hidden_state'] for features in features_list],
            batch_first=True,
            padding_value=0.0
        ).clone().detach()
    elif "image_path_tensor" in first_features:
        features["image_path_tensor"] = rnn_utils.pad_sequence(
            [features["image_path_tensor"] for features in features_list],
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
    load_image_path = not getattr(agent, "cache_hidden_state", True)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=load_image_path,
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=load_image_path,
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
    agent.initialize()

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningDiT(
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
        if IndexedPtCacheDataset.looks_like(cfg.cache_path):
            logger.info("Using indexed .pt chunk cache dataset")
            train_data = IndexedPtCacheDataset(cache_path=cfg.cache_path, log_names=cfg.train_logs)
            val_data = IndexedPtCacheDataset(cache_path=cfg.cache_path, log_names=cfg.val_logs)
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
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)
        train_data = TokenizedDataset(train_data)
        val_data = TokenizedDataset(val_data)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, collate_fn=custom_collate_fn,  **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, collate_fn=custom_collate_fn, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    checkpoint_cfg = cfg.get("checkpoint", {})
    checkpoint_every_n_epochs = int(checkpoint_cfg.get("every_n_epochs", 1) or 0)
    checkpoint_every_n_train_steps = int(checkpoint_cfg.get("every_n_train_steps", 0) or 0)
    checkpoint_save_on_train_epoch_end = checkpoint_cfg.get("save_on_train_epoch_end", True)
    callbacks = []
    if checkpoint_every_n_epochs > 0:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                filename="{epoch}-{step}",
                save_top_k=-1,
                every_n_epochs=checkpoint_every_n_epochs,
                save_on_train_epoch_end=checkpoint_save_on_train_epoch_end,
            )
        )
    if checkpoint_every_n_train_steps > 0:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=Path(cfg.output_dir) / "step_checkpoints",
                filename="step-{step}",
                save_top_k=-1,
                every_n_train_steps=checkpoint_every_n_train_steps,
                every_n_epochs=0,
            )
        )
    callbacks.extend(
        [
            pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
            ReCogDriveTrainingProgressCallback(),
        ]
    )
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=callbacks)

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
