from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
import uuid
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import pickle

from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.planning.training.dataset import Dataset
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.agents.abstract_agent import AbstractAgent

import os
import torch.distributed as dist
import torch


logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def _read_token_manifest(path: Union[str, Path]) -> List[Tuple[str, str]]:
    """
    Read a tab-separated log/token manifest.

    Each non-comment line must be either `log_name<TAB>token` or
    `log_name token`. Keeping the manifest outside Hydra avoids passing very
    large token lists through the command line.
    """
    entries: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Malformed token manifest line {line_no} in {path}: {raw_line!r}")
            log_name, token = parts
            entries.append((log_name, token))
    if not entries:
        raise ValueError(f"Token manifest is empty: {path}")
    return entries


def _apply_token_manifest_shard(scene_filter: SceneFilter, cfg: DictConfig) -> None:
    manifest_path = OmegaConf.select(cfg, "cache_token_manifest")
    if manifest_path is None:
        manifest_path = os.environ.get("NAVSIM_CACHE_TOKEN_MANIFEST")
    if not manifest_path:
        return

    shard_count = int(
        OmegaConf.select(cfg, "cache_token_shard_count")
        or os.environ.get("NAVSIM_CACHE_TOKEN_SHARD_COUNT", "1")
    )
    shard_index = int(
        OmegaConf.select(cfg, "cache_token_shard_index")
        or os.environ.get("NAVSIM_CACHE_TOKEN_SHARD_INDEX", "0")
    )
    if shard_count < 1:
        raise ValueError(f"cache_token_shard_count must be >= 1, got {shard_count}")
    if not (0 <= shard_index < shard_count):
        raise ValueError(f"cache_token_shard_index must be in [0, {shard_count}), got {shard_index}")

    entries = _read_token_manifest(manifest_path)
    selected_entries = [entry for idx, entry in enumerate(entries) if idx % shard_count == shard_index]
    if not selected_entries:
        raise ValueError(
            f"Token manifest shard {shard_index}/{shard_count} is empty: {manifest_path}"
        )

    if scene_filter.tokens is not None:
        allowed_tokens = set(scene_filter.tokens)
        selected_entries = [(log_name, token) for log_name, token in selected_entries if token in allowed_tokens]

    scene_filter.tokens = [token for _, token in selected_entries]
    scene_filter.log_names = sorted({log_name for log_name, _ in selected_entries})
    logger.info(
        "Applied token manifest shard %s/%s from %s: %s tokens across %s logs.",
        shard_index,
        shard_count,
        manifest_path,
        len(scene_filter.tokens),
        len(scene_filter.log_names),
    )


def cache_features(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Optional[Any]]:
    """
    Helper function to cache features and targets of learnable agent.
    :param args: arguments for caching
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    agent: AbstractAgent = instantiate(cfg.agent)

    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True
    )
    logger.info(f"Extracted {len(scene_loader.tokens)} scenarios for thread_id={thread_id}, node_id={node_id}.")

    dataset = Dataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )
    return []

def broadcast_object(obj: Any, device: torch.device, src: int = 0) -> Any:
    """
    Helper function to broadcast an object from the source rank to all other processes.
    :param obj: Object to broadcast.
    :param device: Device to use for tensor operations.
    :param src: Source rank.
    :return: Broadcasted object.
    """
    if dist.get_rank() == src:
        buffer = pickle.dumps(obj)
        tensor = torch.ByteTensor(list(buffer)).to(device)
        size_tensor = torch.tensor(len(tensor)).to(device)
        dist.broadcast(size_tensor, src=src)
        dist.broadcast(tensor, src=src)
    else:
        size_tensor = torch.tensor(0).to(device)
        dist.broadcast(size_tensor, src=src)
        tensor = torch.ByteTensor(size_tensor.item()).to(device)
        dist.broadcast(tensor, src=src)
        buffer = tensor.cpu().numpy().tobytes()
        obj = pickle.loads(buffer)
    return obj

class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for dataset caching script.
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
    device = torch.device(f'cuda:{local_rank}')  

    logger.info("Global Seed set to 0")
    pl.seed_everything(0)

    logger.info("Building Worker")
    worker: WorkerPool = instantiate(cfg.worker)

    logger.info("Building SceneLoader")
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    _apply_token_manifest_shard(scene_filter, cfg)
    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    if rank == 0:
        tokens_to_evaluate = list(set(scene_loader.tokens))
        tokens_to_evaluate = sorted(tokens_to_evaluate) 
    else:
        tokens_to_evaluate = []

    tokens_to_evaluate = broadcast_object(tokens_to_evaluate, device=device, src=0)

    sampler = InferenceSampler(len(tokens_to_evaluate))

    data_points = []
    for idx in sampler:
        token = tokens_to_evaluate[idx]
        log_file = scene_loader.token_to_log_file[token]  
        data_points.append({
            "cfg": cfg,
            "log_file": log_file,
            "tokens": [token], 
        })


    _ = cache_features(data_points)
    logger.info(f"Finished caching {len(scene_loader)} scenarios for training/validation dataset")


if __name__ == "__main__":
    main()
