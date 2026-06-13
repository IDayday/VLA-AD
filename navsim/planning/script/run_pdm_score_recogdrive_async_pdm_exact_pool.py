from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import multiprocessing as mp
import os
import pickle
import threading
import traceback
import uuid
import warnings

import hydra
import pandas as pd
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.recogdrive.recogdrive_features import warn_if_dummy_expert_cache
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.metric_caching.fast_metric_cache_loader import FastMetricCacheLoader, load_metric_cache_auto
from navsim.planning.metric_caching.metric_cache import MetricCache
from nuplan.planning.script.builders.logging_builder import build_logger
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"

_PDM_THREAD_LOCAL = threading.local()
_PROCESS_SCORE_PARAMS: Optional[Dict[str, Any]] = None
_PROCESS_TOOLS: Optional[Tuple[TrajectorySampling, PDMSimulator, PDMScorer]] = None


class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size: int):
        self._size = int(size)
        assert size > 0
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size: int, world_size: int, rank: int) -> range:
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]
        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[: rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self) -> int:
        return len(self._local_indices)


def _cfg_int(cfg: DictConfig, key: str, env_key: str, default: int) -> int:
    value = cfg.get(key, None)
    if value is None:
        value = os.environ.get(env_key, default)
    return int(value)


def _cfg_str(cfg: DictConfig, key: str, env_key: str, default: str) -> str:
    value = cfg.get(key, None)
    if value is None:
        value = os.environ.get(env_key, default)
    return str(value)


def _metric_cache_loader_from_cfg(cfg: DictConfig):
    fast_metric_cache_path = str(cfg.get("fast_metric_cache_path", "") or os.environ.get("RECOGDRIVE_FAST_METRIC_CACHE_PATH", ""))
    if fast_metric_cache_path:
        return FastMetricCacheLoader(Path(fast_metric_cache_path))
    return MetricCacheLoader(Path(cfg.metric_cache_path))


def _score_params_from_cfg(cfg: DictConfig) -> Dict[str, Any]:
    scorer_config = cfg.scorer.get("config", {})
    return {
        "num_poses": int(cfg.proposal_sampling.num_poses),
        "interval_length": float(cfg.proposal_sampling.interval_length),
        "scorer_config": {
            "progress_weight": float(scorer_config.get("progress_weight", 5.0)),
            "ttc_weight": float(scorer_config.get("ttc_weight", 5.0)),
            "comfortable_weight": float(scorer_config.get("comfortable_weight", 2.0)),
            "driving_direction_weight": float(scorer_config.get("driving_direction_weight", 0.0)),
            "driving_direction_horizon": float(scorer_config.get("driving_direction_horizon", 1.0)),
            "driving_direction_compliance_threshold": float(
                scorer_config.get("driving_direction_compliance_threshold", 2.0)
            ),
            "driving_direction_violation_threshold": float(
                scorer_config.get("driving_direction_violation_threshold", 6.0)
            ),
            "stopped_speed_threshold": float(scorer_config.get("stopped_speed_threshold", 5e-3)),
            "progress_distance_threshold": float(scorer_config.get("progress_distance_threshold", 5.0)),
        },
    }


def _make_pdm_tools(score_params: Dict[str, Any]) -> Tuple[TrajectorySampling, PDMSimulator, PDMScorer]:
    proposal_sampling = TrajectorySampling(
        num_poses=int(score_params["num_poses"]),
        interval_length=float(score_params["interval_length"]),
    )
    scorer_config = PDMScorerConfig(**score_params["scorer_config"])
    return proposal_sampling, PDMSimulator(proposal_sampling), PDMScorer(proposal_sampling, scorer_config)


def _get_thread_pdm_tools(score_params: Dict[str, Any]) -> Tuple[TrajectorySampling, PDMSimulator, PDMScorer]:
    tools = getattr(_PDM_THREAD_LOCAL, "tools", None)
    if tools is None:
        tools = _make_pdm_tools(score_params)
        _PDM_THREAD_LOCAL.tools = tools
    return tools


def _init_process_pdm_tools(score_params: Dict[str, Any]) -> None:
    global _PROCESS_SCORE_PARAMS, _PROCESS_TOOLS
    _PROCESS_SCORE_PARAMS = score_params
    _PROCESS_TOOLS = _make_pdm_tools(score_params)


def _get_process_pdm_tools() -> Tuple[TrajectorySampling, PDMSimulator, PDMScorer]:
    global _PROCESS_SCORE_PARAMS, _PROCESS_TOOLS
    if _PROCESS_SCORE_PARAMS is None:
        raise RuntimeError("PDM process worker was not initialized with score parameters.")
    if _PROCESS_TOOLS is None:
        _PROCESS_TOOLS = _make_pdm_tools(_PROCESS_SCORE_PARAMS)
    return _PROCESS_TOOLS


def _score_one_pdm_scalar_exact(
    *,
    token: str,
    metric_cache_path: str,
    poses: Any,
    rank: int,
    index: int,
    score_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score one trajectory with the original scalar pdm_score implementation."""
    score_row: Dict[str, Any] = {"token": token, "valid": True, "rank": rank, "_index": index}
    try:
        if score_params is None:
            future_sampling, simulator, scorer = _get_process_pdm_tools()
        else:
            future_sampling, simulator, scorer = _get_thread_pdm_tools(score_params)
        metric_cache: MetricCache = load_metric_cache_auto(Path(metric_cache_path))
        pdm_result = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=Trajectory(poses),
            future_sampling=future_sampling,
            simulator=simulator,
            scorer=scorer,
        )
        score_row.update(asdict(pdm_result))
    except Exception:
        logger.warning("----------- exact PDM scoring failed for token %s:", token)
        traceback.print_exc()
        score_row["valid"] = False
    return score_row


def _drain_pending(
    pending: List[Future],
    rows: List[Dict[str, Any]],
    *,
    wait_for_one: bool = False,
    wait_all: bool = False,
) -> List[Future]:
    if not pending:
        return pending
    if wait_all:
        done = list(pending)
        remaining: List[Future] = []
    else:
        timeout = None if wait_for_one else 0
        done_set, remaining_set = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
        done = list(done_set)
        remaining = list(remaining_set)
    for future in done:
        rows.append(future.result())
    return remaining


def _build_rank_scene_loader(cfg: DictConfig, log_names: List[str], tokens: List[str], agent: AbstractAgent) -> SceneLoader:
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    return SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True,
    )


def run_pdm_score_async_exact_pool(args: List[Dict[str, Any]]) -> bytes:
    """
    Run ReCogDrive inference on this rank and score PDMS asynchronously.

    This script is intentionally exact: every completed trajectory is scored through the original
    scalar ``navsim.evaluate.pdm_score.pdm_score`` path. The only changed behavior versus
    ``run_pdm_score_recogdrive_async_pdm.py`` is the optional process-pool executor around that
    scalar call.
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    rank = dist.get_rank()
    thread_id = str(uuid.uuid4())
    logger.info("Starting exact async PDM worker thread_id=%s, node_id=%s, rank=%s", thread_id, node_id, rank)

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]
    score_params = _score_params_from_cfg(cfg)

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    metric_cache_loader = _metric_cache_loader_from_cfg(cfg)
    scene_loader = _build_rank_scene_loader(cfg, log_names, tokens, agent)

    tokens_to_evaluate = sorted(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    pdm_workers = max(0, _cfg_int(cfg, "async_pdm_workers", "RECOGDRIVE_ASYNC_PDM_WORKERS", 4))
    queue_size = max(
        1,
        _cfg_int(cfg, "async_pdm_queue_size", "RECOGDRIVE_ASYNC_PDM_QUEUE_SIZE", max(2 * pdm_workers, 1)),
    )
    progress_every = max(0, _cfg_int(cfg, "async_pdm_progress_every", "RECOGDRIVE_ASYNC_PDM_PROGRESS_EVERY", 100))
    backend = _cfg_str(cfg, "async_pdm_backend", "RECOGDRIVE_ASYNC_PDM_BACKEND", "thread").lower()
    process_start_method = _cfg_str(
        cfg,
        "async_pdm_process_start_method",
        "RECOGDRIVE_ASYNC_PDM_PROCESS_START_METHOD",
        "spawn",
    )
    if backend not in {"thread", "process"}:
        raise ValueError(f"async_pdm_backend must be 'thread' or 'process', got {backend!r}.")

    logger.info(
        "Rank %s evaluating %s scenarios with async_pdm_backend=%s, workers=%s, queue_size=%s",
        rank,
        len(tokens_to_evaluate),
        backend,
        pdm_workers,
        queue_size,
    )

    rows: List[Dict[str, Any]] = []
    pending: List[Future] = []
    executor: Optional[ThreadPoolExecutor | ProcessPoolExecutor]
    if pdm_workers <= 0:
        executor = None
    elif backend == "process":
        executor = ProcessPoolExecutor(
            max_workers=pdm_workers,
            mp_context=mp.get_context(process_start_method),
            initializer=_init_process_pdm_tools,
            initargs=(score_params,),
        )
    else:
        executor = ThreadPoolExecutor(max_workers=pdm_workers, thread_name_prefix=f"pdm-r{rank}")

    try:
        for idx, token in enumerate(tokens_to_evaluate):
            if rank == 0 and (progress_every > 0) and ((idx + 1) % progress_every == 0):
                logger.info(
                    "Rank %s submitted/processed scenario %s / %s in thread_id=%s, node_id=%s",
                    rank,
                    idx + 1,
                    len(tokens_to_evaluate),
                    thread_id,
                    node_id,
                )

            if token not in metric_cache_loader.metric_cache_paths:
                rows.append({"token": token, "valid": False, "rank": rank, "_index": idx})
                continue

            try:
                agent_input = scene_loader.get_agent_input_from_token(token)
                trajectory = agent.compute_trajectory(agent_input)
                poses = trajectory.poses
                metric_cache_path = str(metric_cache_loader.metric_cache_paths[token])
                if executor is None:
                    rows.append(
                        _score_one_pdm_scalar_exact(
                            token=token,
                            metric_cache_path=metric_cache_path,
                            poses=poses,
                            rank=rank,
                            index=idx,
                            score_params=score_params,
                        )
                    )
                elif backend == "process":
                    pending.append(
                        executor.submit(
                            _score_one_pdm_scalar_exact,
                            token=token,
                            metric_cache_path=metric_cache_path,
                            poses=poses,
                            rank=rank,
                            index=idx,
                            score_params=None,
                        )
                    )
                else:
                    pending.append(
                        executor.submit(
                            _score_one_pdm_scalar_exact,
                            token=token,
                            metric_cache_path=metric_cache_path,
                            poses=poses,
                            rank=rank,
                            index=idx,
                            score_params=score_params,
                        )
                    )
                if len(pending) >= queue_size:
                    pending = _drain_pending(pending, rows, wait_for_one=True)
            except Exception:
                logger.warning("----------- Agent failed for token %s:", token)
                traceback.print_exc()
                rows.append({"token": token, "valid": False, "rank": rank, "_index": idx})

            pending = _drain_pending(pending, rows)

        pending = _drain_pending(pending, rows, wait_all=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    rows.sort(key=lambda row: int(row.get("_index", 0)))
    for row in rows:
        row.pop("_index", None)
    return pickle.dumps(rows)


def broadcast_object(obj: Any, device: torch.device, src: int = 0) -> Any:
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


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))

    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    build_logger(cfg)

    if rank == 0:
        if cfg.agent.get("expert_feature_source", "none") == "dummy":
            warnings.warn(
                "ReCogDrive evaluation is using expert_feature_source='dummy'. "
                "Dummy features are for computation-flow validation only; no benchmark metrics "
                "should be reported in dummy mode.",
                RuntimeWarning,
            )
        warn_if_dummy_expert_cache(
            cfg.agent.get("expert_cache_dir", None),
            use_expert_features=cfg.agent.get("use_expert_features", False),
            context="ReCogDrive exact-pool evaluation",
        )

    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    if rank == 0:
        metric_cache_loader = _metric_cache_loader_from_cfg(cfg)
        tokens_to_evaluate = sorted(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
        num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
        num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
        if num_missing_metric_cache_tokens > 0:
            logger.warning("Missing metric cache for %s tokens. Skipping these tokens.", num_missing_metric_cache_tokens)
        if num_unused_metric_cache_tokens > 0:
            logger.warning("Unused metric cache for %s tokens. Skipping these tokens.", num_unused_metric_cache_tokens)
    else:
        tokens_to_evaluate = []

    tokens_to_evaluate = broadcast_object(tokens_to_evaluate, device=device, src=0)
    logger.info("Starting exact async pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))

    sampler = InferenceSampler(len(tokens_to_evaluate))
    data_points = []
    for idx in sampler:
        token = tokens_to_evaluate[idx]
        log_file = scene_loader.token_to_log_file[token]
        data_points.append({"cfg": cfg, "log_file": log_file, "tokens": [token]})

    serialized_score_rows = run_pdm_score_async_exact_pool(data_points)
    local_rows = pickle.loads(serialized_score_rows)
    gathered_rows: List[Optional[List[Dict[str, Any]]]] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_rows, local_rows)

    if dist.get_rank() == 0:
        final_results: List[Dict[str, Any]] = []
        for rows in gathered_rows:
            if rows:
                final_results.extend(rows)
        pdm_score_df = pd.DataFrame(final_results)

        num_successful_scenarios = pdm_score_df["valid"].sum()
        num_failed_scenarios = len(pdm_score_df) - num_successful_scenarios
        average_row = pdm_score_df.drop(columns=["token", "valid", "rank"]).mean(skipna=True)
        average_row["token"] = "average"
        average_row["valid"] = pdm_score_df["valid"].all()
        average_row["rank"] = "0"
        pdm_score_df.loc[len(pdm_score_df)] = average_row

        save_path = Path(cfg.output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        output_csv = save_path / f"{timestamp}.csv"
        pdm_score_df.to_csv(output_csv)

        logger.info(
            """
            Finished running exact async PDM evaluation.
                Number of successful scenarios: %s.
                Number of failed scenarios: %s.
                Final average score of valid results: %s.
                Results are stored in: %s.
            """,
            num_successful_scenarios,
            num_failed_scenarios,
            pdm_score_df["score"].mean(),
            output_csv,
        )


if __name__ == "__main__":
    main()
