from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging
import multiprocessing as mp
import os
import pickle
import traceback
import uuid
import warnings
import time

import hydra
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.recogdrive.recogdrive_features import warn_if_dummy_expert_cache
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, pdm_score
from navsim.planning.metric_caching.fast_metric_cache_loader import FastMetricCacheLoader, load_metric_cache_auto
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.script.run_pdm_score_recogdrive_async_pdm_exact_pool import (
    CONFIG_NAME,
    CONFIG_PATH,
    InferenceSampler,
    _apply_eval_token_shard,
    _cfg_bool,
    _cfg_int,
    _cfg_str,
    _distributed_timeout,
    _drain_pending,
    _init_process_pdm_tools,
    _make_pdm_tools,
    _metric_cache_loader_from_cfg,
    _score_params_from_cfg,
    broadcast_object,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

logger = logging.getLogger(__name__)

_PDM_THREAD_LOCAL: Any = None


def _metric_cache_loader_from_cfg_local(cfg: DictConfig):
    fast_metric_cache_path = str(cfg.get("fast_metric_cache_path", "") or os.environ.get("RECOGDRIVE_FAST_METRIC_CACHE_PATH", ""))
    if fast_metric_cache_path:
        return FastMetricCacheLoader(Path(fast_metric_cache_path))
    return MetricCacheLoader(Path(cfg.metric_cache_path))


def _score_one_candidate_exact(
    *,
    token: str,
    metric_cache_path: str,
    poses: Any,
    rank: int,
    index: int,
    candidate_index: int,
    score_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "token": token,
        "valid": True,
        "rank": rank,
        "_index": index,
        "candidate_index": candidate_index,
    }
    try:
        if score_params is None:
            from navsim.planning.script.run_pdm_score_recogdrive_async_pdm_exact_pool import _get_process_pdm_tools

            future_sampling, simulator, scorer = _get_process_pdm_tools()
        else:
            # Keep thread-mode tools process-local without importing the other module's private thread-local state.
            global _PDM_THREAD_LOCAL
            if _PDM_THREAD_LOCAL is None:
                _PDM_THREAD_LOCAL = _make_pdm_tools(score_params)
            future_sampling, simulator, scorer = _PDM_THREAD_LOCAL
        metric_cache: MetricCache = load_metric_cache_auto(Path(metric_cache_path))
        pdm_result = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=Trajectory(np.asarray(poses, dtype=np.float32)),
            future_sampling=future_sampling,
            simulator=simulator,
            scorer=scorer,
        )
        row.update(asdict(pdm_result))
    except Exception:
        logger.warning("----------- exact PDM scoring failed for token %s candidate %s:", token, candidate_index)
        traceback.print_exc()
        row["valid"] = False
    return row


def _pairwise_means(poses: List[np.ndarray]) -> tuple[float, float]:
    if len(poses) < 2:
        return 0.0, 0.0
    ade_values: List[float] = []
    fde_values: List[float] = []
    for i in range(len(poses)):
        xy_i = poses[i][:, :2]
        for j in range(i + 1, len(poses)):
            xy_j = poses[j][:, :2]
            dist = np.linalg.norm(xy_i - xy_j, axis=-1)
            ade_values.append(float(np.mean(dist)))
            fde_values.append(float(dist[-1]))
    return float(np.mean(ade_values)), float(np.mean(fde_values))


def _candidate_errors(poses: List[np.ndarray], gt_poses: np.ndarray) -> tuple[List[float], List[float]]:
    ade_values: List[float] = []
    fde_values: List[float] = []
    gt_xy = gt_poses[:, :2]
    for candidate in poses:
        dist = np.linalg.norm(candidate[:, :2] - gt_xy, axis=-1)
        ade_values.append(float(np.mean(dist)))
        fde_values.append(float(dist[-1]))
    return ade_values, fde_values


def _gt_relative_poses(metric_cache: MetricCache, score_params: Dict[str, Any]) -> np.ndarray:
    future_sampling = TrajectorySampling(
        num_poses=int(score_params["num_poses"]),
        interval_length=float(score_params["interval_length"]),
    )
    states = get_trajectory_as_array(metric_cache.trajectory, future_sampling, metric_cache.ego_state.time_point)
    abs_future_poses = states[1:, StateIndex.STATE_SE2]
    return convert_absolute_to_relative_se2_array(metric_cache.ego_state.rear_axle, abs_future_poses).astype(np.float32)


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


def _compute_features_once(agent: AbstractAgent, agent_input: Any) -> Dict[str, torch.Tensor]:
    features: Dict[str, torch.Tensor] = {}
    for builder in agent.get_feature_builders():
        features.update(builder.compute_features(agent_input))
    return {key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value for key, value in features.items()}


def _sample_candidate_poses(
    agent: AbstractAgent,
    features: Dict[str, torch.Tensor],
    *,
    k: int,
    deterministic: bool,
    base_seed: int,
    token_index: int,
) -> List[np.ndarray]:
    poses: List[np.ndarray] = []
    for candidate_index in range(k):
        if deterministic:
            torch.manual_seed(int(base_seed) + token_index * 1009)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(base_seed) + token_index * 1009)
        else:
            seed = int(base_seed) + token_index * 1009 + candidate_index * 9176
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            predictions = agent.forward(features)
            pose = predictions["pred_traj"].detach().float().cpu().squeeze(0).numpy().astype(np.float32)
        poses.append(pose)
    return poses


def run_policy_diversity(args: List[Dict[str, Any]]) -> bytes:
    node_id = int(os.environ.get("NODE_RANK", 0))
    rank = dist.get_rank()
    thread_id = str(uuid.uuid4())
    logger.info("Starting policy diversity diagnostics thread_id=%s, node_id=%s, rank=%s", thread_id, node_id, rank)

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]
    score_params = _score_params_from_cfg(cfg)

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    metric_cache_loader = _metric_cache_loader_from_cfg_local(cfg)
    scene_loader = _build_rank_scene_loader(cfg, log_names, tokens, agent)

    tokens_to_evaluate = sorted(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    diversity_k = max(1, _cfg_int(cfg, "diversity_k", "RECOGDRIVE_DIVERSITY_K", 8))
    diversity_seed = _cfg_int(cfg, "diversity_seed", "RECOGDRIVE_DIVERSITY_SEED", 260306049)
    deterministic = _cfg_bool(cfg, "diversity_deterministic", "RECOGDRIVE_DIVERSITY_DETERMINISTIC", False)
    pdm_workers = max(0, _cfg_int(cfg, "async_pdm_workers", "RECOGDRIVE_ASYNC_PDM_WORKERS", 4))
    queue_size = max(1, _cfg_int(cfg, "async_pdm_queue_size", "RECOGDRIVE_ASYNC_PDM_QUEUE_SIZE", max(2 * pdm_workers, 1)))
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
        "Rank %s evaluating %s scenarios with diversity_k=%s, deterministic=%s, async_backend=%s, workers=%s",
        rank,
        len(tokens_to_evaluate),
        diversity_k,
        deterministic,
        backend,
        pdm_workers,
    )

    rows: List[Dict[str, Any]] = []
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
                    "Rank %s processed diversity scenario %s / %s in thread_id=%s, node_id=%s",
                    rank,
                    idx + 1,
                    len(tokens_to_evaluate),
                    thread_id,
                    node_id,
                )
            if token not in metric_cache_loader.metric_cache_paths:
                rows.append({"token": token, "valid": False, "rank": rank, "diversity_k": float(diversity_k)})
                continue
            try:
                agent_input = scene_loader.get_agent_input_from_token(token)
                features = _compute_features_once(agent, agent_input)
                candidate_poses = _sample_candidate_poses(
                    agent,
                    features,
                    k=diversity_k,
                    deterministic=deterministic,
                    base_seed=diversity_seed,
                    token_index=idx,
                )
                metric_cache_path = str(metric_cache_loader.metric_cache_paths[token])
                scene_pending: List[Future] = []
                scene_candidate_rows: List[Dict[str, Any]] = []
                for candidate_index, poses in enumerate(candidate_poses):
                    kwargs = {
                        "token": token,
                        "metric_cache_path": metric_cache_path,
                        "poses": poses,
                        "rank": rank,
                        "index": idx,
                        "candidate_index": candidate_index,
                    }
                    if executor is None:
                        scene_candidate_rows.append(_score_one_candidate_exact(**kwargs, score_params=score_params))
                    elif backend == "process":
                        scene_pending.append(executor.submit(_score_one_candidate_exact, **kwargs, score_params=None))
                    else:
                        scene_pending.append(executor.submit(_score_one_candidate_exact, **kwargs, score_params=score_params))
                    if len(scene_pending) >= queue_size:
                        scene_pending = _drain_pending(scene_pending, scene_candidate_rows, wait_for_one=True)
                scene_pending = _drain_pending(scene_pending, scene_candidate_rows, wait_all=True)

                scene_candidate_rows.sort(key=lambda row: int(row.get("candidate_index", 0)))
                valid_candidate_rows = [
                    row for row in scene_candidate_rows if str(row.get("valid", True)).lower() in {"1", "true", "yes"}
                ]
                if not valid_candidate_rows:
                    rows.append({"token": token, "valid": False, "rank": rank, "diversity_k": float(diversity_k)})
                    continue

                score_by_idx = {
                    int(row["candidate_index"]): float(row["score"]) for row in valid_candidate_rows if "score" in row
                }
                valid_indices = [idx_ for idx_ in range(diversity_k) if idx_ in score_by_idx]
                valid_poses = [candidate_poses[idx_] for idx_ in valid_indices]

                metric_cache = load_metric_cache_auto(Path(metric_cache_path))
                gt_poses = _gt_relative_poses(metric_cache, score_params)
                candidate_ade, candidate_fde = _candidate_errors(valid_poses, gt_poses)
                mean_pade, mean_pfde = _pairwise_means(valid_poses)
                scores = [score_by_idx[idx_] for idx_ in valid_indices]
                rows.append(
                    {
                        "token": token,
                        "valid": len(valid_candidate_rows) == diversity_k,
                        "rank": rank,
                        "diversity_k": float(diversity_k),
                        "quality_min_ade": float(np.min(candidate_ade)),
                        "quality_min_fde": float(np.min(candidate_fde)),
                        "diversity_mean_pade": mean_pade,
                        "diversity_mean_pfde": mean_pfde,
                        "mean_ade": float(np.mean(candidate_ade)),
                        "mean_fde": float(np.mean(candidate_fde)),
                        "candidate_ade_json": json.dumps(candidate_ade),
                        "candidate_fde_json": json.dumps(candidate_fde),
                        "candidate_valid_count": float(len(valid_candidate_rows)),
                        "candidate_scores_json": json.dumps(scores),
                        "perf_mean_pdms": float(np.mean(scores)),
                        "perf_best_pdms": float(np.max(scores)),
                        "perf_min_pdms": float(np.min(scores)),
                        "perf_std_pdms": float(np.std(scores)),
                    }
                )
            except Exception:
                logger.warning("----------- Policy diversity failed for token %s:", token)
                traceback.print_exc()
                rows.append({"token": token, "valid": False, "rank": rank, "diversity_k": float(diversity_k)})

    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    return pickle.dumps(rows)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))

    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank, timeout=_distributed_timeout())
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    build_logger(cfg)

    if rank == 0:
        if cfg.agent.get("expert_feature_source", "none") == "dummy":
            warnings.warn(
                "ReCogDrive policy diversity evaluation is using expert_feature_source='dummy'. "
                "Dummy features are for computation-flow validation only.",
                RuntimeWarning,
            )
        warn_if_dummy_expert_cache(
            cfg.agent.get("expert_cache_dir", None),
            use_expert_features=cfg.agent.get("use_expert_features", False),
            context="ReCogDrive policy diversity diagnostics",
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
        tokens_to_evaluate = _apply_eval_token_shard(tokens_to_evaluate, cfg, rank=rank)
    else:
        tokens_to_evaluate = []

    tokens_to_evaluate = broadcast_object(tokens_to_evaluate, device=device, src=0)
    logger.info("Starting policy diversity diagnostics of %s scenarios...", str(len(tokens_to_evaluate)))

    sampler = InferenceSampler(len(tokens_to_evaluate))
    data_points = []
    for idx in sampler:
        token = tokens_to_evaluate[idx]
        log_file = scene_loader.token_to_log_file[token]
        data_points.append({"cfg": cfg, "log_file": log_file, "tokens": [token]})

    serialized_rows = run_policy_diversity(data_points)
    local_rows = pickle.loads(serialized_rows)
    gathered_rows: List[Optional[List[Dict[str, Any]]]] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_rows, local_rows)

    if dist.get_rank() == 0:
        final_results: List[Dict[str, Any]] = []
        for rows in gathered_rows:
            if rows:
                final_results.extend(rows)
        df = pd.DataFrame(final_results)
        num_successful_scenarios = int(df["valid"].astype(str).str.lower().isin({"1", "true", "yes"}).sum())
        num_failed_scenarios = int(len(df) - num_successful_scenarios)
        numeric_for_average = df.drop(columns=[col for col in ("token", "valid", "rank") if col in df.columns])
        average_row = numeric_for_average.apply(pd.to_numeric, errors="coerce").mean(skipna=True)
        average_row["token"] = "average"
        average_row["valid"] = num_failed_scenarios == 0
        average_row["rank"] = "0"
        df.loc[len(df)] = average_row

        save_path = Path(cfg.output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        output_csv = save_path / f"{timestamp}.csv"
        df.to_csv(output_csv)

        valid_df = df[df["token"].astype(str) != "average"].copy()
        logger.info(
            """
            Finished policy diversity diagnostics.
                Number of successful scenarios: %s.
                Number of failed scenarios: %s.
                mean-pADE / mean-pFDE: %s / %s.
                minADE / minFDE: %s / %s.
                mean-PDMS: %s.
                Results are stored in: %s.
            """,
            num_successful_scenarios,
            num_failed_scenarios,
            pd.to_numeric(valid_df["diversity_mean_pade"], errors="coerce").mean(skipna=True),
            pd.to_numeric(valid_df["diversity_mean_pfde"], errors="coerce").mean(skipna=True),
            pd.to_numeric(valid_df["quality_min_ade"], errors="coerce").mean(skipna=True),
            pd.to_numeric(valid_df["quality_min_fde"], errors="coerce").mean(skipna=True),
            pd.to_numeric(valid_df["perf_mean_pdms"], errors="coerce").mean(skipna=True),
            output_csv,
        )


if __name__ == "__main__":
    main()
