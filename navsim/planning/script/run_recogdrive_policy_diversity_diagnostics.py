from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import pickle
import time
import traceback
import uuid
import warnings

import hydra
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.recogdrive.recogdrive_features import warn_if_dummy_expert_cache
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.planning.script.run_pdm_score_recogdrive_async_pdm_exact_pool import (
    CONFIG_NAME,
    CONFIG_PATH,
    InferenceSampler,
    _apply_eval_token_file,
    _apply_eval_token_shard,
    _build_rank_scene_loader,
    _cfg_bool,
    _cfg_int,
    _cfg_str,
    _distributed_timeout,
    _init_process_pdm_tools,
    _metric_cache_loader_from_cfg,
    _restrict_scene_filter_to_eval_token_file,
    _score_one_pdm_scalar_exact,
    _score_params_from_cfg,
    broadcast_object,
)
from navsim.planning.script.run_pdm_score_recogdrive_best_of_n_exact_pool import (
    _compute_candidate_trajectories,
)
from nuplan.planning.script.builders.logging_builder import build_logger

logger = logging.getLogger(__name__)


def _candidate_seed(base_seed: int, token: str, candidate_index: int) -> int:
    payload = f"{int(base_seed)}:{token}:{int(candidate_index)}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16) % (2**31 - 1)


def _finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _score_one_candidate_pdm_scalar_exact(
    *,
    candidate_index: int,
    candidate_seed: int,
    token: str,
    metric_cache_path: str,
    poses: Any,
    rank: int,
    index: int,
    score_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = _score_one_pdm_scalar_exact(
        token=token,
        metric_cache_path=metric_cache_path,
        poses=poses,
        rank=rank,
        index=index,
        score_params=score_params,
        profile=False,
        local_timing=None,
    )
    row["candidate_index"] = int(candidate_index)
    row["candidate_seed"] = int(candidate_seed)
    return row


def _trajectory_diagnostics(candidate_poses: np.ndarray, gt_poses: np.ndarray) -> Dict[str, float]:
    """Paper-style @k diagnostics using xy displacement only."""
    if candidate_poses.ndim != 3 or candidate_poses.shape[-1] < 2:
        raise ValueError(f"Expected candidate poses [k, t, >=2], got {candidate_poses.shape}.")
    if gt_poses.ndim != 2 or gt_poses.shape[-1] < 2:
        raise ValueError(f"Expected GT poses [t, >=2], got {gt_poses.shape}.")

    horizon = min(candidate_poses.shape[1], gt_poses.shape[0])
    candidates_xy = candidate_poses[:, :horizon, :2].astype(np.float64)
    gt_xy = gt_poses[:horizon, :2].astype(np.float64)

    gt_distances = np.linalg.norm(candidates_xy - gt_xy[None, :, :], axis=-1)
    ade_per_candidate = gt_distances.mean(axis=1)
    fde_per_candidate = gt_distances[:, -1]

    num_candidates = candidates_xy.shape[0]
    if num_candidates < 2:
        mean_pairwise_ade = 0.0
        mean_pairwise_fde = 0.0
    else:
        pairwise_ade: List[float] = []
        pairwise_fde: List[float] = []
        for i in range(num_candidates):
            for j in range(i + 1, num_candidates):
                distances = np.linalg.norm(candidates_xy[i] - candidates_xy[j], axis=-1)
                pairwise_ade.append(float(distances.mean()))
                pairwise_fde.append(float(distances[-1]))
        mean_pairwise_ade = float(np.mean(pairwise_ade))
        mean_pairwise_fde = float(np.mean(pairwise_fde))

    return {
        "quality_min_ade": float(np.min(ade_per_candidate)),
        "quality_min_fde": float(np.min(fde_per_candidate)),
        "diversity_mean_pade": mean_pairwise_ade,
        "diversity_mean_pfde": mean_pairwise_fde,
        "mean_ade": float(np.mean(ade_per_candidate)),
        "mean_fde": float(np.mean(fde_per_candidate)),
        "candidate_ade_json": json.dumps([float(value) for value in ade_per_candidate]),
        "candidate_fde_json": json.dumps([float(value) for value in fde_per_candidate]),
    }


def _candidate_score_stats(candidate_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores: List[Optional[float]] = []
    for row in sorted(candidate_rows, key=lambda item: int(item.get("candidate_index", 0))):
        score = _finite_float(row.get("score"))
        valid = bool(row.get("valid", False)) and score is not None
        scores.append(score if valid else None)

    valid_scores = [score for score in scores if score is not None]
    if not valid_scores:
        return {
            "candidate_valid_count": 0,
            "candidate_scores_json": json.dumps(scores),
            "perf_mean_pdms": None,
            "perf_best_pdms": None,
            "perf_min_pdms": None,
            "perf_std_pdms": None,
        }

    mean_score = float(sum(valid_scores) / len(valid_scores))
    best_score = float(max(valid_scores))
    return {
        "candidate_valid_count": int(len(valid_scores)),
        "candidate_scores_json": json.dumps(scores),
        "perf_mean_pdms": mean_score,
        "perf_best_pdms": best_score,
        "perf_min_pdms": float(min(valid_scores)),
        "perf_std_pdms": float(math.sqrt(sum((score - mean_score) ** 2 for score in valid_scores) / len(valid_scores))),
    }


def run_policy_diversity_diagnostics(args: List[Dict[str, Any]]) -> bytes:
    node_id = int(os.environ.get("NODE_RANK", 0))
    rank = dist.get_rank()
    thread_id = str(uuid.uuid4())
    logger.info("Starting policy diversity diagnostics thread_id=%s, node_id=%s, rank=%s", thread_id, node_id, rank)

    log_names = [item["log_file"] for item in args]
    tokens = [token for item in args for token in item["tokens"]]
    cfg: DictConfig = args[0]["cfg"]
    score_params = _score_params_from_cfg(cfg)

    k = max(1, _cfg_int(cfg, "diversity_k", "RECOGDRIVE_DIVERSITY_K", 8))
    seed_base = _cfg_int(cfg, "diversity_seed", "RECOGDRIVE_DIVERSITY_SEED", 260306049)
    deterministic = _cfg_bool(cfg, "diversity_deterministic", "RECOGDRIVE_DIVERSITY_DETERMINISTIC", False)

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    metric_cache_loader = _metric_cache_loader_from_cfg(cfg)
    scene_loader = _build_rank_scene_loader(cfg, log_names, tokens, agent)

    tokens_to_evaluate = sorted(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    pdm_workers = max(0, _cfg_int(cfg, "async_pdm_workers", "RECOGDRIVE_ASYNC_PDM_WORKERS", 4))
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
        k,
        deterministic,
        backend,
        pdm_workers,
    )

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
        executor = ThreadPoolExecutor(max_workers=pdm_workers, thread_name_prefix=f"div-pdm-r{rank}")

    rows: List[Dict[str, Any]] = []
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

            candidate_seeds = [_candidate_seed(seed_base, token, candidate_idx) for candidate_idx in range(k)]
            row: Dict[str, Any] = {
                "token": token,
                "valid": False,
                "rank": rank,
                "_index": idx,
                "diversity_k": int(k),
            }
            try:
                agent_input = scene_loader.get_agent_input_from_token(token)
                scene = scene_loader.get_scene_from_token(token)
                trajectories = _compute_candidate_trajectories(
                    agent,
                    agent_input,
                    best_of_n=k,
                    seeds=candidate_seeds,
                    deterministic=deterministic,
                )
                candidate_poses = np.stack([trajectory.poses for trajectory in trajectories], axis=0)
                gt_poses = scene.get_future_trajectory(num_trajectory_frames=candidate_poses.shape[1]).poses
                row.update(_trajectory_diagnostics(candidate_poses, gt_poses))

                metric_cache_path = str(metric_cache_loader.metric_cache_paths[token])
                candidate_rows: List[Dict[str, Any]] = []
                futures = []
                for candidate_idx, trajectory in enumerate(trajectories):
                    score_index = idx * k + candidate_idx
                    if executor is None:
                        candidate_rows.append(
                            _score_one_candidate_pdm_scalar_exact(
                                candidate_index=candidate_idx,
                                candidate_seed=candidate_seeds[candidate_idx],
                                token=token,
                                metric_cache_path=metric_cache_path,
                                poses=trajectory.poses,
                                rank=rank,
                                index=score_index,
                                score_params=score_params,
                            )
                        )
                    elif backend == "process":
                        futures.append(
                            executor.submit(
                                _score_one_candidate_pdm_scalar_exact,
                                candidate_index=candidate_idx,
                                candidate_seed=candidate_seeds[candidate_idx],
                                token=token,
                                metric_cache_path=metric_cache_path,
                                poses=trajectory.poses,
                                rank=rank,
                                index=score_index,
                                score_params=None,
                            )
                        )
                    else:
                        futures.append(
                            executor.submit(
                                _score_one_candidate_pdm_scalar_exact,
                                candidate_index=candidate_idx,
                                candidate_seed=candidate_seeds[candidate_idx],
                                token=token,
                                metric_cache_path=metric_cache_path,
                                poses=trajectory.poses,
                                rank=rank,
                                index=score_index,
                                score_params=score_params,
                            )
                        )

                if futures:
                    wait(futures)
                    for future in futures:
                        candidate_rows.append(future.result())
                row.update(_candidate_score_stats(candidate_rows))
                row["valid"] = bool(row.get("candidate_valid_count", 0) == k)
            except Exception as exc:
                logger.warning("----------- Policy diversity diagnostics failed for token %s:", token)
                traceback.print_exc()
                row["error"] = type(exc).__name__
            rows.append(row)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    rows.sort(key=lambda item: int(item.get("_index", 0)))
    for row in rows:
        row.pop("_index", None)
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
                "ReCogDrive diversity diagnostics is using expert_feature_source='dummy'. "
                "Dummy features are for computation-flow validation only.",
                RuntimeWarning,
            )
        warn_if_dummy_expert_cache(
            cfg.agent.get("expert_cache_dir", None),
            use_expert_features=cfg.agent.get("use_expert_features", False),
            context="ReCogDrive policy diversity diagnostics",
        )

    metric_cache_loader_for_filter = _metric_cache_loader_from_cfg(cfg)
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    _restrict_scene_filter_to_eval_token_file(scene_filter, cfg, metric_cache_loader_for_filter)
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    if rank == 0:
        metric_cache_loader = metric_cache_loader_for_filter
        tokens_to_evaluate = sorted(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
        num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
        num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
        if num_missing_metric_cache_tokens > 0:
            logger.warning("Missing metric cache for %s tokens. Skipping these tokens.", num_missing_metric_cache_tokens)
        if num_unused_metric_cache_tokens > 0:
            logger.warning("Unused metric cache for %s tokens. Skipping these tokens.", num_unused_metric_cache_tokens)
        tokens_to_evaluate = _apply_eval_token_file(tokens_to_evaluate, cfg, rank=rank)
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

    serialized_rows = run_policy_diversity_diagnostics(data_points)
    local_rows = pickle.loads(serialized_rows)
    gathered_rows: List[Optional[List[Dict[str, Any]]]] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_rows, local_rows)

    if dist.get_rank() == 0:
        final_results: List[Dict[str, Any]] = []
        for rows in gathered_rows:
            if rows:
                final_results.extend(rows)
        diagnostics_df = pd.DataFrame(final_results)

        valid = (
            diagnostics_df["valid"].astype(bool)
            if "valid" in diagnostics_df.columns
            else pd.Series(True, index=diagnostics_df.index)
        )
        num_successful_scenarios = int(valid.sum())
        num_failed_scenarios = int(len(diagnostics_df) - num_successful_scenarios)
        excluded_average_cols = [
            col
            for col in ("token", "valid", "rank", "candidate_scores_json", "candidate_ade_json", "candidate_fde_json", "error")
            if col in diagnostics_df.columns
        ]
        numeric_for_average = diagnostics_df.drop(columns=excluded_average_cols).apply(pd.to_numeric, errors="coerce")
        average_row = numeric_for_average.mean(skipna=True)
        average_row["token"] = "average"
        average_row["valid"] = bool(valid.all())
        average_row["rank"] = "0"
        for col in ("candidate_scores_json", "candidate_ade_json", "candidate_fde_json", "error"):
            if col in diagnostics_df.columns:
                average_row[col] = ""
        diagnostics_df.loc[len(diagnostics_df)] = average_row

        save_path = Path(cfg.output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        output_csv = save_path / f"{timestamp}.csv"
        diagnostics_df.to_csv(output_csv)

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
            pd.to_numeric(diagnostics_df[diagnostics_df["token"].astype(str) != "average"]["diversity_mean_pade"], errors="coerce").mean(),
            pd.to_numeric(diagnostics_df[diagnostics_df["token"].astype(str) != "average"]["diversity_mean_pfde"], errors="coerce").mean(),
            pd.to_numeric(diagnostics_df[diagnostics_df["token"].astype(str) != "average"]["quality_min_ade"], errors="coerce").mean(),
            pd.to_numeric(diagnostics_df[diagnostics_df["token"].astype(str) != "average"]["quality_min_fde"], errors="coerce").mean(),
            pd.to_numeric(diagnostics_df[diagnostics_df["token"].astype(str) != "average"]["perf_mean_pdms"], errors="coerce").mean(),
            output_csv,
        )


if __name__ == "__main__":
    main()
