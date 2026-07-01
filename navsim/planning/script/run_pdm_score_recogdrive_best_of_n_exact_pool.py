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
import pandas as pd
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.recogdrive.recogdrive_features import warn_if_dummy_expert_cache
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SensorConfig, Trajectory
from nuplan.planning.script.builders.logging_builder import build_logger

from navsim.planning.script.run_pdm_score_recogdrive_async_pdm_exact_pool import (
    CONFIG_NAME,
    CONFIG_PATH,
    InferenceSampler,
    _apply_eval_token_shard,
    _apply_eval_token_file,
    _restrict_scene_filter_to_eval_token_file,
    _build_rank_scene_loader,
    _cfg_bool,
    _cfg_int,
    _cfg_str,
    _distributed_timeout,
    _init_process_pdm_tools,
    _log_profile_summary,
    _metric_cache_loader_from_cfg,
    _score_one_pdm_scalar_exact,
    _score_params_from_cfg,
    broadcast_object,
)

logger = logging.getLogger(__name__)


def _finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _candidate_seed(base_seed: int, token: str, candidate_index: int) -> int:
    payload = f"{int(base_seed)}:{token}:{int(candidate_index)}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16) % (2**31 - 1)


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
    profile: bool = False,
    local_timing: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    row = _score_one_pdm_scalar_exact(
        token=token,
        metric_cache_path=metric_cache_path,
        poses=poses,
        rank=rank,
        index=index,
        score_params=score_params,
        profile=profile,
        local_timing=local_timing,
    )
    row["candidate_index"] = int(candidate_index)
    row["candidate_seed"] = int(candidate_seed)
    return row


def _invalid_candidate_row(
    *,
    token: str,
    rank: int,
    index: int,
    candidate_index: int,
    candidate_seed: int,
    error: str,
) -> Dict[str, Any]:
    return {
        "token": token,
        "valid": False,
        "rank": rank,
        "_index": index,
        "candidate_index": int(candidate_index),
        "candidate_seed": int(candidate_seed),
        "candidate_error": error,
    }


def _select_best_candidate_row(
    candidate_rows: List[Dict[str, Any]],
    *,
    token: str,
    rank: int,
    index: int,
    best_of_n: int,
) -> Dict[str, Any]:
    scored_rows: List[tuple[float, int, Dict[str, Any]]] = []
    candidate_scores: List[Optional[float]] = []
    for row in sorted(candidate_rows, key=lambda item: int(item.get("candidate_index", 0))):
        score = _finite_float(row.get("score"))
        valid = bool(row.get("valid", False)) and score is not None
        candidate_scores.append(score if valid else None)
        if valid:
            scored_rows.append((float(score), int(row.get("candidate_index", 0)), row))

    valid_scores = [score for score in candidate_scores if score is not None]
    common = {
        "token": token,
        "rank": rank,
        "best_of_n": int(best_of_n),
        "candidate_count": int(len(candidate_rows)),
        "candidate_valid_count": int(len(valid_scores)),
        "candidate_scores_json": json.dumps(candidate_scores),
    }
    if valid_scores:
        mean_score = sum(valid_scores) / len(valid_scores)
        common.update(
            {
                "candidate_score_mean": mean_score,
                "candidate_score_min": min(valid_scores),
                "candidate_score_max": max(valid_scores),
                "candidate_score_std": math.sqrt(
                    sum((score - mean_score) ** 2 for score in valid_scores) / len(valid_scores)
                ),
                "candidate0_score": candidate_scores[0] if candidate_scores else None,
            }
        )
    else:
        common.update(
            {
                "candidate_score_mean": None,
                "candidate_score_min": None,
                "candidate_score_max": None,
                "candidate_score_std": None,
                "candidate0_score": None,
            }
        )

    if not scored_rows:
        return {
            **common,
            "valid": False,
            "_index": index,
            "best_candidate_index": None,
            "best_candidate_seed": None,
            "best_delta_vs_candidate0": None,
        }

    best_score, _, best_row = max(scored_rows, key=lambda item: (item[0], -item[1]))
    selected = dict(best_row)
    selected.pop("_index", None)
    selected.update(common)
    selected["valid"] = True
    selected["token"] = token
    selected["rank"] = rank
    selected["best_candidate_index"] = int(best_row.get("candidate_index", 0))
    selected["best_candidate_seed"] = int(best_row.get("candidate_seed", 0))
    selected["best_delta_vs_candidate0"] = (
        best_score - float(candidate_scores[0]) if candidate_scores and candidate_scores[0] is not None else None
    )
    selected["_index"] = index
    return selected


def _compute_candidate_trajectories(
    agent: AbstractAgent,
    agent_input: Any,
    *,
    best_of_n: int,
    seeds: List[int],
    deterministic: bool,
) -> List[Trajectory]:
    if hasattr(agent, "compute_trajectory_candidates"):
        return agent.compute_trajectory_candidates(
            agent_input,
            num_candidates=best_of_n,
            seeds=seeds,
            deterministic=deterministic,
        )

    trajectories: List[Trajectory] = []
    for seed in seeds:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        trajectories.append(agent.compute_trajectory(agent_input))
    return trajectories


def run_pdm_score_best_of_n_exact_pool(args: List[Dict[str, Any]]) -> bytes:
    node_id = int(os.environ.get("NODE_RANK", 0))
    rank = dist.get_rank()
    thread_id = str(uuid.uuid4())
    logger.info("Starting best-of-N exact PDM worker thread_id=%s, node_id=%s, rank=%s", thread_id, node_id, rank)

    log_names = [item["log_file"] for item in args]
    tokens = [token for item in args for token in item["tokens"]]
    cfg: DictConfig = args[0]["cfg"]
    score_params = _score_params_from_cfg(cfg)

    best_of_n = max(1, _cfg_int(cfg, "best_of_n", "RECOGDRIVE_BEST_OF_N", 6))
    seed_base = _cfg_int(cfg, "best_of_n_seed", "RECOGDRIVE_BEST_OF_N_SEED", 260306049)
    deterministic = _cfg_bool(cfg, "best_of_n_deterministic", "RECOGDRIVE_BEST_OF_N_DETERMINISTIC", False)

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    metric_cache_loader = _metric_cache_loader_from_cfg(cfg)
    scene_loader = _build_rank_scene_loader(cfg, log_names, tokens, agent)

    tokens_to_evaluate = sorted(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    pdm_workers = max(0, _cfg_int(cfg, "async_pdm_workers", "RECOGDRIVE_ASYNC_PDM_WORKERS", 4))
    progress_every = max(0, _cfg_int(cfg, "async_pdm_progress_every", "RECOGDRIVE_ASYNC_PDM_PROGRESS_EVERY", 100))
    profile = _cfg_bool(cfg, "async_pdm_profile", "RECOGDRIVE_ASYNC_PDM_PROFILE", False)
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
        "Rank %s evaluating %s scenarios with best_of_n=%s, deterministic=%s, "
        "async_pdm_backend=%s, workers=%s",
        rank,
        len(tokens_to_evaluate),
        best_of_n,
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
        executor = ThreadPoolExecutor(max_workers=pdm_workers, thread_name_prefix=f"bon-pdm-r{rank}")

    try:
        for idx, token in enumerate(tokens_to_evaluate):
            if rank == 0 and (progress_every > 0) and ((idx + 1) % progress_every == 0):
                logger.info(
                    "Rank %s processed best-of-N scenario %s / %s in thread_id=%s, node_id=%s",
                    rank,
                    idx + 1,
                    len(tokens_to_evaluate),
                    thread_id,
                    node_id,
                )

            candidate_seeds = [_candidate_seed(seed_base, token, candidate_idx) for candidate_idx in range(best_of_n)]
            if token not in metric_cache_loader.metric_cache_paths:
                rows.append(
                    _select_best_candidate_row(
                        [
                            _invalid_candidate_row(
                                token=token,
                                rank=rank,
                                index=idx,
                                candidate_index=candidate_idx,
                                candidate_seed=candidate_seeds[candidate_idx],
                                error="missing_metric_cache",
                            )
                            for candidate_idx in range(best_of_n)
                        ],
                        token=token,
                        rank=rank,
                        index=idx,
                        best_of_n=best_of_n,
                    )
                )
                continue

            local_timing: Dict[str, float] = {}
            candidate_rows: List[Dict[str, Any]] = []
            try:
                scene_start = time.perf_counter()
                agent_input = scene_loader.get_agent_input_from_token(token)
                if profile:
                    local_timing["_timing_scene_load_s"] = time.perf_counter() - scene_start

                agent_start = time.perf_counter()
                trajectories = _compute_candidate_trajectories(
                    agent,
                    agent_input,
                    best_of_n=best_of_n,
                    seeds=candidate_seeds,
                    deterministic=deterministic,
                )
                if profile:
                    elapsed = time.perf_counter() - agent_start
                    local_timing["_timing_agent_compute_s"] = elapsed
                    local_timing["_timing_agent_compute_per_candidate_s"] = elapsed / float(best_of_n)

                metric_cache_path = str(metric_cache_loader.metric_cache_paths[token])
                futures = []
                for candidate_idx, trajectory in enumerate(trajectories):
                    index = idx * best_of_n + candidate_idx
                    poses = trajectory.poses
                    if executor is None:
                        candidate_rows.append(
                            _score_one_candidate_pdm_scalar_exact(
                                candidate_index=candidate_idx,
                                candidate_seed=candidate_seeds[candidate_idx],
                                token=token,
                                metric_cache_path=metric_cache_path,
                                poses=poses,
                                rank=rank,
                                index=index,
                                score_params=score_params,
                                profile=profile,
                                local_timing=local_timing,
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
                                poses=poses,
                                rank=rank,
                                index=index,
                                score_params=None,
                                profile=profile,
                                local_timing=local_timing,
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
                                poses=poses,
                                rank=rank,
                                index=index,
                                score_params=score_params,
                                profile=profile,
                                local_timing=local_timing,
                            )
                        )

                if futures:
                    wait(futures)
                    for future in futures:
                        candidate_rows.append(future.result())
            except Exception as exc:
                logger.warning("----------- Best-of-N agent/scoring failed for token %s:", token)
                traceback.print_exc()
                candidate_rows = [
                    _invalid_candidate_row(
                        token=token,
                        rank=rank,
                        index=idx * best_of_n + candidate_idx,
                        candidate_index=candidate_idx,
                        candidate_seed=candidate_seeds[candidate_idx],
                        error=type(exc).__name__,
                    )
                    for candidate_idx in range(best_of_n)
                ]

            rows.append(
                _select_best_candidate_row(
                    candidate_rows,
                    token=token,
                    rank=rank,
                    index=idx,
                    best_of_n=best_of_n,
                )
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    rows.sort(key=lambda row: int(row.get("_index", 0)))
    if profile:
        _log_profile_summary(rows, rank=rank)
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
                "ReCogDrive best-of-N evaluation is using expert_feature_source='dummy'. "
                "Dummy features are for computation-flow validation only; no benchmark metrics "
                "should be reported in dummy mode.",
                RuntimeWarning,
            )
        warn_if_dummy_expert_cache(
            cfg.agent.get("expert_cache_dir", None),
            use_expert_features=cfg.agent.get("use_expert_features", False),
            context="ReCogDrive best-of-N exact-pool evaluation",
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
    logger.info("Starting best-of-N exact PDM scoring of %s scenarios...", str(len(tokens_to_evaluate)))

    sampler = InferenceSampler(len(tokens_to_evaluate))
    data_points = []
    for idx in sampler:
        token = tokens_to_evaluate[idx]
        log_file = scene_loader.token_to_log_file[token]
        data_points.append({"cfg": cfg, "log_file": log_file, "tokens": [token]})

    serialized_score_rows = run_pdm_score_best_of_n_exact_pool(data_points)
    local_rows = pickle.loads(serialized_score_rows)
    gathered_rows: List[Optional[List[Dict[str, Any]]]] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_rows, local_rows)

    if dist.get_rank() == 0:
        final_results: List[Dict[str, Any]] = []
        for rows in gathered_rows:
            if rows:
                final_results.extend(rows)
        pdm_score_df = pd.DataFrame(final_results)

        valid = pdm_score_df["valid"].astype(bool) if "valid" in pdm_score_df.columns else pd.Series(True, index=pdm_score_df.index)
        num_successful_scenarios = int(valid.sum())
        num_failed_scenarios = int(len(pdm_score_df) - num_successful_scenarios)
        excluded_average_cols = [col for col in ("token", "valid", "rank", "candidate_scores_json") if col in pdm_score_df.columns]
        numeric_for_average = pdm_score_df.drop(columns=excluded_average_cols).apply(pd.to_numeric, errors="coerce")
        average_row = numeric_for_average.mean(skipna=True)
        average_row["token"] = "average"
        average_row["valid"] = bool(valid.all())
        average_row["rank"] = "0"
        if "candidate_scores_json" in pdm_score_df.columns:
            average_row["candidate_scores_json"] = ""
        pdm_score_df.loc[len(pdm_score_df)] = average_row

        save_path = Path(cfg.output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        output_csv = save_path / f"{timestamp}.csv"
        pdm_score_df.to_csv(output_csv)

        logger.info(
            """
            Finished running best-of-N exact PDM evaluation.
                Number of successful scenarios: %s.
                Number of failed scenarios: %s.
                Final average best-of-N score of valid results: %s.
                Results are stored in: %s.
            """,
            num_successful_scenarios,
            num_failed_scenarios,
            pd.to_numeric(pdm_score_df[pdm_score_df["token"].astype(str) != "average"]["score"], errors="coerce").mean(),
            output_csv,
        )


if __name__ == "__main__":
    main()
