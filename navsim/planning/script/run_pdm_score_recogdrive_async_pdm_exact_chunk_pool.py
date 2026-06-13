from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import multiprocessing as mp
import os
import pickle
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
from navsim.common.dataloader import SceneFilter, SceneLoader
from navsim.common.dataclasses import SensorConfig
from nuplan.planning.script.builders.logging_builder import build_logger
from navsim.planning.script.run_pdm_score_recogdrive_async_pdm_exact_pool import (
    InferenceSampler,
    _build_rank_scene_loader,
    _cfg_int,
    _cfg_str,
    _apply_eval_token_shard,
    _init_process_pdm_tools,
    _metric_cache_loader_from_cfg,
    _score_one_pdm_scalar_exact,
    _score_params_from_cfg,
    broadcast_object,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def _score_pdm_chunk_scalar_exact(
    tasks: List[Dict[str, Any]],
    *,
    score_params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Score a chunk of trajectories with the exact scalar PDM path.

    This is a scheduling optimization only. Each item is still scored by
    ``_score_one_pdm_scalar_exact()``, which calls ``navsim.evaluate.pdm_score.pdm_score()``
    once per token and preserves the official PDMS/submetric semantics.
    """
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        rows.append(
            _score_one_pdm_scalar_exact(
                token=str(task["token"]),
                metric_cache_path=str(task["metric_cache_path"]),
                poses=task["poses"],
                rank=int(task["rank"]),
                index=int(task["index"]),
                score_params=score_params,
            )
        )
    return rows


def _drain_chunk_pending(
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
        rows.extend(future.result())
    return remaining


def _submit_score_chunk(
    *,
    executor: Optional[ThreadPoolExecutor | ProcessPoolExecutor],
    backend: str,
    chunk: List[Dict[str, Any]],
    score_params: Dict[str, Any],
    rows: List[Dict[str, Any]],
    pending: List[Future],
) -> List[Future]:
    if not chunk:
        return pending
    chunk_payload = list(chunk)
    if executor is None:
        rows.extend(_score_pdm_chunk_scalar_exact(chunk_payload, score_params=score_params))
    elif backend == "process":
        pending.append(executor.submit(_score_pdm_chunk_scalar_exact, chunk_payload, score_params=None))
    else:
        pending.append(executor.submit(_score_pdm_chunk_scalar_exact, chunk_payload, score_params=score_params))
    return pending


def run_pdm_score_async_exact_chunk_pool(args: List[Dict[str, Any]]) -> bytes:
    """
    Run ReCogDrive inference and score exact scalar PDMS through chunked CPU tasks.

    Compared with ``run_pdm_score_recogdrive_async_pdm_exact_pool.py``, this groups several
    completed trajectories into one process-pool Future to reduce executor/pickling overhead.
    It deliberately does not batch PDM math: every token still goes through the original scalar
    ``pdm_score()`` call.
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    rank = dist.get_rank()
    thread_id = str(uuid.uuid4())
    logger.info("Starting exact chunked async PDM worker thread_id=%s, node_id=%s, rank=%s", thread_id, node_id, rank)

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
    task_chunk_size = max(
        1,
        _cfg_int(cfg, "async_pdm_task_chunk_size", "RECOGDRIVE_ASYNC_PDM_TASK_CHUNK_SIZE", 4),
    )
    progress_every = max(0, _cfg_int(cfg, "async_pdm_progress_every", "RECOGDRIVE_ASYNC_PDM_PROGRESS_EVERY", 100))
    backend = _cfg_str(cfg, "async_pdm_backend", "RECOGDRIVE_ASYNC_PDM_BACKEND", "process").lower()
    process_start_method = _cfg_str(
        cfg,
        "async_pdm_process_start_method",
        "RECOGDRIVE_ASYNC_PDM_PROCESS_START_METHOD",
        "spawn",
    )
    if backend not in {"thread", "process"}:
        raise ValueError(f"async_pdm_backend must be 'thread' or 'process', got {backend!r}.")

    logger.info(
        "Rank %s evaluating %s scenarios with async_pdm_backend=%s, workers=%s, queue_size=%s, task_chunk_size=%s",
        rank,
        len(tokens_to_evaluate),
        backend,
        pdm_workers,
        queue_size,
        task_chunk_size,
    )

    rows: List[Dict[str, Any]] = []
    pending: List[Future] = []
    chunk: List[Dict[str, Any]] = []
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
                chunk.append(
                    {
                        "token": token,
                        "metric_cache_path": str(metric_cache_loader.metric_cache_paths[token]),
                        "poses": trajectory.poses,
                        "rank": rank,
                        "index": idx,
                    }
                )
                if len(chunk) >= task_chunk_size:
                    pending = _submit_score_chunk(
                        executor=executor,
                        backend=backend,
                        chunk=chunk,
                        score_params=score_params,
                        rows=rows,
                        pending=pending,
                    )
                    chunk = []
                    if len(pending) >= queue_size:
                        pending = _drain_chunk_pending(pending, rows, wait_for_one=True)
            except Exception:
                logger.warning("----------- Agent failed for token %s:", token)
                traceback.print_exc()
                rows.append({"token": token, "valid": False, "rank": rank, "_index": idx})

            pending = _drain_chunk_pending(pending, rows)

        pending = _submit_score_chunk(
            executor=executor,
            backend=backend,
            chunk=chunk,
            score_params=score_params,
            rows=rows,
            pending=pending,
        )
        pending = _drain_chunk_pending(pending, rows, wait_all=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    rows.sort(key=lambda row: int(row.get("_index", 0)))
    for row in rows:
        row.pop("_index", None)
    return pickle.dumps(rows)


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
            context="ReCogDrive exact chunk-pool evaluation",
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
    logger.info("Starting exact chunked async pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))

    sampler = InferenceSampler(len(tokens_to_evaluate))
    data_points = []
    for idx in sampler:
        token = tokens_to_evaluate[idx]
        log_file = scene_loader.token_to_log_file[token]
        data_points.append({"cfg": cfg, "log_file": log_file, "tokens": [token]})

    serialized_score_rows = run_pdm_score_async_exact_chunk_pool(data_points)
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
            Finished running exact chunked async PDM evaluation.
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
