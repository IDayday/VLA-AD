from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import csv
import json
import logging
import os
import traceback

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneFilter, SceneLoader
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.script.run_pdm_score_recogdrive_async_pdm_exact_pool import (
    _make_pdm_tools,
    _metric_cache_loader_from_cfg,
    _score_params_from_cfg,
)
from nuplan.planning.script.builders.logging_builder import build_logger

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"

PDM_KEYS: Tuple[str, ...] = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
)


def _cfg_int(cfg: DictConfig, key: str, env_key: str, default: int) -> int:
    value = cfg.get(key, None)
    if value is None:
        value = os.environ.get(env_key, default)
    return int(value)


def _cfg_float(cfg: DictConfig, key: str, env_key: str, default: float) -> float:
    value = cfg.get(key, None)
    if value is None:
        value = os.environ.get(env_key, default)
    return float(value)


def _cfg_path(cfg: DictConfig, key: str, env_key: str, default: str) -> Path:
    value = cfg.get(key, None)
    if value is None:
        value = os.environ.get(env_key, default)
    return Path(str(value))


def _build_scene_loader_for_tokens(
    cfg: DictConfig,
    tokens: List[str],
    agent: AbstractAgent,
) -> SceneLoader:
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.tokens = tokens
    return SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True,
    )


def _select_tokens(cfg: DictConfig, agent: AbstractAgent, max_tokens: int) -> List[str]:
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = _metric_cache_loader_from_cfg(cfg)
    tokens = sorted(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    if max_tokens > 0:
        tokens = tokens[:max_tokens]
    if not tokens:
        raise RuntimeError("No tokens selected for ReCogDrive batch equivalence validation.")
    # Build the token-filtered scene loader once here to fail early on sensor/log issues.
    _build_scene_loader_for_tokens(cfg, tokens, agent)
    return tokens


def _pad_1d_tensors(tensors: List[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
    max_len = max(int(t.numel()) for t in tensors)
    out = torch.full((len(tensors), max_len), pad_value, dtype=tensors[0].dtype)
    for idx, tensor in enumerate(tensors):
        flat = tensor.reshape(-1)
        out[idx, : flat.numel()] = flat
    return out


def _collate_feature_rows(rows: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = set(rows[0].keys())
    for row in rows[1:]:
        if set(row.keys()) != keys:
            missing = sorted(keys - set(row.keys()))
            extra = sorted(set(row.keys()) - keys)
            raise KeyError(f"Feature keys differ across batch rows: missing={missing}, extra={extra}")

    collated: Dict[str, torch.Tensor] = {}
    for key in sorted(keys):
        tensors = [row[key] for row in rows]
        if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError(f"Feature {key!r} is not tensor-valued for every row.")
        if key == "image_path_tensor":
            collated[key] = _pad_1d_tensors(tensors, pad_value=0)
        else:
            shapes = {tuple(tensor.shape) for tensor in tensors}
            if len(shapes) != 1:
                raise ValueError(f"Cannot stack feature {key!r}; shapes differ: {sorted(shapes)}")
            collated[key] = torch.stack(tensors, dim=0)
    return collated


def _compute_feature_rows(agent: AbstractAgent, agent_inputs: List[AgentInput]) -> List[Dict[str, torch.Tensor]]:
    builders = agent.get_feature_builders()
    rows: List[Dict[str, torch.Tensor]] = []
    for agent_input in agent_inputs:
        features: Dict[str, torch.Tensor] = {}
        for builder in builders:
            features.update(builder.compute_features(agent_input))
        rows.append(features)
    return rows


def _score_scalar_pdm(
    *,
    metric_cache: MetricCache,
    poses: np.ndarray,
    tools,
) -> Dict[str, float]:
    future_sampling, simulator, scorer = tools
    result = pdm_score(
        metric_cache=metric_cache,
        model_trajectory=Trajectory(np.asarray(poses, dtype=np.float32)),
        future_sampling=future_sampling,
        simulator=simulator,
        scorer=scorer,
    )
    return {key: float(value) for key, value in asdict(result).items()}


def _write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    max_tokens = _cfg_int(cfg, "batch_equiv_max_tokens", "RECOGDRIVE_BATCH_EQUIV_MAX_TOKENS", 8)
    batch_size = max(1, _cfg_int(cfg, "batch_equiv_batch_size", "RECOGDRIVE_BATCH_EQUIV_BATCH_SIZE", 4))
    trajectory_tol = _cfg_float(
        cfg,
        "batch_equiv_trajectory_abs_tolerance",
        "RECOGDRIVE_BATCH_EQUIV_TRAJECTORY_ABS_TOLERANCE",
        0.0,
    )
    pdm_tol = _cfg_float(
        cfg,
        "batch_equiv_pdm_abs_tolerance",
        "RECOGDRIVE_BATCH_EQUIV_PDM_ABS_TOLERANCE",
        0.0,
    )
    output_json = _cfg_path(
        cfg,
        "batch_equiv_output_json",
        "RECOGDRIVE_BATCH_EQUIV_OUTPUT_JSON",
        str(Path(cfg.output_dir) / "recogdrive_batch_inference_equivalence.json"),
    )
    output_csv = _cfg_path(
        cfg,
        "batch_equiv_output_csv",
        "RECOGDRIVE_BATCH_EQUIV_OUTPUT_CSV",
        str(Path(cfg.output_dir) / "recogdrive_batch_inference_equivalence_rows.csv"),
    )

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    agent.eval()
    if bool(getattr(agent, "cache_hidden_state", False)):
        raise ValueError("Batch equivalence validation requires agent.cache_hidden_state=False.")

    tokens = _select_tokens(cfg, agent, max_tokens=max_tokens)
    scene_loader = _build_scene_loader_for_tokens(cfg, tokens, agent)
    metric_cache_loader = _metric_cache_loader_from_cfg(cfg)
    score_tools = _make_pdm_tools(_score_params_from_cfg(cfg))

    rows: List[Dict[str, Any]] = []
    max_traj_diff = 0.0
    max_pdm_diff = 0.0
    failures: List[Dict[str, Any]] = []

    logger.info(
        "Validating ReCogDrive batch inference equivalence on %s tokens with batch_size=%s",
        len(tokens),
        batch_size,
    )
    for start in range(0, len(tokens), batch_size):
        batch_tokens = tokens[start : start + batch_size]
        agent_inputs = [scene_loader.get_agent_input_from_token(token) for token in batch_tokens]

        single_poses: List[np.ndarray] = []
        for token, agent_input in zip(batch_tokens, agent_inputs):
            try:
                with torch.no_grad():
                    trajectory = agent.compute_trajectory(agent_input)
                single_poses.append(np.asarray(trajectory.poses, dtype=np.float32))
            except Exception as exc:
                traceback.print_exc()
                raise RuntimeError(f"Single-sample inference failed for token {token}") from exc

        feature_rows = _compute_feature_rows(agent, agent_inputs)
        batched_features = _collate_feature_rows(feature_rows)
        with torch.no_grad():
            predictions = agent.forward(batched_features)
            batched_poses_tensor = predictions["pred_traj"].float().cpu()
        batched_poses = np.asarray(batched_poses_tensor, dtype=np.float32)
        if batched_poses.shape[0] != len(batch_tokens):
            raise RuntimeError(
                f"Batched inference returned {batched_poses.shape[0]} trajectories for {len(batch_tokens)} tokens."
            )

        for local_idx, token in enumerate(batch_tokens):
            metric_cache = metric_cache_loader.get_from_token(token)
            single = single_poses[local_idx]
            batched = batched_poses[local_idx]
            traj_diff = float(np.max(np.abs(single - batched)))
            max_traj_diff = max(max_traj_diff, traj_diff)
            single_pdm = _score_scalar_pdm(metric_cache=metric_cache, poses=single, tools=score_tools)
            batched_pdm = _score_scalar_pdm(metric_cache=metric_cache, poses=batched, tools=score_tools)
            pdm_diffs = {key: abs(single_pdm[key] - batched_pdm[key]) for key in PDM_KEYS}
            token_max_pdm_diff = max(pdm_diffs.values()) if pdm_diffs else 0.0
            max_pdm_diff = max(max_pdm_diff, token_max_pdm_diff)

            row: Dict[str, Any] = {
                "token": token,
                "trajectory_max_abs_diff": traj_diff,
                "pdm_max_abs_diff": token_max_pdm_diff,
                "single_score": single_pdm["score"],
                "batched_score": batched_pdm["score"],
            }
            for key in PDM_KEYS:
                row[f"{key}_single"] = single_pdm[key]
                row[f"{key}_batched"] = batched_pdm[key]
                row[f"{key}_abs_diff"] = pdm_diffs[key]
            rows.append(row)
            if traj_diff > trajectory_tol or token_max_pdm_diff > pdm_tol:
                failures.append(row)

    summary = {
        "tokens": tokens,
        "num_tokens": len(tokens),
        "batch_size": batch_size,
        "trajectory_abs_tolerance": trajectory_tol,
        "pdm_abs_tolerance": pdm_tol,
        "max_trajectory_abs_diff": max_traj_diff,
        "max_pdm_abs_diff": max_pdm_diff,
        "passed": not failures,
        "num_failures": len(failures),
        "failures": failures[:20],
        "rows_csv": str(output_csv),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_rows(output_csv, rows)
    logger.info("Batch equivalence summary: %s", json.dumps(summary, sort_keys=True))
    if failures:
        raise SystemExit(
            "ReCogDrive batch inference equivalence check failed. "
            f"max_trajectory_abs_diff={max_traj_diff}, max_pdm_abs_diff={max_pdm_diff}."
        )


if __name__ == "__main__":
    main()
