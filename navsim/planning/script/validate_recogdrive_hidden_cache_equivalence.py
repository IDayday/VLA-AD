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
from navsim.planning.training.dataset import load_feature_target_from_pickle
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


def _tokens_from_hidden_cache(cache_root: Path) -> List[str]:
    tokens: List[str] = []
    for path in cache_root.glob("*/*/internvl_feature.gz"):
        tokens.append(path.parent.name)
    for path in cache_root.glob("*/internvl_feature.gz"):
        tokens.append(path.parent.name)
    return sorted(set(tokens))


def _select_tokens(cfg: DictConfig, max_tokens: int, cache_root: Path) -> List[str]:
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = _metric_cache_loader_from_cfg(cfg)
    available = set(scene_loader.tokens) & set(metric_cache_loader.tokens)
    cached_tokens = _tokens_from_hidden_cache(cache_root)
    if cached_tokens:
        tokens = [token for token in cached_tokens if token in available]
    else:
        tokens = sorted(available)
    if max_tokens > 0:
        tokens = tokens[:max_tokens]
    if not tokens:
        raise RuntimeError("No tokens selected for ReCogDrive hidden-cache equivalence validation.")
    return tokens


def _cache_feature_path(cache_root: Path, log_name: str, token: str) -> Path:
    candidates = (
        cache_root / log_name / token / "internvl_feature.gz",
        cache_root / token / "internvl_feature.gz",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing hidden-state cache for token={token} log={log_name}: {candidates[0]}")


def _load_cached_features(cache_root: Path, log_name: str, token: str) -> Dict[str, torch.Tensor]:
    path = _cache_feature_path(cache_root, log_name, token)
    features = load_feature_target_from_pickle(path)
    if "last_hidden_state" not in features:
        raise KeyError(f"Cached feature file lacks last_hidden_state: {path}")
    return {
        key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value
        for key, value in features.items()
    }


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

    cache_root = _cfg_path(cfg, "hidden_cache_equiv_cache_path", "RECOGDRIVE_HIDDEN_CACHE_EQUIV_CACHE_PATH", "")
    if not str(cache_root):
        raise ValueError("Set RECOGDRIVE_HIDDEN_CACHE_EQUIV_CACHE_PATH or +hidden_cache_equiv_cache_path.")
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Hidden-state cache path does not exist: {cache_root}")

    max_tokens = _cfg_int(cfg, "hidden_cache_equiv_max_tokens", "RECOGDRIVE_HIDDEN_CACHE_EQUIV_MAX_TOKENS", 8)
    trajectory_tol = _cfg_float(
        cfg,
        "hidden_cache_equiv_trajectory_abs_tolerance",
        "RECOGDRIVE_HIDDEN_CACHE_EQUIV_TRAJECTORY_ABS_TOLERANCE",
        0.0,
    )
    pdm_tol = _cfg_float(
        cfg,
        "hidden_cache_equiv_pdm_abs_tolerance",
        "RECOGDRIVE_HIDDEN_CACHE_EQUIV_PDM_ABS_TOLERANCE",
        0.0,
    )
    output_json = _cfg_path(
        cfg,
        "hidden_cache_equiv_output_json",
        "RECOGDRIVE_HIDDEN_CACHE_EQUIV_OUTPUT_JSON",
        str(Path(cfg.output_dir) / "recogdrive_hidden_cache_equivalence.json"),
    )
    output_csv = _cfg_path(
        cfg,
        "hidden_cache_equiv_output_csv",
        "RECOGDRIVE_HIDDEN_CACHE_EQUIV_OUTPUT_CSV",
        str(Path(cfg.output_dir) / "recogdrive_hidden_cache_equivalence_rows.csv"),
    )

    cfg.agent.cache_hidden_state = False
    no_cache_agent: AbstractAgent = instantiate(cfg.agent)
    no_cache_agent.initialize()
    no_cache_agent.eval()

    cfg.agent.cache_hidden_state = True
    cfg.agent.cache_mode = False
    cache_agent: AbstractAgent = instantiate(cfg.agent)
    cache_agent.initialize()
    cache_agent.eval()

    tokens = _select_tokens(cfg, max_tokens=max_tokens, cache_root=cache_root)
    scene_loader = _build_scene_loader_for_tokens(cfg, tokens, no_cache_agent)
    metric_cache_loader = _metric_cache_loader_from_cfg(cfg)
    score_tools = _make_pdm_tools(_score_params_from_cfg(cfg))

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    max_traj_diff = 0.0
    max_pdm_diff = 0.0

    logger.info("Validating ReCogDrive hidden-cache equivalence on %s tokens", len(tokens))
    for token in tokens:
        agent_input: AgentInput = scene_loader.get_agent_input_from_token(token)
        log_name = scene_loader.token_to_log_file[token]
        try:
            with torch.no_grad():
                no_cache_traj = no_cache_agent.compute_trajectory(agent_input)
                cached_features = _load_cached_features(cache_root, log_name, token)
                cache_pred = cache_agent.forward(cached_features)
                cache_poses = cache_pred["pred_traj"].float().cpu().squeeze(0).numpy()
        except Exception as exc:
            traceback.print_exc()
            raise RuntimeError(f"Hidden-cache equivalence inference failed for token {token}") from exc

        no_cache_poses = np.asarray(no_cache_traj.poses, dtype=np.float32)
        cache_poses = np.asarray(cache_poses, dtype=np.float32)
        traj_diff = float(np.max(np.abs(no_cache_poses - cache_poses)))
        max_traj_diff = max(max_traj_diff, traj_diff)

        metric_cache = metric_cache_loader.get_from_token(token)
        no_cache_pdm = _score_scalar_pdm(metric_cache=metric_cache, poses=no_cache_poses, tools=score_tools)
        cache_pdm = _score_scalar_pdm(metric_cache=metric_cache, poses=cache_poses, tools=score_tools)
        pdm_diffs = {key: abs(no_cache_pdm[key] - cache_pdm[key]) for key in PDM_KEYS}
        token_max_pdm_diff = max(pdm_diffs.values()) if pdm_diffs else 0.0
        max_pdm_diff = max(max_pdm_diff, token_max_pdm_diff)

        row: Dict[str, Any] = {
            "token": token,
            "log_name": log_name,
            "trajectory_max_abs_diff": traj_diff,
            "pdm_max_abs_diff": token_max_pdm_diff,
            "no_cache_score": no_cache_pdm["score"],
            "cache_score": cache_pdm["score"],
        }
        for key in PDM_KEYS:
            row[f"{key}_no_cache"] = no_cache_pdm[key]
            row[f"{key}_cache"] = cache_pdm[key]
            row[f"{key}_abs_diff"] = pdm_diffs[key]
        rows.append(row)
        if traj_diff > trajectory_tol or token_max_pdm_diff > pdm_tol:
            failures.append(row)

    summary = {
        "cache_root": str(cache_root),
        "tokens": tokens,
        "num_tokens": len(tokens),
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
    logger.info("Hidden-cache equivalence summary: %s", json.dumps(summary, sort_keys=True))
    if failures:
        raise SystemExit(
            "ReCogDrive hidden-cache equivalence check failed. "
            f"max_trajectory_abs_diff={max_traj_diff}, max_pdm_abs_diff={max_pdm_diff}."
        )


if __name__ == "__main__":
    main()
