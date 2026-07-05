from __future__ import annotations

import csv
import json
import logging
import os
import sys
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.common.dataclasses import SensorConfig
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.metric_caching.fast_metric_cache_loader import FastMetricCacheLoader, load_metric_cache_auto
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from nuplan.planning.script.builders.logging_builder import build_logger

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def _cfg_str(cfg: DictConfig, key: str, env_key: str, default: str) -> str:
    value = cfg.get(key, None)
    if value is None:
        value = os.environ.get(env_key, default)
    return str(value)


def _read_eval_token_file(cfg: DictConfig) -> List[str]:
    token_file = _cfg_str(cfg, "eval_token_file", "RECOGDRIVE_EVAL_TOKEN_FILE", "").strip()
    if not token_file:
        return []
    token_path = Path(token_file)
    if not token_path.is_file():
        raise FileNotFoundError(f"eval_token_file does not exist: {token_path}")
    return [
        line.strip().split(",")[0]
        for line in token_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _metric_cache_loader_from_cfg(cfg: DictConfig):
    fast_metric_cache_path = str(
        cfg.get("fast_metric_cache_path", "") or os.environ.get("RECOGDRIVE_FAST_METRIC_CACHE_PATH", "")
    )
    if fast_metric_cache_path:
        return FastMetricCacheLoader(Path(fast_metric_cache_path))
    return MetricCacheLoader(Path(cfg.metric_cache_path))


def _load_metric_cache(loader: Any, token: str) -> MetricCache:
    if isinstance(loader, FastMetricCacheLoader):
        return load_metric_cache_auto(Path(loader.metric_cache_paths[token]))
    return loader.get_from_token(token)


def _restrict_scene_filter(scene_filter: SceneFilter, cfg: DictConfig, metric_cache_loader: Any) -> List[str]:
    requested_tokens = _read_eval_token_file(cfg)
    if requested_tokens:
        scene_filter.tokens = requested_tokens
        log_names = set()
        for token in requested_tokens:
            metric_cache_path = metric_cache_loader.metric_cache_paths.get(token)
            if metric_cache_path is None:
                continue
            path = Path(metric_cache_path)
            if len(path.parents) >= 3:
                candidate_log = path.parents[2].name
                if "_veh-" in candidate_log:
                    log_names.add(candidate_log)
        if log_names:
            scene_filter.log_names = sorted(log_names)
        return requested_tokens
    return []


def _write_pdm_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "token",
        "scene_token",
        "log_name",
        "valid",
        "score",
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "ego_progress",
        "time_to_collision_within_bound",
        "comfort",
        "driving_direction_compliance",
        "error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)
    output_dir_raw = _cfg_str(cfg, "trajectory_output_dir", "RECOGDRIVE_TRAJECTORY_OUTPUT_DIR", "").strip()
    if not output_dir_raw:
        raise ValueError("Set +trajectory_output_dir=... or RECOGDRIVE_TRAJECTORY_OUTPUT_DIR.")
    output_dir = Path(output_dir_raw).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_cache_loader = _metric_cache_loader_from_cfg(cfg)
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    requested_tokens = _restrict_scene_filter(scene_filter, cfg, metric_cache_loader)

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True,
    )
    tokens = sorted(set(scene_loader.tokens) & set(metric_cache_loader.metric_cache_paths))
    if requested_tokens:
        wanted = set(requested_tokens)
        tokens = [token for token in tokens if token in wanted]
    if not tokens:
        raise RuntimeError("No tokens selected for trajectory export.")

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)

    predictions: List[Dict[str, Any]] = []
    pdm_rows: List[Dict[str, Any]] = []
    for idx, token in enumerate(tokens, 1):
        logger.info("Exporting trajectory %s/%s token=%s", idx, len(tokens), token)
        log_name = scene_loader.token_to_log_file.get(token, "")
        record: Dict[str, Any] = {
            "token": token,
            "log_name": log_name,
            "valid": False,
        }
        try:
            agent_input = scene_loader.get_agent_input_from_token(token)
            trajectory = agent.compute_trajectory(agent_input)
            pred_poses = trajectory.poses.tolist()
            scene = scene_loader.get_scene_from_token(token)
            gt_trajectory = scene.get_future_trajectory(num_trajectory_frames=8)
            metric_cache = _load_metric_cache(metric_cache_loader, token)
            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            result_dict = asdict(pdm_result)
            record.update(
                {
                    "scene_token": scene.scene_metadata.scene_token,
                    "map_name": scene.scene_metadata.map_name,
                    "pred_traj": pred_poses,
                    "gt_traj": gt_trajectory.poses.tolist(),
                    "pdm": result_dict,
                    "valid": True,
                }
            )
            pdm_rows.append(
                {
                    "token": token,
                    "scene_token": scene.scene_metadata.scene_token,
                    "log_name": log_name,
                    "valid": True,
                    **result_dict,
                }
            )
        except Exception as exc:
            logger.warning("Failed token=%s: %r", token, exc)
            traceback.print_exc()
            record["error"] = repr(exc)
            pdm_rows.append({"token": token, "log_name": log_name, "valid": False, "error": repr(exc)})
        predictions.append(record)

    valid_rows = [row for row in pdm_rows if row.get("valid")]
    metrics: Dict[str, Any] = {
        "checkpoint": str(cfg.agent.get("checkpoint_path", "")),
        "num_tokens": len(tokens),
        "num_valid": len(valid_rows),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "eval_token_file": _cfg_str(cfg, "eval_token_file", "RECOGDRIVE_EVAL_TOKEN_FILE", ""),
    }
    for key in [
        "score",
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "ego_progress",
        "time_to_collision_within_bound",
        "comfort",
        "driving_direction_compliance",
    ]:
        values = [float(row[key]) for row in valid_rows if key in row and row[key] != ""]
        metrics[key] = sum(values) / len(values) if values else None

    (output_dir / "predictions.json").write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_pdm_csv(output_dir / "pdm_results.csv", pdm_rows)
    pd.DataFrame(pdm_rows).to_csv(output_dir / "pdm_results_full.csv", index=False)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
