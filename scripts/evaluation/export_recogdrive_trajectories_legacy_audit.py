from __future__ import annotations

import csv
import json
import logging
import lzma
import os
import pickle
import sys
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import hydra
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from nuplan.planning.script.builders.logging_builder import build_logger

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def _cfg_str(cfg: DictConfig, key: str, env_key: str, default: str = "") -> str:
    value = cfg.get(key, None)
    if value is None:
        value = os.environ.get(env_key, default)
    return str(value)


def _read_token_file(path_str: str) -> List[str]:
    if not path_str:
        return []
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"token file does not exist: {path}")
    tokens: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens.append(line.split(",")[0].strip())
    return tokens


def _load_metric_cache(loader: MetricCacheLoader, token: str) -> MetricCache:
    metric_cache_path = loader.metric_cache_paths[token]
    with lzma.open(metric_cache_path, "rb") as f:
        return pickle.load(f)


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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)
    output_dir = Path(_cfg_str(cfg, "trajectory_output_dir", "RECOGDRIVE_TRAJECTORY_OUTPUT_DIR")).resolve()
    if not str(output_dir):
        raise ValueError("Set +trajectory_output_dir=... or RECOGDRIVE_TRAJECTORY_OUTPUT_DIR.")
    output_dir.mkdir(parents=True, exist_ok=True)

    token_file = _cfg_str(cfg, "eval_token_file", "RECOGDRIVE_EVAL_TOKEN_FILE")
    requested_tokens = _read_token_file(token_file)

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if requested_tokens:
        scene_filter.tokens = requested_tokens

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True,
    )
    tokens = sorted(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    if requested_tokens:
        requested = set(requested_tokens)
        tokens = [token for token in tokens if token in requested]
    if not tokens:
        raise RuntimeError("No selected tokens are present in both SceneLoader and metric cache.")

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    if simulator.proposal_sampling != scorer.proposal_sampling:
        raise AssertionError("Simulator and scorer proposal sampling must match.")

    predictions: List[Dict[str, Any]] = []
    pdm_rows: List[Dict[str, Any]] = []
    for idx, token in enumerate(tokens, 1):
        logger.info("Exporting trajectory %s/%s token=%s", idx, len(tokens), token)
        log_name = scene_loader.token_to_log_file.get(token, "")
        record: Dict[str, Any] = {"token": token, "log_name": log_name, "valid": False}
        try:
            agent_input = scene_loader.get_agent_input_from_token(token)
            trajectory = agent.compute_trajectory(agent_input)
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
            result = asdict(pdm_result)
            scene_token = getattr(scene.scene_metadata, "scene_token", "")
            map_name = getattr(scene.scene_metadata, "map_name", "")
            record.update(
                {
                    "scene_token": scene_token,
                    "map_name": map_name,
                    "pred_traj": trajectory.poses.tolist(),
                    "gt_traj": gt_trajectory.poses.tolist(),
                    "pdm": result,
                    "valid": True,
                }
            )
            pdm_rows.append(
                {
                    "token": token,
                    "scene_token": scene_token,
                    "log_name": log_name,
                    "valid": True,
                    **result,
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
        "eval_token_file": token_file,
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
        values = [float(row[key]) for row in valid_rows if row.get(key, "") != ""]
        metrics[key] = sum(values) / len(values) if values else None

    (output_dir / "predictions.json").write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "hydra_config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    _write_pdm_csv(output_dir / "pdm_results.csv", pdm_rows)
    pd.DataFrame(pdm_rows).to_csv(output_dir / "pdm_results_full.csv", index=False)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
