#!/usr/bin/env python3
"""Isolated persistent NAVSIM v2 scorer used by LFP-GRPO training ranks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _activate_navsim(root: Path) -> None:
    root = root.expanduser().resolve()
    if not (root / "navsim" / "evaluate" / "pdm_score.py").is_file():
        raise FileNotFoundError(f"Not an official NAVSIM v2 source tree: {root}")
    sys.path.insert(0, str(root))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--authkey", required=True)
    parser.add_argument("--navsim-root", required=True, type=Path)
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--config-name", default="default_run_pdm_score")
    parser.add_argument("--metric-cache-path", required=True, type=Path)
    parser.add_argument("--overrides-json", default="[]")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _activate_navsim(args.navsim_root)

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from navsim.common.dataclasses import Trajectory
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
        ego_is_two_frame_extended_comfort,
    )
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex

    overrides = list(json.loads(args.overrides_json))
    overrides.extend((f"metric_cache_path={args.metric_cache_path.resolve()}", "traffic_agents=non_reactive"))
    with initialize_config_dir(
        config_dir=str(args.config_dir.resolve()),
        version_base=None,
        job_name="lfp_v2_metric_worker",
    ):
        cfg = compose(config_name=args.config_name, overrides=overrides)
    simulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    traffic_policy = instantiate(cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling)
    metric_loader = MetricCacheLoader(args.metric_cache_path.resolve())
    previous_state_cache: Dict[tuple[str, bytes], tuple[np.ndarray, float]] = {}

    def score_trajectory(token: str, trajectory: np.ndarray):
        metric_cache = metric_loader.get_from_token(token)
        row, simulated_states = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=Trajectory(np.asarray(trajectory, dtype=np.float32)),
            future_sampling=simulator.proposal_sampling,
            simulator=simulator,
            scorer=scorer,
            traffic_agents_policy=traffic_policy,
        )
        return row.iloc[0].to_dict(), np.asarray(simulated_states), float(metric_cache.timepoint.time_s)

    connection = Client(args.socket, family="AF_UNIX", authkey=bytes.fromhex(args.authkey))
    connection.send({"ready": True})
    while True:
        request = connection.recv()
        if request.get("command") == "close":
            connection.send({"closed": True})
            break
        if request.get("command") != "score":
            connection.send({"error": f"Unknown worker command: {request.get('command')!r}"})
            continue
        try:
            trajectories = np.asarray(request["trajectories"], dtype=np.float32)
            tokens = [str(token) for token in request["tokens"]]
            previous_tokens = [str(token) for token in request["previous_tokens"]]
            previous_trajectories = np.asarray(request["previous_trajectories"], dtype=np.float32)
            if not (
                len(trajectories)
                == len(tokens)
                == len(previous_tokens)
                == len(previous_trajectories)
            ):
                raise ValueError("NAVSIM v2 scorer request arrays have inconsistent lengths.")

            output: Dict[str, list[float]] = {
                "score": [],
                "ego_progress": [],
                "time_to_collision_within_bound": [],
                "lane_keeping": [],
                "history_comfort": [],
                "two_frame_extended_comfort": [],
                "no_at_fault_collisions": [],
                "drivable_area_compliance": [],
                "driving_direction_compliance": [],
                "traffic_light_compliance": [],
            }
            for token, trajectory, previous_token, previous_trajectory in zip(
                tokens,
                trajectories,
                previous_tokens,
                previous_trajectories,
            ):
                row, current_states, current_time = score_trajectory(token, trajectory)
                digest = hashlib.sha1(np.ascontiguousarray(previous_trajectory).tobytes()).digest()
                cache_key = (previous_token, digest)
                previous_state = previous_state_cache.get(cache_key)
                if previous_state is None:
                    _, previous_states, previous_time = score_trajectory(previous_token, previous_trajectory)
                    previous_state = (previous_states, previous_time)
                    previous_state_cache[cache_key] = previous_state
                previous_states, previous_time = previous_state
                observation_interval = current_time - previous_time
                if not 0.0 < observation_interval < 0.55:
                    raise ValueError(
                        f"Invalid v2 adjacent-frame interval for {previous_token}->{token}: "
                        f"{observation_interval}."
                    )
                overlap_start = round(observation_interval / simulator.proposal_sampling.interval_length)
                current_overlap = current_states[:-overlap_start]
                previous_overlap = previous_states[overlap_start:]
                if len(current_overlap) == 0 or len(current_overlap) != len(previous_overlap):
                    raise ValueError(f"No aligned v2 extended-comfort overlap for token {token!r}.")
                time_points = np.arange(len(current_overlap)) * simulator.proposal_sampling.interval_length
                extended = float(
                    ego_is_two_frame_extended_comfort(
                        current_overlap[None, :],
                        previous_overlap[None, :],
                        time_points,
                    )[0]
                )
                weighted = np.asarray(row["weighted_metrics"], dtype=np.float64).copy()
                weights = np.asarray(row["weighted_metrics_array"], dtype=np.float64).copy()
                weighted[WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT] = extended
                denominator = float(weights.sum())
                if denominator <= 0.0:
                    raise ValueError("Official EPDMS weighted denominator is non-positive.")
                scalar = float(row["multiplicative_metrics_prod"]) * float((weighted * weights).sum()) / denominator
                values = {
                    "score": scalar,
                    "ego_progress": row["ego_progress"],
                    "time_to_collision_within_bound": row["time_to_collision_within_bound"],
                    "lane_keeping": row["lane_keeping"],
                    "history_comfort": row["history_comfort"],
                    "two_frame_extended_comfort": extended,
                    "no_at_fault_collisions": row["no_at_fault_collisions"],
                    "drivable_area_compliance": row["drivable_area_compliance"],
                    "driving_direction_compliance": row["driving_direction_compliance"],
                    "traffic_light_compliance": row["traffic_light_compliance"],
                }
                for key, value in values.items():
                    output[key].append(float(value))
            connection.send({"metrics": output})
        except Exception as error:
            connection.send({"error": str(error), "traceback": traceback.format_exc()})
    connection.close()


if __name__ == "__main__":
    main()
