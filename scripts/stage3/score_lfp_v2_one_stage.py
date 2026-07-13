#!/usr/bin/env python3
"""Score fixed trajectories with the official NAVSIM v2 one-stage EPDMS backend."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd


def _activate_official_navsim(navsim_root: Path) -> None:
    root = navsim_root.expanduser().resolve()
    if not (root / "navsim" / "evaluate" / "pdm_score.py").is_file():
        raise FileNotFoundError(f"Not an official NAVSIM v2 source tree: {root}")
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != root]
    sys.path.insert(0, str(root))
    for module_name in list(sys.modules):
        if module_name == "navsim" or module_name.startswith("navsim."):
            del sys.modules[module_name]


def _compose_config(config_dir: Path, config_name: str, overrides: Iterable[str]):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        config_dir=str(config_dir.expanduser().resolve()),
        version_base=None,
        job_name="lfp_v2_one_stage",
    ):
        return compose(config_name=config_name, overrides=list(overrides))


def _infer_adjacent_rows(
    rows: pd.DataFrame,
    scene_frame_type_original: Any,
    reactive_mapping: Iterable[Any],
) -> Dict[str, str]:
    adjacent: Dict[str, str] = {}
    original = rows[rows["frame_type"] == scene_frame_type_original]
    for _, group in original.groupby("log_name"):
        ordered = group.sort_values("start_time").reset_index(drop=True)
        for index in range(1, len(ordered)):
            current = ordered.iloc[index]
            previous = ordered.iloc[index - 1]
            delta = float(current["start_time"]) - float(previous["start_time"])
            if 0.0 < delta <= 0.55:
                adjacent[str(current["token"])] = str(previous["token"])
    available = set(str(token) for token in rows["token"].tolist())
    for mapping_row in reactive_mapping:
        if len(mapping_row) != 3:
            continue
        original, previous, followups = mapping_row
        if str(original) in available and str(previous) in available:
            adjacent[str(original)] = str(previous)
        for pair in followups:
            if len(pair) == 2 and str(pair[0]) in available and str(pair[1]) in available:
                adjacent[str(pair[0])] = str(pair[1])
    return adjacent


def _finalize_scores(rows: pd.DataFrame, weighted_metric_index: Any) -> pd.DataFrame:
    output = rows.copy()
    extended = output["two_frame_extended_comfort"].to_numpy(dtype=np.float64)
    weighted = np.stack(output["weighted_metrics"].to_numpy())
    weights = np.stack(output["weighted_metrics_array"].to_numpy())
    extended_index = weighted_metric_index.TWO_FRAME_EXTENDED_COMFORT
    missing = np.isnan(extended)
    weighted[missing, extended_index] = 0.0
    weights[missing, extended_index] = 0.0
    weighted[~missing, extended_index] = extended[~missing]
    denominator = weights.sum(axis=1)
    if np.any(denominator <= 0.0):
        raise ValueError("Official EPDMS weighted-metric denominator is non-positive.")
    output["score"] = (
        output["multiplicative_metrics_prod"].to_numpy(dtype=np.float64)
        * (weighted * weights).sum(axis=1)
        / denominator
    )
    return output.drop(
        columns=["weighted_metrics", "weighted_metrics_array", "multiplicative_metrics_prod"]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_path", required=True, type=Path)
    parser.add_argument("--navsim_root", required=True, type=Path)
    parser.add_argument("--config_dir", type=Path, default=None)
    parser.add_argument("--config_name", default="default_run_pdm_score")
    parser.add_argument("--metric_cache_path", required=True, type=Path)
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _activate_official_navsim(args.navsim_root)

    from hydra.utils import instantiate
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.geometry.convert import relative_to_absolute_poses
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.common.enums import SceneFrameType
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.scoring.scene_aggregator import SceneAggregator
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex

    config_dir = args.config_dir or (
        args.navsim_root / "navsim" / "planning" / "script" / "config" / "pdm_scoring"
    )
    overrides = list(args.override)
    overrides.extend(
        [
            f"metric_cache_path={args.metric_cache_path.expanduser().resolve()}",
            "traffic_agents=non_reactive",
        ]
    )
    cfg = _compose_config(config_dir, args.config_name, overrides)
    simulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    traffic_policy = instantiate(cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling)
    metric_loader = MetricCacheLoader(args.metric_cache_path.expanduser().resolve())

    with args.submission_path.expanduser().open("rb") as handle:
        payload = pickle.load(handle)
    predictions = payload.get("predictions", payload)
    if isinstance(predictions, list):
        if len(predictions) != 1:
            raise ValueError("LFP v2 one-stage scoring supports exactly one deterministic seed.")
        predictions = predictions[0]
    if not isinstance(predictions, dict) or not predictions:
        raise ValueError("Submission must contain a non-empty token -> Trajectory mapping.")

    rows = []
    for token, trajectory in predictions.items():
        metric_cache = metric_loader.get_from_token(str(token))
        score_row, simulated_states = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=trajectory,
            future_sampling=simulator.proposal_sampling,
            simulator=simulator,
            scorer=scorer,
            traffic_agents_policy=traffic_policy,
        )
        score_row["token"] = str(token)
        score_row["valid"] = True
        score_row["log_name"] = metric_cache.log_name
        score_row["frame_type"] = metric_cache.scene_type
        score_row["start_time"] = metric_cache.timepoint.time_s
        endpoint = StateSE2(*trajectory.poses[-1].tolist())
        absolute_endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [endpoint])[0]
        score_row["endpoint_x"] = absolute_endpoint.x
        score_row["endpoint_y"] = absolute_endpoint.y
        score_row["start_point_x"] = metric_cache.ego_state.rear_axle.x
        score_row["start_point_y"] = metric_cache.ego_state.rear_axle.y
        score_row["ego_simulated_states"] = [simulated_states]
        rows.append(score_row)

    score_df = pd.concat(rows, ignore_index=True)
    score_df["two_frame_extended_comfort"] = np.nan
    updates = []
    reactive_mapping = getattr(cfg.train_test_split, "reactive_all_mapping", ())
    adjacent = _infer_adjacent_rows(score_df, SceneFrameType.ORIGINAL, reactive_mapping)
    score_df["previous_token"] = score_df["token"].map(adjacent)
    indexed = score_df.set_index("token")
    for current, previous in adjacent.items():
        aggregator = SceneAggregator(
            now_frame=current,
            previous_frame=previous,
            score_df=indexed,
            proposal_sampling=simulator.proposal_sampling,
        )
        updates.append(aggregator.aggregate_scores(one_stage_only=True))
    if updates:
        indexed.update(pd.concat(updates, ignore_index=True).set_index("token"))
    score_df = indexed.reset_index()
    score_df = _finalize_scores(score_df, WeightedMetricIndex)
    score_df = score_df.drop(columns=["ego_simulated_states"])

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    score_df.to_csv(args.output_path, index=False)
    print(
        f"wrote {len(score_df)} official NAVSIM v2 one-stage rows to {args.output_path}; "
        f"extended_comfort_available={int(score_df['two_frame_extended_comfort'].notna().sum())}"
    )


if __name__ == "__main__":
    main()
