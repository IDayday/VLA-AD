#!/usr/bin/env python3
"""Score cached ReCogDrive trajectories with official NAVSIM-v2 EPDMS."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


_WORKER_STATE: Dict[str, Any] = {}


def _activate_official_navsim(navsim_root: Path) -> Path:
    root = navsim_root.expanduser().resolve()
    if not (root / "navsim" / "evaluate" / "pdm_score.py").is_file():
        raise FileNotFoundError(f"Not an official NAVSIM v2 source tree: {root}")

    filtered_path = []
    for entry in sys.path:
        candidate = Path(entry or ".").resolve()
        if candidate != root and (candidate / "navsim" / "__init__.py").is_file():
            continue
        filtered_path.append(entry)
    sys.path[:] = filtered_path
    sys.path.insert(0, str(root))
    for module_name in list(sys.modules):
        if module_name == "navsim" or module_name.startswith("navsim."):
            del sys.modules[module_name]
    return root


def _compose_config(config_dir: Path, config_name: str, overrides: Sequence[str]) -> Any:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        config_dir=str(config_dir.expanduser().resolve()),
        version_base=None,
        job_name="recogdrive_navsim_v2_navtest_epdms",
    ):
        return compose(config_name=config_name, overrides=list(overrides))


def _rebase_metric_cache_paths(metric_cache_loader: Any, metric_cache_root: Path) -> None:
    metric_cache_root = metric_cache_root.resolve()
    root_name = metric_cache_root.name
    rebased: Dict[str, Path] = {}
    for token, raw_path in metric_cache_loader.metric_cache_paths.items():
        path = Path(raw_path)
        if path.is_file():
            rebased[str(token)] = path
            continue
        if root_name in path.parts:
            suffix = Path(*path.parts[path.parts.index(root_name) + 1 :])
            candidate = metric_cache_root / suffix
            if candidate.is_file():
                rebased[str(token)] = candidate
                continue
        direct = metric_cache_root / str(token) / "metric_cache.pkl"
        rebased[str(token)] = direct if direct.is_file() else path
    metric_cache_loader.metric_cache_paths = rebased


def _worker_init(
    navsim_root: str,
    config_dir: str,
    config_name: str,
    metric_cache_path: str,
    overrides: Sequence[str],
) -> None:
    _activate_official_navsim(Path(navsim_root))
    from hydra.utils import instantiate
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.geometry.convert import relative_to_absolute_poses
    from navsim.common.dataclasses import Trajectory
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.evaluate.pdm_score import pdm_score

    cfg = _compose_config(Path(config_dir), config_name, overrides)
    simulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    traffic_policy = instantiate(cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling)
    cache_root = Path(metric_cache_path).expanduser().resolve()
    cache_loader = MetricCacheLoader(cache_root)
    _rebase_metric_cache_paths(cache_loader, cache_root)
    _WORKER_STATE.update(
        {
            "Trajectory": Trajectory,
            "StateSE2": StateSE2,
            "relative_to_absolute_poses": relative_to_absolute_poses,
            "pdm_score": pdm_score,
            "simulator": simulator,
            "scorer": scorer,
            "traffic_policy": traffic_policy,
            "cache_loader": cache_loader,
        }
    )


def _score_one(item: Tuple[str, Sequence[Sequence[float]]]) -> Dict[str, Any]:
    token, raw_poses = item
    try:
        poses = np.asarray(raw_poses, dtype=np.float64)
        trajectory = _WORKER_STATE["Trajectory"](poses=poses)
        metric_cache = _WORKER_STATE["cache_loader"].get_from_token(token)
        try:
            detections_tracks = metric_cache.observation.detections_tracks
        except AttributeError:
            detections_tracks = None
        cached_observation_fallback = detections_tracks is None
        if cached_observation_fallback:
            score_row, simulated_states = _score_with_cached_observation(
                metric_cache, trajectory
            )
        else:
            score_row, simulated_states = _WORKER_STATE["pdm_score"](
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=_WORKER_STATE["simulator"].proposal_sampling,
                simulator=_WORKER_STATE["simulator"],
                scorer=_WORKER_STATE["scorer"],
                traffic_agents_policy=_WORKER_STATE["traffic_policy"],
            )
        record = score_row.iloc[0].to_dict()
        endpoint = _WORKER_STATE["StateSE2"](*poses[-1].tolist())
        absolute_endpoint = _WORKER_STATE["relative_to_absolute_poses"](
            metric_cache.ego_state.rear_axle, [endpoint]
        )[0]
        scene_type = metric_cache.scene_type
        record.update(
            {
                "token": token,
                "valid": True,
                "log_name": str(metric_cache.log_name),
                "frame_type_name": getattr(scene_type, "name", str(scene_type)).upper(),
                "start_time": float(metric_cache.timepoint.time_s),
                "endpoint_x": float(absolute_endpoint.x),
                "endpoint_y": float(absolute_endpoint.y),
                "start_point_x": float(metric_cache.ego_state.rear_axle.x),
                "start_point_y": float(metric_cache.ego_state.rear_axle.y),
                "start_heading": float(metric_cache.ego_state.rear_axle.heading),
                "ego_simulated_states": np.asarray(simulated_states),
                "cached_observation_fallback": cached_observation_fallback,
            }
        )
        return record
    except Exception as exc:
        return {
            "token": token,
            "valid": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def _score_with_cached_observation(
    metric_cache: Any, model_trajectory: Any
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Run the official scorer against a materialized cached observation.

    The published pruned navtrain cache intentionally omits replay tracks after
    materializing the equivalent PDM occupancy observation. NAVSIM's official
    scorer accepts those cached occupancy maps when ``None`` is passed for the
    optional simulated-agent update.
    """
    from navsim.common.enums import SceneFrameType
    from navsim.evaluate.pdm_score import (
        get_trajectory_as_array,
        transform_trajectory,
    )
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
        WeightedMetricIndex,
    )

    simulator = _WORKER_STATE["simulator"]
    scorer = _WORKER_STATE["scorer"]
    future_sampling = simulator.proposal_sampling
    initial_ego_state = metric_cache.ego_state
    pdm_states = get_trajectory_as_array(
        metric_cache.trajectory,
        future_sampling,
        initial_ego_state.time_point,
    )
    transformed = transform_trajectory(model_trajectory, initial_ego_state)
    predicted_states = get_trajectory_as_array(
        transformed,
        future_sampling,
        initial_ego_state.time_point,
    )
    trajectory_states = np.stack((pdm_states, predicted_states))
    simulated_states = simulator.simulate_proposals(
        trajectory_states, initial_ego_state
    )
    result = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        metric_cache.map_parameters,
        None,
        metric_cache.past_human_trajectory,
    )[1]

    if (
        scorer._config.human_penalty_filter
        and metric_cache.scene_type == SceneFrameType.ORIGINAL
    ):
        human_trajectory = transform_trajectory(
            metric_cache.human_trajectory, initial_ego_state
        )
        human_states = get_trajectory_as_array(
            human_trajectory,
            future_sampling,
            initial_ego_state.time_point,
        )
        human_simulated = simulator.simulate_proposals(
            human_states[None], initial_ego_state
        )
        human_result = scorer.score_proposals(
            human_simulated,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
            metric_cache.map_parameters,
            None,
        )[0]
        skip_columns = {
            "multiplicative_metrics_prod",
            "weighted_metrics",
            "weighted_metrics_array",
            "pdm_score",
        }
        modified = False
        for column in human_result.columns:
            if (
                column not in skip_columns
                and human_result[column].iloc[0] == 0
            ):
                result.at[0, column] = 1
                modified = True
        if modified:
            result.at[0, "multiplicative_metrics_prod"] = (
                result.at[0, "no_at_fault_collisions"]
                * result.at[0, "drivable_area_compliance"]
                * result.at[0, "driving_direction_compliance"]
                * result.at[0, "traffic_light_compliance"]
            )
            weighted = result.at[0, "weighted_metrics"].copy()
            weighted[WeightedMetricIndex.PROGRESS] = result.at[
                0, "ego_progress"
            ]
            weighted[WeightedMetricIndex.TTC] = result.at[
                0, "time_to_collision_within_bound"
            ]
            weighted[WeightedMetricIndex.LANE_KEEPING] = result.at[
                0, "lane_keeping"
            ]
            weighted[WeightedMetricIndex.HISTORY_COMFORT] = result.at[
                0, "history_comfort"
            ]
            result.at[0, "weighted_metrics"] = weighted
    return result, np.asarray(simulated_states[1])


def _score_chunk(
    items: Sequence[Tuple[str, Sequence[Sequence[float]]]],
) -> List[Dict[str, Any]]:
    """Score a small batch while keeping the executor submission queue bounded."""
    return [_score_one(item) for item in items]


def _prediction_paths(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    direct = sorted(root.glob("shard_*/predictions.json"))
    if direct:
        return direct
    return sorted(root.rglob("predictions.json"))


def _load_predictions(root: Path, max_samples: int) -> List[Tuple[str, Sequence[Sequence[float]]]]:
    paths = _prediction_paths(root)
    if not paths:
        raise FileNotFoundError(f"No predictions.json files found under {root}")
    predictions: Dict[str, Sequence[Sequence[float]]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"Expected a prediction list in {path}")
        for row in payload:
            token = str(row.get("sample_token") or "")
            poses = row.get("pred_traj")
            if not token or poses is None:
                raise ValueError(f"Malformed prediction row in {path}")
            array = np.asarray(poses, dtype=np.float64)
            if array.shape != (8, 3) or not np.isfinite(array).all():
                raise ValueError(f"Prediction {token} has invalid trajectory shape/values: {array.shape}")
            if token in predictions:
                raise ValueError(f"Duplicate prediction token: {token}")
            predictions[token] = poses
    ordered = sorted(predictions.items())
    if max_samples > 0:
        ordered = ordered[:max_samples]
    return ordered


def _infer_adjacent_mapping(rows: pd.DataFrame, time_gap_s: float) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    original = rows[rows["frame_type_name"].str.contains("ORIGINAL", na=False)]
    for _, group in original.groupby("log_name"):
        ordered = group.sort_values("start_time").reset_index(drop=True)
        for index in range(1, len(ordered)):
            current = ordered.iloc[index]
            previous = ordered.iloc[index - 1]
            delta = float(current["start_time"]) - float(previous["start_time"])
            if 0.0 < delta <= time_gap_s:
                mapping[str(current["token"])] = str(previous["token"])
    return mapping


def _finalize_epdms(rows: pd.DataFrame, navsim_root: Path, time_gap_s: float) -> pd.DataFrame:
    _activate_official_navsim(navsim_root)
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
        ego_is_two_frame_extended_comfort,
    )
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex

    output = rows.copy()
    output["two_frame_extended_comfort"] = np.nan
    indexed = output.set_index("token")
    mapping = _infer_adjacent_mapping(output, time_gap_s)
    interval_length = float(_WORKER_STATE["proposal_sampling"].interval_length)
    grouped_pairs: Dict[int, List[Tuple[str, str]]] = {}
    for current, previous in mapping.items():
        delta = float(indexed.at[current, "start_time"]) - float(
            indexed.at[previous, "start_time"]
        )
        if not 0.0 < delta < time_gap_s:
            raise RuntimeError(f"Invalid temporal interval for {current}: {delta}")
        overlap = int(round(delta / interval_length))
        if overlap <= 0:
            raise RuntimeError(f"Non-positive overlap for {current}: {overlap}")
        grouped_pairs.setdefault(overlap, []).append((current, previous))
    for overlap, pairs in grouped_pairs.items():
        for start in range(0, len(pairs), 4096):
            batch = pairs[start : start + 4096]
            current_states = np.stack(
                [
                    np.asarray(
                        indexed.at[current, "ego_simulated_states"],
                        dtype=np.float64,
                    )[:-overlap]
                    for current, _ in batch
                ]
            )
            previous_states = np.stack(
                [
                    np.asarray(
                        indexed.at[previous, "ego_simulated_states"],
                        dtype=np.float64,
                    )[overlap:]
                    for _, previous in batch
                ]
            )
            time_point_s = (
                np.arange(current_states.shape[1], dtype=np.float64)
                * interval_length
            )
            values = ego_is_two_frame_extended_comfort(
                current_states,
                previous_states,
                time_point_s,
            ).astype(np.float64)
            indexed.loc[
                [current for current, _ in batch],
                "two_frame_extended_comfort",
            ] = values
    output = indexed.reset_index()

    extended = output["two_frame_extended_comfort"].to_numpy(dtype=np.float64)
    weighted_metrics = np.stack(output["weighted_metrics"].to_numpy())
    metric_weights = np.stack(output["weighted_metrics_array"].to_numpy())
    extended_index = WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT
    missing = np.isnan(extended)
    weighted_metrics[missing, extended_index] = 0.0
    metric_weights[missing, extended_index] = 0.0
    weighted_metrics[~missing, extended_index] = extended[~missing]
    denominator = metric_weights.sum(axis=1)
    if np.any(denominator <= 0.0):
        raise RuntimeError("NAVSIM v2 EPDMS weighted denominator is non-positive")
    output["score"] = (
        output["multiplicative_metrics_prod"].to_numpy(dtype=np.float64)
        * (weighted_metrics * metric_weights).sum(axis=1)
        / denominator
    )
    return output.drop(
        columns=["weighted_metrics", "weighted_metrics_array", "multiplicative_metrics_prod"]
    )


def _git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _mean(frame: pd.DataFrame, key: str) -> float | None:
    if key not in frame:
        return None
    value = pd.to_numeric(frame[key], errors="coerce").mean(skipna=True)
    return None if pd.isna(value) else float(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", required=True, type=Path)
    parser.add_argument("--navsim-root", required=True, type=Path)
    parser.add_argument("--metric-cache-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--config-name", default="default_run_pdm_score")
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 32))
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument(
        "--pending-chunks",
        type=int,
        default=0,
        help="Maximum submitted chunks; zero uses four per worker.",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--time-gap-threshold", type=float, default=0.55)
    parser.add_argument("--split", default="navtest")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    navsim_root = args.navsim_root.expanduser().resolve()
    config_dir = args.config_dir or navsim_root / "navsim" / "planning" / "script" / "config" / "pdm_scoring"
    metric_cache_path = args.metric_cache_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = _load_predictions(args.predictions_dir.expanduser().resolve(), args.max_samples)
    if args.workers <= 0 or args.chunk_size <= 0 or args.pending_chunks < 0:
        raise ValueError(
            "workers and chunk-size must be positive; pending-chunks must be non-negative"
        )

    overrides = [
        f"train_test_split={args.split}",
        "scorer=pdm_scorer",
        "scorer.config.human_penalty_filter=true",
        "traffic_agents=non_reactive",
        f"metric_cache_path={metric_cache_path}",
    ]
    overrides.extend(args.override)
    initializer_args = (
        str(navsim_root),
        str(config_dir),
        args.config_name,
        str(metric_cache_path),
        overrides,
    )

    records: List[Dict[str, Any]] = []
    context = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=_worker_init,
        initargs=initializer_args,
    ) as executor:
        chunks = (
            predictions[start : start + args.chunk_size]
            for start in range(0, len(predictions), args.chunk_size)
        )
        pending_limit = args.pending_chunks or max(args.workers * 4, 1)
        pending: set[concurrent.futures.Future] = set()
        exhausted = False
        while pending or not exhausted:
            while len(pending) < pending_limit and not exhausted:
                try:
                    chunk = next(chunks)
                except StopIteration:
                    exhausted = True
                    break
                pending.add(executor.submit(_score_chunk, chunk))
            if not pending:
                break
            done, pending = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            previous_count = len(records)
            for future in done:
                records.extend(future.result())
            if (
                len(records) // 500 != previous_count // 500
                or len(records) == len(predictions)
            ):
                print(f"scored={len(records)}/{len(predictions)}", flush=True)

    failed = [record for record in records if not record.get("valid")]
    if failed:
        (output_dir / "failures.json").write_text(
            json.dumps(failed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if not args.allow_failures:
            raise RuntimeError(
                f"NAVSIM v2 scoring failed for {len(failed)}/{len(records)} trajectories; "
                f"see {output_dir / 'failures.json'}"
            )

    valid_rows = pd.DataFrame([record for record in records if record.get("valid")])
    if valid_rows.empty:
        raise RuntimeError("NAVSIM v2 scoring produced no valid rows")

    # Build the proposal sampling required by two-frame extended comfort in this process.
    _activate_official_navsim(navsim_root)
    from hydra.utils import instantiate

    cfg = _compose_config(config_dir, args.config_name, overrides)
    _WORKER_STATE["proposal_sampling"] = instantiate(cfg.simulator.proposal_sampling)
    scored = _finalize_epdms(valid_rows, navsim_root, args.time_gap_threshold)
    scored = scored.drop(columns=["ego_simulated_states"], errors="ignore")
    scored.to_csv(output_dir / f"navsim_v2_{args.split}_epdms.csv", index=False)

    metric_names = {
        "EPDMS": "score",
        "NC": "no_at_fault_collisions",
        "DAC": "drivable_area_compliance",
        "DDC": "driving_direction_compliance",
        "TLC": "traffic_light_compliance",
        "EP": "ego_progress",
        "TTC": "time_to_collision_within_bound",
        "LK": "lane_keeping",
        "HC": "history_comfort",
        "EC": "two_frame_extended_comfort",
    }
    summary: Dict[str, Any] = {
        "protocol": f"navsim_v2_{args.split}_one_stage_epdms",
        "split": str(args.split),
        "official_navsim_revision": _git_revision(navsim_root),
        "num_predictions": len(predictions),
        "successful": len(scored),
        "failed": len(failed),
        "extended_comfort_available": int(scored["two_frame_extended_comfort"].notna().sum()),
        "human_penalty_filter": True,
        "cached_observation_fallback_count": int(
            scored["cached_observation_fallback"].sum()
        ),
    }
    for label, column in metric_names.items():
        summary[label] = _mean(scored, column)
    summary["score_mean"] = summary["EPDMS"]
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
