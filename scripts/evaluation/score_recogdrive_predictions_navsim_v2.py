#!/usr/bin/env python3
"""Score cached ReCogDrive trajectories with NAVSIM v2 navtest EPDMS."""

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
                "ego_simulated_states": np.asarray(simulated_states),
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
    from navsim.planning.simulation.planner.pdm_planner.scoring.scene_aggregator import SceneAggregator
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex

    output = rows.copy()
    output["two_frame_extended_comfort"] = np.nan
    indexed = output.set_index("token")
    updates = []
    for current, previous in _infer_adjacent_mapping(output, time_gap_s).items():
        aggregator = SceneAggregator(
            now_frame=current,
            previous_frame=previous,
            score_df=indexed,
            proposal_sampling=_WORKER_STATE.get("proposal_sampling"),
        )
        updates.append(aggregator.aggregate_scores(one_stage_only=True))

    # SceneAggregator only reads proposal_sampling attributes. Build the official object lazily
    # when this process did not initialize a scorer worker itself.
    if updates:
        update_rows = pd.concat(updates, ignore_index=True).set_index("token")
        indexed.update(update_rows)
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
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--time-gap-threshold", type=float, default=0.55)
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
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    overrides = [
        "train_test_split=navtest",
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
        for index, record in enumerate(executor.map(_score_one, predictions, chunksize=8), 1):
            records.append(record)
            if index % 500 == 0 or index == len(predictions):
                print(f"scored={index}/{len(predictions)}", flush=True)

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
    scored.to_csv(output_dir / "navsim_v2_navtest_epdms.csv", index=False)

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
        "protocol": "navsim_v2_navtest_one_stage_epdms",
        "official_navsim_root": str(navsim_root),
        "official_navsim_revision": _git_revision(navsim_root),
        "metric_cache_path": str(metric_cache_path),
        "num_predictions": len(predictions),
        "successful": len(scored),
        "failed": len(failed),
        "extended_comfort_available": int(scored["two_frame_extended_comfort"].notna().sum()),
        "human_penalty_filter": True,
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
