from __future__ import annotations

from dataclasses import asdict
from typing import List, Sequence, Union

import numpy as np
import numpy.typing as npt

from navsim.common.dataclasses import PDMResults, Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_state_to_state_array
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


TrajectoryLike = Union[Trajectory, npt.NDArray[np.floating]]


def _coerce_trajectory(model_trajectory: TrajectoryLike) -> Trajectory:
    if isinstance(model_trajectory, Trajectory):
        return model_trajectory
    return Trajectory(np.asarray(model_trajectory, dtype=np.float64))


def _as_trajectory_list(model_trajectories: Union[TrajectoryLike, Sequence[TrajectoryLike]]) -> List[Trajectory]:
    if isinstance(model_trajectories, Trajectory):
        return [model_trajectories]
    if isinstance(model_trajectories, np.ndarray):
        if model_trajectories.ndim == 2:
            return [_coerce_trajectory(model_trajectories)]
        if model_trajectories.ndim == 3:
            return [_coerce_trajectory(trajectory) for trajectory in model_trajectories]
        raise ValueError(f"Expected trajectory array with shape [H, 3] or [N, H, 3], got {model_trajectories.shape}.")
    return [_coerce_trajectory(trajectory) for trajectory in model_trajectories]


def _principal_value(angle: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _trajectory_arrays_to_state_array_fast(
    metric_cache: MetricCache,
    model_trajectories: npt.NDArray[np.floating],
    future_sampling: TrajectorySampling,
    source_sampling: TrajectorySampling,
) -> npt.NDArray[np.float64]:
    """Vectorized equivalent of transform_trajectory + get_trajectory_as_array for SE2 reference states."""
    rel = np.asarray(model_trajectories, dtype=np.float64)
    if rel.ndim != 3 or rel.shape[-1] != 3:
        raise ValueError(f"Expected trajectories with shape [N, H, 3], got {rel.shape}.")

    initial_ego_state = metric_cache.ego_state
    origin = initial_ego_state.rear_axle
    origin_xy = np.asarray([origin.x, origin.y], dtype=np.float64)
    origin_heading = float(origin.heading)
    cos_h = np.cos(origin_heading)
    sin_h = np.sin(origin_heading)

    local_xy = rel[..., :2]
    global_xy = np.empty_like(local_xy, dtype=np.float64)
    global_xy[..., 0] = origin_xy[0] + cos_h * local_xy[..., 0] - sin_h * local_xy[..., 1]
    global_xy[..., 1] = origin_xy[1] + sin_h * local_xy[..., 0] + cos_h * local_xy[..., 1]
    global_heading = _principal_value(origin_heading + rel[..., 2])

    initial_state = ego_state_to_state_array(initial_ego_state)
    key_xy = np.concatenate(
        [np.broadcast_to(initial_state[StateIndex.POINT], (rel.shape[0], 1, 2)), global_xy],
        axis=1,
    )
    key_heading = np.concatenate(
        [np.full((rel.shape[0], 1), initial_state[StateIndex.HEADING], dtype=np.float64), global_heading],
        axis=1,
    )
    key_heading = np.unwrap(key_heading, axis=1)

    key_times = np.concatenate(
        [
            np.asarray([0.0], dtype=np.float64),
            np.arange(1, rel.shape[1] + 1, dtype=np.float64) * source_sampling.interval_length,
        ]
    )
    out_times = (
        np.arange(0, future_sampling.num_poses + 1, dtype=np.float64) * future_sampling.interval_length
    )
    out_times = np.clip(out_times, key_times[0], key_times[-1])
    left = np.searchsorted(key_times, out_times, side="right") - 1
    left = np.clip(left, 0, len(key_times) - 2)
    right = left + 1
    denom = np.maximum(key_times[right] - key_times[left], 1e-12)
    alpha = ((out_times - key_times[left]) / denom).astype(np.float64)

    interp_xy = key_xy[:, left] * (1.0 - alpha)[None, :, None] + key_xy[:, right] * alpha[None, :, None]
    interp_heading = key_heading[:, left] * (1.0 - alpha)[None, :] + key_heading[:, right] * alpha[None, :]
    interp_heading = _principal_value(interp_heading)

    states = np.zeros((rel.shape[0], future_sampling.num_poses + 1, StateIndex.size()), dtype=np.float64)
    states[:, :, StateIndex.POINT] = interp_xy
    states[:, :, StateIndex.HEADING] = interp_heading
    states[:, 0, :] = initial_state
    return states


def _extract_pairwise_scalar_equivalent_results(scorer: PDMScorer) -> List[PDMResults]:
    """Extract batched metrics but preserve scalar pdm_score progress normalization semantics."""
    num_proposals = int(scorer._num_proposals)
    if num_proposals < 2:
        return []

    multi_metrics = scorer._multi_metrics
    weighted_metrics = scorer._weighted_metrics
    progress_raw = scorer._progress_raw
    if multi_metrics is None or weighted_metrics is None or progress_raw is None:
        raise RuntimeError("PDMScorer did not populate metric arrays.")

    multiplicative = multi_metrics.prod(axis=0)
    raw_progress = progress_raw * multiplicative
    baseline_progress = float(raw_progress[0])
    pred_progress = raw_progress[1:]
    pred_multiplicative = multiplicative[1:]

    pairwise_max_progress = np.maximum(baseline_progress, pred_progress)
    normalized_progress = np.ones_like(pred_progress, dtype=np.float64)
    progress_mask = pairwise_max_progress > scorer._config.progress_distance_threshold
    normalized_progress[progress_mask] = pred_progress[progress_mask] / pairwise_max_progress[progress_mask]
    normalized_progress[~progress_mask & (pred_multiplicative == 0.0)] = 0.0

    weights = scorer._config.weighted_metrics_array
    weighted_sum = (
        normalized_progress * weights[WeightedMetricIndex.PROGRESS]
        + weighted_metrics[WeightedMetricIndex.TTC, 1:] * weights[WeightedMetricIndex.TTC]
        + weighted_metrics[WeightedMetricIndex.COMFORTABLE, 1:] * weights[WeightedMetricIndex.COMFORTABLE]
        + weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION, 1:] * weights[WeightedMetricIndex.DRIVING_DIRECTION]
    ) / weights.sum()
    scores = pred_multiplicative * weighted_sum

    results: List[PDMResults] = []
    for pred_idx in range(1, num_proposals):
        out_idx = pred_idx - 1
        results.append(
            PDMResults(
                no_at_fault_collisions=float(multi_metrics[MultiMetricIndex.NO_COLLISION, pred_idx]),
                drivable_area_compliance=float(multi_metrics[MultiMetricIndex.DRIVABLE_AREA, pred_idx]),
                ego_progress=float(normalized_progress[out_idx]),
                time_to_collision_within_bound=float(weighted_metrics[WeightedMetricIndex.TTC, pred_idx]),
                comfort=float(weighted_metrics[WeightedMetricIndex.COMFORTABLE, pred_idx]),
                driving_direction_compliance=float(weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION, pred_idx]),
                score=float(scores[out_idx]),
            )
        )
    return results


def pdm_score_batch_same_cache_fast(
    metric_cache: MetricCache,
    model_trajectories: npt.NDArray[np.floating],
    future_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    source_sampling: TrajectorySampling = Trajectory.trajectory_sampling,
) -> List[PDMResults]:
    """
    Faster array-only variant for multiple candidates from one token.

    This bypasses NuPlan trajectory objects for candidate reference states, while keeping the same
    simulator/scorer and scalar-equivalent final score aggregation. Validate against ``pdm_score``
    before using it in a training/evaluation path.
    """
    trajectories = np.asarray(model_trajectories, dtype=np.float64)
    if trajectories.ndim == 2:
        trajectories = trajectories[None]
    if trajectories.shape[0] == 0:
        return []

    initial_ego_state = metric_cache.ego_state
    pdm_states = get_trajectory_as_array(metric_cache.trajectory, future_sampling, initial_ego_state.time_point)
    pred_states = _trajectory_arrays_to_state_array_fast(
        metric_cache,
        trajectories,
        future_sampling,
        source_sampling,
    )

    trajectory_states = np.concatenate([pdm_states[None, ...], pred_states], axis=0)
    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)
    scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
    )
    return _extract_pairwise_scalar_equivalent_results(scorer)


def pdm_score_batch_same_cache(
    metric_cache: MetricCache,
    model_trajectories: Union[TrajectoryLike, Sequence[TrajectoryLike]],
    future_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
) -> List[PDMResults]:
    """
    Score multiple candidate trajectories for one token/metric_cache in one simulator/scorer pass.

    The returned values preserve the scalar ``pdm_score`` semantics: each candidate's final score is
    aggregated as if it were scored against only the PDM baseline trajectory. This matters because the
    original scorer normalizes progress by the max progress among proposals in the current call.
    """
    trajectories = _as_trajectory_list(model_trajectories)
    if not trajectories:
        return []

    initial_ego_state = metric_cache.ego_state
    pdm_states = get_trajectory_as_array(metric_cache.trajectory, future_sampling, initial_ego_state.time_point)

    pred_states = []
    for trajectory in trajectories:
        pred_trajectory = transform_trajectory(trajectory, initial_ego_state)
        pred_states.append(get_trajectory_as_array(pred_trajectory, future_sampling, initial_ego_state.time_point))

    trajectory_states = np.concatenate([pdm_states[None, ...], np.stack(pred_states, axis=0)], axis=0)
    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)
    scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
    )
    return _extract_pairwise_scalar_equivalent_results(scorer)


def pdm_score_batch_same_cache_as_dicts(
    metric_cache: MetricCache,
    model_trajectories: Union[TrajectoryLike, Sequence[TrajectoryLike]],
    future_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
) -> List[dict]:
    return [
        asdict(result)
        for result in pdm_score_batch_same_cache(metric_cache, model_trajectories, future_sampling, simulator, scorer)
    ]
