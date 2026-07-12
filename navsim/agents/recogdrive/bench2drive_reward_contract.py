"""Bench2Drive-aligned reward components for a future Stage3 research run.

This module is intentionally separate from the public ReCogDrive/NAVSIM PDMS
path.  Bench2Drive does not define PDMS as an official metric.  The functions
below compose pre-computed Bench2Drive event and motion signals into:

* an auditable four-objective vector for later multi-objective optimization;
* a conventional scalar baseline for plain GRPO; and
* diagnostics that can be compared with the official closed-loop evaluator.

Geometry, actor replay, traffic-light queries, and controller simulation do not
belong here.  A cache builder must compute those signals from Bench2Drive data
using the same controller contract as closed-loop deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import numpy as np


BENCH2DRIVE_REWARD_CONTRACT_ID = "recogdrive_b2d_stage3_multi_objective_reward_v1"
DEFAULT_TTC_CRITICAL_SECONDS = 1.0
DEFAULT_TTC_SATURATION_SECONDS = 3.0
DEFAULT_STATIONARY_PROGRESS_TOLERANCE_M = 0.5
DEFAULT_HARD_DRIVABLE_MINIMUM = 0.99
DEFAULT_COMFORT_OFFICIAL_WEIGHT = 0.70
DEFAULT_SCALAR_WEIGHTS = (0.25, 0.25, 0.25, 0.25)

# Bench2Drive V0.0.3 infraction factors. Minimum-speed events are deliberately
# absent because the official toolkit excludes them from Driving Score.
INFRACTION_FACTORS = {
    "collision_pedestrian": 0.50,
    "collision_vehicle": 0.60,
    "collision_static": 0.65,
    "red_light": 0.70,
    "stop_sign": 0.80,
    "scenario_timeout": 0.70,
    "yield_emergency_vehicle": 0.70,
}

# Official Bench2Drive Driving Smoothness limits.
COMFORT_LIMITS = {
    "longitudinal_acceleration_min": -4.05,
    "longitudinal_acceleration_max": 2.40,
    "absolute_lateral_acceleration_max": 4.89,
    "absolute_yaw_rate_max": 0.95,
    "absolute_yaw_acceleration_max": 1.93,
    "absolute_longitudinal_jerk_max": 4.13,
    "absolute_magnitude_jerk_max": 8.37,
}


def _validate_unit_interval(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}")
    return value


def _validate_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return value


def _validate_bool(name: str, value: bool) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean, got {value!r}")
    return bool(value)


@dataclass(frozen=True)
class Bench2DriveInfractionEvents:
    """Events used by the official Bench2Drive Driving Score penalty."""

    collision_pedestrian: int = 0
    collision_vehicle: int = 0
    collision_static: int = 0
    red_light: int = 0
    stop_sign: int = 0
    scenario_timeout: int = 0
    yield_emergency_vehicle: int = 0
    outside_route_lanes_percentage: float = 0.0


def official_infraction_penalty(events: Bench2DriveInfractionEvents) -> float:
    """Return the V0.0.3 multiplicative infraction penalty in ``[0, 1]``.

    Route deviation and vehicle blocking terminate a route but do not have a
    separate multiplicative factor in the official statistics manager.  They
    must therefore be represented by route completion and the feasibility flag
    in :class:`Bench2DriveRewardSignals`, not inserted into this calculation.
    """

    penalty = 1.0
    for name, factor in INFRACTION_FACTORS.items():
        count = getattr(events, name)
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {count!r}")
        penalty *= factor ** int(count)

    outside_percentage = float(events.outside_route_lanes_percentage)
    if not math.isfinite(outside_percentage) or not 0.0 <= outside_percentage <= 100.0:
        raise ValueError(
            "outside_route_lanes_percentage must be finite and in [0, 100], "
            f"got {outside_percentage!r}"
        )
    penalty *= 1.0 - outside_percentage / 100.0
    return float(np.clip(penalty, 0.0, 1.0))


@dataclass(frozen=True)
class Bench2DriveComfortResult:
    """Official-threshold and dense-margin comfort diagnostics."""

    official_pass_ratio: float
    margin_score: float
    reward_score: float
    num_segments: int


def _as_metric_array(name: str, values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D sequence, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def score_comfort_segments(
    *,
    longitudinal_acceleration_min: Sequence[float],
    longitudinal_acceleration_max: Sequence[float],
    absolute_lateral_acceleration_max: Sequence[float],
    absolute_yaw_rate_max: Sequence[float],
    absolute_yaw_acceleration_max: Sequence[float],
    absolute_longitudinal_jerk_max: Sequence[float],
    absolute_magnitude_jerk_max: Sequence[float],
    official_weight: float = DEFAULT_COMFORT_OFFICIAL_WEIGHT,
) -> Bench2DriveComfortResult:
    """Score non-overlapping motion segments using official B2D limits.

    The caller is responsible for extracting the extrema from the simulated
    ego trace with the official two-second segmentation and filtering rules.
    ``official_pass_ratio`` matches the threshold decision: a segment passes
    only when every variable remains strictly inside its bound.  The additional
    ``margin_score`` supplies dense training signal inside the valid region; it
    is a research proxy and is never reported as official Driving Smoothness.
    """

    arrays = {
        "longitudinal_acceleration_min": _as_metric_array(
            "longitudinal_acceleration_min", longitudinal_acceleration_min
        ),
        "longitudinal_acceleration_max": _as_metric_array(
            "longitudinal_acceleration_max", longitudinal_acceleration_max
        ),
        "absolute_lateral_acceleration_max": _as_metric_array(
            "absolute_lateral_acceleration_max", absolute_lateral_acceleration_max
        ),
        "absolute_yaw_rate_max": _as_metric_array(
            "absolute_yaw_rate_max", absolute_yaw_rate_max
        ),
        "absolute_yaw_acceleration_max": _as_metric_array(
            "absolute_yaw_acceleration_max", absolute_yaw_acceleration_max
        ),
        "absolute_longitudinal_jerk_max": _as_metric_array(
            "absolute_longitudinal_jerk_max", absolute_longitudinal_jerk_max
        ),
        "absolute_magnitude_jerk_max": _as_metric_array(
            "absolute_magnitude_jerk_max", absolute_magnitude_jerk_max
        ),
    }
    lengths = {array.size for array in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"comfort metric arrays must have equal lengths, got {sorted(lengths)}")

    weight = _validate_unit_interval("official_weight", official_weight)
    lon_min = arrays["longitudinal_acceleration_min"]
    lon_max = arrays["longitudinal_acceleration_max"]
    lat = np.abs(arrays["absolute_lateral_acceleration_max"])
    yaw_rate = np.abs(arrays["absolute_yaw_rate_max"])
    yaw_accel = np.abs(arrays["absolute_yaw_acceleration_max"])
    lon_jerk = np.abs(arrays["absolute_longitudinal_jerk_max"])
    magnitude_jerk = np.abs(arrays["absolute_magnitude_jerk_max"])

    passed = (
        (lon_min > COMFORT_LIMITS["longitudinal_acceleration_min"])
        & (lon_max < COMFORT_LIMITS["longitudinal_acceleration_max"])
        & (lat < COMFORT_LIMITS["absolute_lateral_acceleration_max"])
        & (yaw_rate < COMFORT_LIMITS["absolute_yaw_rate_max"])
        & (yaw_accel < COMFORT_LIMITS["absolute_yaw_acceleration_max"])
        & (lon_jerk < COMFORT_LIMITS["absolute_longitudinal_jerk_max"])
        & (magnitude_jerk < COMFORT_LIMITS["absolute_magnitude_jerk_max"])
    )

    braking_margin = 1.0 - np.maximum(-lon_min, 0.0) / abs(
        COMFORT_LIMITS["longitudinal_acceleration_min"]
    )
    acceleration_margin = 1.0 - np.maximum(lon_max, 0.0) / COMFORT_LIMITS[
        "longitudinal_acceleration_max"
    ]
    longitudinal_margin = np.minimum(braking_margin, acceleration_margin)
    per_metric_margins = np.stack(
        [
            longitudinal_margin,
            1.0 - lat / COMFORT_LIMITS["absolute_lateral_acceleration_max"],
            1.0 - yaw_rate / COMFORT_LIMITS["absolute_yaw_rate_max"],
            1.0 - yaw_accel / COMFORT_LIMITS["absolute_yaw_acceleration_max"],
            1.0 - lon_jerk / COMFORT_LIMITS["absolute_longitudinal_jerk_max"],
            1.0 - magnitude_jerk / COMFORT_LIMITS["absolute_magnitude_jerk_max"],
        ],
        axis=-1,
    )
    margin_score = float(np.clip(per_metric_margins, 0.0, 1.0).mean())
    official_pass_ratio = float(passed.mean())
    reward_score = weight * official_pass_ratio + (1.0 - weight) * margin_score
    return Bench2DriveComfortResult(
        official_pass_ratio=official_pass_ratio,
        margin_score=margin_score,
        reward_score=float(np.clip(reward_score, 0.0, 1.0)),
        num_segments=int(lon_min.size),
    )


@dataclass(frozen=True)
class Bench2DriveRewardSignals:
    """Pre-computed signals for one three-second candidate trajectory.

    Unit-interval fields may be fractional when computed over a trajectory
    footprint. ``traffic_rule_compliance`` aggregates red-light, stop-sign and
    emergency-vehicle-yield checks. ``scenario_task_compliance`` is the local
    proxy for the route's B2D ability (merge, overtake, emergency brake, give
    way, or traffic signs). ``route_corridor_compliance`` must use a
    scenario-aware permitted corridor, not a naive lane-center distance.
    """

    collision_free: bool
    minimum_ttc_seconds: float
    drivable_area_compliance: float
    route_corridor_compliance: float
    traffic_rule_compliance: float
    command_following: float
    scenario_task_compliance: float
    candidate_progress_m: float
    reference_progress_m: float
    ego_mean_speed_mps: float
    surrounding_mean_speed_mps: Optional[float]
    comfort_pass_ratio: float
    comfort_margin_score: float
    official_infraction_penalty: float = 1.0
    route_deviation_free: bool = True
    infraction_free_for_success: bool = True
    finite_trajectory: bool = True


@dataclass(frozen=True)
class Bench2DriveRewardComponents:
    """Normalized objective vector plus scalarization diagnostics."""

    safety: float
    compliance: float
    efficiency: float
    comfort: float
    hard_feasible: bool
    ttc_score: float
    progress_score: float
    speed_score: float
    official_infraction_penalty: float

    @property
    def pareto_vector(self) -> Tuple[float, float, float, float]:
        """Return ``(safety, compliance, efficiency, comfort)``."""

        return (self.safety, self.compliance, self.efficiency, self.comfort)


def _ttc_score(minimum_ttc_seconds: float, critical_seconds: float, saturation_seconds: float) -> float:
    ttc = float(minimum_ttc_seconds)
    if math.isnan(ttc) or ttc == -math.inf:
        raise ValueError(f"minimum_ttc_seconds must not be NaN or -inf, got {ttc!r}")
    if ttc < 0.0:
        raise ValueError(f"minimum_ttc_seconds must be non-negative, got {ttc!r}")
    if not 0.0 <= critical_seconds < saturation_seconds:
        raise ValueError("TTC thresholds must satisfy 0 <= critical < saturation")
    if ttc == math.inf:
        return 1.0
    return float(np.clip((ttc - critical_seconds) / (saturation_seconds - critical_seconds), 0.0, 1.0))


def _progress_score(candidate_progress_m: float, reference_progress_m: float, stationary_tolerance_m: float) -> float:
    candidate = _validate_nonnegative("candidate_progress_m", candidate_progress_m)
    reference = _validate_nonnegative("reference_progress_m", reference_progress_m)
    tolerance = _validate_nonnegative("stationary_tolerance_m", stationary_tolerance_m)
    if reference <= tolerance:
        return 1.0 if candidate <= tolerance else 0.0
    return float(np.clip(candidate / reference, 0.0, 1.0))


def _speed_score(ego_mean_speed_mps: float, surrounding_mean_speed_mps: Optional[float]) -> float:
    ego_speed = _validate_nonnegative("ego_mean_speed_mps", ego_mean_speed_mps)
    # The official B2D minimum-speed criterion assigns 100% when no background
    # vehicle speed is available. Preserve that fallback in the local proxy.
    if surrounding_mean_speed_mps is None:
        return 1.0
    surrounding_speed = _validate_nonnegative(
        "surrounding_mean_speed_mps", surrounding_mean_speed_mps
    )
    if surrounding_speed <= 1e-6:
        return 1.0
    return float(np.clip(ego_speed / surrounding_speed, 0.0, 1.0))


def compose_bench2drive_reward(
    signals: Bench2DriveRewardSignals,
    *,
    ttc_critical_seconds: float = DEFAULT_TTC_CRITICAL_SECONDS,
    ttc_saturation_seconds: float = DEFAULT_TTC_SATURATION_SECONDS,
    stationary_progress_tolerance_m: float = DEFAULT_STATIONARY_PROGRESS_TOLERANCE_M,
    hard_drivable_minimum: float = DEFAULT_HARD_DRIVABLE_MINIMUM,
    comfort_official_weight: float = DEFAULT_COMFORT_OFFICIAL_WEIGHT,
) -> Bench2DriveRewardComponents:
    """Compose the B2D-aligned four-objective reward vector.

    Safety is graded by TTC but collision-gated. Compliance is the geometric
    mean of drivable-area, route-corridor, traffic-rule, command-following and
    scenario-task scores. Efficiency is the geometric mean of route progress
    and the capped B2D surrounding-traffic speed ratio. Comfort combines the
    exact threshold pass ratio with a dense distance-to-threshold margin.
    """

    drivable = _validate_unit_interval(
        "drivable_area_compliance", signals.drivable_area_compliance
    )
    route = _validate_unit_interval(
        "route_corridor_compliance", signals.route_corridor_compliance
    )
    traffic = _validate_unit_interval(
        "traffic_rule_compliance", signals.traffic_rule_compliance
    )
    command = _validate_unit_interval("command_following", signals.command_following)
    scenario_task = _validate_unit_interval(
        "scenario_task_compliance", signals.scenario_task_compliance
    )
    comfort_pass = _validate_unit_interval(
        "comfort_pass_ratio", signals.comfort_pass_ratio
    )
    comfort_margin = _validate_unit_interval(
        "comfort_margin_score", signals.comfort_margin_score
    )
    penalty = _validate_unit_interval(
        "official_infraction_penalty", signals.official_infraction_penalty
    )
    hard_drivable = _validate_unit_interval("hard_drivable_minimum", hard_drivable_minimum)
    comfort_weight = _validate_unit_interval(
        "comfort_official_weight", comfort_official_weight
    )

    ttc = _ttc_score(
        signals.minimum_ttc_seconds,
        float(ttc_critical_seconds),
        float(ttc_saturation_seconds),
    )
    progress = _progress_score(
        signals.candidate_progress_m,
        signals.reference_progress_m,
        stationary_progress_tolerance_m,
    )
    speed = _speed_score(signals.ego_mean_speed_mps, signals.surrounding_mean_speed_mps)

    collision_free = _validate_bool("collision_free", signals.collision_free)
    route_deviation_free = _validate_bool(
        "route_deviation_free", signals.route_deviation_free
    )
    infraction_free = _validate_bool(
        "infraction_free_for_success", signals.infraction_free_for_success
    )
    finite_trajectory = _validate_bool("finite_trajectory", signals.finite_trajectory)

    safety = float(collision_free) * ttc
    compliance = float(
        (drivable * route * traffic * command * scenario_task) ** 0.2
    )
    efficiency = float(math.sqrt(progress * speed))
    comfort = comfort_weight * comfort_pass + (1.0 - comfort_weight) * comfort_margin
    hard_feasible = bool(
        finite_trajectory
        and collision_free
        and route_deviation_free
        and infraction_free
        and drivable >= hard_drivable
        and traffic >= 1.0 - 1e-6
    )

    return Bench2DriveRewardComponents(
        safety=float(np.clip(safety, 0.0, 1.0)),
        compliance=float(np.clip(compliance, 0.0, 1.0)),
        efficiency=float(np.clip(efficiency, 0.0, 1.0)),
        comfort=float(np.clip(comfort, 0.0, 1.0)),
        hard_feasible=hard_feasible,
        ttc_score=ttc,
        progress_score=progress,
        speed_score=speed,
        official_infraction_penalty=penalty,
    )


def scalar_grpo_baseline(
    components: Bench2DriveRewardComponents,
    *,
    weights: Sequence[float] = DEFAULT_SCALAR_WEIGHTS,
    infeasible_floor: float = -1.0,
    infeasible_residual_scale: float = 0.25,
) -> float:
    """Return a declared scalar baseline without performing Pareto ranking.

    Every feasible candidate outranks every infeasible one.  The small residual
    term still orders an all-infeasible GRPO group, avoiding a constant-reward
    batch.  This scalar is a research baseline, not DS, SR, PDMS, or any other
    official Bench2Drive metric.
    """

    weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.shape != (4,) or not np.isfinite(weight_array).all():
        raise ValueError("weights must contain four finite values")
    if (weight_array < 0.0).any() or weight_array.sum() <= 0.0:
        raise ValueError("weights must be non-negative and have a positive sum")
    residual_scale = float(infeasible_residual_scale)
    floor = float(infeasible_floor)
    if not math.isfinite(floor):
        raise ValueError("infeasible_floor must be finite")
    if not math.isfinite(residual_scale) or not 0.0 <= residual_scale < 1.0:
        raise ValueError("infeasible_residual_scale must be finite and in [0, 1)")

    weight_array /= weight_array.sum()
    utility = float(np.dot(weight_array, np.asarray(components.pareto_vector)))
    utility *= components.official_infraction_penalty
    if components.hard_feasible:
        return utility
    return floor + residual_scale * utility
