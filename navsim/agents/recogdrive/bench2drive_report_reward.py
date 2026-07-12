"""Report-aligned Bench2Drive reward state and aggregation.

NAVSIM can directly optimize PDMS because PDMS is also its final benchmark
metric. Bench2Drive instead reports route-level Driving Score, Success Rate,
Driving Efficiency, Driving Smoothness, and Multi-Ability. This module stores
the sufficient statistics required to reconstruct those official outputs and
exposes their non-redundant decomposition as a Pareto-ready objective vector.

The module does not simulate a candidate trajectory. A Bench2Drive rollout or
cache builder must project each candidate into an ``after`` report state using
the official event, checkpoint, and smoothness semantics. Group-relative
training compares these after-state potentials; sequential training may use
the corresponding potential differences.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from navsim.agents.recogdrive.bench2drive_reward_contract import (
    Bench2DriveInfractionEvents,
    INFRACTION_FACTORS,
    official_infraction_penalty,
)


BENCH2DRIVE_REPORT_REWARD_ID = "recogdrive_b2d_stage3_report_aligned_reward_v2"

BENCH2DRIVE_ABILITY_NAMES = (
    "Merging",
    "Overtaking",
    "Emergency Brake",
    "Give Way",
    "Traffic Signs",
)

EFFICIENCY_MAX_VALID_PERCENT = 1000.0
EXPECTED_EFFICIENCY_CHECKS = 20
DEFAULT_REPORT_SCALAR_WEIGHTS = (2.0, 1.0, 1.0)  # RC, Efficiency, Smoothness
INFEASIBLE_RESIDUAL_SCALE = 0.25


def _unit_interval(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}")
    return value


def _nonnegative_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return value


def _nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


def _strict_bool(name: str, value: bool) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean, got {value!r}")
    return bool(value)


def official_infraction_free(
    events: Bench2DriveInfractionEvents,
    *,
    route_deviation: bool = False,
    vehicle_blocked: bool = False,
    route_timeout: bool = False,
) -> bool:
    """Return the strict no-infraction condition used by B2D route success.

    Minimum-speed events are intentionally absent: Bench2Drive V0.0.3 ignores
    them for route success/ability and reports them separately as Efficiency.
    Route deviation, vehicle blocking, and route timeout do not all have a
    multiplicative DS factor, but each invalidates strict route success.
    """

    flags = (
        _strict_bool("route_deviation", route_deviation),
        _strict_bool("vehicle_blocked", vehicle_blocked),
        _strict_bool("route_timeout", route_timeout),
    )
    has_multiplicative_event = any(
        _nonnegative_int(name, getattr(events, name)) > 0 for name in INFRACTION_FACTORS
    )
    outside_percentage = float(events.outside_route_lanes_percentage)
    if not math.isfinite(outside_percentage) or not 0.0 <= outside_percentage <= 100.0:
        raise ValueError(
            "outside_route_lanes_percentage must be finite and in [0, 100], "
            f"got {outside_percentage!r}"
        )
    return not has_multiplicative_event and outside_percentage == 0.0 and not any(flags)


def normalize_official_efficiency(efficiency_percentage: float) -> float:
    """Linearly map the official B2D Efficiency percentage to ``[0, 1]``.

    B2D permits values above 100% and filters individual values above 1000%.
    Dividing by that public cutoff is an affine transform: unlike clipping or
    a logarithm, it also preserves arithmetic means and therefore preserves
    model ordering under the official route aggregation. The raw percentage
    remains the value used for final reporting.
    """

    percentage = _nonnegative_finite("efficiency_percentage", efficiency_percentage)
    if percentage > EFFICIENCY_MAX_VALID_PERCENT:
        raise ValueError(
            "efficiency_percentage exceeds the official 1000% validity cutoff: "
            f"{percentage!r}"
        )
    return percentage / EFFICIENCY_MAX_VALID_PERCENT


@dataclass(frozen=True)
class Bench2DriveReportState:
    """Sufficient statistics for one route at an intermediate or terminal state."""

    route_completion: float
    infraction_penalty: float
    infraction_free: bool
    route_finished: bool
    target_reached: bool
    efficiency_percentage_sum: float = 0.0
    efficiency_check_count: int = 0
    smooth_segments_passed: int = 0
    smooth_segment_count: int = 0
    ability_labels: Tuple[str, ...] = ()
    traffic_sign_junction_success: Optional[bool] = None

    def __post_init__(self) -> None:
        _unit_interval("route_completion", self.route_completion)
        _unit_interval("infraction_penalty", self.infraction_penalty)
        _strict_bool("infraction_free", self.infraction_free)
        _strict_bool("route_finished", self.route_finished)
        target_reached = _strict_bool("target_reached", self.target_reached)

        efficiency_sum = _nonnegative_finite(
            "efficiency_percentage_sum", self.efficiency_percentage_sum
        )
        efficiency_count = _nonnegative_int(
            "efficiency_check_count", self.efficiency_check_count
        )
        if efficiency_count == 0 and efficiency_sum != 0.0:
            raise ValueError("efficiency_percentage_sum must be zero when there are no checks")
        if efficiency_sum > efficiency_count * EFFICIENCY_MAX_VALID_PERCENT + 1e-9:
            raise ValueError("efficiency sum implies a checkpoint value above the 1000% cutoff")

        smooth_passed = _nonnegative_int(
            "smooth_segments_passed", self.smooth_segments_passed
        )
        smooth_count = _nonnegative_int("smooth_segment_count", self.smooth_segment_count)
        if smooth_passed > smooth_count:
            raise ValueError("smooth_segments_passed cannot exceed smooth_segment_count")

        labels = tuple(self.ability_labels)
        if len(labels) != len(set(labels)):
            raise ValueError(f"ability_labels contains duplicates: {labels!r}")
        unknown = sorted(set(labels) - set(BENCH2DRIVE_ABILITY_NAMES))
        if unknown:
            raise ValueError(f"unknown Bench2Drive ability labels: {unknown}")
        object.__setattr__(self, "ability_labels", labels)

        traffic_sign_success = self.traffic_sign_junction_success
        if traffic_sign_success is not None:
            _strict_bool("traffic_sign_junction_success", traffic_sign_success)
            if "Traffic Signs" not in labels:
                raise ValueError(
                    "traffic_sign_junction_success requires the Traffic Signs ability label"
                )

        if target_reached and self.route_completion < 1.0 - 1e-6:
            raise ValueError("target_reached requires route_completion == 1")

    @classmethod
    def from_official_events(
        cls,
        *,
        route_completion_percentage: float,
        events: Bench2DriveInfractionEvents,
        route_finished: bool,
        target_reached: bool,
        efficiency_percentages: Sequence[float] = (),
        smooth_segment_passes: Sequence[bool] = (),
        ability_labels: Sequence[str] = (),
        traffic_sign_junction_success: Optional[bool] = None,
        route_deviation: bool = False,
        vehicle_blocked: bool = False,
        route_timeout: bool = False,
    ) -> "Bench2DriveReportState":
        """Build a state from official event/checkpoint/segment observations."""

        route_completion_percentage = float(route_completion_percentage)
        if not math.isfinite(route_completion_percentage) or not (
            0.0 <= route_completion_percentage <= 100.0
        ):
            raise ValueError("route_completion_percentage must be finite and in [0, 100]")

        efficiency_values = []
        for value in efficiency_percentages:
            value = _nonnegative_finite("efficiency checkpoint percentage", value)
            if value > EFFICIENCY_MAX_VALID_PERCENT:
                continue
            efficiency_values.append(value)

        smooth_values = [
            _strict_bool("smooth segment pass", value) for value in smooth_segment_passes
        ]
        return cls(
            route_completion=route_completion_percentage / 100.0,
            infraction_penalty=official_infraction_penalty(events),
            infraction_free=official_infraction_free(
                events,
                route_deviation=route_deviation,
                vehicle_blocked=vehicle_blocked,
                route_timeout=route_timeout,
            ),
            route_finished=route_finished,
            target_reached=target_reached,
            efficiency_percentage_sum=sum(efficiency_values),
            efficiency_check_count=len(efficiency_values),
            smooth_segments_passed=sum(smooth_values),
            smooth_segment_count=len(smooth_values),
            ability_labels=tuple(ability_labels),
            traffic_sign_junction_success=traffic_sign_junction_success,
        )

    @property
    def driving_score_normalized(self) -> float:
        """Exact per-route contribution to official DS, in ``[0, 1]``."""

        return self.route_completion * self.infraction_penalty

    @property
    def driving_score_percentage(self) -> float:
        return 100.0 * self.driving_score_normalized

    @property
    def success(self) -> bool:
        """Strict terminal route success used by SR and Multi-Ability."""

        return bool(
            self.route_finished
            and self.target_reached
            and self.route_completion >= 1.0 - 1e-6
            and self.infraction_free
        )

    @property
    def efficiency_percentage(self) -> Optional[float]:
        """Official route Efficiency mean, or ``None`` before the first check."""

        if self.efficiency_check_count == 0:
            return None
        return self.efficiency_percentage_sum / self.efficiency_check_count

    @property
    def smoothness_ratio(self) -> Optional[float]:
        """Official route Smoothness ratio, or ``None`` without a full segment."""

        if self.smooth_segment_count == 0:
            return None
        return self.smooth_segments_passed / self.smooth_segment_count


@dataclass(frozen=True)
class Bench2DriveReportObjectives:
    """Non-redundant objectives that reconstruct B2D report metrics."""

    infraction_compliance: float
    route_completion: float
    driving_efficiency: float
    driving_smoothness: float
    strict_infraction_free: bool
    sr_feasible: bool
    efficiency_valid: bool
    smoothness_valid: bool
    driving_score_normalized: float
    terminal_success: bool

    @property
    def pareto_vector(self) -> Tuple[float, float, float, float]:
        """Return ``(infraction, completion, efficiency, smoothness)``."""

        return (
            self.infraction_compliance,
            self.route_completion,
            self.driving_efficiency,
            self.driving_smoothness,
        )


def report_objectives(state: Bench2DriveReportState) -> Bench2DriveReportObjectives:
    """Project a route state into report-aligned Pareto objective potentials."""

    efficiency_percentage = state.efficiency_percentage
    efficiency_valid = efficiency_percentage is not None
    efficiency = (
        normalize_official_efficiency(efficiency_percentage)
        if efficiency_percentage is not None
        else 0.0
    )
    smoothness_ratio = state.smoothness_ratio
    smoothness_valid = smoothness_ratio is not None
    smoothness = smoothness_ratio if smoothness_ratio is not None else 0.0
    terminal_success = state.success
    return Bench2DriveReportObjectives(
        infraction_compliance=state.infraction_penalty,
        route_completion=state.route_completion,
        driving_efficiency=efficiency,
        driving_smoothness=smoothness,
        strict_infraction_free=state.infraction_free,
        sr_feasible=(
            state.infraction_free and (not state.route_finished or terminal_success)
        ),
        efficiency_valid=efficiency_valid,
        smoothness_valid=smoothness_valid,
        driving_score_normalized=state.driving_score_normalized,
        terminal_success=terminal_success,
    )


def constrained_dominates(
    candidate: Bench2DriveReportObjectives,
    other: Bench2DriveReportObjectives,
    *,
    tolerance: float = 1e-12,
) -> bool:
    """Return whether ``candidate`` dominates under the SR feasibility rule.

    Bench2Drive route success requires zero official infractions (minimum-speed
    observations excluded) and successful terminal completion. A candidate
    that preserves that possibility must therefore dominate one that
    irreversibly loses it. If both candidates have the same feasibility status,
    ordinary Pareto dominance is applied to the four report-aligned objectives.
    This is a reward contract primitive, not a Pareto-GRPO optimizer.
    """

    tolerance = _nonnegative_finite("tolerance", tolerance)
    if candidate.sr_feasible != other.sr_feasible:
        return candidate.sr_feasible
    candidate_vector = np.asarray(candidate.pareto_vector, dtype=np.float64)
    other_vector = np.asarray(other.pareto_vector, dtype=np.float64)
    no_worse = np.all(candidate_vector >= other_vector - tolerance)
    strictly_better = np.any(candidate_vector > other_vector + tolerance)
    return bool(no_worse and strictly_better)


def report_aligned_scalar(
    objectives: Bench2DriveReportObjectives,
    *,
    weights: Sequence[float] = DEFAULT_REPORT_SCALAR_WEIGHTS,
) -> float:
    """Return the B2D Report-Aligned Score (B2D-RAS) scalar baseline.

    The base utility
    ``P * weighted_mean(RC, Efficiency, Smoothness)`` mirrors the
    penalty-times-quality structure of PDMS while using only quantities tied
    to B2D's final report. Route completion has weight 2 because DS/SR are the
    primary B2D metrics; Efficiency and Smoothness each have weight 1.

    SR-feasible states keep the base utility in ``[0, 1]``. Infeasible states
    map to ``[-1, -0.75]`` while retaining a small residual utility, so every
    SR-feasible candidate outranks every infeasible candidate and an
    all-infeasible group still has learning signal. B2D-RAS is a research
    reward, not an official Bench2Drive metric.
    """

    weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.shape != (3,) or not np.isfinite(weight_array).all():
        raise ValueError("weights must contain three finite values")
    if (weight_array < 0.0).any() or weight_array.sum() <= 0.0:
        raise ValueError("weights must be non-negative and have a positive sum")
    quality = np.dot(
        weight_array,
        np.asarray(
            [
                objectives.route_completion,
                objectives.driving_efficiency,
                objectives.driving_smoothness,
            ],
            dtype=np.float64,
        ),
    ) / weight_array.sum()
    base_utility = float(objectives.infraction_compliance * quality)
    if objectives.sr_feasible:
        return base_utility
    return -1.0 + INFEASIBLE_RESIDUAL_SCALE * base_utility


@dataclass(frozen=True)
class Bench2DriveRewardTransition:
    """Candidate after-state potential and telescoping reward differences."""

    after_objectives: Bench2DriveReportObjectives
    objective_delta: Tuple[float, float, float, float]
    driving_score_delta: float
    scalar_after: float
    scalar_delta: float

    @property
    def pareto_vector(self) -> Tuple[float, float, float, float]:
        """Use after-state potentials for group-relative candidate comparison."""

        return self.after_objectives.pareto_vector


def score_report_transition(
    before: Bench2DriveReportState,
    after: Bench2DriveReportState,
) -> Bench2DriveRewardTransition:
    """Score a candidate transition while enforcing accumulator invariants."""

    if before.route_finished:
        raise ValueError("cannot transition from a finished route")
    if after.route_completion + 1e-9 < before.route_completion:
        raise ValueError("route completion cannot decrease")
    if after.infraction_penalty > before.infraction_penalty + 1e-9:
        raise ValueError("infraction penalty cannot recover after an event")
    if not before.infraction_free and after.infraction_free:
        raise ValueError("strict infraction-free state cannot recover")
    if after.efficiency_check_count < before.efficiency_check_count:
        raise ValueError("efficiency checkpoint count cannot decrease")
    if after.efficiency_percentage_sum + 1e-9 < before.efficiency_percentage_sum:
        raise ValueError("efficiency checkpoint sum cannot decrease")
    if after.smooth_segment_count < before.smooth_segment_count:
        raise ValueError("smoothness segment count cannot decrease")
    if after.smooth_segments_passed < before.smooth_segments_passed:
        raise ValueError("smoothness pass count cannot decrease")
    if before.ability_labels != after.ability_labels:
        raise ValueError("ability labels cannot change within a route")

    before_objectives = report_objectives(before)
    after_objectives = report_objectives(after)
    objective_delta = tuple(
        after_value - before_value
        for before_value, after_value in zip(
            before_objectives.pareto_vector, after_objectives.pareto_vector
        )
    )
    scalar_before = report_aligned_scalar(before_objectives)
    scalar_after = report_aligned_scalar(after_objectives)
    return Bench2DriveRewardTransition(
        after_objectives=after_objectives,
        objective_delta=objective_delta,
        driving_score_delta=(
            after.driving_score_normalized - before.driving_score_normalized
        ),
        scalar_after=scalar_after,
        scalar_delta=scalar_after - scalar_before,
    )


@dataclass(frozen=True)
class Bench2DriveAggregateReport:
    """Official-format aggregate metrics and denominator diagnostics."""

    total_routes: int
    evaluated_routes: int
    missing_routes: int
    driving_score: float
    success_rate: float
    driving_efficiency: Optional[float]
    efficiency_route_coverage: float
    driving_smoothness: Optional[float]
    smoothness_route_coverage: float
    ability_scores: Mapping[str, Optional[float]]
    ability_counts: Mapping[str, int]
    ability_mean: Optional[float]


def aggregate_bench2drive_report(
    states: Sequence[Bench2DriveReportState],
    *,
    total_routes: int = 220,
) -> Bench2DriveAggregateReport:
    """Aggregate route states following official Bench2Drive denominators.

    Missing routes contribute zero to DS and SR. Efficiency and Smoothness are
    averaged only over routes with valid observations, matching the official
    tools; coverage is returned explicitly to reveal omission-based reward
    hacking. Multi-Ability follows the official route-success aggregation. For
    Traffic Signs, the optional junction outcome is an additional contribution,
    matching the current official ability script.
    """

    total_routes = _nonnegative_int("total_routes", total_routes)
    if total_routes == 0:
        raise ValueError("total_routes must be positive")
    states = tuple(states)
    if len(states) > total_routes:
        raise ValueError("number of route states exceeds total_routes")

    driving_score = sum(state.driving_score_percentage for state in states) / total_routes
    success_rate = 100.0 * sum(state.success for state in states) / total_routes

    efficiency_values = [
        state.efficiency_percentage
        for state in states
        if state.efficiency_percentage is not None
    ]
    driving_efficiency = (
        float(np.mean(efficiency_values)) if efficiency_values else None
    )
    efficiency_coverage = 100.0 * len(efficiency_values) / total_routes

    smoothness_values = [
        state.smoothness_ratio
        for state in states
        if state.smoothness_ratio is not None
    ]
    driving_smoothness = (
        100.0 * float(np.mean(smoothness_values)) if smoothness_values else None
    )
    smoothness_coverage = 100.0 * len(smoothness_values) / total_routes

    ability_successes: Dict[str, list[bool]] = {
        ability: [] for ability in BENCH2DRIVE_ABILITY_NAMES
    }
    for state in states:
        for ability in state.ability_labels:
            ability_successes[ability].append(state.success)
        if state.traffic_sign_junction_success is not None:
            ability_successes["Traffic Signs"].append(
                bool(state.traffic_sign_junction_success)
            )

    ability_scores: Dict[str, Optional[float]] = {}
    ability_counts: Dict[str, int] = {}
    for ability, successes in ability_successes.items():
        ability_counts[ability] = len(successes)
        ability_scores[ability] = (
            100.0 * sum(successes) / len(successes) if successes else None
        )
    valid_ability_scores = [
        score for score in ability_scores.values() if score is not None
    ]
    ability_mean = (
        float(np.mean(valid_ability_scores))
        if len(valid_ability_scores) == len(BENCH2DRIVE_ABILITY_NAMES)
        else None
    )

    return Bench2DriveAggregateReport(
        total_routes=total_routes,
        evaluated_routes=len(states),
        missing_routes=total_routes - len(states),
        driving_score=driving_score,
        success_rate=success_rate,
        driving_efficiency=driving_efficiency,
        efficiency_route_coverage=efficiency_coverage,
        driving_smoothness=driving_smoothness,
        smoothness_route_coverage=smoothness_coverage,
        ability_scores=ability_scores,
        ability_counts=ability_counts,
        ability_mean=ability_mean,
    )
