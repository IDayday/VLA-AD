from dataclasses import replace
import json
from pathlib import Path

import pytest

from navsim.agents.recogdrive.bench2drive_report_reward import (
    BENCH2DRIVE_ABILITY_NAMES,
    BENCH2DRIVE_REPORT_REWARD_ID,
    DEFAULT_REPORT_SCALAR_WEIGHTS,
    EFFICIENCY_MAX_VALID_PERCENT,
    INFEASIBLE_RESIDUAL_SCALE,
    Bench2DriveReportState,
    aggregate_bench2drive_report,
    constrained_dominates,
    normalize_official_efficiency,
    report_aligned_scalar,
    report_objectives,
    score_report_transition,
)
from navsim.agents.recogdrive.bench2drive_reward_contract import (
    Bench2DriveInfractionEvents,
)


def _state(**overrides):
    state = Bench2DriveReportState(
        route_completion=0.5,
        infraction_penalty=1.0,
        infraction_free=True,
        route_finished=False,
        target_reached=False,
        efficiency_percentage_sum=200.0,
        efficiency_check_count=2,
        smooth_segments_passed=2,
        smooth_segment_count=2,
        ability_labels=("Merging",),
    )
    return replace(state, **overrides)


def test_route_state_reconstructs_official_ds_efficiency_and_smoothness():
    events = Bench2DriveInfractionEvents(
        collision_vehicle=1,
        red_light=1,
        outside_route_lanes_percentage=10.0,
    )
    state = Bench2DriveReportState.from_official_events(
        route_completion_percentage=80.0,
        events=events,
        route_finished=True,
        target_reached=False,
        efficiency_percentages=(120.0, 1200.0, 80.0),
        smooth_segment_passes=(True, False),
        ability_labels=("Emergency Brake",),
    )
    expected_penalty = 0.6 * 0.7 * 0.9
    assert state.infraction_penalty == pytest.approx(expected_penalty)
    assert state.driving_score_percentage == pytest.approx(80.0 * expected_penalty)
    assert state.efficiency_percentage == pytest.approx(100.0)
    assert state.smoothness_ratio == pytest.approx(0.5)
    assert not state.infraction_free
    assert not state.success


def test_strict_success_is_the_source_of_sr_and_multi_ability():
    success = _state(
        route_completion=1.0,
        route_finished=True,
        target_reached=True,
    )
    assert success.success
    assert not replace(success, infraction_free=False).success
    assert not replace(success, route_finished=False).success


def test_efficiency_transform_is_linear_and_preserves_official_aggregation():
    score_100 = normalize_official_efficiency(100.0)
    score_250 = normalize_official_efficiency(250.0)
    score_1000 = normalize_official_efficiency(1000.0)
    assert score_100 == pytest.approx(0.1)
    assert score_250 == pytest.approx(0.25)
    assert score_1000 == pytest.approx(1.0)
    assert (score_100 + score_250) / 2 == pytest.approx(
        normalize_official_efficiency((100.0 + 250.0) / 2)
    )
    with pytest.raises(ValueError, match="1000%"):
        normalize_official_efficiency(EFFICIENCY_MAX_VALID_PERCENT + 0.1)


def test_pareto_vector_uses_only_non_redundant_report_quantities():
    state = _state(
        route_completion=0.8,
        infraction_penalty=0.7,
        infraction_free=False,
        efficiency_percentage_sum=500.0,
        efficiency_check_count=2,
        smooth_segments_passed=1,
        smooth_segment_count=2,
    )
    objectives = report_objectives(state)
    assert objectives.pareto_vector == pytest.approx(
        (
            0.7,
            0.8,
            normalize_official_efficiency(250.0),
            0.5,
        )
    )
    assert objectives.driving_score_normalized == pytest.approx(0.56)
    assert not objectives.strict_infraction_free
    assert not objectives.sr_feasible


def test_report_aligned_scalar_matches_penalty_times_quality_formula():
    objectives = report_objectives(_state())
    expected = (
        objectives.infraction_compliance
        * (
            2.0 * objectives.route_completion
            + objectives.driving_efficiency
            + objectives.driving_smoothness
        )
        / 4.0
    )
    assert report_aligned_scalar(objectives) == pytest.approx(expected)


def test_sr_constraint_dominance_and_scalar_barrier_cannot_be_compensated():
    feasible = report_objectives(
        _state(
            route_completion=0.1,
            efficiency_percentage_sum=0.0,
            efficiency_check_count=0,
            smooth_segments_passed=0,
            smooth_segment_count=0,
        )
    )
    infeasible = report_objectives(
        _state(
            route_completion=1.0,
            infraction_penalty=0.99,
            infraction_free=False,
            route_finished=True,
            target_reached=True,
            efficiency_percentage_sum=1000.0,
            efficiency_check_count=1,
            smooth_segments_passed=1,
            smooth_segment_count=1,
        )
    )
    assert constrained_dominates(feasible, infeasible)
    assert not constrained_dominates(infeasible, feasible)
    assert report_aligned_scalar(feasible) >= 0.0
    assert -1.0 <= report_aligned_scalar(infeasible) <= -0.75

    incomplete_terminal = report_objectives(
        _state(route_finished=True, route_completion=0.9)
    )
    assert incomplete_terminal.strict_infraction_free
    assert not incomplete_terminal.sr_feasible
    assert report_aligned_scalar(incomplete_terminal) < 0.0

    weaker_infeasible = replace(
        infeasible,
        route_completion=0.5,
        driving_score_normalized=0.495,
    )
    assert constrained_dominates(infeasible, weaker_infeasible)
    assert report_aligned_scalar(infeasible) > report_aligned_scalar(weaker_infeasible)


def test_safety_progress_efficiency_smoothness_conflict_is_not_collapsed():
    cautious = report_objectives(
        _state(
            route_completion=0.45,
            infraction_penalty=1.0,
            efficiency_percentage_sum=100.0,
            efficiency_check_count=1,
            smooth_segments_passed=1,
            smooth_segment_count=1,
        )
    )
    aggressive = report_objectives(
        _state(
            route_completion=0.9,
            infraction_penalty=0.6,
            infraction_free=False,
            efficiency_percentage_sum=250.0,
            efficiency_check_count=1,
            smooth_segments_passed=0,
            smooth_segment_count=1,
        )
    )
    assert cautious.infraction_compliance > aggressive.infraction_compliance
    assert cautious.driving_smoothness > aggressive.driving_smoothness
    assert cautious.route_completion < aggressive.route_completion
    assert cautious.driving_efficiency < aggressive.driving_efficiency
    assert not all(a >= b for a, b in zip(cautious.pareto_vector, aggressive.pareto_vector))
    assert not all(a >= b for a, b in zip(aggressive.pareto_vector, cautious.pareto_vector))


def test_transition_potential_deltas_telescope_and_enforce_invariants():
    start = _state(
        route_completion=0.1,
        efficiency_percentage_sum=0.0,
        efficiency_check_count=0,
        smooth_segments_passed=0,
        smooth_segment_count=0,
    )
    middle = _state(
        route_completion=0.4,
        efficiency_percentage_sum=100.0,
        efficiency_check_count=1,
        smooth_segments_passed=1,
        smooth_segment_count=1,
    )
    end = _state(
        route_completion=0.8,
        infraction_penalty=0.7,
        infraction_free=False,
        route_finished=True,
        efficiency_percentage_sum=250.0,
        efficiency_check_count=2,
        smooth_segments_passed=1,
        smooth_segment_count=2,
    )
    first = score_report_transition(start, middle)
    second = score_report_transition(middle, end)
    assert first.driving_score_delta + second.driving_score_delta == pytest.approx(
        end.driving_score_normalized - start.driving_score_normalized
    )
    assert first.scalar_delta + second.scalar_delta == pytest.approx(
        report_aligned_scalar(report_objectives(end))
        - report_aligned_scalar(report_objectives(start))
    )
    assert second.pareto_vector == report_objectives(end).pareto_vector

    violated = replace(end, route_finished=False)
    with pytest.raises(ValueError, match="cannot recover"):
        score_report_transition(violated, replace(violated, infraction_penalty=1.0))


def test_aggregate_matches_official_denominators_and_reports_coverage():
    all_abilities = tuple(BENCH2DRIVE_ABILITY_NAMES)
    perfect = Bench2DriveReportState(
        route_completion=1.0,
        infraction_penalty=1.0,
        infraction_free=True,
        route_finished=True,
        target_reached=True,
        efficiency_percentage_sum=100.0,
        efficiency_check_count=1,
        smooth_segments_passed=1,
        smooth_segment_count=1,
        ability_labels=all_abilities,
        traffic_sign_junction_success=True,
    )
    failed = Bench2DriveReportState(
        route_completion=0.5,
        infraction_penalty=0.6,
        infraction_free=False,
        route_finished=True,
        target_reached=False,
        efficiency_percentage_sum=200.0,
        efficiency_check_count=1,
        smooth_segments_passed=0,
        smooth_segment_count=1,
        ability_labels=all_abilities,
        traffic_sign_junction_success=False,
    )
    report = aggregate_bench2drive_report((perfect, failed), total_routes=3)
    assert report.driving_score == pytest.approx((100.0 + 30.0) / 3.0)
    assert report.success_rate == pytest.approx(100.0 / 3.0)
    assert report.driving_efficiency == pytest.approx(150.0)
    assert report.driving_smoothness == pytest.approx(50.0)
    assert report.efficiency_route_coverage == pytest.approx(200.0 / 3.0)
    assert report.smoothness_route_coverage == pytest.approx(200.0 / 3.0)
    assert report.missing_routes == 1
    assert report.ability_counts["Merging"] == 2
    assert report.ability_counts["Traffic Signs"] == 4
    assert report.ability_scores["Traffic Signs"] == pytest.approx(50.0)
    assert report.ability_mean == pytest.approx(50.0)


def test_missing_efficiency_and_smoothness_are_zero_for_optimization_not_hidden():
    state = _state(
        efficiency_percentage_sum=0.0,
        efficiency_check_count=0,
        smooth_segments_passed=0,
        smooth_segment_count=0,
    )
    objectives = report_objectives(state)
    assert objectives.driving_efficiency == 0.0
    assert objectives.driving_smoothness == 0.0
    assert not objectives.efficiency_valid
    assert not objectives.smoothness_valid


def test_v2_config_and_code_constants_are_locked_together():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "bench2drive_recogdrive_stage3_report_reward_v2.json"
    )
    config = json.loads(config_path.read_text())
    assert config["contract_id"] == BENCH2DRIVE_REPORT_REWARD_ID
    assert tuple(config["pareto_objectives"]["order"]) == (
        "infraction_compliance",
        "route_completion",
        "driving_efficiency",
        "driving_smoothness",
    )
    assert tuple(config["report_aligned_scalar_baseline"]["weights"].values()) == (
        DEFAULT_REPORT_SCALAR_WEIGHTS
    )
    assert (
        config["report_aligned_scalar_baseline"]["infeasible_residual_scale"]
        == INFEASIBLE_RESIDUAL_SCALE
    )
    assert config["implementation"]["active_trainer_wired"] is False
