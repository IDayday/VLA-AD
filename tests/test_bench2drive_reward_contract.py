from dataclasses import replace
import json
from pathlib import Path

import pytest

from navsim.agents.recogdrive.bench2drive_reward_contract import (
    BENCH2DRIVE_REWARD_CONTRACT_ID,
    COMFORT_LIMITS,
    DEFAULT_COMFORT_OFFICIAL_WEIGHT,
    DEFAULT_HARD_DRIVABLE_MINIMUM,
    DEFAULT_SCALAR_WEIGHTS,
    DEFAULT_STATIONARY_PROGRESS_TOLERANCE_M,
    DEFAULT_TTC_CRITICAL_SECONDS,
    DEFAULT_TTC_SATURATION_SECONDS,
    INFRACTION_FACTORS,
    Bench2DriveInfractionEvents,
    Bench2DriveRewardSignals,
    compose_bench2drive_reward,
    official_infraction_penalty,
    scalar_grpo_baseline,
    score_comfort_segments,
)


def _signals(**overrides):
    base = Bench2DriveRewardSignals(
        collision_free=True,
        minimum_ttc_seconds=3.0,
        drivable_area_compliance=1.0,
        route_corridor_compliance=1.0,
        traffic_rule_compliance=1.0,
        command_following=1.0,
        scenario_task_compliance=1.0,
        candidate_progress_m=10.0,
        reference_progress_m=10.0,
        ego_mean_speed_mps=8.0,
        surrounding_mean_speed_mps=8.0,
        comfort_pass_ratio=1.0,
        comfort_margin_score=0.8,
    )
    return replace(base, **overrides)


def test_official_infraction_penalty_matches_b2d_multiplication():
    events = Bench2DriveInfractionEvents(
        collision_vehicle=1,
        red_light=1,
        stop_sign=1,
        outside_route_lanes_percentage=10.0,
    )
    assert official_infraction_penalty(events) == pytest.approx(0.6 * 0.7 * 0.8 * 0.9)


def test_comfort_keeps_official_pass_ratio_and_dense_margin_separate():
    result = score_comfort_segments(
        longitudinal_acceleration_min=[-1.0, -1.0],
        longitudinal_acceleration_max=[1.0, 1.0],
        absolute_lateral_acceleration_max=[1.0, 5.0],
        absolute_yaw_rate_max=[0.1, 0.1],
        absolute_yaw_acceleration_max=[0.2, 0.2],
        absolute_longitudinal_jerk_max=[0.5, 0.5],
        absolute_magnitude_jerk_max=[1.0, 1.0],
    )
    assert result.num_segments == 2
    assert result.official_pass_ratio == pytest.approx(0.5)
    assert 0.0 < result.margin_score < 1.0
    assert result.reward_score == pytest.approx(
        0.7 * result.official_pass_ratio + 0.3 * result.margin_score
    )


def test_safety_efficiency_comfort_conflict_remains_in_reward_vector():
    cautious = compose_bench2drive_reward(
        _signals(
            minimum_ttc_seconds=3.0,
            candidate_progress_m=4.0,
            ego_mean_speed_mps=4.0,
            comfort_pass_ratio=1.0,
            comfort_margin_score=1.0,
        )
    )
    aggressive = compose_bench2drive_reward(
        _signals(
            minimum_ttc_seconds=1.5,
            candidate_progress_m=10.0,
            ego_mean_speed_mps=8.0,
            comfort_pass_ratio=0.0,
            comfort_margin_score=0.2,
        )
    )

    assert cautious.safety > aggressive.safety
    assert cautious.comfort > aggressive.comfort
    assert cautious.efficiency < aggressive.efficiency
    assert not all(a >= b for a, b in zip(cautious.pareto_vector, aggressive.pareto_vector))
    assert not all(a >= b for a, b in zip(aggressive.pareto_vector, cautious.pareto_vector))


def test_scalar_baseline_never_lets_infeasible_candidate_beat_feasible_one():
    feasible = compose_bench2drive_reward(_signals())
    collision = compose_bench2drive_reward(
        _signals(
            collision_free=False,
            minimum_ttc_seconds=0.0,
            official_infraction_penalty=0.6,
        )
    )
    assert feasible.hard_feasible
    assert not collision.hard_feasible
    assert scalar_grpo_baseline(feasible) > scalar_grpo_baseline(collision)

    official_infraction = compose_bench2drive_reward(
        _signals(
            infraction_free_for_success=False,
            official_infraction_penalty=0.8,
        )
    )
    assert not official_infraction.hard_feasible
    assert scalar_grpo_baseline(feasible) > scalar_grpo_baseline(official_infraction)


def test_stationary_reference_is_not_mistaken_for_inefficiency():
    matched_stop = compose_bench2drive_reward(
        _signals(
            candidate_progress_m=0.0,
            reference_progress_m=0.0,
            ego_mean_speed_mps=0.0,
            surrounding_mean_speed_mps=None,
        )
    )
    moving_through_stop = compose_bench2drive_reward(
        _signals(
            candidate_progress_m=2.0,
            reference_progress_m=0.0,
            ego_mean_speed_mps=2.0,
            surrounding_mean_speed_mps=None,
        )
    )
    assert matched_stop.progress_score == 1.0
    assert matched_stop.speed_score == 1.0
    assert matched_stop.efficiency == 1.0
    assert moving_through_stop.progress_score == 0.0
    assert moving_through_stop.efficiency == 0.0


def test_contract_config_cannot_label_pdms_as_official_b2d_metric():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "bench2drive_recogdrive_stage3_reward_contract_v1.json"
    )
    config = json.loads(config_path.read_text())
    assert config["contract_id"] == BENCH2DRIVE_REWARD_CONTRACT_ID
    assert config["official_bench2drive_evaluation"]["pdms_is_official"] is False
    assert "PDMS" not in config["official_bench2drive_evaluation"]["primary_metrics"]
    assert config["baseline_isolation"]["modify_active_stage1_stage2_pipeline"] is False
    reward = config["reward_vector"]
    assert reward["safety"]["ttc_critical_seconds"] == DEFAULT_TTC_CRITICAL_SECONDS
    assert reward["safety"]["ttc_saturation_seconds"] == DEFAULT_TTC_SATURATION_SECONDS
    assert (
        reward["efficiency"]["stationary_progress_tolerance_m"]
        == DEFAULT_STATIONARY_PROGRESS_TOLERANCE_M
    )
    assert reward["comfort"]["official_weight"] == DEFAULT_COMFORT_OFFICIAL_WEIGHT
    assert config["hard_feasibility"]["hard_drivable_minimum"] == DEFAULT_HARD_DRIVABLE_MINIMUM
    assert tuple(config["plain_grpo_scalar_baseline"]["weights"].values()) == DEFAULT_SCALAR_WEIGHTS
    assert config["official_infraction_penalty"]["collision_vehicle"] == INFRACTION_FACTORS[
        "collision_vehicle"
    ]
    assert reward["comfort"]["limits"]["absolute_yaw_rate_radps"] == COMFORT_LIMITS[
        "absolute_yaw_rate_max"
    ]


def test_invalid_normalized_signal_is_rejected():
    with pytest.raises(ValueError, match="drivable_area_compliance"):
        compose_bench2drive_reward(_signals(drivable_area_compliance=1.01))
