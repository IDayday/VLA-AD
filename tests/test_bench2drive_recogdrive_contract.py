from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

from navsim.agents.recogdrive.bench2drive_contract import (
    BENCH2DRIVE_CAMERA_LABELS,
    bench2drive_command_name,
    build_bench2drive_stage1_question,
    planner_to_stage1_text_trajectory,
    relative_lidar_poses,
)


def _annotation(world2lidar):
    return {"sensors": {"LIDAR_TOP": {"world2lidar": world2lidar}}}


def test_released_stage1_question_has_six_ordered_views_and_exact_state_text() -> None:
    question = build_bench2drive_stage1_question(
        speed=3.625,
        acceleration_xy=[6.99, -0.02],
        command=4,
    )
    assert question.count("<image>") == 6
    positions = [question.index(f"<{label}>") for label in BENCH2DRIVE_CAMERA_LABELS]
    assert positions == sorted(positions)
    assert "Ego speed: 3.62 m/s, acceleration: (6.99, -0.02) m/s^2" in question
    assert "Active navigation command: [LANE FOLLOW]" in question
    assert "Predict 6 future trajectory points" in question


def test_released_command_spellings_are_preserved() -> None:
    assert bench2drive_command_name(1) == "TURN LEFT"
    assert bench2drive_command_name(2) == "TURN RIGHT"
    assert bench2drive_command_name(3) == "GO STRAIGHT"
    assert bench2drive_command_name(5) == "CHANGE LANE LEFT"


def test_relative_lidar_pose_uses_planner_native_forward_lateral_axes() -> None:
    current = _annotation(np.eye(4))
    target_from_world = np.eye(4)
    target_from_world[0, 3] = -2.0
    target_from_world[1, 3] = 0.5
    target = _annotation(target_from_world)

    result = relative_lidar_poses(current, [target])
    np.testing.assert_allclose(result[0], [2.0, -0.5, 0.0], atol=1e-6)


def test_stage1_text_axis_conversion_is_not_used_as_planner_action() -> None:
    planner = np.array([[2.0, -0.5, 0.1]], dtype=np.float32)
    text = planner_to_stage1_text_trajectory(planner)
    np.testing.assert_allclose(text[0, :2], [-0.5, 2.0], atol=1e-6)
    assert math.isclose(float(text[0, 2]), math.pi / 2.0 - 0.1, abs_tol=1e-6)


def test_formal_stage2_config_excludes_research_branches() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs/bench2drive_recogdrive_stage2_closest_public_2b.yaml").read_text(encoding="utf-8")
    )
    assert config["contract_id"] == "recogdrive_b2d_closest_public_multiview_10hz_6x0p5s_v1"
    assert config["action_horizon"] == 6
    assert config["use_expert_features"] is False
    assert config["use_bit_drive"] is False
    assert config["use_risk_vla"] is False
    assert config["use_last_rd"] is False
    assert config["grpo"] is False
