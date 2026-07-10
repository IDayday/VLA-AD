from __future__ import annotations

import math

import numpy as np

from bench2drive_eval.team_code.recogdrive_b2d_common import (
    build_status_feature,
    command_one_hot,
    relative_poses,
    sample_pose_history,
    trajectory_to_pid_waypoints,
)
from bench2drive_eval.team_code.recogdrive_b2d_pid_controller import PIDController


class DummyRoadOption:
    def __init__(self, value: int) -> None:
        self.value = value


def test_relative_poses_matches_current_origin() -> None:
    poses = np.array(
        [
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
            [13.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    rel = relative_poses(poses[-1], poses)
    np.testing.assert_allclose(rel[-1], np.zeros(3), atol=1e-6)
    np.testing.assert_allclose(rel[:, 0], [-3.0, -2.0, -1.0, 0.0], atol=1e-6)


def test_relative_poses_rotates_by_compass_like_cache_builder() -> None:
    origin = np.array([0.0, 0.0, math.pi / 2], dtype=np.float64)
    poses = np.array([[0.0, 1.0, math.pi / 2]], dtype=np.float64)
    rel = relative_poses(origin, poses)
    np.testing.assert_allclose(rel[0, :2], [1.0, 0.0], atol=1e-6)
    assert abs(float(rel[0, 2])) < 1e-6


def test_command_one_hot_maps_bench2drive_road_options() -> None:
    np.testing.assert_allclose(command_one_hot(DummyRoadOption(1)), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(command_one_hot(DummyRoadOption(2)), [0.0, 0.0, 1.0])
    np.testing.assert_allclose(command_one_hot(DummyRoadOption(4)), [0.0, 1.0, 0.0])
    np.testing.assert_allclose(command_one_hot(6), [0.0, 0.0, 1.0])


def test_sample_pose_history_pads_with_earliest_available_pose() -> None:
    poses = [np.array([float(i), 0.0, 0.0]) for i in range(3)]
    sampled = sample_pose_history(poses, history_frames=4, interval_steps=10)
    assert sampled.shape == (4, 3)
    np.testing.assert_allclose(sampled[:3, 0], [0.0, 0.0, 0.0])
    assert sampled[-1, 0] == 2.0


def test_status_feature_and_pid_waypoint_conversion() -> None:
    status = build_status_feature(command=4, speed=3.0, acceleration_xyz=[1.0, 2.0, 9.8], compass=0.0)
    np.testing.assert_allclose(status[:5], [0.0, 1.0, 0.0, 3.0, 0.0])
    assert status.shape == (8,)

    trajectory = np.array([[1.0, 2.0, 0.0], [2.0, 3.0, 0.0]], dtype=np.float32)
    waypoints = trajectory_to_pid_waypoints(trajectory, flip_y=True)
    np.testing.assert_allclose(waypoints, [[1.0, -2.0], [2.0, -3.0]])


def test_pid_controller_returns_bounded_control() -> None:
    waypoints = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]], dtype=np.float32)
    steer, throttle, brake, metadata = PIDController().control_pid(waypoints, 0.0, np.array([10.0, 0.0]))
    assert -1.0 <= steer <= 1.0
    assert 0.0 <= throttle <= 0.75
    assert isinstance(brake, bool)
    assert metadata["desired_speed"] > 0.0
