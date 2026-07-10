from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np


def normalize_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def relative_poses(origin: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Convert global Bench2Drive [x, y, compass] poses to current-ego-relative poses.

    This mirrors scripts/build_bench2drive_recogdrive_chunk_cache.py so closed-loop
    inputs match the cache used for ReCogDrive planner fine-tuning.
    """

    theta = -float(origin[2])
    rotation = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=np.float64,
    )
    rel = poses.astype(np.float64, copy=True)
    rel -= origin.reshape(1, 3)
    rel[:, :2] = rel[:, :2] @ rotation.T
    rel[:, 2] = normalize_angle(rel[:, 2])
    return rel.astype(np.float32)


def local_xy(origin_heading: float, xy: Iterable[float]) -> np.ndarray:
    theta = -float(origin_heading)
    rotation = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=np.float64,
    )
    return np.asarray(list(xy), dtype=np.float64) @ rotation.T


def road_option_value(command: Any) -> int:
    value = getattr(command, "value", command)
    if isinstance(value, np.ndarray):
        value = value.item()
    return int(value)


def command_one_hot(command: Any) -> np.ndarray:
    value = road_option_value(command)
    one_hot = np.zeros(3, dtype=np.float32)
    if value in {1, 5}:  # left / change lane left
        one_hot[0] = 1.0
    elif value in {2, 6}:  # right / change lane right
        one_hot[2] = 1.0
    else:  # lane follow / straight / unknown
        one_hot[1] = 1.0
    return one_hot


def build_status_feature(
    *,
    command: Any,
    speed: float,
    acceleration_xyz: Sequence[float],
    compass: float,
) -> np.ndarray:
    high_command = command_one_hot(command)
    acceleration_xy = local_xy(compass, acceleration_xyz[:2]).astype(np.float32)
    acceleration = np.array([acceleration_xy[0], acceleration_xy[1], 0.0], dtype=np.float32)
    velocity = np.array([float(speed), 0.0], dtype=np.float32)
    return np.concatenate([high_command, velocity, acceleration], axis=0).astype(np.float32)


def sample_pose_history(
    pose_history: Sequence[np.ndarray],
    *,
    history_frames: int = 4,
    interval_steps: int = 10,
) -> np.ndarray:
    if not pose_history:
        raise ValueError("pose_history is empty.")
    if history_frames < 1:
        raise ValueError("history_frames must be positive.")
    if interval_steps < 1:
        raise ValueError("interval_steps must be positive.")

    last = len(pose_history) - 1
    samples = []
    for offset in reversed(range(history_frames)):
        idx = max(0, last - offset * interval_steps)
        samples.append(np.asarray(pose_history[idx], dtype=np.float64))
    return np.stack(samples, axis=0)


def trajectory_to_pid_waypoints(trajectory: np.ndarray, *, flip_y: bool = True) -> np.ndarray:
    waypoints = np.asarray(trajectory, dtype=np.float32)[..., :2].copy()
    if flip_y:
        waypoints[..., 1] *= -1.0
    return waypoints
