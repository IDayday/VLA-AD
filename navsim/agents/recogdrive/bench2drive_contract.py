"""Closest-public Bench2Drive data contract used by the ReCogDrive baseline.

The public ReCogDrive repository does not contain its Bench2Drive planner
training code.  This module therefore keeps the decisions that can be derived
from the released Bench2Drive trajectory JSONL in one auditable place.  It is
deliberately limited to the baseline contract; research-specific reward or
Pareto logic does not belong here.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


BENCH2DRIVE_CONTRACT_ID = "recogdrive_b2d_closest_public_multiview_10hz_6x0p5s_v1"

BENCH2DRIVE_CAMERA_ORDER = (
    "rgb_front",
    "rgb_front_left",
    "rgb_front_right",
    "rgb_back_left",
    "rgb_back_right",
    "rgb_back",
)

BENCH2DRIVE_CAMERA_LABELS = (
    "FRONT VIEW",
    "FRONT LEFT VIEW",
    "FRONT RIGHT VIEW",
    "BACK LEFT VIEW",
    "BACK RIGHT VIEW",
    "BACK VIEW",
)

BENCH2DRIVE_SYSTEM_MESSAGE = (
    "You are a vehicle trajectory prediction model for autonomous driving. "
    "Your task is to predict the ego vehicle's 4-second trajectory based on "
    "images from front camera, ego vehicle states, and discrete navigation "
    "commands. Predictions will be evaluated under the Bench2Drive closed-loop "
    "protocol covering 220 short routes across 44 interactive scenarios. "
    "Evaluation metrics include Driving Score, Success Rate, Multi-Ability "
    "Scores, Driving Efficiency, and Driving Smoothness as defined in the "
    "Bench2Drive evaluation toolkit. Ensure your predicted trajectories adhere "
    "to the fair and comprehensive assessment standards of Bench2Drive."
)

BENCH2DRIVE_COMMAND_NAMES = {
    0: "VOID",
    1: "TURN LEFT",
    2: "TURN RIGHT",
    3: "GO STRAIGHT",
    4: "LANE FOLLOW",
    5: "CHANGE LANE LEFT",
    6: "CHANGE LANE RIGHT",
}


def bench2drive_command_name(command: Any) -> str:
    """Return the command spelling used by the released trajectory JSONL."""

    value = getattr(command, "value", command)
    if isinstance(value, np.ndarray):
        value = value.item()
    try:
        command_id = int(value)
    except (TypeError, ValueError):
        return "VOID"
    return BENCH2DRIVE_COMMAND_NAMES.get(command_id, "VOID")


def build_bench2drive_stage1_question(
    *,
    speed: float,
    acceleration_xy: Sequence[float],
    command: Any,
) -> str:
    """Reproduce the human prompt in the released Bench2Drive Traj JSONL."""

    if len(acceleration_xy) < 2:
        raise ValueError("acceleration_xy must contain at least two values")
    speed_value = float(speed)
    acceleration_x = float(acceleration_xy[0])
    acceleration_y = float(acceleration_xy[1])
    if not np.isfinite([speed_value, acceleration_x, acceleration_y]).all():
        raise ValueError("speed and acceleration must be finite")

    image_prefix = "\n".join(
        f"<{label}>:\n<image>" for label in BENCH2DRIVE_CAMERA_LABELS
    )
    return (
        f"{image_prefix}\n"
        "As an autonomous driving system, predict the vehicle's trajectory based on:\n"
        "1. Visual perception from front camera\n"
        f"2. Ego speed: {speed_value:.2f} m/s, acceleration: "
        f"({acceleration_x:.2f}, {acceleration_y:.2f}) m/s^2\n"
        f"3. Active navigation command: [{bench2drive_command_name(command)}]\n"
        "Output requirements:\n"
        "- Predict 6 future trajectory points as [x,y,delta] pairs\n"
        "- Combined format [6,3] with (x:float, y:float, delta:float)\n"
        "- Maintain numerical precision to 2 decimal places"
    )


def bench2drive_camera_paths(clip_dir: Path, frame_index: int) -> list[Path]:
    frame_name = f"{int(frame_index):05d}.jpg"
    return [clip_dir / "camera" / camera / frame_name for camera in BENCH2DRIVE_CAMERA_ORDER]


def world2lidar_matrix(annotation: Mapping[str, Any]) -> np.ndarray:
    """Extract the raw 4x4 world-to-LIDAR transform from an annotation."""

    try:
        value = annotation["sensors"]["LIDAR_TOP"]["world2lidar"]
    except (KeyError, TypeError) as exc:
        raise KeyError("annotation is missing sensors.LIDAR_TOP.world2lidar") from exc
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"world2lidar must have shape (4, 4), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("world2lidar contains non-finite values")
    return matrix


def relative_lidar_poses(
    current_annotation: Mapping[str, Any],
    target_annotations: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Express target LIDAR poses in the current LIDAR frame.

    The output convention is ``[forward, lateral, relative_heading]``.  It is
    the native action convention expected by ReCogDrive's published
    ``norm_odo`` bounds.  The released textual Stage1 targets swap the first
    two axes and add a pi/2 heading offset; those textual coordinates must not
    be fed directly to the diffusion planner.
    """

    current_world2lidar = world2lidar_matrix(current_annotation)
    poses = []
    for target in target_annotations:
        target_world2lidar = world2lidar_matrix(target)
        current_from_target = current_world2lidar @ np.linalg.inv(target_world2lidar)
        heading = math.atan2(float(current_from_target[1, 0]), float(current_from_target[0, 0]))
        poses.append(
            [
                float(current_from_target[0, 3]),
                float(current_from_target[1, 3]),
                heading,
            ]
        )
    result = np.asarray(poses, dtype=np.float32)
    if result.shape != (len(target_annotations), 3):
        raise RuntimeError(f"unexpected relative pose shape: {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("relative LIDAR poses contain non-finite values")
    return result


def planner_to_stage1_text_trajectory(trajectory: np.ndarray) -> np.ndarray:
    """Convert planner-native poses to the released JSONL's textual axes."""

    poses = np.asarray(trajectory, dtype=np.float32)
    if poses.ndim != 2 or poses.shape[-1] != 3:
        raise ValueError(f"trajectory must have shape (N, 3), got {poses.shape}")
    result = np.empty_like(poses)
    result[:, 0] = poses[:, 1]
    result[:, 1] = poses[:, 0]
    result[:, 2] = math.pi / 2.0 - poses[:, 2]
    return result
