from __future__ import annotations

from collections import deque
from typing import Dict, Tuple

import numpy as np


class PID:
    def __init__(self, k_p: float = 1.0, k_i: float = 0.0, k_d: float = 0.0, n: int = 20):
        self._k_p = k_p
        self._k_i = k_i
        self._k_d = k_d
        self._window = deque([0.0 for _ in range(n)], maxlen=n)

    def step(self, error: float) -> float:
        self._window.append(float(error))
        if len(self._window) >= 2:
            integral = float(np.mean(self._window))
            derivative = float(self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0
        return self._k_p * error + self._k_i * integral + self._k_d * derivative


class PIDController:
    """Bench2DriveZoo trajectory PID controller.

    The gains and waypoint-to-speed convention match Thinklab-SJTU
    Bench2DriveZoo team_code/pid_controller.py.
    """

    def __init__(
        self,
        turn_kp: float = 0.75,
        turn_ki: float = 0.75,
        turn_kd: float = 0.3,
        turn_n: int = 40,
        speed_kp: float = 5.0,
        speed_ki: float = 0.5,
        speed_kd: float = 1.0,
        speed_n: int = 40,
        max_throttle: float = 0.75,
        brake_speed: float = 0.4,
        brake_ratio: float = 1.1,
        clip_delta: float = 0.25,
        aim_dist: float = 4.0,
        angle_thresh: float = 0.3,
        dist_thresh: float = 10.0,
    ):
        self.turn_controller = PID(k_p=turn_kp, k_i=turn_ki, k_d=turn_kd, n=turn_n)
        self.speed_controller = PID(k_p=speed_kp, k_i=speed_ki, k_d=speed_kd, n=speed_n)
        self.max_throttle = max_throttle
        self.brake_speed = brake_speed
        self.brake_ratio = brake_ratio
        self.clip_delta = clip_delta
        self.aim_dist = aim_dist
        self.angle_thresh = angle_thresh
        self.dist_thresh = dist_thresh

    def control_pid(self, waypoints: np.ndarray, speed: float, target: np.ndarray) -> Tuple[float, float, bool, Dict[str, object]]:
        if len(waypoints) < 2:
            raise ValueError(f"PIDController requires at least 2 waypoints, got {len(waypoints)}.")

        waypoints = np.asarray(waypoints, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        speed = float(speed)

        num_pairs = len(waypoints) - 1
        best_norm = 1e5
        desired_speed = 0.0
        aim = waypoints[0]
        for i in range(num_pairs):
            desired_speed += float(np.linalg.norm(waypoints[i + 1] - waypoints[i]) * 2.0 / num_pairs)
            norm = float(np.linalg.norm((waypoints[i + 1] + waypoints[i]) / 2.0))
            if abs(self.aim_dist - best_norm) > abs(self.aim_dist - norm):
                aim = waypoints[i]
                best_norm = norm

        aim_last = waypoints[-1] - waypoints[-2]
        angle = float(np.degrees(np.pi / 2.0 - np.arctan2(aim[1], aim[0])) / 90.0)
        angle_last = float(np.degrees(np.pi / 2.0 - np.arctan2(aim_last[1], aim_last[0])) / 90.0)
        angle_target = float(np.degrees(np.pi / 2.0 - np.arctan2(target[1], target[0])) / 90.0)

        use_target_to_aim = abs(angle_target) < abs(angle)
        use_target_to_aim = use_target_to_aim or (
            abs(angle_target - angle_last) > self.angle_thresh and target[1] < self.dist_thresh
        )
        angle_final = angle_target if use_target_to_aim else angle

        steer = float(np.clip(self.turn_controller.step(angle_final), -1.0, 1.0))
        brake = bool(desired_speed < self.brake_speed or (desired_speed > 1e-6 and speed / desired_speed > self.brake_ratio))
        delta = float(np.clip(desired_speed - speed, 0.0, self.clip_delta))
        throttle = float(np.clip(self.speed_controller.step(delta), 0.0, self.max_throttle))
        if brake:
            throttle = 0.0

        metadata = {
            "speed": speed,
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
            "aim": aim.astype(float).tolist(),
            "target": target.astype(float).tolist(),
            "desired_speed": desired_speed,
            "angle": angle,
            "angle_last": angle_last,
            "angle_target": angle_target,
            "angle_final": angle_final,
            "delta": delta,
        }
        return steer, throttle, brake, metadata
