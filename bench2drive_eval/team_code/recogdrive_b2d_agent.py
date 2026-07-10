from __future__ import annotations

import datetime as _datetime
import json
import math
import os
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import requests
import yaml
from PIL import Image

import carla
from leaderboard.autoagents import autonomous_agent

from recogdrive_b2d_common import (
    build_status_feature,
    command_one_hot,
    relative_poses,
    sample_pose_history,
    trajectory_to_pid_waypoints,
)
from recogdrive_b2d_pid_controller import PIDController
from recogdrive_b2d_route_planner import EARTH_RADIUS_EQUA, RoutePlanner


SAVE_PATH = os.environ.get("SAVE_PATH")
IS_BENCH2DRIVE = os.environ.get("IS_BENCH2DRIVE")


def get_entry_point() -> str:
    return "ReCogDriveB2DAgent"


def _load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Bench2Drive ReCogDrive agent config not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise TypeError(f"Agent config must contain a mapping: {config_path}")
    return data


def _cfg_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class ReCogDriveB2DAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file: str) -> None:
        self.track = autonomous_agent.Track.SENSORS
        self.save_path: Optional[Path] = None
        self.metric_info: Dict[int, Dict[str, Any]] = {}
        if "+" in path_to_conf_file:
            config_file, appended = path_to_conf_file.split("+", 1)
            appended_save_name = appended.rsplit("+", 1)[-1]
        else:
            config_file, appended_save_name = path_to_conf_file, None
        self.cfg = _load_config(config_file)
        self.route_save_name = str(appended_save_name or self.cfg.get("save_name") or Path(config_file).stem)
        self.mode = str(self.cfg.get("mode", "remote"))
        if self.mode != "remote":
            raise ValueError(
                "Only mode='remote' is enabled for Bench2Drive closed-loop. "
                "Run scripts/bench2drive/serve_recogdrive_b2d.py in the navsim environment."
            )

        self.server_url = str(self.cfg.get("server_url", "http://127.0.0.1:8765")).rstrip("/")
        self.request_timeout = float(self.cfg.get("request_timeout", 240.0))
        self.history_frames = int(self.cfg.get("history_frames", 4))
        self.history_interval_steps = int(self.cfg.get("history_interval_steps", 10))
        self.inference_interval_steps = int(self.cfg.get("inference_interval_steps", 1))
        self.visual_refresh_interval_steps = max(1, int(self.cfg.get("visual_refresh_interval_steps", 1)))
        self.flip_y_for_pid = bool(self.cfg.get("flip_y_for_pid", True))
        self.max_speed_throttle_cutoff = float(self.cfg.get("max_speed_throttle_cutoff", 5.0))
        self.image_quality = int(self.cfg.get("image_quality", 90))
        self.sensor_profile = str(self.cfg.get("sensor_profile", "front_only")).strip().lower()
        self.save_debug_images = _cfg_bool(self.cfg.get("save_debug_images", False))
        self.save_debug_meta = _cfg_bool(self.cfg.get("save_debug_meta", False))
        self.metric_flush_interval_steps = int(self.cfg.get("metric_flush_interval_steps", 100))
        self.repeat_control_when_not_infer = _cfg_bool(
            self.cfg.get("repeat_control_when_not_infer", self.inference_interval_steps > 1)
        )

        self.pidcontroller = PIDController()
        self.pose_history: deque[np.ndarray] = deque(maxlen=max(1, self.history_frames * self.history_interval_steps * 4))
        self.step = -1
        self.initialized = False
        self.last_trajectory: Optional[np.ndarray] = None
        self.last_control: Optional[carla.VehicleControl] = None
        self.last_pid_metadata: Dict[str, Any] = {}
        self.visual_cache_key = f"{self.route_save_name}_{os.getpid()}_{uuid.uuid4().hex}"

        vla_ad_root = Path(os.environ.get("VLA_AD_ROOT", ".")).resolve()
        default_image_root = vla_ad_root / "outputs" / "bench2drive_recogdrive_closed_loop" / "images"
        self.image_dir = Path(self.cfg.get("image_dir", default_image_root))
        if not self.image_dir.is_absolute():
            self.image_dir = vla_ad_root / self.image_dir
        self.image_dir.mkdir(parents=True, exist_ok=True)

        if SAVE_PATH:
            save_name = self.route_save_name
            if not IS_BENCH2DRIVE:
                now = _datetime.datetime.now()
                save_name = f"{save_name}_{now:%m%d_%H%M%S}"
            self.save_path = Path(SAVE_PATH) / save_name
            self.save_path.mkdir(parents=True, exist_ok=False)
            if self.save_debug_images:
                (self.save_path / "rgb_front").mkdir()
            if self.save_debug_meta:
                (self.save_path / "meta").mkdir()

    def _init(self) -> None:
        try:
            locx = self._global_plan_world_coord[0][0].location.x
            locy = self._global_plan_world_coord[0][0].location.y
            lon = self._global_plan[0][0]["lon"]
            lat = self._global_plan[0][0]["lat"]

            from scipy.optimize import fsolve

            def equations(vars_):
                x, y = vars_
                eq1 = lon * math.cos(x * math.pi / 180.0) - (locx * x * 180.0) / (
                    math.pi * EARTH_RADIUS_EQUA
                ) - math.cos(x * math.pi / 180.0) * y
                eq2 = (
                    math.log(math.tan((lat + 90.0) * math.pi / 360.0))
                    * EARTH_RADIUS_EQUA
                    * math.cos(x * math.pi / 180.0)
                    + locy
                    - math.cos(x * math.pi / 180.0)
                    * EARTH_RADIUS_EQUA
                    * math.log(math.tan((90.0 + x) * math.pi / 360.0))
                )
                return [eq1, eq2]

            solution = fsolve(equations, [0.0, 0.0])
            self.lat_ref, self.lon_ref = float(solution[0]), float(solution[1])
        except Exception as exc:
            print(f"[ReCogDriveB2D] Failed to solve lat/lon reference, fallback to 42/2: {exc}", flush=True)
            self.lat_ref, self.lon_ref = 42.0, 2.0

        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._global_plan, gps=True)
        self.initialized = True

    def sensors(self):
        front_camera = {
            "type": "sensor.camera.rgb",
            "x": 0.80,
            "y": 0.0,
            "z": 1.60,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "width": 1600,
            "height": 900,
            "fov": 70,
            "id": "CAM_FRONT",
        }
        state_sensors = [
            {
                "type": "sensor.other.imu",
                "x": -1.4,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.05,
                "id": "IMU",
            },
            {
                "type": "sensor.other.gnss",
                "x": -1.4,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.01,
                "id": "GPS",
            },
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "SPEED"},
        ]
        sensors = [front_camera]
        if self.sensor_profile in {"full", "all", "bench2drive_zoo", "zoo"}:
            sensors.extend(
                [
                    {
                        "type": "sensor.camera.rgb",
                        "x": 0.27,
                        "y": -0.55,
                        "z": 1.60,
                        "roll": 0.0,
                        "pitch": 0.0,
                        "yaw": -55.0,
                        "width": 1600,
                        "height": 900,
                        "fov": 70,
                        "id": "CAM_FRONT_LEFT",
                    },
                    {
                        "type": "sensor.camera.rgb",
                        "x": 0.27,
                        "y": 0.55,
                        "z": 1.60,
                        "roll": 0.0,
                        "pitch": 0.0,
                        "yaw": 55.0,
                        "width": 1600,
                        "height": 900,
                        "fov": 70,
                        "id": "CAM_FRONT_RIGHT",
                    },
                    {
                        "type": "sensor.camera.rgb",
                        "x": -2.0,
                        "y": 0.0,
                        "z": 1.60,
                        "roll": 0.0,
                        "pitch": 0.0,
                        "yaw": 180.0,
                        "width": 1600,
                        "height": 900,
                        "fov": 110,
                        "id": "CAM_BACK",
                    },
                    {
                        "type": "sensor.camera.rgb",
                        "x": -0.32,
                        "y": -0.55,
                        "z": 1.60,
                        "roll": 0.0,
                        "pitch": 0.0,
                        "yaw": -110.0,
                        "width": 1600,
                        "height": 900,
                        "fov": 70,
                        "id": "CAM_BACK_LEFT",
                    },
                    {
                        "type": "sensor.camera.rgb",
                        "x": -0.32,
                        "y": 0.55,
                        "z": 1.60,
                        "roll": 0.0,
                        "pitch": 0.0,
                        "yaw": 110.0,
                        "width": 1600,
                        "height": 900,
                        "fov": 70,
                        "id": "CAM_BACK_RIGHT",
                    },
                ]
            )
        sensors.extend(state_sensors)
        if IS_BENCH2DRIVE and self.sensor_profile in {"full", "all", "bench2drive_zoo", "zoo", "front_bev"}:
            sensors.append(
                {
                    "type": "sensor.camera.rgb",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 50.0,
                    "roll": 0.0,
                    "pitch": -90.0,
                    "yaw": 0.0,
                    "width": 512,
                    "height": 512,
                    "fov": 50.0,
                    "id": "bev",
                }
            )
        return sensors

    def _rgb(self, input_data: Dict[str, Any], key: str) -> np.ndarray:
        return cv2.cvtColor(input_data[key][1][:, :, :3], cv2.COLOR_BGR2RGB)

    def _local_command_xy(self, near_node: np.ndarray, pos: np.ndarray, compass: float) -> np.ndarray:
        can_bus_x = pos[0]
        can_bus_y = -pos[1]
        command_near_xy = np.array([near_node[0] - can_bus_x, -near_node[1] - can_bus_y], dtype=np.float32)
        rotation_matrix = np.array(
            [[np.cos(compass), -np.sin(compass)], [np.sin(compass), np.cos(compass)]],
            dtype=np.float32,
        )
        return rotation_matrix @ command_near_xy

    def _write_front_image(self, front_rgb: np.ndarray) -> Path:
        route_dir = self.image_dir / self.route_save_name
        route_dir.mkdir(parents=True, exist_ok=True)
        image_path = route_dir / f"{self.step:06d}.jpg"
        Image.fromarray(front_rgb).save(image_path, quality=self.image_quality)
        return image_path

    def _request_trajectory(
        self, image_path: Optional[Path], history: np.ndarray, command, status: np.ndarray
    ) -> np.ndarray:
        payload = {
            "sample_token": f"{self.route_save_name}_{self.step:06d}",
            "visual_cache_key": self.visual_cache_key,
            "visual_refresh": image_path is not None,
            "history_trajectory": history.tolist(),
            "high_command_one_hot": command_one_hot(command).tolist(),
            "status_feature": status.tolist(),
        }
        if image_path is not None:
            payload["image_path"] = str(image_path)
        response = requests.post(f"{self.server_url}/predict", json=payload, timeout=self.request_timeout)
        if response.status_code >= 400:
            raise RuntimeError(f"ReCogDrive server HTTP {response.status_code}: {response.text[:1000]}")
        result = response.json()
        if "trajectory" not in result:
            raise KeyError(f"ReCogDrive server response missing 'trajectory': {result}")
        trajectory = np.asarray(result["trajectory"], dtype=np.float32)
        if trajectory.shape != (8, 3):
            raise ValueError(f"Expected trajectory shape (8, 3), got {trajectory.shape}")
        if not np.isfinite(trajectory).all():
            raise ValueError("ReCogDrive server returned non-finite trajectory.")
        return trajectory

    def tick(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.step += 1
        front_rgb = self._rgb(input_data, "CAM_FRONT")
        gps = input_data["GPS"][1][:2]
        speed = float(input_data["SPEED"][1]["speed"])
        compass = float(input_data["IMU"][1][-1])
        acceleration = np.asarray(input_data["IMU"][1][:3], dtype=np.float32)
        angular_velocity = np.asarray(input_data["IMU"][1][3:6], dtype=np.float32)
        if math.isnan(compass):
            compass = 0.0
            acceleration = np.zeros(3, dtype=np.float32)
            angular_velocity = np.zeros(3, dtype=np.float32)

        pos = self._route_planner.gps_to_location(np.asarray(gps, dtype=np.float64))
        near_node, near_command = self._route_planner.run_step(pos)
        pose = np.asarray([pos[0], pos[1], compass], dtype=np.float64)
        self.pose_history.append(pose)
        history_global = sample_pose_history(
            list(self.pose_history),
            history_frames=self.history_frames,
            interval_steps=self.history_interval_steps,
        )
        history = relative_poses(pose, history_global)
        status = build_status_feature(
            command=near_command,
            speed=speed,
            acceleration_xyz=acceleration,
            compass=compass,
        )
        return {
            "front_rgb": front_rgb,
            "gps": gps,
            "pos": pos,
            "speed": speed,
            "compass": compass,
            "acceleration": acceleration,
            "angular_velocity": angular_velocity,
            "command_near": near_command,
            "command_near_xy": near_node,
            "local_command_xy": self._local_command_xy(near_node, pos, compass),
            "history_trajectory": history,
            "status_feature": status,
        }

    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        tick_data = self.tick(input_data)

        should_infer = (
            self.last_trajectory is None
            or self.inference_interval_steps <= 1
            or self.step % self.inference_interval_steps == 0
        )
        if should_infer:
            should_refresh_visual = (
                self.last_trajectory is None
                or self.visual_refresh_interval_steps <= 1
                or self.step % self.visual_refresh_interval_steps == 0
            )
            image_path = self._write_front_image(tick_data["front_rgb"]) if should_refresh_visual else None
            self.last_trajectory = self._request_trajectory(
                image_path=image_path,
                history=tick_data["history_trajectory"],
                command=tick_data["command_near"],
                status=tick_data["status_feature"],
            )

        repeated_control = (not should_infer) and self.repeat_control_when_not_infer and self.last_control is not None
        if repeated_control:
            control = carla.VehicleControl()
            control.steer = float(self.last_control.steer)
            control.throttle = float(self.last_control.throttle)
            control.brake = float(self.last_control.brake)
            self.last_pid_metadata = {}
        else:
            waypoints = trajectory_to_pid_waypoints(self.last_trajectory, flip_y=self.flip_y_for_pid)
            steer_traj, throttle_traj, brake_traj, metadata_traj = self.pidcontroller.control_pid(
                waypoints,
                tick_data["speed"],
                tick_data["local_command_xy"],
            )
            if brake_traj < 0.05:
                brake_traj = False
            if throttle_traj > float(brake_traj):
                brake_traj = False
            if tick_data["speed"] > self.max_speed_throttle_cutoff:
                throttle_traj = 0.0

            control = carla.VehicleControl()
            control.steer = float(np.clip(steer_traj, -1.0, 1.0))
            control.throttle = float(np.clip(throttle_traj, 0.0, 0.75))
            control.brake = float(np.clip(float(brake_traj), 0.0, 1.0))
            self.last_pid_metadata = dict(metadata_traj)

        self.last_pid_metadata.update(
            {
                "agent": "recogdrive_remote",
                "step": self.step,
                "repeated_control": repeated_control,
                "steer": control.steer,
                "throttle": control.throttle,
                "brake": control.brake,
                "plan": self.last_trajectory.tolist(),
                "history_trajectory": tick_data["history_trajectory"].tolist(),
                "status_feature": tick_data["status_feature"].tolist(),
            }
        )
        self.last_control = control
        self.metric_info[self.step] = self.get_metric_info()
        if self.save_path is not None:
            self.save(tick_data)
            if self.metric_flush_interval_steps > 0 and self.step % self.metric_flush_interval_steps == 0:
                self._flush_metric_info()
        return control

    def save(self, tick_data: Dict[str, Any]) -> None:
        assert self.save_path is not None
        frame = self.step // 10
        if self.step % 10 == 0 and self.save_debug_images:
            Image.fromarray(tick_data["front_rgb"]).save(self.save_path / "rgb_front" / f"{frame:04d}.png")
        if self.step % 10 == 0 and self.save_debug_meta:
            (self.save_path / "meta" / f"{frame:04d}.json").write_text(
                json.dumps(self.last_pid_metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def _flush_metric_info(self) -> None:
        assert self.save_path is not None
        (self.save_path / "metric_info.json").write_text(
            json.dumps(self.metric_info, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def destroy(self) -> None:
        if getattr(self, "save_path", None) is not None:
            self._flush_metric_info()
