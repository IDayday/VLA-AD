from __future__ import annotations

from collections import deque
import math
from typing import Iterable, Optional

import numpy as np


EARTH_RADIUS_EQUA = 6378137.0


class RoutePlanner:
    """Small route planner copied in spirit from Bench2DriveZoo.

    It stores the downsampled GPS global plan in metric map coordinates and
    returns the near target node plus RoadOption for every control tick.
    """

    def __init__(self, min_distance: float, max_distance: float, lat_ref: float = 42.0, lon_ref: float = 2.0):
        self.route = deque()
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.lat_ref = lat_ref
        self.lon_ref = lon_ref

    def set_route(self, global_plan: Iterable, gps: bool = False, global_plan_world: Optional[Iterable] = None) -> None:
        self.route.clear()
        if global_plan_world:
            for (pos, cmd), (pos_world, _) in zip(global_plan, global_plan_world):
                metric_pos = self.gps_to_location(np.array([pos["lat"], pos["lon"]])) if gps else np.array(
                    [pos.location.x, pos.location.y], dtype=np.float64
                )
                self.route.append((metric_pos, cmd, pos_world))
            return

        for pos, cmd in global_plan:
            metric_pos = self.gps_to_location(np.array([pos["lat"], pos["lon"]])) if gps else np.array(
                [pos.location.x, pos.location.y], dtype=np.float64
            )
            self.route.append((metric_pos, cmd))

    def run_step(self, gps: np.ndarray):
        if len(self.route) == 0:
            raise RuntimeError("RoutePlanner route is empty. Did set_global_plan run?")
        if len(self.route) == 1:
            return self.route[0]

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0
        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break
            cumulative_distance += float(np.linalg.norm(self.route[i][0] - self.route[i - 1][0]))
            distance = float(np.linalg.norm(self.route[i][0] - gps))
            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()

        return self.route[1]

    def gps_to_location(self, gps: np.ndarray) -> np.ndarray:
        lat, lon = gps[:2]
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat + 90.0) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y], dtype=np.float64)
