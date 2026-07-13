from __future__ import annotations

from typing import List

import numpy as np
import shapely
import shapely.vectorized
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex, EgoAreaIndex


class FastPDMScorer(PDMScorer):
    """
    Experimental PDM scorer with bbox-filtered ego-area lookup.

    The base scorer computes ``points_in_polygons`` by testing every map polygon against every ego
    corner/center point. For AWAC oracle scoring, each token has many proposal trajectories, so this
    dense polygon x point loop dominates runtime. This override first queries the existing STRtree with
    the ego-points bounding box, then runs the same vectorized ``contains`` checks only on polygons whose
    bounds intersect that box. A polygon that contains any ego point must intersect this box, so this
    preserves the original contains semantics while reducing the polygon count.
    """

    def _calculate_ego_area(self) -> None:
        n_proposals, n_horizon, n_points, _ = self._ego_coords.shape
        if n_points <= int(BBCoordsIndex.CENTER):
            raise ValueError(f"Expected ego coords to include center point, got n_points={n_points}.")

        flat_points = self._ego_coords.reshape(-1, 2)
        min_x, min_y = flat_points.min(axis=0)
        max_x, max_y = flat_points.max(axis=0)
        candidate_polygon_idcs = np.asarray(
            self._drivable_area_map.query(shapely.box(min_x, min_y, max_x, max_y), predicate="intersects"),
            dtype=np.int64,
        )

        drivable_area_idcs = self._drivable_area_map.get_indices_of_map_type(
            [
                SemanticMapLayer.ROADBLOCK,
                SemanticMapLayer.INTERSECTION,
                SemanticMapLayer.DRIVABLE_AREA,
                SemanticMapLayer.CARPARK_AREA,
            ]
        )
        drivable_lane_idcs = self._drivable_area_map.get_indices_of_map_type(
            [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        )
        drivable_on_route_idcs: List[int] = [
            idx for idx in drivable_lane_idcs if self._drivable_area_map.tokens[idx] in self._route_lane_ids
        ]

        if candidate_polygon_idcs.size == 0:
            self._ego_areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA] = True
            self._ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC] = True
            return

        in_polygons = np.zeros((candidate_polygon_idcs.size, flat_points.shape[0]), dtype=np.bool_)
        for out_idx, polygon_idx in enumerate(candidate_polygon_idcs):
            polygon = self._drivable_area_map._geometries[int(polygon_idx)]
            in_polygons[out_idx] = shapely.vectorized.contains(polygon, flat_points[:, 0], flat_points[:, 1])

        in_polygons = in_polygons.reshape(candidate_polygon_idcs.size, n_proposals, n_horizon, n_points)
        in_polygons = in_polygons.transpose(1, 2, 0, 3)

        candidate_to_local = {int(polygon_idx): local_idx for local_idx, polygon_idx in enumerate(candidate_polygon_idcs)}
        drivable_area_local_idcs = [
            candidate_to_local[idx] for idx in drivable_area_idcs if idx in candidate_to_local
        ]
        drivable_lane_local_idcs = [
            candidate_to_local[idx] for idx in drivable_lane_idcs if idx in candidate_to_local
        ]
        drivable_on_route_local_idcs = [
            candidate_to_local[idx] for idx in drivable_on_route_idcs if idx in candidate_to_local
        ]

        corners_in_polygon = in_polygons[..., : int(BBCoordsIndex.CENTER)]
        center_in_polygon = in_polygons[..., int(BBCoordsIndex.CENTER)]

        if drivable_lane_local_idcs:
            lane_corner_counts = corners_in_polygon[:, :, drivable_lane_local_idcs].sum(axis=-1)
            multiple_lanes_mask = (lane_corner_counts > 0).sum(axis=-1) > 1
            not_single_lanes_mask = np.all(lane_corner_counts != int(BBCoordsIndex.CENTER), axis=-1)
            self._ego_areas[
                np.logical_and(multiple_lanes_mask, not_single_lanes_mask),
                EgoAreaIndex.MULTIPLE_LANES,
            ] = True

        corner_inside_drivable = np.zeros((n_proposals, n_horizon, int(BBCoordsIndex.CENTER)), dtype=np.bool_)
        if drivable_area_local_idcs:
            corner_inside_drivable = corners_in_polygon[:, :, drivable_area_local_idcs].sum(axis=-2) > 0
        nondrivable_area_mask = corner_inside_drivable.sum(axis=-1) < int(BBCoordsIndex.CENTER)
        self._ego_areas[nondrivable_area_mask, EgoAreaIndex.NON_DRIVABLE_AREA] = True

        center_on_route = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        if drivable_on_route_local_idcs:
            center_on_route = center_in_polygon[:, :, drivable_on_route_local_idcs].sum(axis=-1) > 0
        self._ego_areas[~center_on_route, EgoAreaIndex.ONCOMING_TRAFFIC] = True
