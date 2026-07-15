#!/usr/bin/env python3
"""Split Bench2Drive routes into workload-balanced, count-bounded shards."""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path


def route_length(route: ET.Element) -> float:
    points = []
    waypoints = route.find("waypoints")
    if waypoints is not None:
        for position in waypoints.findall("position"):
            points.append(tuple(float(position.get(axis, "0")) for axis in ("x", "y", "z")))
    return sum(math.dist(left, right) for left, right in zip(points, points[1:]))


def normalize_route_id(value: str) -> str:
    route_id = str(value)
    if route_id.startswith("RouteScenario_"):
        route_id = route_id[len("RouteScenario_") :]
    if route_id.endswith("_rep0"):
        route_id = route_id[: -len("_rep0")]
    return route_id


def load_duration_costs(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("_checkpoint", {}).get("records", [])
    costs: dict[str, float] = {}
    for record in records:
        route_id = normalize_route_id(record.get("route_id", ""))
        duration = float(record.get("meta", {}).get("duration_game", 0.0))
        if route_id and math.isfinite(duration) and duration > 0:
            costs[route_id] = duration
    if not costs:
        raise ValueError(f"No positive meta.duration_game route costs found in {path}")
    return costs


def shard_routes(
    routes: list[ET.Element],
    shard_count: int,
    duration_costs: dict[str, float] | None = None,
) -> list[list[ET.Element]]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if shard_count > len(routes):
        raise ValueError(f"Cannot split {len(routes)} routes into {shard_count} non-empty shards")

    base, remainder = divmod(len(routes), shard_count)
    capacities = [base + int(index < remainder) for index in range(shard_count)]
    shards: list[list[tuple[int, ET.Element, float]]] = [[] for _ in range(shard_count)]
    loads = [0.0] * shard_count
    lengths = [route_length(route) for route in routes]
    known_costs = [
        duration_costs[normalize_route_id(route.get("id", ""))]
        for route in routes
        if duration_costs and normalize_route_id(route.get("id", "")) in duration_costs
    ]
    known_lengths = [
        length
        for route, length in zip(routes, lengths)
        if duration_costs and normalize_route_id(route.get("id", "")) in duration_costs
    ]
    fallback_scale = sum(known_costs) / sum(known_lengths) if known_lengths and sum(known_lengths) > 0 else 1.0
    weighted_routes = []
    for index, (route, length) in enumerate(zip(routes, lengths)):
        route_id = normalize_route_id(route.get("id", ""))
        cost = duration_costs.get(route_id, length * fallback_scale) if duration_costs else length
        weighted_routes.append((index, route, cost))

    # Longest-processing-time greedy assignment gives a tight makespan while
    # the capacities keep route setup overhead nearly equal across shards.
    for original_index, route, cost in sorted(weighted_routes, key=lambda item: item[2], reverse=True):
        candidates = [
            index for index in range(shard_count) if len(shards[index]) < capacities[index]
        ]
        target = min(candidates, key=lambda index: (loads[index], len(shards[index]), index))
        shards[target].append((original_index, route, cost))
        loads[target] += cost

    # Preserve the official XML order within each independent evaluator.
    return [
        [route for _, route, _ in sorted(shard, key=lambda item: item[0])]
        for shard in shards
    ]


def write_shards(
    base_route: Path,
    shard_count: int,
    algo: str,
    planner_type: str,
    cost_json: Path | None = None,
) -> None:
    source = base_route.with_suffix(".xml")
    routes = ET.parse(source).getroot().findall("route")
    duration_costs = load_duration_costs(cost_json) if cost_json else None
    shards = shard_routes(routes, shard_count, duration_costs)
    for index, shard in enumerate(shards):
        root = ET.Element("routes")
        root.extend(shard)
        output = Path(f"{base_route}_{index}_{algo}_{planner_type}.xml")
        ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_route", type=Path, help="Route XML path without the .xml suffix")
    parser.add_argument("shard_count", type=int)
    parser.add_argument("algo")
    parser.add_argument("planner_type")
    parser.add_argument(
        "--cost-json",
        type=Path,
        help="Prior merged route JSON; meta.duration_game is used only for scheduling",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_shards(
        args.base_route,
        args.shard_count,
        args.algo,
        args.planner_type,
        cost_json=args.cost_json,
    )


if __name__ == "__main__":
    main()
