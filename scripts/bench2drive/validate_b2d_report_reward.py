#!/usr/bin/env python3
"""Validate the report-aligned reward aggregator on official B2D artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from navsim.agents.recogdrive.bench2drive_report_reward import (
    Bench2DriveReportState,
    aggregate_bench2drive_report,
)


EFFICIENCY_PATTERN = re.compile(r"\b\d+\.?\d*%")
EFFICIENCY_RESULT_PATTERN = re.compile(r"Driving Efficiency=([-+eE.\d]+)")
SMOOTHNESS_RESULT_PATTERN = re.compile(r"Driving Smoothness=([-+eE.\d]+)")
SMOOTHNESS_KEYS = (
    "acceleration",
    "angular_velocity",
    "forward_vector",
    "right_vector",
    "location",
    "rotation",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-json", type=Path, required=True)
    parser.add_argument("--metric-dir", type=Path)
    parser.add_argument(
        "--bench2drive-root",
        type=Path,
        default=Path("/mnt/data/Bench2Drive"),
    )
    parser.add_argument("--official-ability-json", type=Path)
    parser.add_argument("--official-efficiency-smoothness-log", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    return parser.parse_args()


def parse_efficiency_percentages(messages: Sequence[str]) -> Tuple[float, ...]:
    """Parse and filter checkpoint percentages exactly like the B2D tool."""

    values = []
    for message in messages:
        match = EFFICIENCY_PATTERN.search(message)
        if match is None:
            raise ValueError(f"cannot parse minimum-speed message: {message!r}")
        value = float(match.group().rstrip("%"))
        if value <= 1000.0:
            values.append(value)
    return tuple(values)


def record_is_infraction_free(record: Dict[str, Any]) -> bool:
    """Apply the official strict-success infraction rule."""

    return not any(
        values
        for name, values in record["infractions"].items()
        if name != "min_speed_infractions"
    )


def parse_efficiency_smoothness_log(text: str) -> Tuple[float, float]:
    """Parse official post-processing output onto report percentage scales."""

    efficiency_match = EFFICIENCY_RESULT_PATTERN.search(text)
    smoothness_match = SMOOTHNESS_RESULT_PATTERN.search(text)
    if efficiency_match is None or smoothness_match is None:
        raise ValueError("official log is missing Efficiency or Smoothness")
    efficiency = float(efficiency_match.group(1))
    smoothness = 100.0 * float(smoothness_match.group(1))
    return efficiency, smoothness


def _load_official_smoothness_module(bench2drive_root: Path):
    path = bench2drive_root / "tools" / "efficiency_smoothness_benchmark.py"
    if not path.is_file():
        raise FileNotFoundError(f"official smoothness tool not found: {path}")
    spec = importlib.util.spec_from_file_location("b2d_official_smoothness", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official smoothness tool: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_route_smoothness_counts(
    metric_path: Path,
    official_module,
) -> Tuple[int, int]:
    """Return passed/total segments while delegating scoring to B2D code."""

    metric_rows = json.loads(metric_path.read_text())
    arrays = {
        key: np.asarray([row[key] for row in metric_rows.values()])
        for key in SMOOTHNESS_KEYS
    }
    episode_length = len(arrays["angular_velocity"])
    if episode_length == 0:
        return 0, 0
    segment_count = 1 if episode_length <= 20 else episode_length // 20
    ratio = float(official_module.seg_compute_comfort_metric(**arrays))
    passed = int(round(ratio * segment_count))
    if not 0 <= passed <= segment_count:
        raise RuntimeError(f"invalid official smoothness result for {metric_path}")
    return passed, segment_count


def build_report_states(
    records: Sequence[Dict[str, Any]],
    *,
    metric_dir: Optional[Path] = None,
    official_smoothness_module=None,
) -> Tuple[Bench2DriveReportState, ...]:
    states = []
    for record in records:
        efficiency = parse_efficiency_percentages(
            record["infractions"]["min_speed_infractions"]
        )
        smooth_passed = 0
        smooth_count = 0
        if metric_dir is not None:
            if official_smoothness_module is None:
                raise ValueError("official_smoothness_module is required with metric_dir")
            metric_path = metric_dir / record["save_name"] / "metric_info.json"
            smooth_passed, smooth_count = compute_route_smoothness_counts(
                metric_path, official_smoothness_module
            )

        route_completion = float(record["scores"]["score_route"]) / 100.0
        states.append(
            Bench2DriveReportState(
                route_completion=route_completion,
                infraction_penalty=float(record["scores"]["score_penalty"]),
                infraction_free=record_is_infraction_free(record),
                route_finished=True,
                target_reached=route_completion >= 1.0 - 1e-9,
                efficiency_percentage_sum=sum(efficiency),
                efficiency_check_count=len(efficiency),
                smooth_segments_passed=smooth_passed,
                smooth_segment_count=smooth_count,
            )
        )
    return tuple(states)


def _metric_delta(computed: Optional[float], reference: Optional[float]) -> Optional[float]:
    if computed is None or reference is None:
        return None
    return computed - reference


def main() -> None:
    args = _parse_args()
    merged = json.loads(args.merged_json.read_text())
    records = merged["_checkpoint"]["records"]
    official_module = (
        _load_official_smoothness_module(args.bench2drive_root)
        if args.metric_dir is not None
        else None
    )
    states = build_report_states(
        records,
        metric_dir=args.metric_dir,
        official_smoothness_module=official_module,
    )
    total_routes = int(merged.get("eval num", len(records)))
    report = aggregate_bench2drive_report(states, total_routes=total_routes)

    reference = {
        "driving_score": float(merged["driving score"]),
        "success_rate": 100.0 * float(merged["success rate"]),
    }
    if args.official_efficiency_smoothness_log is not None:
        efficiency, smoothness = parse_efficiency_smoothness_log(
            args.official_efficiency_smoothness_log.read_text()
        )
        reference["driving_efficiency"] = efficiency
        reference["driving_smoothness"] = smoothness
    ability_reference = None
    if args.official_ability_json is not None:
        ability_reference = json.loads(args.official_ability_json.read_text())

    computed = {
        "driving_score": report.driving_score,
        "success_rate": report.success_rate,
        "driving_efficiency": report.driving_efficiency,
        "driving_smoothness": report.driving_smoothness,
        "efficiency_route_coverage": report.efficiency_route_coverage,
        "smoothness_route_coverage": report.smoothness_route_coverage,
    }
    deltas = {
        name: _metric_delta(computed.get(name), reference_value)
        for name, reference_value in reference.items()
    }
    passed = all(
        delta is not None and abs(delta) <= args.tolerance for delta in deltas.values()
    )
    result = {
        "contract": "recogdrive_b2d_stage3_report_aligned_reward_v2",
        "merged_json": str(args.merged_json.resolve()),
        "metric_dir": str(args.metric_dir.resolve()) if args.metric_dir else None,
        "computed": computed,
        "official_reference": reference,
        "official_ability_reference": ability_reference,
        "exact_metric_deltas": deltas,
        "exact_report_gate_passed": passed,
        "note": (
            "Ability values are retained as an official reference only; exact per-route "
            "ability reconstruction additionally requires route XML and Traffic-Signs "
            "junction processing."
        ),
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
