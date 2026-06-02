#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


EPS = 1e-9


def as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_metrics(eval_dir: Path) -> Dict[str, Dict[str, Any]]:
    rows = {}
    for row in read_jsonl(eval_dir / "per_sample_metrics.jsonl"):
        key = str(row.get("sample_token") or row.get("scene_token") or "")
        if not key:
            continue
        rows[key] = row
    return rows


def load_predictions(eval_dir: Path) -> Dict[str, Dict[str, Any]]:
    rows = {}
    for row in read_jsonl(eval_dir / "predictions.jsonl"):
        key = str(row.get("sample_token") or row.get("scene_token") or "")
        if key:
            rows[key] = row
    return rows


def point(traj: List[List[float]], idx: int) -> List[float]:
    idx = max(0, min(idx, len(traj) - 1))
    return [float(value) for value in traj[idx][:3]]


def rule_use_bit(
    base_pred: Dict[str, Any],
    bit_pred: Dict[str, Any],
    *,
    early_threshold: float,
    terminal_threshold: float,
    early_points: int,
) -> bool:
    base = base_pred.get("pred_traj")
    bit = bit_pred.get("pred_traj")
    if not isinstance(base, list) or not isinstance(bit, list) or not base or not bit:
        return False
    horizon = min(len(base), len(bit))
    points = min(max(1, early_points), horizon)
    early_delta = sum(point(bit, idx)[0] - point(base, idx)[0] for idx in range(points)) / float(points)
    terminal_delta = point(bit, horizon - 1)[0] - point(base, horizon - 1)[0]
    return early_delta <= early_threshold and terminal_delta <= terminal_threshold


def metric_row(source: Dict[str, Any], selection_source: str, key: str) -> Dict[str, Any]:
    return {
        "sample_token": source.get("sample_token"),
        "scene_token": source.get("scene_token"),
        "selection_source": selection_source,
        "key": key,
        "score": as_float(source.get("score")),
        "drivable_area_compliance": as_float(source.get("drivable_area_compliance")),
        "no_at_fault_collisions": as_float(source.get("no_at_fault_collisions")),
        "time_to_collision_within_bound": as_float(source.get("time_to_collision_within_bound")),
        "ego_progress": as_float(source.get("ego_progress")),
        "comfort": as_float(source.get("comfort")),
        "valid": source.get("valid", True),
    }


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def zero_count(values: List[float]) -> int:
    return int(sum(1 for value in values if value <= EPS))


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [float(row["score"]) for row in rows if row.get("score") is not None]
    dac = [float(row["drivable_area_compliance"]) for row in rows if row.get("drivable_area_compliance") is not None]
    nc = [float(row["no_at_fault_collisions"]) for row in rows if row.get("no_at_fault_collisions") is not None]
    ttc = [float(row["time_to_collision_within_bound"]) for row in rows if row.get("time_to_collision_within_bound") is not None]
    ego = [float(row["ego_progress"]) for row in rows if row.get("ego_progress") is not None]
    comfort = [float(row["comfort"]) for row in rows if row.get("comfort") is not None]
    return {
        "num_samples": len(rows),
        "num_pdm_valid": len(scores),
        "mean_pdms": sum(scores) / len(scores) if scores else None,
        "median_pdms": statistics.median(scores) if scores else None,
        "p10_pdms": percentile(scores, 10),
        "zero_score_count": zero_count(scores),
        "drivable_area_compliance_zero_count": zero_count(dac),
        "no_at_fault_collision_zero_count": zero_count(nc),
        "time_to_collision_zero_count": zero_count(ttc),
        "ego_progress_mean": sum(ego) / len(ego) if ego else None,
        "comfort_mean": sum(comfort) / len(comfort) if comfort else None,
        "selection_bit_count": sum(1 for row in rows if row.get("selection_source") == "bit"),
        "selection_base_count": sum(1 for row in rows if row.get("selection_source") == "base"),
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = ["sample_token", "scene_token", "selection_source", "score", "drivable_area_compliance", "no_at_fault_collisions", "time_to_collision_within_bound", "ego_progress", "comfort"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a rule-based BiT safety fallback from existing eval predictions.")
    parser.add_argument("--a0-dir", type=Path, required=True)
    parser.add_argument("--bit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--early-x-delta-threshold", type=float, default=1.0)
    parser.add_argument("--terminal-x-delta-threshold", type=float, default=2.0)
    parser.add_argument("--early-points", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    a0_metrics = load_metrics(args.a0_dir)
    bit_metrics = load_metrics(args.bit_dir)
    a0_preds = load_predictions(args.a0_dir)
    bit_preds = load_predictions(args.bit_dir)
    rows = []
    for key in sorted(set(a0_metrics) & set(bit_metrics)):
        use_bit = rule_use_bit(
            a0_preds.get(key, {}),
            bit_preds.get(key, {}),
            early_threshold=args.early_x_delta_threshold,
            terminal_threshold=args.terminal_x_delta_threshold,
            early_points=args.early_points,
        )
        source = bit_metrics[key] if use_bit else a0_metrics[key]
        rows.append(metric_row(source, "bit" if use_bit else "base", key))
    agg = aggregate(rows)
    agg.update({
        "a0_dir": str(args.a0_dir),
        "bit_dir": str(args.bit_dir),
        "fallback_mode": "rule",
        "early_x_delta_threshold": args.early_x_delta_threshold,
        "terminal_x_delta_threshold": args.terminal_x_delta_threshold,
    })
    write_jsonl(args.output_dir / "selected_per_sample_metrics.jsonl", rows)
    write_csv(args.output_dir / "selected_per_sample_metrics.csv", rows)
    (args.output_dir / "aggregate_metrics.json").write_text(json.dumps(agg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# BiT Rule Safety Router Evaluation",
        "",
        f"Mean/P10: {agg['mean_pdms']} / {agg['p10_pdms']}",
        f"Zero/DAC0/NC0/TTC0: {agg['zero_score_count']} / {agg['drivable_area_compliance_zero_count']} / {agg['no_at_fault_collision_zero_count']} / {agg['time_to_collision_zero_count']}",
        f"Selected BiT/Base: {agg['selection_bit_count']} / {agg['selection_base_count']}",
    ]
    (args.output_dir / "aggregate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(agg, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
