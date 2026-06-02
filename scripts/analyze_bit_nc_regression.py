#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def discover_file(eval_dir: Path, names: Iterable[str]) -> Optional[Path]:
    for name in names:
        path = eval_dir / name
        if path.is_file():
            return path
    return None


def load_metrics(eval_dir: Path) -> Dict[str, Dict[str, Any]]:
    path = discover_file(eval_dir, ("per_sample_metrics.jsonl", "per_sample_metrics.csv"))
    if path is None:
        raise FileNotFoundError(f"No per-sample metrics found in {eval_dir}")
    if path.suffix == ".jsonl":
        rows = read_jsonl(path)
    else:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = [dict(row) for row in csv.DictReader(f)]
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("sample_token") or row.get("scene_token") or "")
        if not key:
            continue
        out[key] = normalize_metric_row(row)
    return out


def normalize_metric_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_token": row.get("sample_token"),
        "scene_token": row.get("scene_token"),
        "score": as_float(row.get("score") if "score" in row else row.get("pdm_score")),
        "dac": as_float(row.get("drivable_area_compliance")),
        "nc": as_float(row.get("no_at_fault_collisions")),
        "ttc": as_float(row.get("time_to_collision_within_bound")),
        "ego": as_float(row.get("ego_progress")),
        "comfort": as_float(row.get("comfort")),
    }


def load_predictions(eval_dir: Path) -> Dict[str, Dict[str, Any]]:
    path = discover_file(eval_dir, ("predictions.jsonl",))
    if path is None:
        return {}
    out = {}
    for row in read_jsonl(path):
        key = str(row.get("sample_token") or row.get("scene_token") or "")
        if key:
            out[key] = row
    return out


def is_zero(row: Dict[str, Any], metric: str) -> bool:
    value = row.get(metric)
    return value is not None and float(value) <= EPS


def is_nonzero(row: Dict[str, Any], metric: str) -> bool:
    value = row.get(metric)
    return value is not None and float(value) > EPS


def traj(row: Optional[Dict[str, Any]]) -> Optional[List[List[float]]]:
    if not row:
        return None
    value = row.get("pred_traj")
    if not isinstance(value, list) or not value:
        return None
    return value


def safe_point(trajectory: List[List[float]], idx: int) -> List[float]:
    idx = max(0, min(idx, len(trajectory) - 1))
    return [float(x) for x in trajectory[idx][:3]]


def step_distances(trajectory: List[List[float]], points: int = 3) -> List[float]:
    distances = []
    for idx in range(1, min(points + 1, len(trajectory))):
        x0, y0, _ = safe_point(trajectory, idx - 1)
        x1, y1, _ = safe_point(trajectory, idx)
        distances.append(math.hypot(x1 - x0, y1 - y0))
    return distances


def curvature_proxy(trajectory: List[List[float]]) -> float:
    headings = [safe_point(trajectory, idx)[2] for idx in range(len(trajectory))]
    if len(headings) < 3:
        return 0.0
    return float(sum(abs(headings[idx + 1] - 2 * headings[idx] + headings[idx - 1]) for idx in range(1, len(headings) - 1)))


def traj_delta_features(base_pred: Optional[Dict[str, Any]], bit_pred: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = traj(base_pred)
    bit = traj(bit_pred)
    if base is None or bit is None:
        return {}
    horizon = min(len(base), len(bit))
    endpoint_base = safe_point(base, horizon - 1)
    endpoint_bit = safe_point(bit, horizon - 1)
    early_points = [idx for idx in (1, 2, 3) if idx < horizon]
    early_x_deltas = [safe_point(bit, idx)[0] - safe_point(base, idx)[0] for idx in early_points]
    early_y_deltas = [safe_point(bit, idx)[1] - safe_point(base, idx)[1] for idx in early_points]
    base_steps = step_distances(base, points=3)
    bit_steps = step_distances(bit, points=3)
    step_delta = statistics.mean(bit_steps) - statistics.mean(base_steps) if base_steps and bit_steps else None
    mean_abs_x = statistics.mean(abs(safe_point(bit, idx)[0] - safe_point(base, idx)[0]) for idx in range(horizon))
    mean_abs_y = statistics.mean(abs(safe_point(bit, idx)[1] - safe_point(base, idx)[1]) for idx in range(horizon))
    endpoint_dx = endpoint_bit[0] - endpoint_base[0]
    endpoint_dy = endpoint_bit[1] - endpoint_base[1]
    endpoint_dh = endpoint_bit[2] - endpoint_base[2]
    return {
        "endpoint_dx": endpoint_dx,
        "endpoint_dy": endpoint_dy,
        "endpoint_dheading": endpoint_dh,
        "mean_abs_x_delta": mean_abs_x,
        "mean_abs_y_delta": mean_abs_y,
        "early_x_delta_p1": early_x_deltas[0] if len(early_x_deltas) > 0 else None,
        "early_x_delta_p2": early_x_deltas[1] if len(early_x_deltas) > 1 else None,
        "early_x_delta_p3": early_x_deltas[2] if len(early_x_deltas) > 2 else None,
        "early_y_delta_mean": statistics.mean(early_y_deltas) if early_y_deltas else None,
        "early_x_delta_mean": statistics.mean(early_x_deltas) if early_x_deltas else None,
        "early_step_distance_delta": step_delta,
        "heading_delta_mean": statistics.mean(
            safe_point(bit, idx)[2] - safe_point(base, idx)[2] for idx in range(horizon)
        ),
        "curvature_proxy_delta": curvature_proxy(bit) - curvature_proxy(base),
        "bit_more_aggressive_longitudinal": bool((statistics.mean(early_x_deltas) if early_x_deltas else 0.0) > 0.0 or endpoint_dx > 0.0),
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def case_row(key: str, a0: Dict[str, Any], bit: Dict[str, Any], features: Dict[str, Any], run: str) -> Dict[str, Any]:
    row = {
        "run": run,
        "sample_token": bit.get("sample_token") or a0.get("sample_token"),
        "scene_token": bit.get("scene_token") or a0.get("scene_token"),
        "a0_score": a0.get("score"),
        "bit_score": bit.get("score"),
        "delta_score": None if a0.get("score") is None or bit.get("score") is None else bit["score"] - a0["score"],
        "a0_dac": a0.get("dac"),
        "bit_dac": bit.get("dac"),
        "a0_nc": a0.get("nc"),
        "bit_nc": bit.get("nc"),
        "a0_ttc": a0.get("ttc"),
        "bit_ttc": bit.get("ttc"),
        "a0_ego": a0.get("ego"),
        "bit_ego": bit.get("ego"),
    }
    row.update(features)
    return row


def summarize_bool(rows: List[Dict[str, Any]], key: str) -> str:
    valid = [row for row in rows if row.get(key) is not None]
    if not valid:
        return "n/a"
    count = sum(1 for row in valid if bool(row.get(key)))
    return f"{count}/{len(valid)} ({count / len(valid):.1%})"


def mean_feature(rows: List[Dict[str, Any]], key: str) -> str:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return "n/a"
    return f"{statistics.mean(values):.4f}"


def analyze_pair(
    run: str,
    a0_metrics: Dict[str, Dict[str, Any]],
    bit_metrics: Dict[str, Dict[str, Any]],
    a0_preds: Dict[str, Dict[str, Any]],
    bit_preds: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    all_rows = []
    nc_regressed = []
    nc_fixed = []
    dac_fixed = []
    dac_regressed = []
    ttc_regressed = []
    overlap = []
    for key in sorted(set(a0_metrics) & set(bit_metrics)):
        a0 = a0_metrics[key]
        bit = bit_metrics[key]
        features = traj_delta_features(a0_preds.get(key), bit_preds.get(key))
        row = case_row(key, a0, bit, features, run)
        row["a0_nc_nonzero_b5_nc0"] = is_nonzero(a0, "nc") and is_zero(bit, "nc")
        row["a0_nc0_b5_nc_nonzero"] = is_zero(a0, "nc") and is_nonzero(bit, "nc")
        row["a0_dac0_b5_dac_nonzero"] = is_zero(a0, "dac") and is_nonzero(bit, "dac")
        row["a0_dac_nonzero_b5_dac0"] = is_nonzero(a0, "dac") and is_zero(bit, "dac")
        row["a0_ttc_nonzero_b5_ttc0"] = is_nonzero(a0, "ttc") and is_zero(bit, "ttc")
        all_rows.append(row)
        if row["a0_nc_nonzero_b5_nc0"]:
            nc_regressed.append(row)
        if row["a0_nc0_b5_nc_nonzero"]:
            nc_fixed.append(row)
        if row["a0_dac0_b5_dac_nonzero"]:
            dac_fixed.append(row)
        if row["a0_dac_nonzero_b5_dac0"]:
            dac_regressed.append(row)
        if row["a0_ttc_nonzero_b5_ttc0"]:
            ttc_regressed.append(row)
        if row["a0_nc_nonzero_b5_nc0"] and row["a0_dac0_b5_dac_nonzero"]:
            overlap.append(row)
    return {
        "all": all_rows,
        "nc_regressed": nc_regressed,
        "nc_fixed": nc_fixed,
        "dac_fixed": dac_fixed,
        "dac_regressed": dac_regressed,
        "ttc_regressed": ttc_regressed,
        "overlap": overlap,
    }


def write_report(path: Path, b5: Dict[str, List[Dict[str, Any]]], b6: Dict[str, List[Dict[str, Any]]]) -> None:
    b5_nc = b5["nc_regressed"]
    b5_dac = b5["dac_fixed"]
    lines = [
        "# BiT NC Regression Diagnosis",
        "",
        "## Transition Counts",
        "",
        "| Run | NC newly broken | NC fixed | DAC fixed | DAC regressed | TTC newly broken | DAC fixed and NC broken |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, data in (("B5", b5), ("B6", b6)):
        lines.append(
            f"| {name} | {len(data['nc_regressed'])} | {len(data['nc_fixed'])} | {len(data['dac_fixed'])} | "
            f"{len(data['dac_regressed'])} | {len(data['ttc_regressed'])} | {len(data['overlap'])} |"
        )
    lines.extend([
        "",
        "## B5 Trajectory Pattern in New NC Cases",
        "",
        f"- More aggressive longitudinal cases: {summarize_bool(b5_nc, 'bit_more_aggressive_longitudinal')}",
        f"- Mean early x delta: {mean_feature(b5_nc, 'early_x_delta_mean')}",
        f"- Mean endpoint dx: {mean_feature(b5_nc, 'endpoint_dx')}",
        f"- Mean endpoint dy: {mean_feature(b5_nc, 'endpoint_dy')}",
        f"- Mean early y delta: {mean_feature(b5_nc, 'early_y_delta_mean')}",
        f"- Mean heading delta: {mean_feature(b5_nc, 'heading_delta_mean')}",
        "",
        "## Direct Answers",
        "",
        f"- Is NC regression concentrated in samples where BiT fixed DAC? {len(b5['overlap'])}/{len(b5_nc)} B5 new NC cases overlap with DAC fixes.",
        f"- Does BiT increase early longitudinal displacement in new NC cases? {summarize_bool(b5_nc, 'bit_more_aggressive_longitudinal')}; mean early x delta is {mean_feature(b5_nc, 'early_x_delta_mean')}.",
        f"- Is terminal conditioning pushing the endpoint farther forward? Mean endpoint dx in B5 new NC cases is {mean_feature(b5_nc, 'endpoint_dx')}.",
        f"- Are NC regressions mostly longitudinal or lateral? Compare endpoint dx={mean_feature(b5_nc, 'endpoint_dx')} with endpoint dy={mean_feature(b5_nc, 'endpoint_dy')} and early y delta={mean_feature(b5_nc, 'early_y_delta_mean')}.",
        "",
        "Interpretation should remain conservative: trajectory deltas are diagnostics, not proof of causal collision mechanism.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze BiT NC regressions against A0.")
    parser.add_argument("--a0-dir", type=Path, required=True)
    parser.add_argument("--b5-dir", type=Path, required=True)
    parser.add_argument("--b6-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    a0_metrics = load_metrics(args.a0_dir)
    b5_metrics = load_metrics(args.b5_dir)
    b6_metrics = load_metrics(args.b6_dir)
    a0_preds = load_predictions(args.a0_dir)
    b5_preds = load_predictions(args.b5_dir)
    b6_preds = load_predictions(args.b6_dir)

    b5 = analyze_pair("b5", a0_metrics, b5_metrics, a0_preds, b5_preds)
    b6 = analyze_pair("b6", a0_metrics, b6_metrics, a0_preds, b6_preds)

    write_jsonl(args.output_dir / "nc_regression_cases.jsonl", b5["nc_regressed"])
    write_jsonl(args.output_dir / "dac_fixed_cases.jsonl", b5["dac_fixed"])
    write_jsonl(args.output_dir / "overlap_dac_fixed_but_nc_broken.jsonl", b5["overlap"])
    write_csv(args.output_dir / "bit_vs_base_delta_features.csv", b5["all"])
    write_report(args.output_dir / "nc_regression_report.md", b5, b6)
    print(json.dumps({
        "b5_nc_regressed": len(b5["nc_regressed"]),
        "b5_dac_fixed": len(b5["dac_fixed"]),
        "b5_overlap": len(b5["overlap"]),
        "output_dir": str(args.output_dir),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
