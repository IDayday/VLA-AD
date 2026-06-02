#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.bit_safety_selector import feature_tensor  # noqa: E402
from scripts.eval_bit_safety_router import aggregate as aggregate_metric_rows  # noqa: E402
from scripts.eval_bit_selector_on_counterfactual import load_selector  # noqa: E402


EPS = 1e-9


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                row["_source_jsonl"] = str(path)
                rows.append(row)
    return rows


def metric_value(row: Dict[str, Any], side: str, key: str) -> float:
    metrics = row.get(f"{side}_metrics") or {}
    value = metrics.get(key)
    return float(value) if value is not None else 0.0


def metric_row(row: Dict[str, Any], source: str) -> Dict[str, Any]:
    metrics = row.get(f"{source}_metrics") or {}
    return {
        "sample_token": row.get("sample_token"),
        "scene_token": row.get("scene_token"),
        "selection_source": source,
        "score": metrics.get("pdm"),
        "drivable_area_compliance": metrics.get("dac"),
        "no_at_fault_collisions": metrics.get("nc"),
        "time_to_collision_within_bound": metrics.get("ttc"),
        "ego_progress": metrics.get("ego"),
        "comfort": metrics.get("comfort"),
        "valid": bool(metrics),
    }


def aggregate_selection(rows: Sequence[Dict[str, Any]], use_bit: Sequence[bool]) -> Dict[str, Any]:
    metric_rows = [metric_row(row, "bit" if bit else "base") for row, bit in zip(rows, use_bit)]
    return aggregate_metric_rows(metric_rows)


def _nanpercentile(values: np.ndarray, q: float) -> float | None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


def build_fast_arrays(rows: Sequence[Dict[str, Any]], probs: Sequence[float]) -> Dict[str, np.ndarray]:
    def side_array(side: str, key: str) -> np.ndarray:
        values: List[float] = []
        for row in rows:
            metrics = row.get(f"{side}_metrics") or {}
            value = metrics.get(key)
            values.append(float(value) if value is not None else np.nan)
        return np.asarray(values, dtype=np.float64)

    def feature_array(name: str) -> np.ndarray:
        values = []
        for row in rows:
            value = (row.get("features") or {}).get(name, 0.0)
            values.append(float(value) if value is not None else 0.0)
        return np.asarray(values, dtype=np.float64)

    arrays = {
        "probs": np.asarray(probs, dtype=np.float64),
        "early_x_delta_mean": feature_array("early_x_delta_mean"),
        "terminal_dx": feature_array("terminal_dx"),
        "curvature_delta": feature_array("curvature_delta"),
    }
    for side in ("base", "bit"):
        for key in ("pdm", "dac", "nc", "ttc", "ego", "comfort"):
            arrays[f"{side}_{key}"] = side_array(side, key)
    arrays["safety_regression"] = (
        ((arrays["base_nc"] > EPS) & (arrays["bit_nc"] <= EPS))
        | ((arrays["base_ttc"] > EPS) & (arrays["bit_ttc"] <= EPS))
    )
    arrays["dac_fix"] = (arrays["base_dac"] <= EPS) & (arrays["bit_dac"] > EPS)
    return arrays


def aggregate_fast(arrays: Dict[str, np.ndarray], use_bit: np.ndarray) -> Dict[str, Any]:
    def chosen(key: str) -> np.ndarray:
        return np.where(use_bit, arrays[f"bit_{key}"], arrays[f"base_{key}"])

    scores = chosen("pdm")
    dac = chosen("dac")
    nc = chosen("nc")
    ttc = chosen("ttc")
    ego = chosen("ego")
    comfort = chosen("comfort")

    finite_scores = scores[np.isfinite(scores)]
    finite_ego = ego[np.isfinite(ego)]
    finite_comfort = comfort[np.isfinite(comfort)]

    def zero_count(values: np.ndarray) -> int:
        values = values[np.isfinite(values)]
        return int(np.sum(values <= EPS))

    return {
        "num_samples": int(use_bit.size),
        "num_pdm_valid": int(finite_scores.size),
        "mean_pdms": float(np.mean(finite_scores)) if finite_scores.size else None,
        "median_pdms": float(np.median(finite_scores)) if finite_scores.size else None,
        "p10_pdms": _nanpercentile(scores, 10),
        "zero_score_count": zero_count(scores),
        "drivable_area_compliance_zero_count": zero_count(dac),
        "no_at_fault_collision_zero_count": zero_count(nc),
        "time_to_collision_zero_count": zero_count(ttc),
        "ego_progress_mean": float(np.mean(finite_ego)) if finite_ego.size else None,
        "comfort_mean": float(np.mean(finite_comfort)) if finite_comfort.size else None,
        "selection_bit_count": int(np.sum(use_bit)),
        "selection_base_count": int(use_bit.size - np.sum(use_bit)),
    }


def safety_regression(row: Dict[str, Any]) -> bool:
    return (
        metric_value(row, "base", "nc") > EPS and metric_value(row, "bit", "nc") <= EPS
    ) or (
        metric_value(row, "base", "ttc") > EPS and metric_value(row, "bit", "ttc") <= EPS
    )


def dac_fix(row: Dict[str, Any]) -> bool:
    return metric_value(row, "base", "dac") <= EPS and metric_value(row, "bit", "dac") > EPS


def linspace(start: float, stop: float, steps: int) -> List[float]:
    if steps <= 1:
        return [start]
    return [start + (stop - start) * i / (steps - 1) for i in range(steps)]


def finite_feature_values(rows: Sequence[Dict[str, Any]], name: str) -> List[float]:
    values = []
    for row in rows:
        value = (row.get("features") or {}).get(name)
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            values.append(value)
    return sorted(values)


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    idx = min(max(int(round(q * (len(values) - 1))), 0), len(values) - 1)
    return float(values[idx])


def candidate_thresholds(rows: Sequence[Dict[str, Any]], name: str, defaults: Iterable[float], *, include_inf: bool) -> List[float]:
    values = finite_feature_values(rows, name)
    candidates = set(float(v) for v in defaults)
    for q in (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99):
        candidates.add(quantile(values, q))
    if include_inf:
        candidates.add(float("inf"))
    return sorted(candidates)


def use_bit_for_thresholds(
    rows: Sequence[Dict[str, Any]],
    probs: Sequence[float],
    *,
    selector_threshold: float,
    early_x_threshold: float,
    terminal_x_threshold: float,
    curvature_threshold: float,
) -> List[bool]:
    output = []
    for row, prob in zip(rows, probs):
        features = row.get("features") or {}
        early = float(features.get("early_x_delta_mean", 0.0))
        terminal = float(features.get("terminal_dx", 0.0))
        curvature = float(features.get("curvature_delta", 0.0))
        ok = (
            float(prob) >= selector_threshold
            and early <= early_x_threshold
            and terminal <= terminal_x_threshold
            and curvature <= curvature_threshold
        )
        output.append(bool(ok))
    return output


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "rank",
        "safety_ok",
        "selector_threshold",
        "early_x_delta_threshold",
        "terminal_x_delta_threshold",
        "curvature_delta_threshold",
        "selected_safety_regressions",
        "selected_dac_fixes",
        "mean_pdms",
        "p10_pdms",
        "zero_score_count",
        "drivable_area_compliance_zero_count",
        "no_at_fault_collision_zero_count",
        "time_to_collision_zero_count",
        "ego_progress_mean",
        "selection_bit_count",
        "selection_base_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune selector safety override thresholds on non-test counterfactual data.")
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--selector-config", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selector-threshold", type=float, default=None)
    parser.add_argument(
        "--selector-threshold-grid",
        default=None,
        help="Optional comma-separated selector probability thresholds to sweep. Overrides --selector-threshold.",
    )
    parser.add_argument("--max-selected-safety-regressions", type=int, default=0)
    parser.add_argument("--allow-dac0-worse-than-base", type=int, default=0)
    parser.add_argument("--allow-nc0-worse-than-base", type=int, default=1)
    parser.add_argument("--allow-ttc0-worse-than-base", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for path in args.val_jsonl:
        rows.extend(read_jsonl(path))
    rows = [row for row in rows if row.get("base_metrics") and row.get("bit_metrics")]
    if not rows:
        raise RuntimeError("No valid counterfactual rows found")
    for path in args.val_jsonl:
        meta_path = path.parent / "metadata.json"
        if meta_path.is_file():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            split = str(metadata.get("split", "")).lower()
            if "test" in split or bool(metadata.get("analysis_only", False)):
                raise RuntimeError(f"Refusing to tune safety override on test/analysis-only data: {path}")

    model, config = load_selector(args.selector, args.selector_config)
    feature_names = config["feature_names"]
    if args.selector_threshold_grid:
        selector_thresholds = [
            float(item.strip())
            for item in args.selector_threshold_grid.split(",")
            if item.strip()
        ]
    else:
        selector_thresholds = [float(args.selector_threshold if args.selector_threshold is not None else config.get("threshold", 0.5))]
    x = feature_tensor((row.get("features", {}) for row in rows), feature_names)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).cpu().tolist()
    arrays = build_fast_arrays(rows, probs)

    base_agg = aggregate_selection(rows, [False] * len(rows))
    early_candidates = candidate_thresholds(rows, "early_x_delta_mean", [-0.2, -0.1, 0.0, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0], include_inf=False)
    terminal_candidates = candidate_thresholds(rows, "terminal_dx", [-1.0, -0.5, 0.0, 0.10, 0.20, 0.50, 1.0, 2.0], include_inf=False)
    curvature_candidates = candidate_thresholds(rows, "curvature_delta", [-0.2, -0.1, 0.0, 0.05, 0.10, 0.20, 0.50, 1.0], include_inf=True)

    sweep_rows: List[Dict[str, Any]] = []
    for selector_threshold in selector_thresholds:
        for early in early_candidates:
            for terminal in terminal_candidates:
                for curvature in curvature_candidates:
                    use_bit = (
                        (arrays["probs"] >= selector_threshold)
                        & (arrays["early_x_delta_mean"] <= early)
                        & (arrays["terminal_dx"] <= terminal)
                        & (arrays["curvature_delta"] <= curvature)
                    )
                    agg = aggregate_fast(arrays, use_bit)
                    selected_safety = int(np.sum(use_bit & arrays["safety_regression"]))
                    selected_dac_fixes = int(np.sum(use_bit & arrays["dac_fix"]))
                    safety_ok = (
                        selected_safety <= args.max_selected_safety_regressions
                        and int(agg.get("no_at_fault_collision_zero_count") or 0) <= int(base_agg.get("no_at_fault_collision_zero_count") or 0) + args.allow_nc0_worse_than_base
                        and int(agg.get("time_to_collision_zero_count") or 0) <= int(base_agg.get("time_to_collision_zero_count") or 0) + args.allow_ttc0_worse_than_base
                        and int(agg.get("drivable_area_compliance_zero_count") or 0) <= int(base_agg.get("drivable_area_compliance_zero_count") or 0) + args.allow_dac0_worse_than_base
                    )
                    sweep_rows.append({
                        "safety_ok": bool(safety_ok),
                        "selector_threshold": selector_threshold,
                        "early_x_delta_threshold": early,
                        "terminal_x_delta_threshold": terminal,
                        "curvature_delta_threshold": curvature,
                        "selected_safety_regressions": selected_safety,
                        "selected_dac_fixes": selected_dac_fixes,
                        **agg,
                    })

    feasible = [row for row in sweep_rows if row["safety_ok"]]
    ranking_pool = feasible if feasible else sweep_rows
    ranking_pool.sort(
        key=lambda row: (
            float(row.get("mean_pdms") or -1.0),
            -int(row.get("zero_score_count") or 10**9),
            -int(row.get("drivable_area_compliance_zero_count") or 10**9),
            -int(row.get("no_at_fault_collision_zero_count") or 10**9),
            -int(row.get("time_to_collision_zero_count") or 10**9),
            int(row.get("selected_dac_fixes") or 0),
        ),
        reverse=True,
    )
    top_rows = []
    for rank, row in enumerate(ranking_pool[: args.top_k], start=1):
        top_rows.append({"rank": rank, **row})
    best = top_rows[0]
    write_csv(args.output_dir / "override_threshold_sweep_top.csv", top_rows)
    payload = {
        "selector": str(args.selector),
        "selector_config": str(args.selector_config),
        "selector_thresholds": selector_thresholds,
        "val_jsonl": [str(path) for path in args.val_jsonl],
        "num_rows": len(rows),
        "feasible_count": len(feasible),
        "base_aggregate": base_agg,
        "best": best,
    }
    (args.output_dir / "best_override_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# BiT Selector Safety Override Tuning",
        "",
        f"Rows: {len(rows)}",
        f"Feasible threshold sets: {len(feasible)} / {len(sweep_rows)}",
        f"Selector thresholds: `{selector_thresholds}`",
        "",
        "| Rank | Mean | P10 | Zero | DAC0 | NC0 | TTC0 | Ego | Use Bit | SafetyReg | DACFix | early_x | terminal_x | curvature |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top_rows[:10]:
        lines.append(
            f"| {row['rank']} | {row.get('mean_pdms')} | {row.get('p10_pdms')} | {row.get('zero_score_count')} | "
            f"{row.get('drivable_area_compliance_zero_count')} | {row.get('no_at_fault_collision_zero_count')} | "
            f"{row.get('time_to_collision_zero_count')} | {row.get('ego_progress_mean')} | {row.get('selection_bit_count')} | "
            f"{row.get('selected_safety_regressions')} | {row.get('selected_dac_fixes')} | "
            f"{row.get('early_x_delta_threshold')} | {row.get('terminal_x_delta_threshold')} | {row.get('curvature_delta_threshold')} |"
        )
    (args.output_dir / "override_threshold_tuning.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "best": best, "feasible_count": len(feasible)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
