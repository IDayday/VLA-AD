#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.bit_safety_selector import (  # noqa: E402
    SELECTOR_FEATURE_NAMES,
    FeatureMLPSelector,
    feature_tensor,
)
from scripts.eval_bit_safety_router import aggregate as aggregate_metric_rows  # noqa: E402


EPS = 1e-9


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def metric_row(row: Dict[str, Any], source: str, selection_source: str) -> Dict[str, Any]:
    metrics = row.get(f"{source}_metrics") or {}
    return {
        "sample_token": row.get("sample_token"),
        "scene_token": row.get("scene_token"),
        "selection_source": selection_source,
        "score": metrics.get("pdm"),
        "drivable_area_compliance": metrics.get("dac"),
        "no_at_fault_collisions": metrics.get("nc"),
        "time_to_collision_within_bound": metrics.get("ttc"),
        "ego_progress": metrics.get("ego"),
        "comfort": metrics.get("comfort"),
        "valid": bool(metrics),
    }


def aggregate_selection(rows: Sequence[Dict[str, Any]], use_bit: Sequence[bool], *, selection_name: str) -> Dict[str, Any]:
    metric_rows = [
        metric_row(row, "bit" if bit else "base", "bit" if bit else "base")
        for row, bit in zip(rows, use_bit)
    ]
    agg = aggregate_metric_rows(metric_rows)
    agg["method"] = selection_name
    return agg


def metric_value(row: Dict[str, Any], side: str, key: str) -> float:
    metrics = row.get(f"{side}_metrics") or {}
    value = metrics.get(key)
    return float(value) if value is not None else 0.0


def safety_oracle_choice(row: Dict[str, Any]) -> bool:
    base_nc = metric_value(row, "base", "nc")
    bit_nc = metric_value(row, "bit", "nc")
    base_ttc = metric_value(row, "base", "ttc")
    bit_ttc = metric_value(row, "bit", "ttc")
    if base_nc > EPS and bit_nc <= EPS:
        return False
    if base_ttc > EPS and bit_ttc <= EPS:
        return False
    return True


def oracle_best_choice(row: Dict[str, Any]) -> bool:
    return metric_value(row, "bit", "pdm") > metric_value(row, "base", "pdm")


def rule_choice(
    row: Dict[str, Any],
    *,
    early_x_delta_threshold: float,
    terminal_x_delta_threshold: float,
    curvature_delta_threshold: Optional[float] = None,
) -> bool:
    features = row.get("features", {})
    use_bit = (
        float(features.get("early_x_delta_mean", 0.0)) <= early_x_delta_threshold
        and float(features.get("terminal_dx", 0.0)) <= terminal_x_delta_threshold
    )
    if curvature_delta_threshold is not None:
        use_bit = use_bit and float(features.get("curvature_delta", 0.0)) <= curvature_delta_threshold
    return bool(use_bit)


def load_selector(selector_path: Path, config_path: Path) -> Tuple[FeatureMLPSelector, Dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        payload = torch.load(selector_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(selector_path, map_location="cpu")
    feature_names = payload.get("feature_names") or config.get("feature_names") or SELECTOR_FEATURE_NAMES
    hidden_dim = int(payload.get("hidden_dim", config.get("hidden_dim", 128)))
    dropout = float(payload.get("dropout", config.get("dropout", 0.0)))
    model = FeatureMLPSelector(len(feature_names), hidden_dim=hidden_dim, dropout=dropout)
    state = payload.get("state_dict", payload)
    model.load_state_dict(state, strict=True)
    model.eval()
    config["feature_names"] = feature_names
    return model, config


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "sample_token",
        "scene_token",
        "label_use_bit",
        "selector_use_bit",
        "selector_prob",
        "selected_source",
        "base_pdm",
        "bit_pdm",
        "base_dac",
        "bit_dac",
        "base_nc",
        "bit_nc",
        "base_ttc",
        "bit_ttc",
        "error_type",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def selector_error(row: Dict[str, Any], use_bit: bool) -> Optional[str]:
    if use_bit and metric_value(row, "base", "nc") > EPS and metric_value(row, "bit", "nc") <= EPS:
        return "false_positive_nc_regression"
    if use_bit and metric_value(row, "base", "ttc") > EPS and metric_value(row, "bit", "ttc") <= EPS:
        return "false_positive_ttc_regression"
    if not use_bit and metric_value(row, "base", "dac") <= EPS and metric_value(row, "bit", "dac") > EPS:
        return "false_negative_dac_fix"
    if not use_bit and metric_value(row, "bit", "pdm") > metric_value(row, "base", "pdm"):
        return "false_negative_pdm_gain"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate learned BiT selector on counterfactual JSONL.")
    parser.add_argument("--counterfactual-jsonl", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--selector-config", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rule-early-x-delta-threshold", type=float, default=0.02)
    parser.add_argument("--rule-terminal-x-delta-threshold", type=float, default=0.2)
    parser.add_argument("--safety-override", action="store_true")
    parser.add_argument("--override-early-x-delta-threshold", type=float, default=0.02)
    parser.add_argument("--override-terminal-x-delta-threshold", type=float, default=0.2)
    parser.add_argument("--override-curvature-delta-threshold", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.counterfactual_jsonl)
    if not rows:
        raise RuntimeError(f"No rows found in {args.counterfactual_jsonl}")
    model, config = load_selector(args.selector, args.selector_config)
    feature_names = config["feature_names"]
    threshold = float(args.threshold if args.threshold is not None else config.get("threshold", 0.5))
    x = feature_tensor((row.get("features", {}) for row in rows), feature_names)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).cpu()
    selector_use_bit = (probs >= threshold).numpy().astype(bool).tolist()
    if args.safety_override:
        selector_use_bit = [
            bool(use_bit and rule_choice(
                row,
                early_x_delta_threshold=args.override_early_x_delta_threshold,
                terminal_x_delta_threshold=args.override_terminal_x_delta_threshold,
                curvature_delta_threshold=args.override_curvature_delta_threshold,
            ))
            for row, use_bit in zip(rows, selector_use_bit)
        ]

    base_use = [False] * len(rows)
    bit_use = [True] * len(rows)
    rule_use = [
        rule_choice(
            row,
            early_x_delta_threshold=args.rule_early_x_delta_threshold,
            terminal_x_delta_threshold=args.rule_terminal_x_delta_threshold,
        )
        for row in rows
    ]
    safety_oracle_use = [safety_oracle_choice(row) for row in rows]
    oracle_best_use = [oracle_best_choice(row) for row in rows]
    summaries = [
        aggregate_selection(rows, base_use, selection_name="base"),
        aggregate_selection(rows, bit_use, selection_name="bit"),
        aggregate_selection(rows, rule_use, selection_name="rule_fallback"),
        aggregate_selection(rows, safety_oracle_use, selection_name="safety_oracle_analysis_only"),
        aggregate_selection(rows, oracle_best_use, selection_name="oracle_best_analysis_only"),
        aggregate_selection(rows, selector_use_bit, selection_name="learned_selector" + ("_safety_override" if args.safety_override else "")),
    ]
    selected_rows = []
    confusion_rows = []
    error_cases = []
    for row, prob, use_bit in zip(rows, probs.tolist(), selector_use_bit):
        source = "bit" if use_bit else "base"
        selected = metric_row(row, source, source)
        selected["selector_prob"] = float(prob)
        selected["label_use_bit"] = row.get("label_use_bit")
        selected_rows.append(selected)
        error_type = selector_error(row, use_bit)
        confusion = {
            "sample_token": row.get("sample_token"),
            "scene_token": row.get("scene_token"),
            "label_use_bit": row.get("label_use_bit"),
            "selector_use_bit": int(use_bit),
            "selector_prob": float(prob),
            "selected_source": source,
            "base_pdm": metric_value(row, "base", "pdm"),
            "bit_pdm": metric_value(row, "bit", "pdm"),
            "base_dac": metric_value(row, "base", "dac"),
            "bit_dac": metric_value(row, "bit", "dac"),
            "base_nc": metric_value(row, "base", "nc"),
            "bit_nc": metric_value(row, "bit", "nc"),
            "base_ttc": metric_value(row, "base", "ttc"),
            "bit_ttc": metric_value(row, "bit", "ttc"),
            "error_type": error_type,
        }
        confusion_rows.append(confusion)
        if error_type is not None:
            error_cases.append({**confusion, "features": row.get("features", {})})

    write_jsonl(args.output_dir / "selected_samples.jsonl", selected_rows)
    write_csv(args.output_dir / "selector_confusion.csv", confusion_rows)
    write_jsonl(args.output_dir / "selector_error_cases.jsonl", error_cases)
    payload = {
        "counterfactual_jsonl": str(args.counterfactual_jsonl),
        "selector": str(args.selector),
        "selector_config": str(args.selector_config),
        "threshold": threshold,
        "safety_override": bool(args.safety_override),
        "summaries": summaries,
    }
    (args.output_dir / "selected_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# BiT Selector Navtest Evaluation",
        "",
        f"Threshold: `{threshold}`",
        f"Safety override: `{args.safety_override}`",
        "",
        "| Method | Mean | P10 | Zero | DAC0 | NC0 | TTC0 | Ego | Use Bit |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['method']} | {row.get('mean_pdms')} | {row.get('p10_pdms')} | "
            f"{row.get('zero_score_count')} | {row.get('drivable_area_compliance_zero_count')} | "
            f"{row.get('no_at_fault_collision_zero_count')} | {row.get('time_to_collision_zero_count')} | "
            f"{row.get('ego_progress_mean')} | {row.get('selection_bit_count')} |"
        )
    lines.extend([
        "",
        "Oracle rows are analysis-only and use per-sample labels/metrics.",
        f"Selector error cases written: `{len(error_cases)}`.",
    ])
    (args.output_dir / "selected_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "threshold": threshold, "summaries": summaries}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
