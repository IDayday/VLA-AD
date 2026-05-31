#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CSV_FIELDS = [
    "run_name",
    "start_checkpoint",
    "condition",
    "precision_mode",
    "true_bf16_weights",
    "optimizer_restored",
    "lr_scheduler",
    "lr_action_head",
    "lr_expert",
    "lr_expert_gate",
    "start_epoch_offset",
    "optimizer_steps",
    "effective_batch_size",
    "PDMS",
    "trajectory_l1",
    "NC",
    "DAC",
    "TTC",
    "comfort",
    "EP",
    "DDC",
    "model_param_dtype_counts",
    "checkpoint_state_dtype_counts",
    "output_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize A4 minimal continuation matrix metrics.")
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def json_cell(value: Any) -> str:
    if value in (None, ""):
        return ""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def run_parts(run_name: str) -> Tuple[str, str]:
    if run_name.startswith("start120_"):
        start = "120k"
        condition = run_name[len("start120_"):]
    elif run_name.startswith("start160_"):
        start = "160k"
        condition = run_name[len("start160_"):]
    else:
        start = ""
        condition = run_name
    return start, condition


def start_checkpoint_path(start: str, meta: Dict[str, Any]) -> str:
    if start == "120k":
        return str(meta.get("ckpt_120k") or "120k")
    if start == "160k":
        return str(meta.get("ckpt_160k") or "160k")
    return ""


def precision_mode(report: Dict[str, Any], run_name: str) -> str:
    if not report:
        return "baseline_eval_fp32"
    requested = report.get("requested_precision")
    true_bf16 = bool(report.get("true_bf16_weights"))
    if requested == "bf16" and true_bf16:
        return "true_bf16_weights"
    if requested == "bf16":
        return "fp32_weights_bf16_autocast"
    return str(requested or "")


def parse_final_optimizer_steps(path: Path) -> Optional[int]:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Optimizer steps:"):
            value = line.split(":", 1)[1].strip()
            try:
                return int(value)
            except ValueError:
                return None
    return None


def effective_batch(train_args: Dict[str, Any]) -> str:
    if not train_args:
        return ""
    batch = train_args.get("batch_size")
    grad_acc = train_args.get("gradient_accumulation_steps")
    world = train_args.get("world_size", 1)
    if batch is None or grad_acc is None:
        return ""
    return str(int(batch) * int(grad_acc) * int(world))


def metric_value(metrics: Dict[str, Any], key: str) -> Any:
    if key == "PDMS":
        return metrics.get("PDMS", metrics.get("pdm_score"))
    return metrics.get(key)


def build_rows(root: Path, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metrics_path in sorted((root / "eval").glob("*/metrics.json")):
        run_name = metrics_path.parent.name
        run_dir = root / "runs" / run_name
        metrics = load_json(metrics_path)
        train_args = load_json(run_dir / "train_args.json")
        precision = load_json(run_dir / "precision_report.json")
        start, condition = run_parts(run_name)
        final_steps = parse_final_optimizer_steps(run_dir / "final_report.md")
        optimizer_steps = final_steps if final_steps is not None else train_args.get("num_optimizer_steps", "")
        output_dir = str(run_dir if run_dir.is_dir() else metrics_path.parent)
        row = {
            "run_name": run_name,
            "start_checkpoint": start_checkpoint_path(start, meta),
            "condition": condition,
            "precision_mode": precision_mode(precision, run_name),
            "true_bf16_weights": precision.get("true_bf16_weights", ""),
            "optimizer_restored": precision.get("optimizer_restored", False if train_args else ""),
            "lr_scheduler": train_args.get("lr_scheduler", ""),
            "lr_action_head": train_args.get("lr_action_head", ""),
            "lr_expert": train_args.get("lr_expert", ""),
            "lr_expert_gate": train_args.get("lr_expert_gate", ""),
            "start_epoch_offset": train_args.get("lr_scheduler_start_epoch", ""),
            "optimizer_steps": optimizer_steps,
            "effective_batch_size": effective_batch(train_args),
            "PDMS": metric_value(metrics, "PDMS"),
            "trajectory_l1": metric_value(metrics, "trajectory_l1"),
            "NC": metric_value(metrics, "NC"),
            "DAC": metric_value(metrics, "DAC"),
            "TTC": metric_value(metrics, "TTC"),
            "comfort": metric_value(metrics, "comfort"),
            "EP": metric_value(metrics, "EP"),
            "DDC": metric_value(metrics, "DDC"),
            "model_param_dtype_counts": json_cell(precision.get("model_param_dtype_counts")),
            "checkpoint_state_dtype_counts": json_cell(precision.get("loaded_checkpoint_state_dtype_counts")),
            "output_dir": output_dir,
        }
        rows.append(row)
    return rows


def as_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return ""
    return f"{number:.6f}"


def signed_fmt(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:+.6f}"


def comparison(rows_by_name: Dict[str, Dict[str, Any]], left: str, right: str) -> Dict[str, Any]:
    left_row = rows_by_name.get(left, {})
    right_row = rows_by_name.get(right, {})
    left_pdms = as_float(left_row.get("PDMS"))
    right_pdms = as_float(right_row.get("PDMS"))
    left_l1 = as_float(left_row.get("trajectory_l1"))
    right_l1 = as_float(right_row.get("trajectory_l1"))
    return {
        "comparison": f"{left} - {right}",
        "left_pdms": left_pdms,
        "right_pdms": right_pdms,
        "pdms_delta": None if left_pdms is None or right_pdms is None else left_pdms - right_pdms,
        "left_l1": left_l1,
        "right_l1": right_l1,
        "trajectory_l1_delta": None if left_l1 is None or right_l1 is None else left_l1 - right_l1,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def markdown_table(rows: List[Dict[str, Any]]) -> List[str]:
    lines = [
        "| run_name | condition | precision_mode | scheduler | opt_steps | eff_batch | PDMS | trajectory_l1 | NC | DAC | TTC | comfort | EP | DDC |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run_name} | {condition} | {precision_mode} | {lr_scheduler} | {optimizer_steps} | {effective_batch_size} | "
            "{PDMS} | {trajectory_l1} | {NC} | {DAC} | {TTC} | {comfort} | {EP} | {DDC} |".format(
                run_name=row.get("run_name", ""),
                condition=row.get("condition", ""),
                precision_mode=row.get("precision_mode", ""),
                lr_scheduler=row.get("lr_scheduler", ""),
                optimizer_steps=row.get("optimizer_steps", ""),
                effective_batch_size=row.get("effective_batch_size", ""),
                PDMS=fmt(row.get("PDMS")),
                trajectory_l1=fmt(row.get("trajectory_l1")),
                NC=fmt(row.get("NC")),
                DAC=fmt(row.get("DAC")),
                TTC=fmt(row.get("TTC")),
                comfort=fmt(row.get("comfort")),
                EP=fmt(row.get("EP")),
                DDC=fmt(row.get("DDC")),
            )
        )
    return lines


def comparison_table(comparisons: List[Dict[str, Any]]) -> List[str]:
    lines = [
        "| comparison | left_PDMS | right_PDMS | delta_PDMS | left_trajectory_l1 | right_trajectory_l1 | delta_trajectory_l1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        lines.append(
            "| {comparison} | {left_pdms} | {right_pdms} | {pdms_delta} | {left_l1} | {right_l1} | {trajectory_l1_delta} |".format(
                comparison=item["comparison"],
                left_pdms=fmt(item["left_pdms"]),
                right_pdms=fmt(item["right_pdms"]),
                pdms_delta=signed_fmt(item["pdms_delta"]),
                left_l1=fmt(item["left_l1"]),
                right_l1=fmt(item["right_l1"]),
                trajectory_l1_delta=signed_fmt(item["trajectory_l1_delta"]),
            )
        )
    return lines


def readout_lines(rows_by_name: Dict[str, Dict[str, Any]], comparisons: Dict[str, Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for prefix in ("start120", "start160"):
        label = "120k" if prefix == "start120" else "160k"
        c2_c1 = comparisons.get(f"{prefix}_C2_minus_C1", {})
        c3_c2 = comparisons.get(f"{prefix}_C3_minus_C2", {})
        delta_c2_c1 = c2_c1.get("pdms_delta")
        delta_c3_c2 = c3_c2.get("pdms_delta")
        if delta_c2_c1 is not None:
            if delta_c2_c1 > 0:
                lines.append(f"- {label}: C2 - C1 PDMS = {delta_c2_c1:+.6f}; fp32 weights + bf16 autocast continuation is higher than true bf16 continuation.")
            else:
                lines.append(f"- {label}: C2 - C1 PDMS = {delta_c2_c1:+.6f}; fp32 weights + bf16 autocast continuation is not higher than true bf16 continuation.")
        if delta_c3_c2 is not None:
            if delta_c3_c2 > 0:
                lines.append(f"- {label}: C3 - C2 PDMS = {delta_c3_c2:+.6f}; higher/constant expert LR continuation is higher than mixed cosine-tail continuation.")
            else:
                lines.append(f"- {label}: C3 - C2 PDMS = {delta_c3_c2:+.6f}; LR restart/constant is not higher here, so structure or alignment floor may need checking.")
    c3_120 = as_float(rows_by_name.get("start120_C3_mixed_const_expertlr", {}).get("PDMS"))
    c3_160 = as_float(rows_by_name.get("start160_C3_mixed_const_expertlr", {}).get("PDMS"))
    if c3_120 is not None and c3_160 is not None:
        delta = c3_120 - c3_160
        if delta > 0:
            lines.append(f"- start120_C3 - start160_C3 PDMS = {delta:+.6f}; earlier switch to mixed precision / expert LR is higher in this run.")
        else:
            lines.append(f"- start120_C3 - start160_C3 PDMS = {delta:+.6f}; earlier switch is not higher in this run.")
    return lines


def write_markdown(path: Path, root: Path, rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    rows_by_name = {row["run_name"]: row for row in rows}
    comparison_specs = {
        "120k": [
            ("start120_C2_mixed_tail", "start120_C1_truebf16_tail", "start120_C2_minus_C1"),
            ("start120_C3_mixed_const_expertlr", "start120_C2_mixed_tail", "start120_C3_minus_C2"),
            ("start120_C3_mixed_const_expertlr", "start120_baseline", "start120_C3_minus_baseline"),
        ],
        "160k": [
            ("start160_C2_mixed_tail", "start160_C1_truebf16_tail", "start160_C2_minus_C1"),
            ("start160_C3_mixed_const_expertlr", "start160_C2_mixed_tail", "start160_C3_minus_C2"),
            ("start160_C3_mixed_const_expertlr", "start160_baseline", "start160_C3_minus_baseline"),
        ],
    }
    comparison_by_key: Dict[str, Dict[str, Any]] = {}
    lines = [
        "# A4 Minimal Continuation Summary",
        "",
        f"Root: `{root}`",
    ]
    eval_max_samples = meta.get("eval_max_samples")
    if eval_max_samples not in (None, ""):
        lines.extend([
            "",
            f"Subset pilot: EVAL_MAX_SAMPLES={eval_max_samples}. Do not treat this as final full-eval evidence.",
        ])
    else:
        lines.extend(["", "Evaluation scope: full eval requested."])
    lines.extend(["", "## Core Metrics", "", *markdown_table(rows), ""])
    lines.append("## Paired Comparisons")
    for label, specs in comparison_specs.items():
        comparisons = []
        for left, right, key in specs:
            item = comparison(rows_by_name, left, right)
            comparison_by_key[key] = item
            comparisons.append(item)
        lines.extend(["", f"### For {label}", "", *comparison_table(comparisons)])
    readouts = readout_lines(rows_by_name, comparison_by_key)
    if readouts:
        lines.extend(["", "## Objective Readouts", "", *readouts])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.root
    meta = load_json(root / "matrix_meta.json")
    rows = build_rows(root, meta)
    root.mkdir(parents=True, exist_ok=True)
    write_csv(root / "summary.csv", rows)
    write_markdown(root / "summary.md", root, rows, meta)
    print(root / "summary.md")
    print(root / "summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
