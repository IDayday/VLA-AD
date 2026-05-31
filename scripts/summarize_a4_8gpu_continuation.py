#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


FIELDS = [
    "run_name",
    "start_checkpoint",
    "condition",
    "true_bf16_weights",
    "lr_scheduler",
    "lr_action_head",
    "lr_expert",
    "lr_expert_gate",
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize A4 8-GPU continuation matrix.")
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


def split_run_name(name: str) -> Tuple[str, str]:
    if name.startswith("start120_"):
        return "120", name[len("start120_"):]
    if name.startswith("start160_"):
        return "160", name[len("start160_"):]
    return "", name


def start_checkpoint(start: str, meta: Dict[str, Any]) -> str:
    if start == "120":
        return str(meta.get("ckpt_120k", ""))
    if start == "160":
        return str(meta.get("ckpt_160k", ""))
    return ""


def final_optimizer_steps(path: Path) -> Optional[int]:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Optimizer steps:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def effective_batch(train_args: Dict[str, Any]) -> str:
    if not train_args:
        return ""
    try:
        return str(
            int(train_args["batch_size"])
            * int(train_args["gradient_accumulation_steps"])
            * int(train_args.get("world_size", 1))
        )
    except (KeyError, TypeError, ValueError):
        return ""


def metric(metrics: Dict[str, Any], key: str) -> Any:
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
        start, condition = split_run_name(run_name)
        steps = final_optimizer_steps(run_dir / "final_report.md")
        row = {
            "run_name": run_name,
            "start_checkpoint": start_checkpoint(start, meta),
            "condition": condition,
            "true_bf16_weights": precision.get("true_bf16_weights", ""),
            "lr_scheduler": train_args.get("lr_scheduler", ""),
            "lr_action_head": train_args.get("lr_action_head", ""),
            "lr_expert": train_args.get("lr_expert", ""),
            "lr_expert_gate": train_args.get("lr_expert_gate", ""),
            "optimizer_steps": steps if steps is not None else train_args.get("num_optimizer_steps", ""),
            "effective_batch_size": effective_batch(train_args),
            "PDMS": metric(metrics, "PDMS"),
            "trajectory_l1": metric(metrics, "trajectory_l1"),
            "NC": metric(metrics, "NC"),
            "DAC": metric(metrics, "DAC"),
            "TTC": metric(metrics, "TTC"),
            "comfort": metric(metrics, "comfort"),
            "EP": metric(metrics, "EP"),
            "DDC": metric(metrics, "DDC"),
            "model_param_dtype_counts": json_cell(precision.get("model_param_dtype_counts")),
            "checkpoint_state_dtype_counts": json_cell(precision.get("loaded_checkpoint_state_dtype_counts")),
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
    return "" if number is None else f"{number:.6f}"


def signed(value: Optional[float]) -> str:
    return "" if value is None else f"{value:+.6f}"


def delta(rows_by_name: Dict[str, Dict[str, Any]], left: str, right: str) -> Dict[str, Any]:
    l = rows_by_name.get(left, {})
    r = rows_by_name.get(right, {})
    lp, rp = as_float(l.get("PDMS")), as_float(r.get("PDMS"))
    ll, rl = as_float(l.get("trajectory_l1")), as_float(r.get("trajectory_l1"))
    return {
        "comparison": f"{left} - {right}",
        "left_pdms": lp,
        "right_pdms": rp,
        "delta_pdms": None if lp is None or rp is None else lp - rp,
        "left_l1": ll,
        "right_l1": rl,
        "delta_l1": None if ll is None or rl is None else ll - rl,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDS})


def core_table(rows: List[Dict[str, Any]]) -> List[str]:
    lines = [
        "| run_name | condition | true_bf16_weights | scheduler | opt_steps | eff_batch | PDMS | trajectory_l1 | NC | DAC | TTC | comfort | EP | DDC |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('run_name', '')} | {row.get('condition', '')} | {row.get('true_bf16_weights', '')} | "
            f"{row.get('lr_scheduler', '')} | {row.get('optimizer_steps', '')} | {row.get('effective_batch_size', '')} | "
            f"{fmt(row.get('PDMS'))} | {fmt(row.get('trajectory_l1'))} | {fmt(row.get('NC'))} | {fmt(row.get('DAC'))} | "
            f"{fmt(row.get('TTC'))} | {fmt(row.get('comfort'))} | {fmt(row.get('EP'))} | {fmt(row.get('DDC'))} |"
        )
    return lines


def delta_table(items: List[Dict[str, Any]]) -> List[str]:
    lines = [
        "| comparison | left_PDMS | right_PDMS | delta_PDMS | left_trajectory_l1 | right_trajectory_l1 | delta_trajectory_l1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in items:
        lines.append(
            f"| {item['comparison']} | {fmt(item['left_pdms'])} | {fmt(item['right_pdms'])} | {signed(item['delta_pdms'])} | "
            f"{fmt(item['left_l1'])} | {fmt(item['right_l1'])} | {signed(item['delta_l1'])} |"
        )
    return lines


def write_md(path: Path, root: Path, rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    rows_by_name = {str(row["run_name"]): row for row in rows}
    lines = ["# A4 8-GPU Continuation Summary", "", f"Root: `{root}`"]
    eval_max = meta.get("eval_max_samples")
    if eval_max not in (None, ""):
        lines.extend(["", f"Subset pilot: EVAL_MAX_SAMPLES={eval_max}. Do not use as final full-eval conclusion."])
    else:
        lines.extend(["", "Evaluation scope: full eval requested."])
    lines.extend(["", "## Core Metrics", "", *core_table(rows), "", "## Paired Deltas"])
    for start in ("120", "160"):
        prefix = f"start{start}"
        if not any(name.startswith(prefix) for name in rows_by_name):
            continue
        items = [
            delta(rows_by_name, f"{prefix}_C2_mixed_tail", f"{prefix}_C1_truebf16_tail"),
            delta(rows_by_name, f"{prefix}_C3_mixed_const_expertlr", f"{prefix}_C2_mixed_tail"),
            delta(rows_by_name, f"{prefix}_C3_mixed_const_expertlr", f"{prefix}_baseline"),
        ]
        lines.extend(["", f"### start{start}", "", *delta_table(items)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    meta = load_json(args.root / "matrix_meta.json")
    rows = build_rows(args.root, meta)
    write_csv(args.root / "summary.csv", rows)
    write_md(args.root / "summary.md", args.root, rows, meta)
    print(args.root / "summary.md")
    print(args.root / "summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
