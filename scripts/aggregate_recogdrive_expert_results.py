#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

METRIC_KEYS = ["pdm_score", "PDMS", "NC", "DAC", "TTC", "comfort", "EP", "progress", "trajectory_l1"]
TRAIN_KEYS = ["total_loss", "diffusion_loss", "jepa_alignment_loss", "vggt_alignment_loss", "jepa_gate", "vggt_gate", "branch_weight_vlm", "branch_weight_jepa", "branch_weight_vggt"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate ReCogDrive expert-token eval metrics and training logs.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def final_train_record(root: Path) -> Dict[str, Any]:
    candidates = list(root.rglob("train_log.jsonl"))
    if not candidates:
        return {}
    path = sorted(candidates)[-1]
    last: Dict[str, Any] = {}
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
                last = json.loads(line)
    last["train_log_path"] = str(path)
    last["training_steps_logged"] = count
    return last


def find_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        try:
            metrics = read_json(metrics_path)
        except Exception as exc:
            print(f"Skipping unreadable metrics {metrics_path}: {exc}")
            continue
        ablation_dir = metrics_path.parent.parent if metrics_path.parent.name == "eval" else metrics_path.parent
        row: Dict[str, Any] = {
            "ablation": ablation_dir.name,
            "metrics_path": str(metrics_path),
            "checkpoint_path": metrics.get("checkpoint"),
            "eval_samples": metrics.get("num_samples"),
            "split": metrics.get("split"),
            "target_teacher_tokens_disabled_in_eval": metrics.get("target_teacher_tokens_disabled_in_eval"),
        }
        for key in METRIC_KEYS:
            row[key] = metrics.get(key)
        train = final_train_record(ablation_dir)
        row["training_samples"] = None
        row["training_steps_logged"] = train.get("training_steps_logged")
        for key in TRAIN_KEYS:
            row[f"final_{key}"] = train.get(key)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["ablation", "training_samples", "eval_samples", "checkpoint_path", "split", "target_teacher_tokens_disabled_in_eval", *METRIC_KEYS, "training_steps_logged", *(f"final_{key}" for key in TRAIN_KEYS), "metrics_path"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def primary_value(row: Dict[str, Any]) -> tuple[Optional[float], bool]:
    pdms = row.get("pdm_score") if row.get("pdm_score") is not None else row.get("PDMS")
    if pdms is not None:
        return float(pdms), True
    if row.get("trajectory_l1") is not None:
        return float(row["trajectory_l1"]), False
    return None, True


def compare(rows: List[Dict[str, Any]], left_contains: str, right_contains: str) -> str:
    left = next((row for row in rows if left_contains in row["ablation"]), None)
    right = next((row for row in rows if right_contains in row["ablation"]), None)
    return compare_rows(left, right)


def compare_rows(left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]) -> str:
    if left is None or right is None:
        return "not available"
    lval, higher_better = primary_value(left)
    rval, _ = primary_value(right)
    if lval is None or rval is None:
        return "not available"
    improved = lval > rval if higher_better else lval < rval
    metric = "PDMS" if higher_better else "trajectory_l1"
    return f"{'improved' if improved else 'did not improve'} ({metric}: {fmt(lval)} vs {fmt(rval)})"


def first_row(rows: List[Dict[str, Any]], contains: str) -> Optional[Dict[str, Any]]:
    return next((row for row in rows if contains in row["ablation"]), None)


def write_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# ReCogDrive Expert Ablation Summary", ""]
    if not rows:
        lines.extend(["No metrics.json files found.", ""])
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    lines.extend([
        "All conclusions below are same-pipeline subset conclusions only unless full NAVSIM evaluation artifacts are present.",
        "",
        "| Ablation | Train Steps | Eval Samples | PDMS | NC | DAC | TTC | Comfort | EP/Progress | Traj L1 | Diff Loss | JEPA Align | VGGT Align | Gates | Branch Weights | Checkpoint |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ])
    for row in rows:
        progress = row.get("EP") if row.get("EP") is not None else row.get("progress")
        gates = f"J {fmt(row.get('final_jepa_gate'))} / V {fmt(row.get('final_vggt_gate'))}"
        branches = f"VLM {fmt(row.get('final_branch_weight_vlm'))} / JEPA {fmt(row.get('final_branch_weight_jepa'))} / VGGT {fmt(row.get('final_branch_weight_vggt'))}"
        pdms = row.get("pdm_score") if row.get("pdm_score") is not None else row.get("PDMS")
        lines.append(
            f"| {row['ablation']} | {fmt(row.get('training_steps_logged'))} | {fmt(row.get('eval_samples'))} | {fmt(pdms)} | {fmt(row.get('NC'))} | {fmt(row.get('DAC'))} | {fmt(row.get('TTC'))} | {fmt(row.get('comfort'))} | {fmt(progress)} | {fmt(row.get('trajectory_l1'))} | {fmt(row.get('final_diffusion_loss'))} | {fmt(row.get('final_jepa_alignment_loss'))} | {fmt(row.get('final_vggt_alignment_loss'))} | {gates} | {branches} | `{row.get('checkpoint_path')}` |"
        )
    lines.extend(["", "## Initial Conclusions", ""])
    base_a0 = first_row(rows, "A0_base_no_expert_eval_256")
    fair_a0 = first_row(rows, "A0_no_expert_finetune_eval_256")
    control = fair_a0 or base_a0
    control_name = "same-budget A0 no-expert finetune" if fair_a0 is not None else "Base-IL A0"
    a1 = first_row(rows, "A1_jepa_only_eval_256")
    a4 = first_row(rows, "A4_full_eval_256")
    lines.append(f"- Control baseline for trained ablations: {control_name}.")
    lines.append(f"- JEPA-only vs {control_name}: {compare_rows(a1, control)}")
    lines.append(f"- Full A4 vs {control_name}: {compare_rows(a4, control)}")
    if base_a0 is not None and fair_a0 is not None:
        lines.append(f"- Same-budget A0 finetune vs Base-IL A0: {compare_rows(fair_a0, base_a0)}")
    lines.append(f"- VGGT-only vs {control_name}: {compare_rows(first_row(rows, 'A2'), control)}")
    lines.append(f"- Full A4 vs context-only A3: {compare(rows, 'A4', 'A3')}")
    lines.append(f"- VGGT 3x4 pooling vs global pooling: {compare(rows, 'A4', 'A5')}")
    max_eval = max((row.get("eval_samples") or 0) for row in rows)
    lines.append(f"- Statistical strength: weak if this is only a {max_eval}-sample subset; do not overclaim without larger/full navtest.")
    lines.append("- Recommended next run: increase train chunks and evaluate on at least 1024 samples, then full navtest if resources allow.")
    target_ok = all(row.get("target_teacher_tokens_disabled_in_eval") is True for row in rows)
    lines.append(f"- Target teacher tokens disabled in eval artifacts: {target_ok}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = find_rows(args.input_root)
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv} and {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
