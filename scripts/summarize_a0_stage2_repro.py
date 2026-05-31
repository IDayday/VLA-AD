#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


METRIC_KEYS = ("PDMS", "trajectory_l1", "NC", "DAC", "TTC", "comfort", "EP", "DDC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize A0 Stage2 reproduction experiments.")
    parser.add_argument("--official-dir", type=Path, required=True, help="A0-official-aligned output dir.")
    parser.add_argument("--local-dir", type=Path, required=True, help="A0-local-fixed output dir.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--official-public-pdms", type=float, default=0.863)
    parser.add_argument("--old-a0-pdms", type=float, default=0.8541272154019499)
    parser.add_argument("--close-threshold", type=float, default=0.005)
    parser.add_argument("--meaningful-threshold", type=float, default=0.003)
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def eval_rows(root: Path, run_name: str, condition: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    eval_root = root / "eval"
    if not eval_root.is_dir():
        eval_root = root
    for metrics_path in sorted(eval_root.glob("*/metrics.json")):
        checkpoint = metrics_path.parent.name
        metrics = read_json(metrics_path)
        row: Dict[str, Any] = {
            "run_name": run_name,
            "checkpoint": checkpoint,
            "condition": condition,
            "output_dir": str(metrics_path.parent),
        }
        for key in METRIC_KEYS:
            row[key] = metrics.get(key, "")
        rows.append(row)
    return rows


def precision_metadata(root: Path) -> Dict[str, Any]:
    report = read_json(root / "precision_report.json")
    args = read_json(root / "train_args.json")
    out: Dict[str, Any] = {
        "precision_mode": report.get("requested_precision", ""),
        "model_param_dtype_counts": json.dumps(report.get("model_param_dtype_counts", {}), sort_keys=True),
        "loader_mode": report.get("loader_mode", ""),
        "scheduler": "",
        "lr": "",
        "epochs": "",
        "effective_batch_size": "",
    }
    if "agent" in args and "trainer" in args:
        batch = int(args.get("dataloader", {}).get("params", {}).get("batch_size", 0) or 0)
        devices = int(args.get("trainer", {}).get("params", {}).get("devices", 1) or 1)
        accum = int(args.get("trainer", {}).get("params", {}).get("accumulate_grad_batches", 1) or 1)
        out.update({
            "lr": args.get("agent", {}).get("lr", ""),
            "scheduler": "WarmupCosLR",
            "epochs": args.get("trainer", {}).get("params", {}).get("max_epochs", ""),
            "effective_batch_size": batch * devices * accum if batch else "",
        })
    else:
        batch = int(args.get("batch_size", 0) or 0)
        world = int(args.get("world_size", 1) or 1)
        accum = int(args.get("gradient_accumulation_steps", 1) or 1)
        out.update({
            "lr": args.get("lr_action_head", ""),
            "scheduler": args.get("lr_scheduler", ""),
            "epochs": args.get("global_epochs", ""),
            "effective_batch_size": batch * world * accum if batch else "",
        })
    return out


def best_row(rows: List[Dict[str, Any]], run_name: str) -> Optional[Dict[str, Any]]:
    candidates = [row for row in rows if row.get("run_name") == run_name and row.get("PDMS") not in ("", None)]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row["PDMS"]))


def fmt(value: Any) -> str:
    if value in ("", None):
        return ""
    return f"{float(value):.6f}"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "run_name",
        "checkpoint",
        "condition",
        "PDMS",
        "trajectory_l1",
        "NC",
        "DAC",
        "TTC",
        "comfort",
        "EP",
        "DDC",
        "precision_mode",
        "model_param_dtype_counts",
        "effective_batch_size",
        "lr",
        "scheduler",
        "epochs",
        "output_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def md_table(rows: List[Dict[str, Any]]) -> List[str]:
    lines = [
        "| run_name | checkpoint | condition | PDMS | trajectory_l1 | NC | DAC | TTC | comfort | EP | DDC | precision | dtype_counts | eff_batch | lr | scheduler | epochs |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run_name} | {checkpoint} | {condition} | {PDMS} | {trajectory_l1} | {NC} | {DAC} | {TTC} | {comfort} | {EP} | {DDC} | {precision_mode} | `{model_param_dtype_counts}` | {effective_batch_size} | {lr} | {scheduler} | {epochs} |".format(
                run_name=row.get("run_name", ""),
                checkpoint=row.get("checkpoint", ""),
                condition=row.get("condition", ""),
                PDMS=fmt(row.get("PDMS")),
                trajectory_l1=fmt(row.get("trajectory_l1")),
                NC=fmt(row.get("NC")),
                DAC=fmt(row.get("DAC")),
                TTC=fmt(row.get("TTC")),
                comfort=fmt(row.get("comfort")),
                EP=fmt(row.get("EP")),
                DDC=fmt(row.get("DDC")),
                precision_mode=row.get("precision_mode", ""),
                model_param_dtype_counts=row.get("model_param_dtype_counts", ""),
                effective_batch_size=row.get("effective_batch_size", ""),
                lr=row.get("lr", ""),
                scheduler=row.get("scheduler", ""),
                epochs=row.get("epochs", ""),
            )
        )
    return lines


def judgement(official_best: Optional[Dict[str, Any]], local_best: Optional[Dict[str, Any]], args: argparse.Namespace) -> List[str]:
    lines: List[str] = []
    official_pdms = float(official_best["PDMS"]) if official_best else None
    local_pdms = float(local_best["PDMS"]) if local_best else None
    if official_best:
        gap = official_pdms - args.official_public_pdms
        lines.append(f"- A0-official-aligned best: `{official_best['checkpoint']}` PDMS `{official_pdms:.6f}`; gap vs public `0.863000` is `{gap:+.6f}`.")
    else:
        lines.append("- A0-official-aligned has no completed metrics yet.")
    if local_best:
        gap = local_pdms - args.official_public_pdms
        lines.append(f"- A0-local-fixed best: `{local_best['checkpoint']}` PDMS `{local_pdms:.6f}`; gap vs public `0.863000` is `{gap:+.6f}`.")
    else:
        lines.append("- A0-local-fixed has no completed metrics yet.")
    if official_pdms is not None and local_pdms is not None:
        if abs(official_pdms - args.official_public_pdms) <= args.close_threshold and official_pdms - local_pdms > args.meaningful_threshold:
            lines.append("- Key judgment: official-aligned is close to 86.3 while local-fixed is lower, so the local chunked training loop is the first suspect.")
        elif local_pdms - args.old_a0_pdms > args.meaningful_threshold:
            lines.append("- Key judgment: local-fixed is meaningfully higher than old A0, so true bf16 weights were an important factor.")
        elif abs(official_pdms - args.official_public_pdms) > args.close_threshold and abs(local_pdms - args.official_public_pdms) > args.close_threshold:
            lines.append("- Key judgment: both runs are below 86.3; prioritize hidden-state cache parity, train split coverage, and sample construction.")
        elif abs(official_pdms - args.official_public_pdms) <= args.close_threshold:
            lines.append("- Key judgment: official-aligned is close to 86.3; future A4 work should use this training protocol as the standard baseline.")
        else:
            lines.append("- Key judgment: results do not isolate one cause yet; inspect per-checkpoint curves and cache/split parity.")
    return lines


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = eval_rows(args.official_dir, "a0_official_aligned", "official-aligned") + eval_rows(args.local_dir, "a0_local_fixed", "local-fixed")
    official_meta = precision_metadata(args.official_dir)
    local_meta = precision_metadata(args.local_dir)
    for row in rows:
        row.update(official_meta if row["run_name"] == "a0_official_aligned" else local_meta)
        if row["run_name"] == "a0_official_aligned" and official_meta.get("loader_mode"):
            row["condition"] = official_meta["loader_mode"]
    rows.sort(key=lambda row: (row["run_name"], row["checkpoint"]))

    write_csv(args.output_dir / "summary.csv", rows)
    official_best = best_row(rows, "a0_official_aligned")
    local_best = best_row(rows, "a0_local_fixed")

    lines = [
        "# A0 Stage2 Reproduction Summary",
        "",
        f"Official public-weight reference PDMS: `{args.official_public_pdms:.6f}`.",
        "",
        f"A0-official-aligned loader mode: `{official_meta.get('loader_mode') or 'official-loader'}`.",
        "",
        "## Metrics",
        "",
        *md_table(rows),
        "",
        "## Best Checkpoints And Gaps",
        "",
        *judgement(official_best, local_best, args),
        "",
        "## Roots",
        "",
        f"- A0-official-aligned: `{args.official_dir}`",
        f"- A0-local-fixed: `{args.local_dir}`",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_dir / "summary.md")
    print(args.output_dir / "summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
