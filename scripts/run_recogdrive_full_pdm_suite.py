#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


RUNS: Dict[str, Dict[str, str | None]] = {
    "base_il": {
        "config": "configs/ablations/recogdrive2b_A0_base_no_expert.yaml",
        "checkpoint": "checkpoints/recogdrive/ReCogDrive-2B-IL",
        "train_dir": None,
    },
    "a0_no_expert": {
        "config": "configs/ablations/recogdrive2b_A0_base_no_expert.yaml",
        "checkpoint": "experiments/recogdrive_expert/main_il_full_v1_A0_no_expert_20260527_1500/best.ckpt",
        "train_dir": "experiments/recogdrive_expert/main_il_full_v1_A0_no_expert_20260527_1500",
    },
    "a1_jepa_only": {
        "config": "configs/ablations/recogdrive2b_A1_jepa_only.yaml",
        "checkpoint": "experiments/recogdrive_expert/main_il_full_v1_A1_jepa_only_20260527_1502/best.ckpt",
        "train_dir": "experiments/recogdrive_expert/main_il_full_v1_A1_jepa_only_20260527_1502",
    },
    "a2_vggt_only": {
        "config": "configs/ablations/recogdrive2b_A2_vggt_only.yaml",
        "checkpoint": "experiments/recogdrive_expert/main_il_full_v1_A2_vggt_only_20260527_1502/best.ckpt",
        "train_dir": "experiments/recogdrive_expert/main_il_full_v1_A2_vggt_only_20260527_1502",
    },
    "a3_context_only": {
        "config": "configs/ablations/recogdrive2b_A3_context_only.yaml",
        "checkpoint": "experiments/recogdrive_expert/main_il_full_v1_A3_context_only_20260527_1502/best.ckpt",
        "train_dir": "experiments/recogdrive_expert/main_il_full_v1_A3_context_only_20260527_1502",
    },
    "a4_jepa_vggt": {
        "config": "configs/recogdrive2b_expert768_il.yaml",
        "checkpoint": "experiments/recogdrive_expert/main_il_full_v1_A4_20260527_1455/best.ckpt",
        "train_dir": "experiments/recogdrive_expert/main_il_full_v1_A4_20260527_1455",
    },
}


SUMMARY_FIELDS = [
    "run",
    "checkpoint",
    "num_samples",
    "num_pdm_valid",
    "pdm_score",
    "NC",
    "DAC",
    "TTC",
    "comfort",
    "EP",
    "DDC",
    "trajectory_l1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for ReCogDrive full runs, evaluate PDM, and aggregate results.")
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/project/VLA-AD"))
    parser.add_argument("--chunk-cache-root", type=Path, default=Path("/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1"))
    parser.add_argument("--chunk-name-pattern", default="navtest_full_chunk_*")
    parser.add_argument("--metric-cache-dir", type=Path, default=Path("/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--runs", default="base_il,a0_no_expert,a1_jepa_only,a2_vggt_only,a3_context_only,a4_jepa_vggt")
    parser.add_argument("--run-spec-json", type=Path, default=None)
    parser.add_argument("--root-report-name", default=None)
    parser.add_argument("--expected-metric-caches", type=int, default=12146)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--python", default="/root/miniconda3/envs/navsim/bin/python")
    return parser.parse_args()


def load_run_specs(args: argparse.Namespace) -> Dict[str, Dict[str, str | None]]:
    if args.run_spec_json is None:
        return RUNS
    path = args.run_spec_json
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain an object mapping run names to specs.")
    required = {"config", "checkpoint", "train_dir"}
    result: Dict[str, Dict[str, str | None]] = {}
    for run, spec in data.items():
        if not isinstance(spec, dict):
            raise TypeError(f"Run spec for {run!r} must be an object.")
        missing = sorted(required - set(spec))
        if missing:
            raise ValueError(f"Run spec for {run!r} missing keys: {missing}")
        result[str(run)] = {
            "config": None if spec["config"] is None else str(spec["config"]),
            "checkpoint": None if spec["checkpoint"] is None else str(spec["checkpoint"]),
            "train_dir": None if spec["train_dir"] is None else str(spec["train_dir"]),
        }
    return result


def metric_cache_count(metric_cache_dir: Path) -> int:
    if not metric_cache_dir.exists():
        return 0
    return sum(1 for _ in metric_cache_dir.rglob("metric_cache.pkl"))


def read_last_training_row(train_dir: Path) -> Dict[str, Any]:
    path = train_dir / "train_log.jsonl"
    last: Dict[str, Any] = {}
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                last = json.loads(line)
    return last


def final_report_failure(train_dir: Path) -> str | None:
    path = train_dir / "final_report.md"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if line.lower().startswith("- failure:"):
            return line.split(":", 1)[1].strip() or "unknown failure"
    return None


def wait_for_ready(
    args: argparse.Namespace,
    run_specs: Dict[str, Dict[str, str | None]],
    selected_runs: List[str],
    output_root: Path,
) -> None:
    while True:
        count = metric_cache_count(args.metric_cache_dir)
        missing: List[str] = []
        for run in selected_runs:
            spec = run_specs[run]
            train_dir_value = spec["train_dir"]
            if train_dir_value is None:
                continue
            train_dir = args.project_root / train_dir_value
            if not (train_dir / "final_report.md").is_file():
                last = read_last_training_row(train_dir)
                step = last.get("step", 0)
                chunk = last.get("chunk", "n/a")
                missing.append(f"{run}: training not finished (step={step}, chunk={chunk})")
            else:
                failure = final_report_failure(train_dir)
                if failure is not None:
                    missing.append(f"{run}: training final report contains failure ({failure})")
            checkpoint = args.project_root / str(spec["checkpoint"])
            if not checkpoint.exists():
                missing.append(f"{run}: missing checkpoint {checkpoint}")
        if count < args.expected_metric_caches:
            missing.append(f"metric cache {count}/{args.expected_metric_caches}")
        heartbeat = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "metric_cache_count": count,
            "expected_metric_caches": args.expected_metric_caches,
            "missing": missing,
        }
        (output_root / "wait_status.json").write_text(json.dumps(heartbeat, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(heartbeat, sort_keys=True), flush=True)
        if not missing:
            return
        if not args.wait:
            raise RuntimeError("Prerequisites are not ready and --wait was not set.")
        time.sleep(args.poll_seconds)


def run_eval(
    args: argparse.Namespace,
    run_specs: Dict[str, Dict[str, str | None]],
    run: str,
    output_root: Path,
) -> Dict[str, Any]:
    spec = run_specs[run]
    out_dir = output_root / run
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python,
        "scripts/eval_recogdrive_expert_pdm.py",
        "--config",
        str(args.project_root / str(spec["config"])),
        "--checkpoint",
        str(args.project_root / str(spec["checkpoint"])),
        "--chunk-cache-root",
        str(args.chunk_cache_root),
        "--chunk-name-pattern",
        args.chunk_name_pattern,
        "--metric-cache-dir",
        str(args.metric_cache_dir),
        "--precision",
        args.precision,
        "--output-dir",
        str(out_dir),
    ]
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{args.project_root}:{env.get('PYTHONPATH', '')}"
    print("RUN", " ".join(cmd), flush=True)
    with (out_dir / "console.log").open("w", encoding="utf-8") as log_fp:
        subprocess.run(cmd, cwd=args.project_root, env=env, stdout=log_fp, stderr=subprocess.STDOUT, check=True)
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics["run"] = run
    return metrics


def write_summary(output_root: Path, rows: List[Dict[str, Any]]) -> None:
    csv_path = output_root / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in SUMMARY_FIELDS})
    lines = [
        "# ReCogDrive Full Navtest PDM Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| Run | Samples | Valid PDM | PDMS | NC | DAC | TTC | Comfort | EP | DDC | Traj L1 | Checkpoint |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {run} | {num_samples} | {num_pdm_valid} | {pdm_score} | {NC} | {DAC} | {TTC} | {comfort} | {EP} | {DDC} | {trajectory_l1} | `{checkpoint}` |".format(
                **{key: row.get(key) for key in SUMMARY_FIELDS}
            )
        )
    lines.extend([
        "",
        "Interpretation guardrail: compare trained expert runs against `a0_no_expert`, not only against `base_il`.",
        "A JEPA/VGGT improvement claim requires A4 to beat the same-budget A0 no-expert control on PDMS, with ablations supporting attribution.",
    ])
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_float(row: Dict[str, Any] | None, key: str) -> float | None:
    if row is None:
        return None
    value = row.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_metric(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "n/a"


def fmt_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.6f}"


def write_interpretation_report(args: argparse.Namespace, output_root: Path, rows: List[Dict[str, Any]]) -> None:
    by_run = {str(row.get("run")): row for row in rows}
    a0 = by_run.get("a0_no_expert")
    a4 = by_run.get("a4_jepa_vggt")
    base = by_run.get("base_il")
    a0_pdms = metric_float(a0, "pdm_score")
    a4_pdms = metric_float(a4, "pdm_score")
    base_pdms = metric_float(base, "pdm_score")
    delta_a4_a0 = None if a0_pdms is None or a4_pdms is None else a4_pdms - a0_pdms
    delta_a4_base = None if base_pdms is None or a4_pdms is None else a4_pdms - base_pdms
    scope = "full navtest" if args.max_samples is None else f"navtest subset max_samples={args.max_samples}"
    if delta_a4_a0 is None:
        conclusion = "结论：证据不足，缺少 A4 或同预算 A0 的 PDMS。"
    elif delta_a4_a0 > 0:
        conclusion = "结论：A4 在该评估范围内高于同预算 A0，支持 JEPA/VGGT 注入带来改进。"
    else:
        conclusion = "结论：A4 在该评估范围内未超过同预算 A0，不能声称 JEPA/VGGT 注入带来改进。"
    if args.max_samples is not None:
        conclusion += " 这是子集结果，最终主张仍以 full navtest 为准。"

    lines = [
        "# ReCogDrive JEPA/VGGT PDM Experiment Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Evaluation scope: {scope}",
        f"Output root: `{output_root}`",
        f"Metric cache: `{args.metric_cache_dir}`",
        f"Chunk pattern: `{args.chunk_name_pattern}`",
        "",
        "## Decision Rule",
        "",
        "只有当 A4 JEPA+VGGT 在 PDMS 上超过同训练预算的 A0 no-expert control 时，才认为外部模型知识注入改进了 ReCogDrive。Base-IL 只作为原始基线，不作为公平训练预算对照。",
        "",
        "## Result Table",
        "",
        "| Run | Samples | Valid PDM | PDMS | NC | DAC | TTC | Comfort | EP | DDC | Traj L1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run} | {num_samples} | {num_pdm_valid} | {pdm_score} | {NC} | {DAC} | {TTC} | {comfort} | {EP} | {DDC} | {trajectory_l1} |".format(
                run=row.get("run", "n/a"),
                num_samples=row.get("num_samples", "n/a"),
                num_pdm_valid=row.get("num_pdm_valid", "n/a"),
                pdm_score=fmt_metric(row.get("pdm_score")),
                NC=fmt_metric(row.get("NC")),
                DAC=fmt_metric(row.get("DAC")),
                TTC=fmt_metric(row.get("TTC")),
                comfort=fmt_metric(row.get("comfort")),
                EP=fmt_metric(row.get("EP")),
                DDC=fmt_metric(row.get("DDC")),
                trajectory_l1=fmt_metric(row.get("trajectory_l1")),
            )
        )
    lines.extend([
        "",
        "## Primary Comparison",
        "",
        f"- A4 PDMS: `{fmt_metric(a4_pdms)}`",
        f"- A0 no-expert PDMS: `{fmt_metric(a0_pdms)}`",
        f"- Delta A4 - A0: `{fmt_delta(delta_a4_a0)}`",
        f"- Delta A4 - Base-IL: `{fmt_delta(delta_a4_base)}`",
        "",
        conclusion,
        "",
        "## Artifacts",
        "",
        f"- Summary CSV: `{output_root / 'summary.csv'}`",
        f"- Summary MD: `{output_root / 'summary.md'}`",
        f"- Per-run outputs: `{output_root}`",
    ])
    text = "\n".join(lines) + "\n"
    (output_root / "interpretation_report.md").write_text(text, encoding="utf-8")
    report_dir = args.project_root / "experiments/recogdrive_expert"
    if args.root_report_name:
        report_name = args.root_report_name
    else:
        report_name = "FINAL_JEPA_VGGT_PDM_REPORT.md" if args.max_samples is None else f"INTERIM_JEPA_VGGT_PDM_REPORT_max{args.max_samples}.md"
    (report_dir / report_name).write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_specs = load_run_specs(args)
    selected_runs = [run.strip() for run in args.runs.split(",") if run.strip()]
    unknown = [run for run in selected_runs if run not in run_specs]
    if unknown:
        raise ValueError(f"Unknown runs: {unknown}")
    output_root = args.output_root
    if output_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = args.project_root / f"experiments/recogdrive_expert/pdm_flow/full_navtest_suite_{stamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "suite_args.json").write_text(json.dumps(vars(args), default=str, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "run_specs.json").write_text(json.dumps(run_specs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    wait_for_ready(args, run_specs, selected_runs, output_root)
    rows = [run_eval(args, run_specs, run, output_root) for run in selected_runs]
    write_summary(output_root, rows)
    write_interpretation_report(args, output_root, rows)
    print(f"Wrote summary to {output_root / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
