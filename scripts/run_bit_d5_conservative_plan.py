#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIT_WORK_ROOT = Path(os.environ.get("BIT_WORK_ROOT", "/mnt/project/bit_drive_left_tail"))
DEFAULT_BIT_EXP_ROOT = Path(os.environ.get("BIT_EXP_ROOT", DEFAULT_BIT_WORK_ROOT / "experiments/bit_drive"))
DEFAULT_SHARED_ROOT = Path(os.environ.get("VLA_AD_SHARED_ROOT", "/mnt/project/VLA-AD"))
SHARED_EXPERIMENT_ROOT = DEFAULT_SHARED_ROOT / "experiments"
DEFAULT_CHECKPOINT_ROOT = Path(os.environ.get("CHECKPOINT_ROOT", DEFAULT_SHARED_ROOT / "checkpoints"))
DEFAULT_SHARED_CHUNK_CACHE_ROOT = Path(
    os.environ.get("SHARED_CHUNK_CACHE_ROOT", DEFAULT_SHARED_ROOT / "cache/recogdrive_expert_chunks/full_v1")
)


RUNS: Dict[str, Dict[str, str]] = {
    "D5A": {
        "name": "D5A_safe_x_kd",
        "config": "configs/bit_drive/v5/bit_v5_D5A_safe_x_kd.yaml",
        "mode": "early_longitudinal",
    },
    "D5B": {
        "name": "D5B_safe_x_step_kd",
        "config": "configs/bit_drive/v5/bit_v5_D5B_safe_x_step_kd.yaml",
        "mode": "early_step",
    },
    "D5C": {
        "name": "D5C_safe_x_step_heading_kd",
        "config": "configs/bit_drive/v5/bit_v5_D5C_safe_x_step_heading_kd.yaml",
        "mode": "early_longitudinal_step_heading",
    },
    "D5D": {
        "name": "D5D_safe_kd_plus_risk_aux",
        "config": "configs/bit_drive/v5/bit_v5_D5D_safe_kd_plus_risk_aux.yaml",
        "mode": "early_step",
    },
    "D5E": {
        "name": "D5E_safe_kd_plus_dac_preserve",
        "config": "configs/bit_drive/v5/bit_v5_D5E_safe_kd_plus_dac_preserve.yaml",
        "mode": "early_step",
    },
}


REFERENCE = {
    "A0 Base": {"mean": 0.8690, "p10": 0.8003, "zero": 62, "dac0": 56, "nc0": 6, "ttc0": 43, "ego": 0.8007},
    "B5/C1 BiT": {"mean": 0.8738, "p10": 0.8158, "zero": 58, "dac0": 49, "nc0": 9, "ttc0": 40, "ego": 0.8008},
    "D1 risk token": {"mean": 0.8732, "p10": 0.8030, "zero": 59, "dac0": 52, "nc0": 7, "ttc0": 46, "ego": 0.8075},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D5 Conservative Risk-Regularized BiT ablations.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--exp-root", type=Path, default=DEFAULT_BIT_EXP_ROOT / "v5_d5")
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--counterfactual-label-jsonl", type=Path, required=True)
    parser.add_argument("--base-il-checkpoint", type=Path, default=DEFAULT_CHECKPOINT_ROOT / "recogdrive/ReCogDrive-2B-IL")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--chunk-cache-root", type=Path, default=DEFAULT_SHARED_CHUNK_CACHE_ROOT)
    parser.add_argument("--train-chunk-name-pattern", default="train_full_chunk_*")
    parser.add_argument("--eval-chunk-name-pattern", default="navtest_full_chunk_*")
    parser.add_argument("--metric-cache-dir", type=Path, default=DEFAULT_SHARED_ROOT / "cache/metric_cache_navtest_full_v1")
    parser.add_argument("--max-train-samples", type=int, default=4096)
    parser.add_argument("--max-eval-samples", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--only", default="D5A,D5B,D5C,D5D,D5E")
    parser.add_argument("--reuse-existing-chunks", action="store_true", default=True)
    parser.add_argument("--skip-train-if-checkpoint-exists", action="store_true")
    parser.add_argument("--skip-eval-if-metrics-exist", action="store_true")
    parser.add_argument("--force-train-with-sparse-labels", action="store_true")
    parser.add_argument("--seed-list", default="20260601,20260602,20260603")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def assert_not_shared_experiment_output(path: Path) -> None:
    resolved = path.resolve()
    shared = SHARED_EXPERIMENT_ROOT.resolve()
    if resolved == shared or shared in resolved.parents:
        raise RuntimeError(f"Refusing to write D5 outputs under shared experiment root: {shared}")


def selected_runs(value: str) -> List[str]:
    names = [item.strip().upper() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in RUNS]
    if unknown:
        raise ValueError(f"Unknown D5 runs: {unknown}")
    return names


def parse_seed_list(value: str) -> List[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seed-list must contain at least one integer seed")
    return seeds


def discover_init_checkpoint(project_root: Path, exp_root: Path, explicit: Optional[Path]) -> Optional[Path]:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    bit_root = exp_root.parent
    candidates.extend(
        [
            bit_root / "v3/C1_lateral_terminal/best.ckpt",
            bit_root / "v2/B5_terminal_only_context/best.ckpt",
            DEFAULT_BIT_EXP_ROOT / "v3/C1_lateral_terminal/best.ckpt",
            DEFAULT_BIT_EXP_ROOT / "v2/B5_terminal_only_context/best.ckpt",
            project_root / "experiments/bit_drive/v3/C1_lateral_terminal/best.ckpt",
            project_root / "experiments/bit_drive/v2/B5_terminal_only_context/best.ckpt",
            DEFAULT_SHARED_ROOT / "experiments/bit_drive/v3/C1_lateral_terminal/best.ckpt",
            DEFAULT_SHARED_ROOT / "experiments/bit_drive/v2/B5_terminal_only_context/best.ckpt",
        ]
    )
    for path in candidates:
        if path is not None and path.is_file():
            return path
    return None


def metric_row(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "mean": data.get("mean_pdms"),
        "p10": data.get("p10_pdms"),
        "zero": data.get("zero_score_count"),
        "dac0": data.get("drivable_area_compliance_zero_count"),
        "nc0": data.get("no_at_fault_collision_zero_count"),
        "ttc0": data.get("time_to_collision_zero_count"),
        "ego": data.get("ego_progress_mean"),
    }


def collect_existing_results(exp_root: Path, max_eval_samples: int) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for run_id in RUNS:
        path = exp_root / f"eval_{run_id}_navtest_{max_eval_samples}" / "aggregate_metrics.json"
        row = metric_row(path)
        if row is not None:
            rows[run_id] = row
    return rows


def acceptance(row: Dict[str, Any]) -> bool:
    return (
        row.get("mean") is not None
        and float(row["mean"]) >= 0.8710
        and float(row["p10"]) >= 0.8003
        and int(row["zero"]) <= 60
        and int(row["dac0"]) <= 51
        and int(row["ttc0"]) <= 43
        and int(row["nc0"]) <= 7
    )


def load_mining_summary(counterfactual_label_jsonl: Path) -> Dict[str, Any]:
    summary_path = counterfactual_label_jsonl.parent / "mining_summary.json"
    if not summary_path.is_file():
        return {"missing": True, "path": str(summary_path)}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def mining_count(summary: Dict[str, Any], key: str, *tag_names: str) -> int:
    if summary.get(key) is not None:
        return int(summary[key])
    tags = summary.get("tag_counts", {}) if isinstance(summary.get("tag_counts"), dict) else {}
    return int(sum(int(tags.get(tag, 0)) for tag in tag_names))


def mining_gate(summary: Dict[str, Any]) -> Dict[str, Any]:
    hard_nc = mining_count(summary, "hard_nc_regressions", "hard_nc_regression", "bit_nc_regression")
    hard_ttc = mining_count(summary, "hard_ttc_regressions", "hard_ttc_regression", "bit_ttc_regression")
    soft = mining_count(summary, "soft_safety_regressions", "soft_safety_regression", "soft_ttc_regression")
    soft = max(soft, hard_nc + hard_ttc)
    dac = mining_count(summary, "dac_fixes", "dac_fix", "bit_dac_fix")
    ideal = hard_nc >= 50 and hard_ttc >= 150 and dac >= 200
    acceptable = hard_nc >= 30 and hard_ttc >= 100 and soft >= 500 and dac >= 200
    return {
        "hard_nc_regressions": hard_nc,
        "hard_ttc_regressions": hard_ttc,
        "soft_safety_regressions": soft,
        "dac_fixes": dac,
        "ideal_gate_passed": ideal,
        "acceptable_gate_passed": acceptable,
        "gate_passed": ideal or acceptable,
    }


def run_cmd(cmd: List[str], *, dry_run: bool, cwd: Path) -> None:
    print("RUN", " ".join(str(part) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run([str(part) for part in cmd], cwd=cwd, check=True)


def report_lines(args: argparse.Namespace, results: Dict[str, Dict[str, Any]], mining: Optional[Dict[str, Any]]) -> List[str]:
    lines = [
        "# BiT D5 Conservative Risk-Regularized Report",
        "",
        "No Target-Constrained GRPO/RL was run.",
        "",
        "## Previous Reference",
        "",
        "| Method | Mean | P10 | Zero | DAC0 | NC0 | TTC0 | Ego |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in REFERENCE.items():
        lines.append(f"| {name} | {row['mean']:.4f} | {row['p10']:.4f} | {row['zero']} | {row['dac0']} | {row['nc0']} | {row['ttc0']} | {row['ego']:.4f} |")
    lines.extend(["", "## Mining Dataset", ""])
    if mining:
        tags = mining.get("tag_counts", {})
        gate = mining_gate(mining)
        lines.extend(
            [
                f"Rows: `{mining.get('total_rows')}`",
                f"Training allowed: `{mining.get('training_allowed')}`",
                f"Splits: `{mining.get('split_counts')}`",
                f"Hard NC regressions: `{gate['hard_nc_regressions']}`",
                f"Hard TTC regressions: `{gate['hard_ttc_regressions']}`",
                f"Soft safety regressions: `{gate['soft_safety_regressions']}`",
                f"DAC fixes: `{gate['dac_fixes']}`",
                f"Mining gate passed: `{gate['gate_passed']}`",
            ]
        )
    else:
        lines.append("Mining summary not found yet.")
    lines.extend(
        [
            "",
            "## D5 Results",
            "",
            "| Run | Mean | P10 | Zero | DAC0 | NC0 | TTC0 | Ego | Acceptance |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    if not results:
        lines.append("| pending | - | - | - | - | - | - | - | not run |")
    for name, row in results.items():
        ok = acceptance(row)
        lines.append(
            f"| {name} | {float(row['mean']):.4f} | {float(row['p10']):.4f} | {int(row['zero'])} | "
            f"{int(row['dac0'])} | {int(row['nc0'])} | {int(row['ttc0'])} | {float(row['ego']):.4f} | "
            f"{'pass' if ok else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "## Acceptance Check",
            "",
            "Acceptance: Mean >= A0 + 0.002, P10 >= A0, Zero <= A0 - 2, DAC0 <= A0 - 5, TTC0 <= A0, NC0 <= A0 + 1.",
            "",
        ]
    )
    if results:
        passing = [name for name, row in results.items() if acceptance(row)]
        lines.append(f"Passing navtest-{args.max_eval_samples} runs: `{passing}`")
    else:
        lines.append("Passing runs: `[]`")
    lines.extend(
        [
            "",
            "## Analysis",
            "",
            "Safe KD is evaluated only on non-test counterfactual safety-regression samples. A pass requires NC/TTC control without losing DAC fixes.",
            "",
            "Over-distillation is checked through DAC0 and Zero: if these move back toward A0/Base while NC/TTC improve, D5 is preserving safety by collapsing BiT behavior rather than fixing the unsafe modes.",
            "",
            "The best loss is the passing run with the strongest left-tail table. If no run passes, D5 remains inconclusive and should not be used as a success claim.",
            "",
            "## Recommendation",
            "",
            "Do not claim success unless matched navtest-1024 acceptance is met, then verify on 2048 and 3-seed stability. If no run passes, continue non-test safety mining, revise D5, pivot to interaction-aware JEPA, or keep BiT as auxiliary only.",
        ]
    )
    return lines


def write_report(args: argparse.Namespace, results: Dict[str, Dict[str, Any]]) -> None:
    mining_path = args.counterfactual_label_jsonl.parent / "mining_summary.json"
    mining = json.loads(mining_path.read_text(encoding="utf-8")) if mining_path.is_file() else None
    report_path = args.report_path or (args.exp_root.parent / "BIT_D5_CONSERVATIVE_REPORT.md")
    assert_not_shared_experiment_output(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines(args, results, mining)) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    assert_not_shared_experiment_output(args.exp_root)
    init_checkpoint = discover_init_checkpoint(args.project_root, args.exp_root, args.init_checkpoint)
    if init_checkpoint is None:
        print("No C1/B5 init checkpoint found; falling back to Base-IL initialization.", flush=True)
    if not args.dry_run and not args.counterfactual_label_jsonl.is_file():
        raise FileNotFoundError(args.counterfactual_label_jsonl)
    args.exp_root.mkdir(parents=True, exist_ok=True)
    mining_summary = load_mining_summary(args.counterfactual_label_jsonl)
    gate = mining_gate(mining_summary)
    if not gate["gate_passed"] and not args.force_train_with_sparse_labels:
        print(
            "D5 training blocked: insufficient safety/DAC labels. "
            f"hard_nc={gate['hard_nc_regressions']} hard_ttc={gate['hard_ttc_regressions']} "
            f"soft_safety={gate['soft_safety_regressions']} dac_fixes={gate['dac_fixes']}",
            flush=True,
        )
        write_report(args, collect_existing_results(args.exp_root, args.max_eval_samples))
        return 2
    results: Dict[str, Dict[str, Any]] = collect_existing_results(args.exp_root, args.max_eval_samples)
    for run_id in selected_runs(args.only):
        spec = RUNS[run_id]
        train_dir = args.exp_root / spec["name"]
        eval_dir = args.exp_root / f"eval_{run_id}_navtest_{args.max_eval_samples}"
        train_ckpt = train_dir / "best.ckpt"
        train_cmd = [
            args.python,
            "scripts/train_bit_drive_chunked.py",
            "--config",
            spec["config"],
            "--chunk-cache-root",
            args.chunk_cache_root,
            "--chunk-name-pattern",
            args.train_chunk_name_pattern,
            "--output-dir",
            train_dir,
            "--max-samples",
            args.max_train_samples,
            "--batch-size",
            args.batch_size,
            "--num-steps",
            args.num_steps,
            "--precision",
            args.precision,
            "--counterfactual-label-jsonl",
            args.counterfactual_label_jsonl,
            "--require-counterfactual-labels",
            "--use-base-traj-kd",
            "--base-traj-kd-mode",
            spec["mode"],
            "--safety-tag-filter",
            "bit_nc_regression,bit_ttc_regression,hard_nc_regression,hard_ttc_regression,soft_safety_regression",
            "--dac-tag-filter",
            "bit_dac_fix,dac_fix",
            "--base-il-checkpoint",
            args.base_il_checkpoint,
        ]
        if init_checkpoint is not None:
            train_cmd.extend(["--resume-from", init_checkpoint])
        if not (args.skip_train_if_checkpoint_exists and train_ckpt.is_file()):
            run_cmd(train_cmd, dry_run=args.dry_run, cwd=args.project_root)
        else:
            print(f"SKIP train {run_id}: {train_ckpt} exists", flush=True)
        eval_cmd = [
            args.python,
            "scripts/eval_bit_drive_pdm.py",
            "--config",
            spec["config"],
            "--checkpoint",
            train_ckpt,
            "--split",
            "navtest",
            "--chunk-cache-root",
            args.chunk_cache_root,
            "--chunk-name-pattern",
            args.eval_chunk_name_pattern,
            "--metric-cache-dir",
            args.metric_cache_dir,
            "--max-samples",
            args.max_eval_samples,
            "--output-dir",
            eval_dir,
            "--precision",
            "fp32",
        ]
        metrics_path = eval_dir / "aggregate_metrics.json"
        if not (args.skip_eval_if_metrics_exist and metrics_path.is_file()):
            run_cmd(eval_cmd, dry_run=args.dry_run, cwd=args.project_root)
        else:
            print(f"SKIP eval {run_id}: {metrics_path} exists", flush=True)
        row = metric_row(metrics_path)
        if row is not None:
            results[run_id] = row
    passing_runs = [run_id for run_id, row in results.items() if run_id in RUNS and acceptance(row)]
    for run_id in passing_runs:
        spec = RUNS[run_id]
        train_ckpt = args.exp_root / spec["name"] / "best.ckpt"
        eval_2048_dir = args.exp_root / f"eval_{run_id}_navtest_2048"
        eval_2048_metrics = eval_2048_dir / "aggregate_metrics.json"
        eval_2048_cmd = [
            args.python,
            "scripts/eval_bit_drive_pdm.py",
            "--config",
            spec["config"],
            "--checkpoint",
            train_ckpt,
            "--split",
            "navtest",
            "--chunk-cache-root",
            args.chunk_cache_root,
            "--chunk-name-pattern",
            args.eval_chunk_name_pattern,
            "--metric-cache-dir",
            args.metric_cache_dir,
            "--max-samples",
            2048,
            "--output-dir",
            eval_2048_dir,
            "--precision",
            "fp32",
        ]
        if not (args.skip_eval_if_metrics_exist and eval_2048_metrics.is_file()):
            run_cmd(eval_2048_cmd, dry_run=args.dry_run, cwd=args.project_root)
        else:
            print(f"SKIP eval 2048 {run_id}: {eval_2048_metrics} exists", flush=True)

        for seed in parse_seed_list(args.seed_list):
            seed_train_dir = args.exp_root / f"{spec['name']}_seed{seed}"
            seed_ckpt = seed_train_dir / "best.ckpt"
            seed_eval_dir = args.exp_root / f"eval_{run_id}_seed{seed}_navtest_{args.max_eval_samples}"
            seed_train_cmd = [
                args.python,
                "scripts/train_bit_drive_chunked.py",
                "--config",
                spec["config"],
                "--chunk-cache-root",
                args.chunk_cache_root,
                "--chunk-name-pattern",
                args.train_chunk_name_pattern,
                "--output-dir",
                seed_train_dir,
                "--max-samples",
                args.max_train_samples,
                "--batch-size",
                args.batch_size,
                "--num-steps",
                args.num_steps,
                "--precision",
                args.precision,
                "--counterfactual-label-jsonl",
                args.counterfactual_label_jsonl,
                "--require-counterfactual-labels",
                "--use-base-traj-kd",
                "--base-traj-kd-mode",
                spec["mode"],
                "--safety-tag-filter",
                "bit_nc_regression,bit_ttc_regression,hard_nc_regression,hard_ttc_regression,soft_safety_regression",
                "--dac-tag-filter",
                "bit_dac_fix,dac_fix",
                "--base-il-checkpoint",
                args.base_il_checkpoint,
                "--seed",
                seed,
            ]
            if init_checkpoint is not None:
                seed_train_cmd.extend(["--resume-from", init_checkpoint])
            if not (args.skip_train_if_checkpoint_exists and seed_ckpt.is_file()):
                run_cmd(seed_train_cmd, dry_run=args.dry_run, cwd=args.project_root)
            else:
                print(f"SKIP train seed {run_id}/{seed}: {seed_ckpt} exists", flush=True)
            seed_eval_cmd = [
                args.python,
                "scripts/eval_bit_drive_pdm.py",
                "--config",
                spec["config"],
                "--checkpoint",
                seed_ckpt,
                "--split",
                "navtest",
                "--chunk-cache-root",
                args.chunk_cache_root,
                "--chunk-name-pattern",
                args.eval_chunk_name_pattern,
                "--metric-cache-dir",
                args.metric_cache_dir,
                "--max-samples",
                args.max_eval_samples,
                "--output-dir",
                seed_eval_dir,
                "--precision",
                "fp32",
                "--seed",
                seed,
            ]
            seed_metrics = seed_eval_dir / "aggregate_metrics.json"
            if not (args.skip_eval_if_metrics_exist and seed_metrics.is_file()):
                run_cmd(seed_eval_cmd, dry_run=args.dry_run, cwd=args.project_root)
            else:
                print(f"SKIP eval seed {run_id}/{seed}: {seed_metrics} exists", flush=True)
    if not args.dry_run:
        write_report(args, results)
    else:
        write_report(args, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
