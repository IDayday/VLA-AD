#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


B_CONFIGS = {
    "B1": "configs/bit_drive/v2/bit_v2_B1_auxiliary_only.yaml",
    "B2": "configs/bit_drive/v2/bit_v2_B2_context_only_low_strength.yaml",
    "B3": "configs/bit_drive/v2/bit_v2_B3_predicted_only_context.yaml",
    "B4": "configs/bit_drive/v2/bit_v2_B4_path_only_context.yaml",
    "B5": "configs/bit_drive/v2/bit_v2_B5_terminal_only_context.yaml",
    "B6": "configs/bit_drive/v2/bit_v2_B6_step2_reverse_low_weight.yaml",
    "B7": "configs/bit_drive/v2/bit_v2_B7_dac_focused_failure_sampling.yaml",
}

B_NAMES = {
    "B1": "B1_auxiliary_only",
    "B2": "B2_context_only_low_strength",
    "B3": "B3_predicted_only_context",
    "B4": "B4_path_only_context",
    "B5": "B5_terminal_only_context",
    "B6": "B6_reverse_low_weight",
    "B7": "B7_dac_focused_failure_sampling",
}


def shell_join(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_or_print(cmd: Sequence[str], *, dry_run: bool) -> None:
    print(shell_join(cmd), flush=True)
    if not dry_run:
        subprocess.run([str(part) for part in cmd], check=True)


def parse_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run BiT v2 B1-B7 ablations.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", default=None, help="Comma-separated subset: B1,B2,...")
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/project/VLA-AD"))
    parser.add_argument("--navsim-root", type=Path, default=Path("/mnt/navsim"))
    parser.add_argument("--exp-root", type=Path, default=Path("/mnt/project/VLA-AD/experiments/bit_drive/v2"))
    parser.add_argument("--base-il-checkpoint", type=Path, default=Path("/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-IL"))
    parser.add_argument("--chunk-cache-root", type=Path, default=Path("/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1"))
    parser.add_argument("--train-chunk-pattern", default="train_full_chunk_*")
    parser.add_argument("--eval-chunk-pattern", default="navtest_full_chunk_*")
    parser.add_argument("--metric-cache-dir", type=Path, default=None)
    parser.add_argument("--max-train-samples", type=int, default=1024)
    parser.add_argument("--max-eval-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--reuse-existing-chunks", action="store_true", default=True)
    parser.add_argument("--skip-train-if-checkpoint-exists", action="store_true")
    parser.add_argument("--skip-eval-if-metrics-exist", action="store_true")
    parser.add_argument("--train-failure-index-path", type=Path, default=None)
    return parser.parse_args()


def train_output_dir(args: argparse.Namespace, bid: str) -> Path:
    return args.exp_root / B_NAMES[bid]


def eval_output_dir(args: argparse.Namespace, bid: str) -> Path:
    return args.exp_root / f"eval_{bid}"


def train_cmd(
    args: argparse.Namespace,
    bid: str,
    *,
    init_checkpoint: Path,
    failure_index_path: Optional[Path] = None,
) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/train_bit_drive_chunked.py",
        "--config",
        B_CONFIGS[bid],
        "--base-il-checkpoint",
        str(init_checkpoint),
        "--chunk-cache-root",
        str(args.chunk_cache_root),
        "--chunk-name-pattern",
        args.train_chunk_pattern,
        "--max-samples",
        str(args.max_train_samples),
        "--batch-size",
        str(args.batch_size),
        "--gradient-accumulation-steps",
        "1",
        "--num-steps",
        str(args.num_steps),
        "--precision",
        args.precision,
        "--freeze-vlm",
        "--output-dir",
        str(train_output_dir(args, bid)),
    ]
    if failure_index_path is not None:
        cmd.extend([
            "--failure-sampling-enabled",
            "--failure-index-path",
            str(failure_index_path),
            "--include-tags",
            "dac_zero",
            "--exclude-tags",
            "nc_zero,ttc_zero",
            "--max-failure-fraction-per-batch",
            "0.30",
        ])
    return cmd


def eval_cmd(args: argparse.Namespace, bid: str, checkpoint: Path) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/eval_bit_drive_pdm.py",
        "--config",
        B_CONFIGS[bid],
        "--checkpoint",
        str(checkpoint),
        "--split",
        "navtest",
        "--chunk-cache-root",
        str(args.chunk_cache_root),
        "--chunk-name-pattern",
        args.eval_chunk_pattern,
        "--max-samples",
        str(args.max_eval_samples),
        "--output-dir",
        str(eval_output_dir(args, bid)),
    ]
    if args.metric_cache_dir is not None:
        cmd.extend(["--metric-cache-dir", str(args.metric_cache_dir)])
    return cmd


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metric(metrics: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key)
    return default if value is None else float(value)


def safe_vs_a0(metrics: Dict[str, Any], a0: Dict[str, Any]) -> bool:
    return (
        metric(metrics, "drivable_area_compliance_zero_count", 10**9) <= metric(a0, "drivable_area_compliance_zero_count", -1)
        and metric(metrics, "zero_score_count", 10**9) <= metric(a0, "zero_score_count", -1)
        and metric(metrics, "no_at_fault_collision_zero_count", 10**9) <= metric(a0, "no_at_fault_collision_zero_count", -1) + 1
        and metric(metrics, "time_to_collision_zero_count", 10**9) <= metric(a0, "time_to_collision_zero_count", -1) + 1
        and metric(metrics, "mean_pdms", -10**9) >= metric(a0, "mean_pdms", 10**9) - 0.002
    )


def select_best_safe(args: argparse.Namespace, candidates: Iterable[str]) -> Optional[str]:
    a0_path = args.project_root / "experiments/bit_drive/eval_a0_base/aggregate_metrics.json"
    a0 = load_json(a0_path)
    if a0 is None:
        return None
    safe_rows = []
    for bid in candidates:
        metrics = load_json(eval_output_dir(args, bid) / "aggregate_metrics.json")
        if metrics is None:
            continue
        if safe_vs_a0(metrics, a0):
            safe_rows.append((bid, metrics))
    if not safe_rows:
        return None
    safe_rows.sort(
        key=lambda item: (
            metric(item[1], "drivable_area_compliance_zero_count", 10**9),
            metric(item[1], "zero_score_count", 10**9),
            -metric(item[1], "mean_pdms", -10**9),
        )
    )
    return safe_rows[0][0]


def aggregate_cmd(args: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        "scripts/aggregate_bit_left_tail_results.py",
        "--input-root",
        str(args.exp_root),
        "--output-csv",
        str(args.exp_root / "bit_v2_left_tail_results.csv"),
        "--output-md",
        str(args.exp_root / "bit_v2_left_tail_results.md"),
    ]


def write_report(args: argparse.Namespace, selected_best: Optional[str]) -> None:
    if args.dry_run:
        return
    a0 = load_json(args.project_root / "experiments/bit_drive/eval_a0_base/aggregate_metrics.json") or {}
    rows: List[tuple[str, Dict[str, Any]]] = []
    for bid in B_CONFIGS:
        metrics = load_json(eval_output_dir(args, bid) / "aggregate_metrics.json")
        if metrics is not None:
            rows.append((bid, metrics))
    lines = [
        "# BiT v2 Report",
        "",
        "## V1 Baseline Context",
        "",
        "| Run | Mean | P10 | Zero | DAC0 | NC0 | TTC0 | Ego |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| A0 | {a0.get('mean_pdms')} | {a0.get('p10_pdms')} | {a0.get('zero_score_count')} | "
        f"{a0.get('drivable_area_compliance_zero_count')} | {a0.get('no_at_fault_collision_zero_count')} | "
        f"{a0.get('time_to_collision_zero_count')} | {a0.get('ego_progress_mean')} |",
        "",
        "Step1 v1 improved DAC/zero on 256 samples but introduced NC/TTC regressions. Step3 broad failure sampling was harmful. No Target-Constrained GRPO is recommended from v1.",
        "",
        "## V2 B1-B7 Results",
        "",
        "| Run | Mean | P10 | Zero | DAC0 | NC0 | TTC0 | Ego | Safe vs A0 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for bid, metrics in rows:
        safe = safe_vs_a0(metrics, a0) if a0 else False
        lines.append(
            f"| {bid} | {metrics.get('mean_pdms')} | {metrics.get('p10_pdms')} | {metrics.get('zero_score_count')} | "
            f"{metrics.get('drivable_area_compliance_zero_count')} | {metrics.get('no_at_fault_collision_zero_count')} | "
            f"{metrics.get('time_to_collision_zero_count')} | {metrics.get('ego_progress_mean')} | {safe} |"
        )
    lines.extend([
        "",
        "## Per-Sample Transition Matrix",
        "",
        "See `/mnt/project/VLA-AD/experiments/bit_drive/diagnostics_v1/transition_matrix.md` for the v1 transition diagnosis.",
        "",
        "## Recommendation",
        "",
    ])
    if selected_best:
        lines.append(
            f"{selected_best} meets the 256-sample safe criteria. Run a matched 1024-sample A0/best-config eval before making any strong claim or considering Step4 later."
        )
    else:
        lines.append(
            "No v2 config has been verified as safe yet. Do not proceed to Target-Constrained GRPO; inspect B1-B5 transitions or redesign conditioning if NC/TTC regressions persist."
        )
    report_path = args.project_root / "experiments/bit_drive/BIT_V2_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_run_stage(args: argparse.Namespace, bid: str, init_checkpoint: Path, failure_index_path: Optional[Path] = None) -> None:
    latest = train_output_dir(args, bid) / "latest.ckpt"
    metrics = eval_output_dir(args, bid) / "aggregate_metrics.json"
    if args.skip_train_if_checkpoint_exists and latest.is_file():
        print(f"# skip train {bid}: {latest} exists")
    else:
        run_or_print(train_cmd(args, bid, init_checkpoint=init_checkpoint, failure_index_path=failure_index_path), dry_run=args.dry_run)
    if args.skip_eval_if_metrics_exist and metrics.is_file():
        print(f"# skip eval {bid}: {metrics} exists")
    else:
        run_or_print(eval_cmd(args, bid, latest), dry_run=args.dry_run)


def main() -> int:
    args = parse_args()
    if args.metric_cache_dir is None:
        candidate = args.project_root / "cache/metric_cache_navtest_full_v1"
        if candidate.exists():
            args.metric_cache_dir = candidate
    args.exp_root.mkdir(parents=True, exist_ok=True)
    selected = parse_list(args.only)
    if not selected:
        selected = ["B1", "B2", "B3", "B4", "B5"]
    unknown = [item for item in selected if item not in B_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown BiT v2 stages: {unknown}")

    print("# BiT v2 ablation plan")
    print(f"# exp_root={args.exp_root}")
    print("# No RL/GRPO commands are generated.")
    if not args.chunk_cache_root.exists() and not args.reuse_existing_chunks:
        print("# Chunk cache is missing; build VLM-only train/navtest chunks before running this plan.")

    for bid in [item for item in ["B1", "B2", "B3", "B4", "B5"] if item in selected]:
        maybe_run_stage(args, bid, args.base_il_checkpoint)

    run_or_print(aggregate_cmd(args), dry_run=args.dry_run)

    best_safe = select_best_safe(args, ["B1", "B2", "B3", "B4", "B5"])
    if best_safe:
        print(f"# best_safe={best_safe}")
    else:
        print("# no safe B1-B5 config found yet")

    if "B6" in selected:
        if best_safe is None and not args.dry_run:
            print("# skip B6: no safe B1-B5 config found")
        else:
            init = train_output_dir(args, best_safe or "B2") / "latest.ckpt"
            maybe_run_stage(args, "B6", init)

    if "B7" in selected:
        if best_safe is None and not args.dry_run:
            print("# skip B7: no safe B1-B5/B6 config found")
        else:
            failure_index = args.train_failure_index_path
            if failure_index is None:
                failure_index = args.exp_root / "failure_index/train_dac_failure_index.jsonl"
            if not args.dry_run and not failure_index.is_file():
                raise FileNotFoundError(
                    f"B7 requires a training-split DAC failure index, not navtest: {failure_index}"
                )
            init_bid = "B6" if (train_output_dir(args, "B6") / "latest.ckpt").is_file() else (best_safe or "B2")
            maybe_run_stage(args, "B7", train_output_dir(args, init_bid) / "latest.ckpt", failure_index_path=failure_index)

    run_or_print(aggregate_cmd(args), dry_run=args.dry_run)
    write_report(args, best_safe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
