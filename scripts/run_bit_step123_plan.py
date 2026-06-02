#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence


def shell_join(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_or_print(cmd: Sequence[str], *, dry_run: bool) -> None:
    print(shell_join(cmd), flush=True)
    if not dry_run:
        subprocess.run([str(part) for part in cmd], check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run the BiT-Drive Step 1-3 experiment plan.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/project/VLA-AD"))
    parser.add_argument("--navsim-root", type=Path, default=Path("/mnt/navsim"))
    parser.add_argument("--exp-root", type=Path, default=Path("/mnt/project/VLA-AD/experiments/bit_drive"))
    parser.add_argument("--chunk-cache-root", type=Path, default=Path("/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1"))
    parser.add_argument("--train-chunk-pattern", default="train_full_chunk_*")
    parser.add_argument("--eval-chunk-pattern", default="navtest_full_chunk_*")
    parser.add_argument("--base-il-checkpoint", type=Path, default=Path("/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-IL"))
    parser.add_argument("--metric-cache-dir", type=Path, default=None)
    parser.add_argument("--max-train-samples", type=int, default=1024)
    parser.add_argument("--max-eval-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--run-a0", action="store_true")
    parser.add_argument("--run-step1", action="store_true")
    parser.add_argument("--run-step2", action="store_true")
    parser.add_argument("--run-step3", action="store_true")
    return parser.parse_args()


def eval_cmd(args: argparse.Namespace, config: str, checkpoint: Path, output_dir: Path) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/eval_bit_drive_pdm.py",
        "--config",
        config,
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
        str(output_dir),
    ]
    if args.metric_cache_dir is not None:
        cmd.extend(["--metric-cache-dir", str(args.metric_cache_dir)])
    return cmd


def train_cmd(
    args: argparse.Namespace,
    config: str,
    output_dir: Path,
    *,
    init_checkpoint: Path,
    failure_sampling: bool = False,
) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/train_bit_drive_chunked.py",
        "--config",
        config,
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
        "bf16",
        "--lr-bit",
        "1e-4",
        "--lr-action-head",
        "2e-5",
        "--freeze-vlm",
        "--output-dir",
        str(output_dir),
    ]
    if failure_sampling:
        cmd.extend([
            "--failure-sampling-enabled",
            "--failure-index-path",
            str(args.exp_root / "failure_index" / "baseline_failure_index.jsonl"),
        ])
    return cmd


def main() -> int:
    args = parse_args()
    if args.metric_cache_dir is None:
        candidate = args.project_root / "cache/metric_cache_navtest_full_v1"
        if candidate.exists():
            args.metric_cache_dir = candidate
    args.exp_root.mkdir(parents=True, exist_ok=True)
    print("# BiT-Drive Step 1-3 Plan")
    print(f"# project_root={args.project_root}")
    print(f"# navsim_root={args.navsim_root}")
    print(f"# exp_root={args.exp_root}")
    print("# No RL/GRPO commands are generated.")

    if not args.chunk_cache_root.exists():
        print("# Chunk cache root is missing; build VLM-only chunks before training/eval:")
        print(shell_join([
            sys.executable,
            "scripts/build_recogdrive_chunk_cache.py",
            "--data-root",
            str(args.navsim_root),
            "--project-root",
            str(args.project_root),
            "--split",
            "navtrain",
            "--chunk-index",
            "0",
            "--chunk-size",
            str(args.max_train_samples),
            "--output-dir",
            str(args.chunk_cache_root / "train_full_chunk_000000"),
            "--build-vlm-hidden",
            "--recogdrive-vlm-path",
            str(args.project_root / "checkpoints/recogdrive/ReCogDrive-VLM-2B"),
            "--precision",
            "bf16",
        ]))

    a0_eval = args.exp_root / "eval_a0_base"
    step1_dir = args.exp_root / "step1_terminal_path"
    step2_dir = args.exp_root / "step2_reverse_consistency"
    step3_dir = args.exp_root / "step3_failure_focused"

    planned = []
    if args.dry_run or args.run_a0:
        planned.append(eval_cmd(args, "configs/bit_drive/bit_ablation_base_no_bit.yaml", args.base_il_checkpoint, a0_eval))
        planned.append([
            sys.executable,
            "scripts/build_bit_failure_index.py",
            "--eval-dir",
            str(a0_eval),
            "--split",
            "navtest",
            "--disallow-training",
            "--output-dir",
            str(args.exp_root / "failure_index"),
        ])
    if args.dry_run or args.run_step1:
        planned.append(train_cmd(args, "configs/bit_drive/bit_step1_terminal_path.yaml", step1_dir, init_checkpoint=args.base_il_checkpoint))
        planned.append(eval_cmd(args, "configs/bit_drive/bit_step1_terminal_path.yaml", step1_dir / "latest.ckpt", args.exp_root / "eval_step1"))
    if args.dry_run or args.run_step2:
        planned.append(train_cmd(args, "configs/bit_drive/bit_step2_reverse_consistency.yaml", step2_dir, init_checkpoint=step1_dir / "latest.ckpt"))
        planned.append(eval_cmd(args, "configs/bit_drive/bit_step2_reverse_consistency.yaml", step2_dir / "latest.ckpt", args.exp_root / "eval_step2"))
    if args.dry_run or args.run_step3:
        planned.append(train_cmd(args, "configs/bit_drive/bit_step3_failure_focused.yaml", step3_dir, init_checkpoint=step2_dir / "latest.ckpt", failure_sampling=True))
        planned.append(eval_cmd(args, "configs/bit_drive/bit_step3_failure_focused.yaml", step3_dir / "latest.ckpt", args.exp_root / "eval_step3"))
    planned.append([
        sys.executable,
        "scripts/aggregate_bit_left_tail_results.py",
        "--input-root",
        str(args.exp_root),
        "--output-csv",
        str(args.exp_root / "bit_left_tail_results.csv"),
        "--output-md",
        str(args.exp_root / "bit_left_tail_results.md"),
    ])

    for cmd in planned:
        run_or_print(cmd, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
