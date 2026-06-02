#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import List


def run(cmd: List[str], *, dry_run: bool) -> None:
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BiT-v4 risk-aware IL experiment plan. No RL/GRPO.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/project/VLA-AD"))
    parser.add_argument("--exp-root", type=Path, default=Path("/mnt/project/VLA-AD/experiments/bit_drive/v4"))
    parser.add_argument("--base-il-checkpoint", type=Path, default=Path("/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-IL"))
    parser.add_argument("--chunk-cache-root", type=Path, default=Path("/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1"))
    parser.add_argument("--metric-cache-dir", type=Path, default=Path("/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1"))
    parser.add_argument("--risk-label-jsonl", type=Path, default=Path("/mnt/project/VLA-AD/experiments/bit_drive/select/P11_enriched_features/counterfactual_train_merged_3336_enriched/counterfactual_samples.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/bit_drive/v4/bit_v4_D1_risk_aware_terminal.yaml"))
    parser.add_argument("--max-train-samples", type=int, default=1024)
    parser.add_argument("--max-eval-samples", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--skip-train-if-checkpoint-exists", action="store_true")
    parser.add_argument("--skip-eval-if-metrics-exist", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    py = "/root/miniconda3/envs/navsim/bin/python"
    args.exp_root.mkdir(parents=True, exist_ok=True)
    train_dir = args.exp_root / "D1_risk_aware_terminal"
    eval_dir = args.exp_root / "eval_D1_navtest_1024"
    ckpt = train_dir / "best.ckpt"

    commands = []
    if not (args.skip_train_if_checkpoint_exists and ckpt.is_file()):
        commands.append([
            py,
            "scripts/train_bit_drive_chunked.py",
            "--config",
            str(args.config),
            "--base-il-checkpoint",
            str(args.base_il_checkpoint),
            "--chunk-cache-root",
            str(args.chunk_cache_root),
            "--chunk-name-pattern",
            "train_full_chunk_*",
            "--risk-label-jsonl",
            str(args.risk_label_jsonl),
            "--require-risk-labels",
            "--output-dir",
            str(train_dir),
            "--max-samples",
            str(args.max_train_samples),
            "--batch-size",
            str(args.batch_size),
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
            "--num-steps",
            str(args.num_steps),
            "--precision",
            args.precision,
            "--log-every",
            "20",
            "--save-every",
            "250",
        ])
    if not (args.skip_eval_if_metrics_exist and (eval_dir / "aggregate_metrics.json").is_file()):
        commands.append([
            py,
            "scripts/eval_bit_drive_pdm.py",
            "--config",
            str(args.config),
            "--checkpoint",
            str(ckpt),
            "--split",
            "navtest",
            "--chunk-cache-root",
            str(args.chunk_cache_root),
            "--chunk-name-pattern",
            "navtest_full_chunk_*",
            "--metric-cache-dir",
            str(args.metric_cache_dir),
            "--max-samples",
            str(args.max_eval_samples),
            "--output-dir",
            str(eval_dir),
        ])

    payload = {
        "dry_run": args.dry_run,
        "note": "BiT-v4 risk-aware IL only. No RL/GRPO. Do not use navtest labels for training.",
        "commands": commands,
    }
    (args.exp_root / "bit_v4_plan.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for cmd in commands:
        run(cmd, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
