#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path("/mnt/project/VLA-AD")
EXP_ROOT = PROJECT_ROOT / "experiments/recogdrive_expert/ablations"
CHUNK_ROOT = PROJECT_ROOT / "cache/recogdrive_expert_chunks"
NAVTEST_CHUNK = CHUNK_ROOT / "navtest_chunk_000000"
BASE_IL = PROJECT_ROOT / "checkpoints/recogdrive/ReCogDrive-2B-IL"

ABLATIONS = {
    "A0": "configs/ablations/recogdrive2b_A0_base_no_expert.yaml",
    "A1": "configs/ablations/recogdrive2b_A1_jepa_only.yaml",
    "A2": "configs/ablations/recogdrive2b_A2_vggt_only.yaml",
    "A3": "configs/ablations/recogdrive2b_A3_context_only.yaml",
    "A4": "configs/ablations/recogdrive2b_A4_full.yaml",
    "A5": "configs/ablations/recogdrive2b_A5_vggt_global_pool.yaml",
}
NAMES = {
    "A0": "A0_base_loaded_no_expert",
    "A1": "A1_jepa_only",
    "A2": "A2_vggt_only",
    "A3": "A3_context_only",
    "A4": "A4_full",
    "A5": "A5_vggt_global_pool",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run the ReCogDrive expert ablation plan.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", default=None, help="Comma-separated ablations, e.g. A1,A2")
    parser.add_argument("--train-chunks", type=int, default=4)
    parser.add_argument("--eval-max-samples", type=int, default=256)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--chunk-cache-root", type=Path, default=CHUNK_ROOT)
    parser.add_argument("--chunk-name-pattern", default="train_chunk_*")
    parser.add_argument("--navtest-chunk", type=Path, default=NAVTEST_CHUNK)
    parser.add_argument("--base-il-checkpoint", type=Path, default=BASE_IL)
    parser.add_argument("--output-root", type=Path, default=EXP_ROOT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--epochs-per-chunk", type=int, default=1)
    return parser.parse_args()


def selected(args: argparse.Namespace) -> List[str]:
    if not args.only:
        return ["A0", "A1", "A2", "A3", "A4", "A5"]
    requested = [item.strip().upper() for item in args.only.split(",") if item.strip()]
    unknown = [item for item in requested if item not in ABLATIONS]
    if unknown:
        raise ValueError(f"Unknown ablations: {unknown}")
    return requested


def train_command(args: argparse.Namespace, code: str, out_dir: Path) -> List[str]:
    return [
        sys.executable, "scripts/train_recogdrive_expert_chunked.py",
        "--config", ABLATIONS[code],
        "--base-il-checkpoint", str(args.base_il_checkpoint),
        "--chunk-cache-root", str(args.chunk_cache_root),
        "--chunk-name-pattern", args.chunk_name_pattern,
        "--output-dir", str(out_dir / "train"),
        "--num-chunks", str(args.train_chunks),
        "--epochs-per-chunk", str(args.epochs_per_chunk),
        "--batch-size", str(args.batch_size),
        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
        "--lr-expert", "1e-4",
        "--lr-action-head", "2e-5",
        "--precision", "bf16",
        "--log-every", "20",
        "--save-every", "500",
    ]


def eval_command(args: argparse.Namespace, code: str, checkpoint: Path, out_dir: Path) -> List[str]:
    return [
        sys.executable, "scripts/eval_recogdrive_expert_pdm.py",
        "--config", ABLATIONS[code],
        "--checkpoint", str(checkpoint),
        "--chunk-cache-dir", str(args.navtest_chunk),
        "--split", "navtest",
        "--max-samples", str(args.eval_max_samples),
        "--output-dir", str(out_dir / "eval"),
    ]


def command_plan(args: argparse.Namespace) -> List[List[str]]:
    commands: List[List[str]] = []
    for code in selected(args):
        out_dir = args.output_root / NAMES[code]
        if code == "A0":
            commands.append(eval_command(args, code, args.base_il_checkpoint, out_dir))
        else:
            commands.append(train_command(args, code, out_dir))
            commands.append(eval_command(args, code, out_dir / "train/best.ckpt", out_dir))
    return commands


def write_shell(path: Path, commands: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "cd /mnt/project/VLA-AD", ""]
    for cmd in commands:
        lines.append(shlex.join(cmd))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def main() -> int:
    args = parse_args()
    commands = command_plan(args)
    shell_path = args.output_root / "run_ablation_plan.sh"
    write_shell(shell_path, commands)
    print(f"Wrote ablation command script: {shell_path}")
    for cmd in commands:
        print(shlex.join(cmd))
    if args.dry_run:
        return 0
    for cmd in commands:
        subprocess.run(cmd, cwd=args.project_root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
