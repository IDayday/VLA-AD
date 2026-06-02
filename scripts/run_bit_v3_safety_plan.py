#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


C_CONFIGS = {
    "C1": "configs/bit_drive/v3/bit_v3_C1_lateral_terminal.yaml",
    "C2": "configs/bit_drive/v3/bit_v3_C2_lateral_path.yaml",
    "C3": "configs/bit_drive/v3/bit_v3_C3_lateral_terminal_path_low_strength.yaml",
    "C4": "configs/bit_drive/v3/bit_v3_C4_no_x_condition_context_only.yaml",
    "C5": "configs/bit_drive/v3/bit_v3_C5_b5_plus_rule_fallback.yaml",
    "C6": "configs/bit_drive/v3/bit_v3_C6_best_plus_learned_or_rule_router.yaml",
}
C_NAMES = {
    "C1": "C1_lateral_terminal",
    "C2": "C2_lateral_path",
    "C3": "C3_lateral_terminal_path_low_strength",
    "C4": "C4_no_x_condition_context_only",
    "C5": "C5_b5_plus_rule_fallback",
    "C6": "C6_best_plus_router",
}


def shell_join(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_or_print(cmd: Sequence[str], *, dry_run: bool) -> None:
    print(shell_join(cmd), flush=True)
    if not dry_run:
        subprocess.run([str(part) for part in cmd], check=True)


def parse_only(value: Optional[str]) -> List[str]:
    if not value:
        return ["C1", "C2", "C3", "C4"]
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run BiT v3 NC-safety ablations.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", default=None)
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/project/VLA-AD"))
    parser.add_argument("--exp-root", type=Path, default=Path("/mnt/project/VLA-AD/experiments/bit_drive/v3"))
    parser.add_argument("--base-il-checkpoint", type=Path, default=Path("/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-IL"))
    parser.add_argument("--chunk-cache-root", type=Path, default=Path("/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1"))
    parser.add_argument("--train-chunk-pattern", default="train_full_chunk_*")
    parser.add_argument("--eval-chunk-pattern", default="navtest_full_chunk_*")
    parser.add_argument("--metric-cache-dir", type=Path, default=Path("/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1"))
    parser.add_argument("--max-train-samples", type=int, default=1024)
    parser.add_argument("--max-eval-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--reuse-existing-chunks", action="store_true", default=True)
    parser.add_argument("--skip-train-if-checkpoint-exists", action="store_true")
    parser.add_argument("--skip-eval-if-metrics-exist", action="store_true")
    parser.add_argument("--a0-1024-dir", type=Path, default=Path("/mnt/project/VLA-AD/experiments/bit_drive/v2/eval_a0_1024_seed20260601"))
    parser.add_argument("--b5-1024-dir", type=Path, default=Path("/mnt/project/VLA-AD/experiments/bit_drive/v2/eval_B5_1024_seed20260601"))
    parser.add_argument("--b6-1024-dir", type=Path, default=Path("/mnt/project/VLA-AD/experiments/bit_drive/v2/eval_B6_1024_seed20260601"))
    return parser.parse_args()


def train_dir(args: argparse.Namespace, cid: str) -> Path:
    return args.exp_root / C_NAMES[cid]


def eval_dir(args: argparse.Namespace, cid: str) -> Path:
    return args.exp_root / f"eval_{cid}"


def train_cmd(args: argparse.Namespace, cid: str) -> List[str]:
    return [
        sys.executable,
        "scripts/train_bit_drive_chunked.py",
        "--config",
        C_CONFIGS[cid],
        "--base-il-checkpoint",
        str(args.base_il_checkpoint),
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
        str(train_dir(args, cid)),
    ]


def eval_cmd(args: argparse.Namespace, cid: str) -> List[str]:
    return [
        sys.executable,
        "scripts/eval_bit_drive_pdm.py",
        "--config",
        C_CONFIGS[cid],
        "--checkpoint",
        str(train_dir(args, cid) / "latest.ckpt"),
        "--split",
        "navtest",
        "--chunk-cache-root",
        str(args.chunk_cache_root),
        "--chunk-name-pattern",
        args.eval_chunk_pattern,
        "--metric-cache-dir",
        str(args.metric_cache_dir),
        "--max-samples",
        str(args.max_eval_samples),
        "--output-dir",
        str(eval_dir(args, cid)),
        "--seed",
        "20260601",
    ]


def aggregate_cmd(args: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        "scripts/aggregate_bit_left_tail_results.py",
        "--input-root",
        str(args.exp_root),
        "--output-csv",
        str(args.exp_root / "bit_v3_left_tail_results.csv"),
        "--output-md",
        str(args.exp_root / "bit_v3_left_tail_results.md"),
    ]


def oracle_cmd(args: argparse.Namespace, bit_dir: Path, name: str) -> List[str]:
    return [
        sys.executable,
        "scripts/analyze_bit_oracle_fallback.py",
        "--a0-dir",
        str(args.a0_1024_dir),
        "--bit-dir",
        str(bit_dir),
        "--b6-dir",
        str(args.b6_1024_dir),
        "--output-dir",
        str(args.exp_root / "oracle_fallback" / name),
    ]


def rule_fallback_cmd(args: argparse.Namespace, bit_dir: Path, name: str) -> List[str]:
    return [
        sys.executable,
        "scripts/eval_bit_safety_router.py",
        "--a0-dir",
        str(args.a0_1024_dir),
        "--bit-dir",
        str(bit_dir),
        "--output-dir",
        str(args.exp_root / "rule_fallback" / name),
        "--early-x-delta-threshold",
        "1.0",
        "--terminal-x-delta-threshold",
        "2.0",
    ]


def load_metrics(path: Path) -> Optional[Dict[str, Any]]:
    metrics_path = path / "aggregate_metrics.json"
    if not metrics_path.is_file():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def acceptance(metrics: Dict[str, Any], a0: Dict[str, Any]) -> bool:
    return (
        float(metrics.get("mean_pdms", -1.0)) >= float(a0.get("mean_pdms", 0.0)) + 0.002
        and float(metrics.get("p10_pdms", -1.0)) >= float(a0.get("p10_pdms", 0.0))
        and int(metrics.get("zero_score_count", 10**9)) <= int(a0.get("zero_score_count", 0)) - 2
        and int(metrics.get("drivable_area_compliance_zero_count", 10**9)) <= int(a0.get("drivable_area_compliance_zero_count", 0)) - 5
        and int(metrics.get("time_to_collision_zero_count", 10**9)) <= int(a0.get("time_to_collision_zero_count", 0))
        and int(metrics.get("no_at_fault_collision_zero_count", 10**9)) <= int(a0.get("no_at_fault_collision_zero_count", 0)) + 1
    )


def best_c(args: argparse.Namespace) -> Optional[str]:
    a0 = load_metrics(args.a0_1024_dir)
    if a0 is None:
        return None
    candidates = []
    for cid in ("C1", "C2", "C3", "C4"):
        metrics = load_metrics(eval_dir(args, cid))
        if metrics is None:
            continue
        candidates.append((cid, acceptance(metrics, a0), metrics))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (not item[1], item[2].get("no_at_fault_collision_zero_count", 10**9), -item[2].get("mean_pdms", -1.0)))
    return candidates[0][0]


def maybe_run(args: argparse.Namespace, cid: str) -> None:
    latest = train_dir(args, cid) / "latest.ckpt"
    metrics = eval_dir(args, cid) / "aggregate_metrics.json"
    if args.skip_train_if_checkpoint_exists and latest.is_file():
        print(f"# skip train {cid}: {latest} exists")
    else:
        run_or_print(train_cmd(args, cid), dry_run=args.dry_run)
    if args.skip_eval_if_metrics_exist and metrics.is_file():
        print(f"# skip eval {cid}: {metrics} exists")
    else:
        run_or_print(eval_cmd(args, cid), dry_run=args.dry_run)


def main() -> int:
    args = parse_args()
    args.exp_root.mkdir(parents=True, exist_ok=True)
    selected = parse_only(args.only)
    unknown = [item for item in selected if item not in C_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown v3 stages: {unknown}")
    print("# BiT v3 safety plan")
    print(f"# exp_root={args.exp_root}")
    print("# No RL/GRPO and no broad failure sampling commands are generated.")

    for cid in [item for item in ("C1", "C2", "C3", "C4") if item in selected]:
        maybe_run(args, cid)
    run_or_print(aggregate_cmd(args), dry_run=args.dry_run)

    best = best_c(args)
    if best:
        print(f"# best_c={best}")
        run_or_print(oracle_cmd(args, eval_dir(args, best), best), dry_run=args.dry_run)
    else:
        print("# best_c unavailable")
    run_or_print(oracle_cmd(args, args.b5_1024_dir, "b5"), dry_run=args.dry_run)

    if "C5" in selected:
        run_or_print(rule_fallback_cmd(args, args.b5_1024_dir, "b5_rule"), dry_run=args.dry_run)
        if best:
            run_or_print(rule_fallback_cmd(args, eval_dir(args, best), f"{best}_rule"), dry_run=args.dry_run)
    if "C6" in selected:
        print("# C6 learned router is optional; train with scripts/train_bit_safety_router.py after NC diagnosis features are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
