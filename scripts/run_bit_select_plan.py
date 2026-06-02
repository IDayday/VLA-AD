#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


PROJECT = Path("/mnt/project/VLA-AD")


def shell_join(cmd: Iterable[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run(cmd: List[object], *, cwd: Path, dry_run: bool, env: dict | None = None) -> None:
    print(shell_join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run([str(part) for part in cmd], cwd=cwd, env=env, check=True)


def parse_csv(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BiT-Select train/val counterfactual and selector experiments.")
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--exp-root", type=Path, default=PROJECT / "experiments/bit_drive/select")
    parser.add_argument("--chunk-cache-root", type=Path, default=PROJECT / "cache/recogdrive_expert_chunks/full_v1")
    parser.add_argument("--train-chunk", default="train_full_chunk_000000")
    parser.add_argument("--val-chunk", default="train_full_chunk_000001")
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--val-samples", type=int, default=256)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--metric-workers", type=int, default=32)
    parser.add_argument("--python", default="/root/miniconda3/envs/navsim/bin/python")
    parser.add_argument("--base-config", type=Path, default=Path("configs/bit_drive/bit_ablation_base_no_bit.yaml"))
    parser.add_argument("--base-checkpoint", type=Path, default=PROJECT / "checkpoints/recogdrive/ReCogDrive-2B-IL")
    parser.add_argument("--bit-config", type=Path, default=Path("configs/bit_drive/v3/bit_v3_C1_lateral_terminal.yaml"))
    parser.add_argument("--bit-checkpoint", type=Path, default=PROJECT / "experiments/bit_drive/v3/C1_lateral_terminal/best.ckpt")
    parser.add_argument("--navtest-counterfactual", type=Path, default=PROJECT / "experiments/bit_drive/select/counterfactual_navtest_1024/counterfactual_samples.jsonl")
    parser.add_argument("--only", default="cache,counterfactual,train,eval")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--cpu-counterfactual", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stages = parse_csv(args.only)
    tag = args.tag or f"train{args.train_samples}_val{args.val_samples}_{args.train_chunk}_{args.val_chunk}"
    run_root = args.exp_root / tag
    train_metric = run_root / f"metric_cache_train_{args.train_samples}"
    val_metric = run_root / f"metric_cache_val_{args.val_samples}"
    train_cf = run_root / f"counterfactual_train_{args.train_samples}"
    val_cf = run_root / f"counterfactual_val_{args.val_samples}"
    selector_dir = run_root / "selector_feature_mlp"
    eval_dir = run_root / "eval_selector_navtest_1024"
    run_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "tag": tag,
        "train_chunk": args.train_chunk,
        "val_chunk": args.val_chunk,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "navtest_counterfactual": str(args.navtest_counterfactual),
    }
    (run_root / "plan_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if "cache" in stages:
        cache_jobs = [
            (args.train_chunk, args.train_samples, train_metric),
            (args.val_chunk, args.val_samples, val_metric),
        ]
        for chunk_name, samples, out_cache in cache_jobs:
            if args.skip_existing and count_jsonl(out_cache / "metadata" / f"{out_cache.name}_metadata_node_0.csv") >= samples:
                print(f"skip existing metric cache {out_cache}", flush=True)
                continue
            cmd = [
                args.python,
                "scripts/run_metric_cache_for_chunk_subset.py",
                "--project-root",
                args.project_root,
                "--chunk-cache-dir",
                args.chunk_cache_root / chunk_name,
                "--max-samples",
                samples,
                "--output-cache",
                out_cache,
                "--split",
                "navtrain",
                "--navsim-log-path",
                "/mnt/navsim/trainval_navsim_logs/trainval",
                "--openscene-root",
                "/mnt/navsim",
                "--maps-root",
                "/mnt/navsim/maps",
                "--max-workers",
                args.metric_workers,
            ]
            run(cmd, cwd=args.project_root, dry_run=args.dry_run)

    if "counterfactual" in stages:
        cf_jobs = [
            (args.train_chunk, args.train_samples, train_metric, train_cf),
            (args.val_chunk, args.val_samples, val_metric, val_cf),
        ]
        for chunk_name, samples, metric_dir, out_dir in cf_jobs:
            if args.skip_existing and (out_dir / "counterfactual_samples.jsonl").is_file():
                print(f"skip existing counterfactual {out_dir}", flush=True)
                continue
            cmd = [
                args.python,
                "scripts/build_bit_counterfactual_dataset.py",
                "--base-config",
                args.base_config,
                "--base-checkpoint",
                args.base_checkpoint,
                "--bit-config",
                args.bit_config,
                "--bit-checkpoint",
                args.bit_checkpoint,
                "--split",
                "navtrain",
                "--chunk-cache-dir",
                args.chunk_cache_root / chunk_name,
                "--metric-cache-dir",
                metric_dir,
                "--max-samples",
                samples,
                "--output-dir",
                out_dir,
                "--seed",
                "20260601",
                "--deterministic",
                "--precision",
                "fp32",
            ]
            env = os.environ.copy()
            if args.cpu_counterfactual:
                env["CUDA_VISIBLE_DEVICES"] = ""
            run(cmd, cwd=args.project_root, dry_run=args.dry_run, env=env)

    if "train" in stages:
        if args.skip_existing and (selector_dir / "selector.pt").is_file():
            print(f"skip existing selector {selector_dir}", flush=True)
        else:
            cmd = [
                args.python,
                "scripts/train_bit_safety_selector.py",
                "--train-jsonl",
                train_cf / "counterfactual_samples.jsonl",
                "--val-jsonl",
                val_cf / "counterfactual_samples.jsonl",
                "--output-dir",
                selector_dir,
                "--epochs",
                "50",
                "--batch-size",
                "256",
                "--lr",
                "1e-3",
                "--weight-decay",
                "1e-4",
                "--threshold-mode",
                "val_sweep",
                "--safety-first",
            ]
            run(cmd, cwd=args.project_root, dry_run=args.dry_run)

    if "eval" in stages:
        cmd = [
            args.python,
            "scripts/eval_bit_selector_on_counterfactual.py",
            "--counterfactual-jsonl",
            args.navtest_counterfactual,
            "--selector",
            selector_dir / "selector.pt",
            "--selector-config",
            selector_dir / "selector_config.json",
            "--output-dir",
            eval_dir,
        ]
        run(cmd, cwd=args.project_root, dry_run=args.dry_run)

    print(json.dumps({"run_root": str(run_root), "tag": tag}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
