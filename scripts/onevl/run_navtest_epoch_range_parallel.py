#!/usr/bin/env python3
"""Run OneVL stage2 navtest evaluation for an epoch range in parallel."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


METRIC_KEYS = ("PDMS", "pdm_score", "NC", "DAC", "TTC", "comfort", "EP", "DDC")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epoch-start", type=int, required=True)
    p.add_argument("--epoch-end", type=int, required=True)
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--metric-cache", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--host-tag", required=True)
    p.add_argument("--repo-root", type=Path, default=Path("/mnt/project/VLA-AD"))
    p.add_argument("--python-bin", default="/root/miniconda3/envs/navsim/bin/python")
    p.add_argument("--gpu-pairs", default="0,1;2,3;4,5;6,7")
    p.add_argument("--precision", default="fp32", choices=("fp32", "bf16", "fp16"))
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_gpu_pairs(text: str) -> list[list[str]]:
    pairs: list[list[str]] = []
    for item in text.split(";"):
        gpus = [part.strip() for part in item.split(",") if part.strip()]
        if gpus:
            pairs.append(gpus)
    if not pairs:
        raise ValueError("--gpu-pairs did not contain any GPU ids")
    return pairs


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def weighted_average(rows: list[dict[str, Any]], key: str) -> float | None:
    total = 0.0
    weight = 0
    for row in rows:
        value = row.get(key)
        n = int(row.get("num_pdm_valid") or 0)
        if value is None or n <= 0:
            continue
        total += float(value) * n
        weight += n
    return total / weight if weight else None


def aggregate_eval(eval_dir: Path, checkpoint: Path, host_tag: str) -> dict[str, Any]:
    shard_metrics = [json.loads(path.read_text()) for path in sorted(eval_dir.glob("shard_*/metrics.json"))]
    if not shard_metrics:
        raise RuntimeError(f"No shard metrics found under {eval_dir}")
    metrics: dict[str, Any] = {
        "host_tag": host_tag,
        "split": "navtest",
        "checkpoint": str(checkpoint),
        "checkpoint_id": checkpoint.stem,
        "num_shards": len(shard_metrics),
        "num_samples": sum(int(row.get("num_samples") or 0) for row in shard_metrics),
        "num_pdm_valid": sum(int(row.get("num_pdm_valid") or 0) for row in shard_metrics),
        "num_pdm_failed": sum(int(row.get("num_pdm_failed") or 0) for row in shard_metrics),
        "num_pdm_missing_metric_cache": sum(int(row.get("num_pdm_missing_metric_cache") or 0) for row in shard_metrics),
        "evaluated_at": now(),
    }
    for key in METRIC_KEYS:
        metrics[key] = weighted_average(shard_metrics, key)
    write_json(eval_dir / "metrics.json", metrics)
    return metrics


def run_checkpoint(args: argparse.Namespace, epoch: int, gpus: list[str]) -> dict[str, Any]:
    ckpt = args.train_dir / f"epoch_{epoch:03d}.ckpt"
    if not ckpt.is_file():
        return {"epoch": epoch, "status": "missing", "checkpoint": str(ckpt)}
    eval_dir = args.output_root / args.host_tag / "eval" / "navtest" / f"epoch_{epoch:03d}"
    metrics_path = eval_dir / "metrics.json"
    if args.skip_existing and metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text())
        return {"epoch": epoch, "status": "skipped_existing", "metrics": metrics}

    logs_dir = args.output_root / args.host_tag / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    commands: list[dict[str, Any]] = []
    procs: list[tuple[int, subprocess.Popen[bytes], Any]] = []
    num_shards = len(gpus)
    for shard_index, gpu in enumerate(gpus):
        shard_dir = eval_dir / f"shard_{shard_index:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python_bin,
            "scripts/eval_recogdrive_expert_pdm.py",
            "--config",
            str(args.config),
            "--checkpoint",
            str(ckpt),
            "--split",
            "navtest",
            "--feature-source",
            "chunk",
            "--chunk-cache-root",
            str(args.cache_root),
            "--chunk-name-pattern",
            "shard_*",
            "--metric-cache-dir",
            str(args.metric_cache),
            "--output-dir",
            str(shard_dir),
            "--precision",
            args.precision,
            "--num-shards",
            str(num_shards),
            "--shard-index",
            str(shard_index),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        log_path = logs_dir / f"epoch_{epoch:03d}_shard_{shard_index:02d}.log"
        log_fp = log_path.open("wb")
        commands.append({"epoch": epoch, "gpu": gpu, "cmd": cmd, "log": str(log_path)})
        procs.append((shard_index, subprocess.Popen(cmd, cwd=args.repo_root, env=env, stdout=log_fp, stderr=subprocess.STDOUT), log_fp))

    with (args.output_root / args.host_tag / "commands.log").open("a", encoding="utf-8") as f:
        for item in commands:
            f.write(json.dumps({"time": now(), **item}, sort_keys=True) + "\n")

    failures: list[dict[str, Any]] = []
    for shard_index, proc, log_fp in procs:
        rc = proc.wait()
        log_fp.close()
        if rc != 0:
            failures.append({"shard_index": shard_index, "returncode": rc})

    if failures:
        result = {"epoch": epoch, "status": "failed", "failures": failures, "checkpoint": str(ckpt)}
        write_json(eval_dir / "failed.json", result)
        return result

    metrics = aggregate_eval(eval_dir, ckpt, args.host_tag)
    return {"epoch": epoch, "status": "done", "metrics": metrics}


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    host_root = args.output_root / args.host_tag
    host_root.mkdir(parents=True, exist_ok=True)
    gpu_pairs = parse_gpu_pairs(args.gpu_pairs)
    epochs = list(range(args.epoch_start, args.epoch_end + 1))
    write_json(
        host_root / "run_config.json",
        {
            "started_at": now(),
            "epoch_start": args.epoch_start,
            "epoch_end": args.epoch_end,
            "train_dir": str(args.train_dir),
            "cache_root": str(args.cache_root),
            "metric_cache": str(args.metric_cache),
            "config": str(args.config),
            "gpu_pairs": gpu_pairs,
            "host_tag": args.host_tag,
        },
    )

    results: list[dict[str, Any]] = []
    failed = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_pairs)) as pool:
        future_to_epoch = {
            pool.submit(run_checkpoint, args, epoch, gpu_pairs[idx % len(gpu_pairs)]): epoch
            for idx, epoch in enumerate(epochs)
        }
        for future in concurrent.futures.as_completed(future_to_epoch):
            result = future.result()
            results.append(result)
            if result.get("status") == "failed":
                failed = True
            write_json(host_root / "progress.json", {"updated_at": now(), "results": sorted(results, key=lambda row: row["epoch"])})

    completed = [row["metrics"] for row in results if row.get("metrics")]
    rankings = sorted(completed, key=lambda row: float(row.get("PDMS") or -1.0), reverse=True)
    write_json(host_root / "summary.json", {"finished_at": now(), "failed": failed, "rankings": rankings, "results": sorted(results, key=lambda row: row["epoch"])})
    for idx, row in enumerate(rankings[:20], start=1):
        print(idx, row.get("checkpoint_id"), row.get("PDMS"), row.get("num_pdm_valid"), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
