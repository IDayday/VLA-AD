#!/usr/bin/env python3
"""Watch OneVL Stage2 checkpoints and evaluate top-k with remote GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import torch


METRIC_KEYS = ("PDMS", "pdm_score", "NC", "DAC", "TTC", "comfort", "EP", "DDC")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--remote-host", default="training-rl-zt3")
    p.add_argument("--remote-python", default="/root/miniconda3/envs/navsim/bin/python")
    p.add_argument("--repo-root", type=Path, default=Path("/mnt/project/VLA-AD"))
    p.add_argument("--min-epoch", type=int, default=50)
    p.add_argument("--min-step", type=int, default=40000)
    p.add_argument("--poll-seconds", type=int, default=300)
    p.add_argument("--stable-seconds", type=int, default=120)
    p.add_argument("--num-shards", type=int, default=8)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--once", action="store_true")
    p.add_argument("--val-cache-root", type=Path, required=True)
    p.add_argument("--val-cache-pattern", default="val6000_chunk_*")
    p.add_argument("--val-metric-cache", type=Path, required=True)
    p.add_argument("--nav-cache-root", type=Path, required=True)
    p.add_argument("--nav-cache-pattern", default="shard_*")
    p.add_argument("--nav-metric-cache", type=Path, required=True)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def is_stable(path: Path, stable_seconds: int) -> bool:
    try:
        stat1 = path.stat()
    except FileNotFoundError:
        return False
    if time.time() - stat1.st_mtime < stable_seconds:
        return False
    time.sleep(1)
    try:
        stat2 = path.stat()
    except FileNotFoundError:
        return False
    return stat1.st_size == stat2.st_size and stat1.st_mtime == stat2.st_mtime


def checkpoint_info(path: Path) -> dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    metrics = obj.get("metrics", {}) if isinstance(obj, dict) else {}
    step = int(obj.get("global_step", metrics.get("step", 0))) if isinstance(obj, dict) else 0
    epoch = int(metrics.get("global_epoch", metrics.get("epoch", -1))) if isinstance(metrics, dict) else -1
    return {"step": step, "epoch": epoch, "metrics": metrics}


def candidate_checkpoints(train_dir: Path, min_epoch: int, min_step: int, stable_seconds: int) -> list[Path]:
    paths = sorted({*train_dir.glob("step_*.ckpt"), *train_dir.glob("epoch_*.ckpt")})
    latest = train_dir / "latest.ckpt"
    if latest.is_file():
        paths.append(latest)
    selected: list[Path] = []
    for path in paths:
        if not is_stable(path, stable_seconds):
            continue
        info = checkpoint_info(path)
        if int(info["epoch"]) >= min_epoch and int(info["step"]) >= min_step:
            selected.append(path)
    return selected


def ssh_run(host: str, command: str, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        local_names = {"local", "localhost", "127.0.0.1", socket.gethostname()}
        if host in local_names:
            proc = subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            return proc.returncode
        proc = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, command],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return proc.returncode


def shard_command(
    *,
    repo_root: Path,
    python_bin: str,
    config: Path,
    checkpoint: Path,
    split: str,
    chunk_root: Path,
    chunk_pattern: str,
    metric_cache: Path,
    eval_dir: Path,
    num_shards: int,
) -> str:
    lines = [
        "set -euo pipefail",
        f"cd {repo_root}",
        f"mkdir -p {eval_dir}/logs",
        "pids=()",
    ]
    for shard in range(num_shards):
        shard_dir = eval_dir / f"shard_{shard:02d}"
        shard_log = eval_dir / "logs" / f"shard_{shard:02d}.log"
        cmd = (
            f"CUDA_VISIBLE_DEVICES={shard} {python_bin} scripts/eval_recogdrive_expert_pdm.py "
            f"--config {config} "
            f"--checkpoint {checkpoint} "
            f"--split {split} "
            f"--feature-source chunk "
            f"--chunk-cache-root {chunk_root} "
            f"--chunk-name-pattern '{chunk_pattern}' "
            f"--metric-cache-dir {metric_cache} "
            f"--output-dir {shard_dir} "
            f"--precision fp32 "
            f"--num-shards {num_shards} "
            f"--shard-index {shard}"
        )
        lines.append(f"({cmd}) > {shard_log} 2>&1 & pids+=(\"$!\")")
    lines.extend(
        [
            "rc=0",
            "for pid in \"${pids[@]}\"; do",
            "  wait \"$pid\" || rc=1",
            "done",
            "exit \"$rc\"",
        ]
    )
    return "bash -lc " + shlex.quote("\n".join(lines))


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


def aggregate_eval(eval_dir: Path, split: str, checkpoint: Path, sha: str) -> dict[str, Any]:
    shard_metrics = []
    for path in sorted(eval_dir.glob("shard_*/metrics.json")):
        shard_metrics.append(json.loads(path.read_text()))
    if not shard_metrics:
        raise RuntimeError(f"No shard metrics under {eval_dir}")
    metrics: dict[str, Any] = {
        "split": split,
        "checkpoint": str(checkpoint),
        "sha256": sha,
        "num_shards": len(shard_metrics),
        "num_samples": sum(int(row.get("num_samples") or 0) for row in shard_metrics),
        "num_pdm_valid": sum(int(row.get("num_pdm_valid") or 0) for row in shard_metrics),
        "num_pdm_failed": sum(int(row.get("num_pdm_failed") or 0) for row in shard_metrics),
        "num_pdm_missing_metric_cache": sum(int(row.get("num_pdm_missing_metric_cache") or 0) for row in shard_metrics),
    }
    for key in METRIC_KEYS:
        metrics[key] = weighted_average(shard_metrics, key)
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def ranking_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    pdms = float(row.get("PDMS") or row.get("pdm_score") or -1.0)
    nc = float(row.get("NC") or 0.0)
    dac = float(row.get("DAC") or 0.0)
    ttc = float(row.get("TTC") or 0.0)
    ep = float(row.get("EP") or 0.0)
    return (pdms, nc * dac, ttc, ep, -float(row.get("step") or 0))


def update_ranking(root: Path, split: str, metrics: dict[str, Any], checkpoint: Path, top_k: int) -> None:
    rank_dir = root / "rankings" / split
    backup_dir = root / "checkpoint_backups" / f"{split}_top5"
    object_dir = root / "checkpoint_backups" / "objects"
    rank_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)
    object_dir.mkdir(parents=True, exist_ok=True)
    current_path = rank_dir / "current_top5.json"
    rows = json.loads(current_path.read_text()) if current_path.is_file() else []
    rows = [row for row in rows if row.get("sha256") != metrics["sha256"]]
    info = checkpoint_info(checkpoint)
    row = {
        **metrics,
        "checkpoint_id": checkpoint.stem,
        "source_path": str(checkpoint),
        "epoch": info["epoch"],
        "step": info["step"],
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    rows.append(row)
    rows = sorted(rows, key=ranking_key, reverse=True)[:top_k]
    current_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    tsv = rank_dir / "current_top5.tsv"
    fields = ["rank", "checkpoint_id", "sha256", "epoch", "step", "PDMS", "NC", "DAC", "TTC", "EP", "source_path"]
    with tsv.open("w", encoding="utf-8") as f:
        f.write("\t".join(fields) + "\n")
        for idx, item in enumerate(rows, start=1):
            f.write("\t".join(str(item.get(field, "")) if field != "rank" else str(idx) for field in fields) + "\n")
    if any(item["sha256"] == metrics["sha256"] for item in rows):
        object_path = object_dir / f"{metrics['sha256']}.ckpt"
        if not object_path.exists():
            shutil.copy2(checkpoint, object_path)
        named = backup_dir / f"{metrics['sha256']}_{checkpoint.name}"
        if not named.exists():
            shutil.copy2(checkpoint, named)


def split_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {
            "split": "val6000",
            "chunk_root": args.val_cache_root,
            "chunk_pattern": args.val_cache_pattern,
            "metric_cache": args.val_metric_cache,
        },
        {
            "split": "navtest",
            "chunk_root": args.nav_cache_root,
            "chunk_pattern": args.nav_cache_pattern,
            "metric_cache": args.nav_metric_cache,
        },
    ]


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    state_path = args.output_root / "state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {"evaluated": {}}
    while True:
        for ckpt in candidate_checkpoints(args.train_dir, args.min_epoch, args.min_step, args.stable_seconds):
            sha = sha256_file(ckpt)
            state["evaluated"].setdefault(sha, {})
            for spec in split_specs(args):
                split = spec["split"]
                if state["evaluated"][sha].get(split) == "done":
                    continue
                eval_dir = args.output_root / "eval" / split / ckpt.stem
                command = shard_command(
                    repo_root=args.repo_root,
                    python_bin=args.remote_python,
                    config=args.config,
                    checkpoint=ckpt,
                    split=split,
                    chunk_root=spec["chunk_root"],
                    chunk_pattern=spec["chunk_pattern"],
                    metric_cache=spec["metric_cache"],
                    eval_dir=eval_dir,
                    num_shards=args.num_shards,
                )
                state["evaluated"][sha][split] = "started"
                state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
                rc = ssh_run(args.remote_host, command, args.output_root / "logs" / f"{ckpt.stem}_{split}.ssh.log")
                if rc != 0:
                    state["evaluated"][sha][split] = f"failed:{rc}"
                    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
                    continue
                metrics = aggregate_eval(eval_dir, split, ckpt, sha)
                update_ranking(args.output_root, split, metrics, ckpt, args.top_k)
                state["evaluated"][sha][split] = "done"
                state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
