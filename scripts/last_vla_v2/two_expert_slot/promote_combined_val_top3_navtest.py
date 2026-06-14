#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path("/mnt/project/VLA-AD_last_vla_dev")
PYTHON_BIN = Path("/root/miniconda3/envs/navsim/bin/python")
LAUNCHER = Path("/mnt/project/skill/stable-gpu-job-launch/scripts/stable_gpu_launcher.py")
METRIC_KEYS = ("PDMS", "NC", "DAC", "TTC", "comfort", "EP", "DDC")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_name(text: str) -> str:
    keep = []
    for char in text:
        keep.append(char if char.isalnum() or char in ("-", "_", ".") else "_")
    return "".join(keep).strip("._") or "item"


def command_string(command: Sequence[Any]) -> str:
    return shlex.join(str(part) for part in command)


def log(args: argparse.Namespace, message: str) -> None:
    line = f"[{utc_now()}] {args.worker_name}: {message}"
    print(line, flush=True)
    path = args.out_root / "logs" / f"{args.worker_name}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def metric_mean(items: Iterable[Dict[str, Any]], key: str, weight_key: str) -> Optional[float]:
    total = 0.0
    weight = 0
    for item in items:
        value = item.get(key)
        w = int(item.get(weight_key) or 0)
        if value is not None and w > 0:
            total += float(value) * w
            weight += w
    return total / weight if weight else None


def read_summary(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                row["PDMS_float"] = float(row.get("PDMS") or "nan")
            except ValueError:
                continue
            if row["PDMS_float"] != row["PDMS_float"]:
                continue
            checkpoint = Path(row.get("checkpoint") or "")
            if not checkpoint.is_file():
                continue
            row["checkpoint_path"] = checkpoint
            row["checkpoint_key"] = safe_name(row.get("checkpoint_name") or checkpoint.name)
            rows.append(row)
    return rows


def combined_top(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], int]:
    by_name: Dict[str, Dict[str, Any]] = {}
    total = 0
    for root in args.val_roots:
        rows = read_summary(root / "summary" / "val6000_summary.csv")
        total += len(rows)
        for row in rows:
            key = row["checkpoint_key"]
            current = by_name.get(key)
            if current is None or float(row["PDMS_float"]) > float(current["PDMS_float"]):
                by_name[key] = row
    ranked = sorted(by_name.values(), key=lambda row: float(row["PDMS_float"]), reverse=True)
    return ranked[: args.top_k], total


def navtest_cache_ready(args: argparse.Namespace) -> bool:
    index = args.navtest_chunk_cache_root / args.navtest_chunk_name_pattern / "index.jsonl"
    if not index.is_file():
        log(args, f"waiting for navtest index: {index}")
        return False
    if args.required_navtest_index_count <= 0:
        return True
    with index.open("r", encoding="utf-8") as f:
        count = sum(1 for _ in f)
    if count < args.required_navtest_index_count:
        log(args, f"waiting for navtest index count {count}/{args.required_navtest_index_count}: {index}")
        return False
    return True


def aggregate(run_dir: Path, num_shards: int, checkpoint: Path, source_row: Dict[str, Any]) -> Dict[str, Any]:
    shard_metrics = []
    missing = []
    for shard in range(num_shards):
        path = run_dir / f"shard_{shard:02d}" / "metrics.json"
        if path.is_file():
            shard_metrics.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"Missing shard metrics: {missing[:3]} ({len(missing)} total)")
    payload: Dict[str, Any] = {
        "split_name": "navtest",
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "source_val_checkpoint": str(source_row.get("checkpoint") or ""),
        "source_val_PDMS": source_row.get("PDMS"),
        "source_val_epoch": source_row.get("epoch"),
        "num_shards": num_shards,
        "num_samples": sum(int(item.get("num_samples") or 0) for item in shard_metrics),
        "num_pdm_valid": sum(int(item.get("num_pdm_valid") or 0) for item in shard_metrics),
        "num_pdm_missing_metric_cache": sum(int(item.get("num_pdm_missing_metric_cache") or 0) for item in shard_metrics),
        "num_pdm_failed": sum(int(item.get("num_pdm_failed") or 0) for item in shard_metrics),
        "trajectory_l1": metric_mean(shard_metrics, "trajectory_l1", "num_samples"),
        "completed_at": utc_now(),
        "eval_dir": str(run_dir),
    }
    for key in METRIC_KEYS:
        payload[key] = metric_mean(shard_metrics, key, "num_pdm_valid")
    payload["pdm_score"] = payload["PDMS"]
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def write_navtest_summary(args: argparse.Namespace) -> None:
    rows = []
    for metrics_path in sorted((args.out_root / "navtest").glob("*/metrics.json")):
        try:
            rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
        except Exception:
            continue
    fields = [
        "split_name",
        "checkpoint_name",
        "checkpoint",
        "PDMS",
        "trajectory_l1",
        "num_pdm_valid",
        "source_val_PDMS",
        "source_val_epoch",
        "completed_at",
        "eval_dir",
    ]
    path = args.out_root / "summary" / "navtest_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: float(row.get("source_val_PDMS") or -1.0), reverse=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def build_jobs(args: argparse.Namespace, checkpoint: Path, run_dir: Path) -> Path:
    jobs_tsv = run_dir / "jobs" / "jobs.tsv"
    jobs_tsv.parent.mkdir(parents=True, exist_ok=True)
    gpus = [item.strip() for item in args.gpus_csv.split(",") if item.strip()]
    with jobs_tsv.open("w", encoding="utf-8") as f:
        for shard in range(args.num_shards):
            shard_dir = run_dir / f"shard_{shard:02d}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            command = [
                str(args.python_bin),
                "scripts/eval_recogdrive_expert_pdm.py",
                "--config",
                str(args.config),
                "--checkpoint",
                str(checkpoint),
                "--split",
                "navtest",
                "--chunk-cache-root",
                str(args.navtest_chunk_cache_root),
                "--chunk-name-pattern",
                args.navtest_chunk_name_pattern,
                "--metric-cache-dir",
                str(args.navtest_metric_cache_dir),
                "--precision",
                args.precision,
                "--trajectory-output-key",
                args.trajectory_output_key,
                "--num-shards",
                str(args.num_shards),
                "--shard-index",
                str(shard),
                "--output-dir",
                str(shard_dir),
            ]
            if args.navtest_vlm_text_anchor_cache_root is not None:
                command.extend(["--vlm-text-anchor-cache-root", str(args.navtest_vlm_text_anchor_cache_root)])
            f.write(f"navtest_{safe_name(checkpoint.stem)}_shard_{shard:02d}\t{gpus[shard % len(gpus)]}\t{command_string(command)}\n")
    return jobs_tsv


def run_navtest(args: argparse.Namespace, row: Dict[str, Any]) -> Dict[str, Any]:
    key = row["checkpoint_key"]
    src = Path(row["checkpoint_path"])
    snapshot = args.out_root / "checkpoint_snapshots" / key / src.name
    hardlink_or_copy(src, snapshot)
    run_dir = args.out_root / "navtest" / key
    jobs_tsv = build_jobs(args, snapshot, run_dir)
    launcher_root = run_dir / "launcher"
    command = [
        str(args.python_bin),
        str(args.launcher),
        "--jobs-tsv",
        str(jobs_tsv),
        "--out-root",
        str(launcher_root),
        "--stagger-seconds",
        str(args.stagger_seconds),
        "--poll-seconds",
        str(args.launcher_poll_seconds),
    ]
    with (args.out_root / "commands.log").open("a", encoding="utf-8") as f:
        f.write(f"[{utc_now()}] {args.worker_name} navtest {command_string(command)}\n")
    log(args, f"start navtest checkpoint={src.name} val_PDMS={row.get('PDMS')}")
    proc = subprocess.run(command, cwd=args.project_root)
    if proc.returncode != 0:
        raise RuntimeError(f"navtest launcher failed with returncode={proc.returncode}")
    metrics = aggregate(run_dir, args.num_shards, snapshot, row)
    write_navtest_summary(args)
    log(args, f"done navtest checkpoint={src.name} PDMS={metrics.get('PDMS')}")
    return metrics


def claim_and_run_one(args: argparse.Namespace, ranked: List[Dict[str, Any]]) -> bool:
    for row in ranked:
        key = row["checkpoint_key"]
        run_dir = args.out_root / "navtest" / key
        if (run_dir / "metrics.json").is_file() or (run_dir / "failed.json").is_file():
            continue
        claim_dir = args.out_root / "claims" / key
        try:
            claim_dir.mkdir(parents=True)
        except FileExistsError:
            continue
        claim = {"worker_name": args.worker_name, "claimed_at": utc_now(), "row": {k: str(v) for k, v in row.items()}}
        (claim_dir / "claim.json").write_text(json.dumps(claim, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            run_navtest(args, row)
        except Exception as exc:
            run_dir.mkdir(parents=True, exist_ok=True)
            payload = {"checkpoint": str(row.get("checkpoint") or ""), "error": repr(exc), "failed_at": utc_now()}
            (run_dir / "failed.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            log(args, f"failed navtest checkpoint={row.get('checkpoint_name')} error={exc!r}")
        return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote combined val6000 top-k checkpoints to full navtest with shared claims.")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--val-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker-name", required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--python-bin", type=Path, default=PYTHON_BIN)
    parser.add_argument("--launcher", type=Path, default=LAUNCHER)
    parser.add_argument("--navtest-chunk-cache-root", type=Path, required=True)
    parser.add_argument("--navtest-chunk-name-pattern", required=True)
    parser.add_argument("--navtest-metric-cache-dir", type=Path, required=True)
    parser.add_argument("--navtest-vlm-text-anchor-cache-root", type=Path, default=None)
    parser.add_argument("--required-navtest-index-count", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-val-completed", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=120.0)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--gpus-csv", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--trajectory-output-key", choices=("pred_traj", "pred_coarse_traj"), default="pred_traj")
    parser.add_argument("--stagger-seconds", type=float, default=1.0)
    parser.add_argument("--launcher-poll-seconds", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "logs").mkdir(parents=True, exist_ok=True)
    (args.out_root / f"{args.worker_name}.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    while True:
        if navtest_cache_ready(args):
            ranked, total = combined_top(args)
            if len(ranked) >= args.min_val_completed:
                log(args, "top candidates: " + ", ".join(f"{row.get('checkpoint_name')}={row.get('PDMS')}" for row in ranked))
                started = claim_and_run_one(args, ranked)
                if not started:
                    write_navtest_summary(args)
                    log(args, f"no unclaimed top-{args.top_k} navtest candidate; val_completed={total}")
            else:
                log(args, f"waiting for val6000 scores {len(ranked)}/{args.min_val_completed}; raw_rows={total}")
        if args.once:
            return 0
        time.sleep(max(args.poll_seconds, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())
