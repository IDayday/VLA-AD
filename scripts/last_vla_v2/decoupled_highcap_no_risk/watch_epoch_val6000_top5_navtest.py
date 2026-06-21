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
TRAIN_CHUNK_ROOT = Path("/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/train_full_highcap_chunks")
TRAIN_METRIC_CACHE = Path("/mnt/project/VLA-AD/cache/metric_cache_train_full")
TRAIN_ANCHOR_CACHE = Path("/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/vlm_text_anchor_2bbase_train_20260609T052233Z")
NAVTEST_CHUNK_ROOT = Path("/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/navtest_full_highcap_chunks")
NAVTEST_METRIC_CACHE = Path("/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1")
METRIC_KEYS = ("PDMS", "NC", "DAC", "TTC", "comfort", "EP", "DDC")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_name(text: str) -> str:
    keep = []
    for char in text:
        keep.append(char if char.isalnum() or char in ("-", "_", ".") else "_")
    return "".join(keep).strip("._") or "item"


def read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def command_string(command: Sequence[Any]) -> str:
    return shlex.join(str(part) for part in command)


def log(args: argparse.Namespace, message: str) -> None:
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    path = args.out_root / "logs" / "watcher.log"
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


def epoch_from_ckpt(path: Path) -> Optional[int]:
    stem = path.stem
    if stem.startswith("epoch_"):
        try:
            return int(stem.split("_", 1)[1])
        except ValueError:
            return None
    if stem.startswith("epoch="):
        try:
            return int(stem.split("=", 1)[1].split("-", 1)[0]) + 1
        except ValueError:
            return None
    return None


def discover_epoch_ckpts(run_root: Path) -> List[Path]:
    direct = list(run_root.glob("epoch_*.ckpt"))
    lightning = list(run_root.glob("lightning_logs/**/checkpoints/epoch=*-step=*.ckpt"))
    ckpts = [path for path in direct + lightning if path.is_file() and epoch_from_ckpt(path) is not None]
    return sorted(set(ckpts), key=lambda path: (epoch_from_ckpt(path) or 0, str(path)))


def stable_file(path: Path, stable_seconds: float) -> bool:
    if not path.is_file():
        return False
    try:
        size_a = path.stat().st_size
        if size_a <= 0 or time.time() - path.stat().st_mtime < stable_seconds:
            return False
        time.sleep(min(max(stable_seconds, 1.0), 15.0))
        return path.is_file() and path.stat().st_size == size_a
    except OSError:
        return False


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


def aggregate(run_dir: Path, num_shards: int, checkpoint: Path, split_name: str) -> Dict[str, Any]:
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
        "split_name": split_name,
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "epoch": epoch_from_ckpt(checkpoint),
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
    write_json(run_dir / "metrics.json", payload)
    return payload


def build_jobs(args: argparse.Namespace, checkpoint: Path, run_dir: Path, split_name: str) -> Path:
    if split_name == "val6000":
        chunk_root = args.val_chunk_cache_root
        chunk_pattern = args.val_chunk_name_pattern
        metric_cache = args.val_metric_cache_dir
        sample_token_file = args.val_sample_token_file
        anchor_root = args.val_vlm_text_anchor_cache_root
    elif split_name == "navtest":
        chunk_root = args.navtest_chunk_cache_root
        chunk_pattern = args.navtest_chunk_name_pattern
        metric_cache = args.navtest_metric_cache_dir
        sample_token_file = None
        anchor_root = args.navtest_vlm_text_anchor_cache_root
    else:
        raise ValueError(split_name)
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
                split_name,
                "--chunk-cache-root",
                str(chunk_root),
                "--chunk-name-pattern",
                chunk_pattern,
                "--metric-cache-dir",
                str(metric_cache),
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
            if sample_token_file is not None:
                command.extend(["--sample-token-file", str(sample_token_file)])
            if anchor_root is not None and anchor_root.exists():
                command.extend(["--vlm-text-anchor-cache-root", str(anchor_root)])
            f.write(f"{split_name}_{safe_name(checkpoint.stem)}_shard_{shard:02d}\t{gpus[shard % len(gpus)]}\t{command_string(command)}\n")
    return jobs_tsv


def run_eval(args: argparse.Namespace, checkpoint: Path, split_name: str) -> Dict[str, Any]:
    ckpt_key = safe_name(str(checkpoint))
    snapshot = args.out_root / "checkpoint_snapshots" / ckpt_key / checkpoint.name
    hardlink_or_copy(checkpoint, snapshot)
    run_dir = args.out_root / split_name / ckpt_key
    jobs_tsv = build_jobs(args, snapshot, run_dir, split_name)
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
        f.write(f"[{utc_now()}] {split_name} {command_string(command)}\n")
    log(args, f"start {split_name} checkpoint={checkpoint}")
    proc = subprocess.run(command, cwd=args.project_root)
    if proc.returncode != 0:
        raise RuntimeError(f"{split_name} launcher failed with returncode={proc.returncode}")
    metrics = aggregate(run_dir, args.num_shards, snapshot, split_name)
    metrics["source_checkpoint"] = str(checkpoint)
    log(args, f"done {split_name} checkpoint={checkpoint.name} PDMS={metrics.get('PDMS')}")
    return metrics


def load_state(args: argparse.Namespace) -> Dict[str, Any]:
    default = {"created_at": utc_now(), "val_completed": {}, "navtest_completed": {}, "failed": {}}
    state = read_json(args.out_root / "state.json", default)
    for key in ("val_completed", "navtest_completed", "failed"):
        state.setdefault(key, {})
    return state


def save_state(args: argparse.Namespace, state: Dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    write_json(args.out_root / "state.json", state)


def write_summary(args: argparse.Namespace, state: Dict[str, Any]) -> None:
    fields = ["split_name", "checkpoint_name", "checkpoint", "epoch", "PDMS", "trajectory_l1", "num_pdm_valid", "completed_at", "eval_dir"]
    for split_key, rows_by_key in [("val6000", state.get("val_completed", {})), ("navtest", state.get("navtest_completed", {}))]:
        path = args.out_root / "summary" / f"{split_key}_summary.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = sorted(rows_by_key.values(), key=lambda row: (row.get("epoch") or 0, row.get("checkpoint_name") or ""))
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fields})


def ckpt_key(path: Path) -> str:
    return str(path.resolve())


def run_cycle(args: argparse.Namespace, state: Dict[str, Any]) -> Dict[str, Any]:
    if not args.val_sample_token_file.is_file():
        log(args, f"waiting for val token file: {args.val_sample_token_file}")
        return state
    half_epoch = int(args.max_epochs * args.navtest_after_fraction)
    if half_epoch < 1:
        half_epoch = 1
    for ckpt in discover_epoch_ckpts(args.run_root):
        epoch = epoch_from_ckpt(ckpt)
        if epoch is None:
            continue
        if args.val_every_n_epochs > 0 and epoch % args.val_every_n_epochs != 0:
            continue
        key = ckpt_key(ckpt)
        if key in state["val_completed"] or key in state["failed"]:
            continue
        if not stable_file(ckpt, args.stable_seconds):
            continue
        try:
            state["val_completed"][key] = run_eval(args, ckpt, "val6000")
        except Exception as exc:
            state["failed"][key] = {"checkpoint": str(ckpt), "stage": "val6000", "error": repr(exc), "failed_at": utc_now()}
        save_state(args, state)
        write_summary(args, state)

    eligible = [
        (key, payload)
        for key, payload in state["val_completed"].items()
        if int(payload.get("epoch") or 0) >= half_epoch and payload.get("PDMS") is not None
    ]
    top = sorted(eligible, key=lambda item: float(item[1]["PDMS"]), reverse=True)[: args.top_k]
    for key, payload in top:
        if key in state["navtest_completed"]:
            continue
        ckpt = Path(payload["checkpoint"])
        if not ckpt.is_file():
            continue
        try:
            state["navtest_completed"][key] = run_eval(args, ckpt, "navtest")
        except Exception as exc:
            state["failed"][key] = {"checkpoint": str(ckpt), "stage": "navtest", "error": repr(exc), "failed_at": utc_now()}
        save_state(args, state)
        write_summary(args, state)
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate epoch checkpoints on val6000 and promote top-k to full navtest.")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--python-bin", type=Path, default=PYTHON_BIN)
    parser.add_argument("--launcher", type=Path, default=LAUNCHER)
    parser.add_argument("--val-sample-token-file", type=Path, required=True)
    parser.add_argument("--val-chunk-cache-root", type=Path, default=TRAIN_CHUNK_ROOT)
    parser.add_argument("--val-chunk-name-pattern", default="train_*")
    parser.add_argument("--val-metric-cache-dir", type=Path, default=TRAIN_METRIC_CACHE)
    parser.add_argument("--val-vlm-text-anchor-cache-root", type=Path, default=TRAIN_ANCHOR_CACHE)
    parser.add_argument("--navtest-chunk-cache-root", type=Path, default=NAVTEST_CHUNK_ROOT)
    parser.add_argument("--navtest-chunk-name-pattern", default="navtest_full_chunk_*")
    parser.add_argument("--navtest-metric-cache-dir", type=Path, default=NAVTEST_METRIC_CACHE)
    parser.add_argument("--navtest-vlm-text-anchor-cache-root", type=Path, default=None)
    parser.add_argument("--val-every-n-epochs", type=int, default=5)
    parser.add_argument("--navtest-after-fraction", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--stable-seconds", type=float, default=90.0)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--gpus-csv", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="fp32")
    parser.add_argument("--trajectory-output-key", choices=("pred_traj", "pred_coarse_traj"), default="pred_traj")
    parser.add_argument("--stagger-seconds", type=float, default=8.0)
    parser.add_argument("--launcher-poll-seconds", type=float, default=20.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "logs").mkdir(parents=True, exist_ok=True)
    (args.out_root / "watcher.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    state = load_state(args)
    while True:
        state = run_cycle(args, state)
        save_state(args, state)
        write_summary(args, state)
        if args.once:
            return 0
        time.sleep(max(args.poll_seconds, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())
