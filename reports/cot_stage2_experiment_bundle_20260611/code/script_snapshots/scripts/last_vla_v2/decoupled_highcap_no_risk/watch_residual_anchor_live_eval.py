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
from typing import Any, Dict, Iterable, List, Optional


EXPERIMENTS = ("pure_recogdrive", "lastvla_A")
METRIC_KEYS = ("PDMS", "NC", "DAC", "TTC", "comfort", "EP", "DDC")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously evaluate new residual-anchor stage2 checkpoints on navtest."
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/project/VLA-AD_last_vla_dev"))
    parser.add_argument("--python-bin", type=Path, default=Path("/root/miniconda3/envs/navsim/bin/python"))
    parser.add_argument("--launcher", type=Path, default=Path("/mnt/project/skill/stable-gpu-job-launch/scripts/stable_gpu_launcher.py"))
    parser.add_argument("--pure-run-root", type=Path, required=True)
    parser.add_argument("--lastvla-run-root", type=Path, required=True)
    parser.add_argument(
        "--pure-config",
        type=Path,
        default=Path("configs/last_vla_v2/decoupled_highcap_no_risk/original_recogdrive_residual_anchor_eval_flat.yaml"),
    )
    parser.add_argument(
        "--lastvla-config",
        type=Path,
        default=Path("configs/last_vla_v2/decoupled_highcap_no_risk/progressive_sft_vlm_text_residual_eval_flat.yaml"),
    )
    parser.add_argument(
        "--navtest-chunk-cache-root",
        type=Path,
        default=Path("/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/navtest_full_highcap_chunks"),
    )
    parser.add_argument(
        "--metric-cache-dir",
        type=Path,
        default=Path("/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1"),
    )
    parser.add_argument("--vlm-text-anchor-cache-root", type=Path, required=True)
    parser.add_argument(
        "--seed-eval-root",
        type=Path,
        action="append",
        default=[Path("/mnt/project/VLA-AD/outputs/last_vla_v2/vlm_text_residual_anchor_ckpt_eval_20260610T051017Z")],
        help="Existing residual-anchor eval root used only to seed completed checkpoint basenames. Can be repeated.",
    )
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--stable-seconds", type=float, default=90.0)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--gpus-csv", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="fp32")
    parser.add_argument("--trajectory-output-key", choices=("pred_traj", "pred_coarse_traj"), default="pred_traj")
    parser.add_argument("--stagger-seconds", type=float, default=8.0)
    parser.add_argument("--launcher-poll-seconds", type=float, default=20.0)
    parser.add_argument("--max-new-per-cycle", type=int, default=0, help="0 means no per-cycle limit.")
    parser.add_argument("--oldest-first", action="store_true", help="Default is newest first for fresh ckpts.")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def log(out_root: Path, message: str) -> None:
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    log_path = out_root / "logs" / "watcher.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


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


def experiment_roots(args: argparse.Namespace) -> Dict[str, Path]:
    return {
        "pure_recogdrive": args.pure_run_root,
        "lastvla_A": args.lastvla_run_root,
    }


def experiment_configs(args: argparse.Namespace) -> Dict[str, Path]:
    return {
        "pure_recogdrive": args.pure_config,
        "lastvla_A": args.lastvla_config,
    }


def discover_ckpts(run_root: Path) -> List[Path]:
    ckpts = [
        path
        for path in run_root.glob("lightning_logs/**/checkpoints/epoch=*-step=*.ckpt")
        if path.is_file()
    ]
    return sorted(ckpts, key=lambda p: (p.stat().st_mtime, p.name))


def ckpt_stable(path: Path, stable_seconds: float) -> bool:
    if not path.is_file():
        return False
    try:
        st = path.stat()
        if st.st_size <= 0:
            return False
        if time.time() - st.st_mtime < stable_seconds:
            return False
        size_a = st.st_size
        time.sleep(min(max(stable_seconds, 1.0), 15.0))
        return path.is_file() and path.stat().st_size == size_a and size_a > 0
    except OSError:
        return False


def safe_name(path: Path) -> str:
    return path.name.replace("/", "_").replace(":", "_").replace(" ", "_").removesuffix(".ckpt")


def seed_completed_from_previous(seed_roots: Optional[List[Path]]) -> Dict[str, Dict[str, Any]]:
    completed: Dict[str, Dict[str, Any]] = {name: {} for name in EXPERIMENTS}
    if not seed_roots:
        return completed
    for seed_root in seed_roots:
        if seed_root is None or not seed_root.exists():
            continue
        for experiment in EXPERIMENTS:
            summary = seed_root / experiment / "summary" / "A_summary.csv"
            if not summary.is_file():
                continue
            with summary.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    ckpt = Path(row.get("checkpoint") or "")
                    if ckpt.name:
                        completed[experiment][ckpt.name] = {
                            "seeded_from": str(summary),
                            "PDMS": row.get("PDMS"),
                            "trajectory_l1": row.get("trajectory_l1"),
                        }
    return completed


def load_state(args: argparse.Namespace) -> Dict[str, Any]:
    state_path = args.out_root / "state.json"
    default = {
        "created_at": utc_now(),
        "completed": seed_completed_from_previous(args.seed_eval_root),
        "failed": {name: {} for name in EXPERIMENTS},
        "running": None,
    }
    state = read_json(state_path, default)
    for key in ("completed", "failed"):
        state.setdefault(key, {})
        for experiment in EXPERIMENTS:
            state[key].setdefault(experiment, {})
    return state


def save_state(args: argparse.Namespace, state: Dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    write_json(args.out_root / "state.json", state)


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def weighted_mean(items: Iterable[Dict[str, Any]], key: str, weight_key: str) -> Optional[float]:
    total = 0.0
    weight = 0
    for item in items:
        value = item.get(key)
        w = int(item.get(weight_key) or 0)
        if value is not None and w > 0:
            total += float(value) * w
            weight += w
    return total / weight if weight else None


def aggregate_metrics(run_dir: Path, num_shards: int, checkpoint: Path, experiment: str, output_key: str) -> Dict[str, Any]:
    shard_metrics: List[Dict[str, Any]] = []
    missing: List[str] = []
    for shard in range(num_shards):
        metrics_path = run_dir / f"shard_{shard:02d}" / "metrics.json"
        if metrics_path.is_file():
            shard_metrics.append(json.loads(metrics_path.read_text(encoding="utf-8")))
        else:
            missing.append(str(metrics_path))
    if missing:
        raise FileNotFoundError(f"Missing shard metrics: {missing[:3]} ... ({len(missing)} total)")
    aggregate: Dict[str, Any] = {
        "experiment": experiment,
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "trajectory_output_key": output_key,
        "num_shards": num_shards,
        "num_samples": sum(int(item.get("num_samples") or 0) for item in shard_metrics),
        "num_pdm_valid": sum(int(item.get("num_pdm_valid") or 0) for item in shard_metrics),
        "num_pdm_missing_metric_cache": sum(int(item.get("num_pdm_missing_metric_cache") or 0) for item in shard_metrics),
        "num_pdm_failed": sum(int(item.get("num_pdm_failed") or 0) for item in shard_metrics),
        "trajectory_l1": weighted_mean(shard_metrics, "trajectory_l1", "num_samples"),
        "shards": [str(run_dir / f"shard_{idx:02d}") for idx in range(num_shards)],
    }
    for key in METRIC_KEYS:
        aggregate[key] = weighted_mean(shard_metrics, key, "num_pdm_valid")
    aggregate["pdm_score"] = aggregate["PDMS"]
    write_json(run_dir / "metrics.json", aggregate)
    return aggregate


def write_summary(out_root: Path, state: Dict[str, Any]) -> None:
    fields = [
        "experiment",
        "checkpoint_name",
        "checkpoint",
        "PDMS",
        "trajectory_l1",
        "num_pdm_valid",
        "num_pdm_missing_metric_cache",
        "num_pdm_failed",
        "completed_at",
        "eval_dir",
    ]
    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for experiment in EXPERIMENTS:
        exp_rows = []
        for ckpt_name, payload in sorted(state.get("completed", {}).get(experiment, {}).items()):
            if "checkpoint" not in payload:
                continue
            row = {"experiment": experiment, "checkpoint_name": ckpt_name, **payload}
            rows.append(row)
            exp_rows.append(row)
        with (summary_dir / f"{experiment}_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in exp_rows:
                writer.writerow({key: row.get(key) for key in fields})
    with (summary_dir / "combined_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def build_jobs(
    args: argparse.Namespace,
    experiment: str,
    checkpoint_snapshot: Path,
    run_dir: Path,
    config: Path,
) -> Path:
    jobs_dir = run_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    jobs_tsv = jobs_dir / "jobs.tsv"
    gpus = [item.strip() for item in args.gpus_csv.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus-csv must contain at least one GPU id")
    with jobs_tsv.open("w", encoding="utf-8") as f:
        for shard in range(args.num_shards):
            shard_dir = run_dir / f"shard_{shard:02d}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            job_name = f"{experiment}_{checkpoint_snapshot.stem}_shard_{shard:02d}"
            gpu = gpus[shard % len(gpus)]
            command = shlex.join(
                [
                    str(args.python_bin),
                    "scripts/eval_recogdrive_expert_pdm.py",
                    "--config",
                    str(config),
                    "--checkpoint",
                    str(checkpoint_snapshot),
                    "--chunk-cache-root",
                    str(args.navtest_chunk_cache_root),
                    "--chunk-name-pattern",
                    "navtest_full_chunk_*",
                    "--metric-cache-dir",
                    str(args.metric_cache_dir),
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
                    "--vlm-text-anchor-cache-root",
                    str(args.vlm_text_anchor_cache_root),
                ]
            )
            f.write(f"{job_name}\t{gpu}\t{command}\n")
    return jobs_tsv


def evaluate_checkpoint(args: argparse.Namespace, experiment: str, ckpt: Path) -> Dict[str, Any]:
    config = experiment_configs(args)[experiment]
    run_dir = args.out_root / "eval" / experiment / f"{safe_name(ckpt)}_{args.trajectory_output_key}"
    snapshot = args.out_root / "checkpoint_snapshots" / experiment / ckpt.name
    hardlink_or_copy(ckpt, snapshot)
    jobs_tsv = build_jobs(args, experiment, snapshot, run_dir, config)
    launcher_root = run_dir / "launcher"
    launcher_root.mkdir(parents=True, exist_ok=True)
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
    log(args.out_root, f"start eval experiment={experiment} ckpt={ckpt.name}")
    with (args.out_root / "commands.log").open("a", encoding="utf-8") as f:
        f.write(f"[{utc_now()}] " + shlex.join(command) + "\n")
    proc = subprocess.run(command, cwd=args.project_root)
    if proc.returncode != 0:
        raise RuntimeError(f"stable launcher failed with returncode={proc.returncode}")
    metrics = aggregate_metrics(run_dir, args.num_shards, snapshot, experiment, args.trajectory_output_key)
    log(args.out_root, f"done eval experiment={experiment} ckpt={ckpt.name} PDMS={metrics.get('PDMS')}")
    return metrics


def validate_inputs(args: argparse.Namespace) -> None:
    required_paths = [
        args.project_root,
        args.python_bin,
        args.launcher,
        args.pure_run_root,
        args.lastvla_run_root,
        args.navtest_chunk_cache_root,
        args.metric_cache_dir,
        args.vlm_text_anchor_cache_root / "index.jsonl",
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(str(path))
    for config in experiment_configs(args).values():
        if not (args.project_root / config).exists() and not config.exists():
            raise FileNotFoundError(str(config))


def pending_ckpts(args: argparse.Namespace, state: Dict[str, Any]) -> List[tuple[str, Path]]:
    roots = experiment_roots(args)
    pending: List[tuple[str, Path]] = []
    for experiment in EXPERIMENTS:
        completed = set(state.get("completed", {}).get(experiment, {}))
        failed = set(state.get("failed", {}).get(experiment, {}))
        for ckpt in discover_ckpts(roots[experiment]):
            if ckpt.name in completed or ckpt.name in failed:
                continue
            if ckpt_stable(ckpt, args.stable_seconds):
                pending.append((experiment, ckpt))
    pending.sort(key=lambda item: item[1].stat().st_mtime, reverse=not args.oldest_first)
    if args.max_new_per_cycle > 0:
        pending = pending[: args.max_new_per_cycle]
    return pending


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "logs").mkdir(parents=True, exist_ok=True)
    (args.out_root / "watcher.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    validate_inputs(args)
    state = load_state(args)
    save_state(args, state)
    write_summary(args.out_root, state)
    log(args.out_root, f"watching pure={args.pure_run_root}")
    log(args.out_root, f"watching lastvla_A={args.lastvla_run_root}")
    log(args.out_root, f"seed_eval_roots={[str(path) for path in (args.seed_eval_root or [])]}")

    while True:
        state = load_state(args)
        todo = pending_ckpts(args, state)
        status = {
            "updated_at": utc_now(),
            "pending": [{"experiment": exp, "checkpoint": str(path), "checkpoint_name": path.name} for exp, path in todo],
            "completed_counts": {exp: len(state.get("completed", {}).get(exp, {})) for exp in EXPERIMENTS},
            "failed_counts": {exp: len(state.get("failed", {}).get(exp, {})) for exp in EXPERIMENTS},
        }
        write_json(args.out_root / "status.json", status)
        if not todo:
            log(args.out_root, "no new stable checkpoints")
            if args.once:
                return 0
            time.sleep(max(args.poll_seconds, 1.0))
            continue
        for experiment, ckpt in todo:
            state = load_state(args)
            state["running"] = {"experiment": experiment, "checkpoint": str(ckpt), "checkpoint_name": ckpt.name, "started_at": utc_now()}
            save_state(args, state)
            try:
                metrics = evaluate_checkpoint(args, experiment, ckpt)
            except Exception as exc:
                state = load_state(args)
                state["running"] = None
                state["failed"][experiment][ckpt.name] = {
                    "checkpoint": str(ckpt),
                    "error": repr(exc),
                    "failed_at": utc_now(),
                }
                save_state(args, state)
                write_summary(args.out_root, state)
                log(args.out_root, f"failed eval experiment={experiment} ckpt={ckpt.name} error={exc!r}")
                continue
            state = load_state(args)
            state["running"] = None
            state["completed"][experiment][ckpt.name] = {
                "checkpoint": str(ckpt),
                "eval_dir": str(args.out_root / "eval" / experiment / f"{safe_name(ckpt)}_{args.trajectory_output_key}"),
                "PDMS": metrics.get("PDMS"),
                "trajectory_l1": metrics.get("trajectory_l1"),
                "num_pdm_valid": metrics.get("num_pdm_valid"),
                "num_pdm_missing_metric_cache": metrics.get("num_pdm_missing_metric_cache"),
                "num_pdm_failed": metrics.get("num_pdm_failed"),
                "completed_at": utc_now(),
            }
            save_state(args, state)
            write_summary(args.out_root, state)
        if args.once:
            return 0
        time.sleep(max(args.poll_seconds, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())
