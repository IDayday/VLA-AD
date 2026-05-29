#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


CONFIGS = {
    "a0_no_expert": "configs/ablations/recogdrive2b_A0_base_no_expert.yaml",
    "a1_jepa_only": "configs/ablations/recogdrive2b_A1_jepa_only.yaml",
    "a2_vggt_only": "configs/ablations/recogdrive2b_A2_vggt_only.yaml",
    "a3_context_only": "configs/ablations/recogdrive2b_A3_context_only.yaml",
    "a4_jepa_vggt": "configs/ablations/recogdrive2b_A4_full.yaml",
    "a4_v2": "configs/ablations/recogdrive2b_A4_v2.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snapshot staged ReCogDrive checkpoints and run PDM evals sequentially.")
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/project/VLA-AD"))
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--runs", default="a0_no_expert,a4_v2")
    parser.add_argument("--checkpoint-steps", default="50000,60000,80000,100000,120000,140000,160000")
    parser.add_argument("--include-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--stable-seconds", type=int, default=20)
    parser.add_argument("--python", default="/root/miniconda3/envs/navsim/bin/python")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--chunk-cache-root", type=Path, default=Path("/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1"))
    parser.add_argument("--chunk-name-pattern", default="navtest_full_chunk_*")
    parser.add_argument("--metric-cache-dir", type=Path, default=Path("/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1"))
    parser.add_argument("--expected-metric-caches", type=int, default=12138)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="fp32")
    parser.add_argument("--cuda-visible-devices", default="7")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def selected_runs(args: argparse.Namespace) -> List[str]:
    return [run.strip() for run in args.runs.split(",") if run.strip()]


def checkpoint_steps(args: argparse.Namespace) -> List[int]:
    return [int(item.strip()) for item in args.checkpoint_steps.split(",") if item.strip()]


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=str, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def last_train_row(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "train_log.jsonl"
    if not path.is_file():
        return {}
    last: Dict[str, Any] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    continue
    return last


def stable_stat(path: Path, stable_seconds: int) -> os.stat_result:
    if not path.is_file():
        raise FileNotFoundError(path)
    first = path.stat()
    time.sleep(stable_seconds)
    second = path.stat()
    if first.st_size != second.st_size or first.st_mtime_ns != second.st_mtime_ns:
        raise RuntimeError(f"{path} changed while waiting for a stable file")
    return second


def checkpoint_meta(path: Path) -> Dict[str, Any]:
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu")
        metrics = ckpt.get("metrics") if isinstance(ckpt, dict) else None
        return {
            "global_step": ckpt.get("global_step") if isinstance(ckpt, dict) else None,
            "metrics": metrics if isinstance(metrics, dict) else {},
        }
    except Exception as exc:
        return {"metadata_error": repr(exc)}


def wait_for_paths(args: argparse.Namespace, paths: Dict[str, Path], stage_dir: Path, *, require_final_report: bool) -> None:
    status_path = stage_dir / "wait_status.json"
    while True:
        missing = []
        run_status = {}
        for run, path in paths.items():
            run_dir = args.experiment_root / run
            row = last_train_row(run_dir)
            exists = path.is_file()
            final_report = run_dir / "final_report.md"
            final_ready = final_report.is_file()
            run_status[run] = {
                "checkpoint": str(path),
                "exists": exists,
                "final_report": str(final_report),
                "final_ready": final_ready,
                "train_step": row.get("step"),
                "train_epoch": row.get("global_epoch", row.get("epoch")),
            }
            if not exists:
                missing.append(f"{run}: missing {path}")
            if require_final_report and not final_ready:
                missing.append(f"{run}: missing final report {final_report}")
        payload = {
            "time": now(),
            "missing": missing,
            "runs": run_status,
        }
        write_json(status_path, payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
        if not missing:
            return
        time.sleep(args.poll_seconds)


def snapshot_paths(args: argparse.Namespace, paths: Dict[str, Path], stage_dir: Path, stage_name: str) -> Dict[str, Dict[str, Any]]:
    snapshots: Dict[str, Dict[str, Any]] = {}
    for run, src in paths.items():
        while True:
            try:
                before = stable_stat(src, args.stable_seconds)
                dst = stage_dir / "checkpoint_snapshots" / f"{run}_{stage_name}.ckpt"
                tmp = dst.with_suffix(dst.suffix + ".tmp")
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, tmp)
                after = src.stat()
                if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError(f"{src} changed during copy")
                os.replace(tmp, dst)
                snapshots[run] = {
                    "source": str(src),
                    "snapshot": str(dst),
                    "source_size": after.st_size,
                    "source_mtime": datetime.fromtimestamp(after.st_mtime).isoformat(timespec="seconds"),
                    "checkpoint_meta": checkpoint_meta(dst),
                }
                break
            except Exception as exc:
                print(f"snapshot retry for {src}: {exc}", flush=True)
                time.sleep(max(5, args.stable_seconds))
    write_json(stage_dir / "snapshot_status.json", {"time": now(), "stage": stage_name, "snapshots": snapshots})
    return snapshots


def run_specs(args: argparse.Namespace, snapshots: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, str | None]]:
    specs: Dict[str, Dict[str, str | None]] = {}
    for run, info in snapshots.items():
        if run not in CONFIGS:
            raise ValueError(f"Unknown run {run!r}; add it to CONFIGS first.")
        specs[run] = {
            "config": str(args.project_root / CONFIGS[run]),
            "checkpoint": str(info["snapshot"]),
            "train_dir": None,
        }
    return specs


def run_eval(args: argparse.Namespace, stage_dir: Path, spec_path: Path, stage_name: str) -> int:
    eval_dir = stage_dir / "pdm_full_navtest"
    cmd = [
        args.python,
        "scripts/run_recogdrive_full_pdm_suite.py",
        "--run-spec-json",
        str(spec_path),
        "--output-root",
        str(eval_dir),
        "--runs",
        ",".join(selected_runs(args)),
        "--chunk-cache-root",
        str(args.chunk_cache_root),
        "--chunk-name-pattern",
        args.chunk_name_pattern,
        "--metric-cache-dir",
        str(args.metric_cache_dir),
        "--expected-metric-caches",
        str(args.expected_metric_caches),
        "--precision",
        args.precision,
        "--root-report-name",
        f"{stage_name}_STAGE1BASE_FULL200_PDM_REPORT.md",
        "--wait",
        "--poll-seconds",
        str(args.poll_seconds),
    ]
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{args.project_root}:{env.get('PYTHONPATH', '')}"
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    write_json(stage_dir / "eval_command.json", {"time": now(), "cmd": cmd, "cuda_visible_devices": args.cuda_visible_devices})
    print("RUN", " ".join(cmd), flush=True)
    with (stage_dir / "eval_console.log").open("w", encoding="utf-8") as log_fp:
        return subprocess.run(cmd, cwd=args.project_root, env=env, stdout=log_fp, stderr=subprocess.STDOUT).returncode


def stage_paths(args: argparse.Namespace, stage_name: str, step: int | None) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for run in selected_runs(args):
        run_dir = args.experiment_root / run
        paths[run] = run_dir / "latest.ckpt" if step is None else run_dir / f"step_{step:08d}.ckpt"
    return paths


def process_stage(args: argparse.Namespace, output_root: Path, stage_name: str, step: int | None) -> int:
    stage_dir = output_root / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    paths = stage_paths(args, stage_name, step)
    wait_for_paths(args, paths, stage_dir, require_final_report=(step is None))
    snapshots = snapshot_paths(args, paths, stage_dir, stage_name)
    specs = run_specs(args, snapshots)
    spec_path = stage_dir / "run_specs.json"
    write_json(spec_path, specs)
    code = run_eval(args, stage_dir, spec_path, stage_name)
    write_json(stage_dir / "stage_status.json", {"time": now(), "stage": stage_name, "returncode": code})
    return code


def main() -> int:
    args = parse_args()
    output_root = args.output_root or (args.experiment_root / "pdm_staged_navtest")
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "watcher_args.json", vars(args))
    stages: List[tuple[str, int | None]] = [(f"step_{step:08d}", step) for step in checkpoint_steps(args)]
    if args.include_final:
        stages.append(("final_latest", None))
    for stage_name, step in stages:
        done_path = output_root / stage_name / "stage_status.json"
        if done_path.is_file():
            print(f"SKIP {stage_name}: {done_path} exists", flush=True)
            continue
        code = process_stage(args, output_root, stage_name, step)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
