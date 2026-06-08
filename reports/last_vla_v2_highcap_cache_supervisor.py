#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any


REPO_ROOT = Path(os.environ.get("PROJECT_ROOT", "/mnt/project/VLA-AD_last_vla_dev"))
PYTHON_BIN = os.environ.get("PYTHON_BIN", "/root/miniconda3/envs/navsim/bin/python")
BASE_CHUNK_ROOT = Path(os.environ.get("BASE_CHUNK_ROOT", "/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1"))
CACHE_ROOT = Path(os.environ.get("HIGHCAP_CACHE_ROOT", "/mnt/project/VLA-AD/cache/last_vla_v2/highcap_no_risk"))
OUT_ROOT = Path(os.environ.get("HIGHCAP_OUT_ROOT", "/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_full_sft"))
A0_INIT_CHECKPOINT = Path(os.environ.get("A0_INIT_CHECKPOINT", "/mnt/project/VLA-AD/outputs/a0_stage2_repro_20260531_003029/a0_official_aligned_a0complete/step_00100000.ckpt"))
VGGT_MODEL_PATH = Path(os.environ.get("VGGT_MODEL_PATH", "/mnt/project/VLA-AD/checkpoints/teachers/VGGT-1B"))
VJEPA_MODEL_PATH = Path(os.environ.get("VJEPA_MODEL_PATH", "/mnt/project/VLA-AD/checkpoints/teachers/vjepa2-vitl-fpc64-256"))
VLM_PATH = Path(os.environ.get("VLM_PATH", "/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B"))

TRAIN_PATTERN = "train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*"
GPU_COUNT = int(os.environ.get("HIGHCAP_CACHE_GPUS", "8"))
MAX_PARALLEL = int(os.environ.get("HIGHCAP_CACHE_PARALLEL", str(GPU_COUNT)))
GEOMETRY_SHARDS = int(os.environ.get("HIGHCAP_GEOMETRY_SHARDS", str(GPU_COUNT)))

JEPA_ROOT = CACHE_ROOT / "train_jepa128_overlay"
GEOMETRY_ROOT = CACHE_ROOT / "train_geometry192_overlay"
FULL_ROOT = CACHE_ROOT / "train_full_highcap_chunks"
MANIFEST_DIR = CACHE_ROOT / "manifests"
LOG_ROOT = OUT_ROOT / "cache_generation_logs"
REPORT_JSON = OUT_ROOT / "reports" / "last_vla_v2_highcap_cache_supervisor_report.json"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_command(cmd: list[str], log_path: Path, *, env: dict[str, str] | None = None, cwd: Path = REPO_ROOT) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    merged_env.setdefault("PYTHONUNBUFFERED", "1")
    merged_env.setdefault("TOKENIZERS_PARALLELISM", "false")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[{now()}] cwd={cwd}\n")
        f.write(f"[{now()}] command={shell_join(cmd)}\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), env=merged_env, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"[{now()}] exit_code={proc.returncode}\n")
        return int(proc.returncode)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def chunk_plan() -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for pattern in TRAIN_PATTERN.split(","):
        for chunk_dir in sorted(BASE_CHUNK_ROOT.glob(pattern)):
            if chunk_dir in seen or not (chunk_dir / "index.jsonl").is_file():
                continue
            seen.add(chunk_dir)
            metadata = read_json(chunk_dir / "metadata.json")
            name = chunk_dir.name
            match = re.search(r"_(\d+)$", name)
            chunk_index = int(match.group(1)) if match else len(chunks)
            records = line_count(chunk_dir / "index.jsonl")
            start = int(metadata["chunk_start"])
            stop = int(metadata.get("chunk_end", start + int(metadata.get("raw_window_size", records))))
            chunks.append(
                {
                    "name": name,
                    "chunk_index": chunk_index,
                    "records": records,
                    "chunk_start": start,
                    "chunk_stop": stop,
                }
            )
    if not chunks:
        raise RuntimeError(f"No base chunk dirs found under {BASE_CHUNK_ROOT}")
    return chunks


def validate_inputs() -> None:
    required = [REPO_ROOT, BASE_CHUNK_ROOT, A0_INIT_CHECKPOINT, VGGT_MODEL_PATH, VJEPA_MODEL_PATH, VLM_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required path(s): " + ", ".join(missing))
    for path in (JEPA_ROOT, GEOMETRY_ROOT, FULL_ROOT):
        if path.exists() and any(path.iterdir()) and os.environ.get("OVERWRITE_HIGHCAP_CACHE", "0") != "1":
            raise RuntimeError(f"Refusing to reuse non-empty output dir without OVERWRITE_HIGHCAP_CACHE=1: {path}")
        if path.exists() and os.environ.get("OVERWRITE_HIGHCAP_CACHE", "0") == "1":
            backup = path.with_name(path.name + ".bak_" + time.strftime("%Y%m%d_%H%M%S"))
            path.rename(backup)
    for path in (CACHE_ROOT, JEPA_ROOT, GEOMETRY_ROOT, FULL_ROOT, MANIFEST_DIR, LOG_ROOT, REPORT_JSON.parent):
        path.mkdir(parents=True, exist_ok=True)


def jepa_command(item: dict[str, Any]) -> list[str]:
    return [
        PYTHON_BIN,
        "scripts/build_recogdrive_chunk_cache.py",
        "--navsim-root",
        "/mnt/navsim",
        "--split",
        "navtrain",
        "--chunk-index",
        str(item["chunk_index"]),
        "--chunk-start",
        str(item["chunk_start"]),
        "--chunk-stop",
        str(item["chunk_stop"]),
        "--chunk-size",
        str(item["records"]),
        "--output-dir",
        str(JEPA_ROOT / item["name"]),
        "--build-jepa",
        "--jepa-model-path",
        str(VJEPA_MODEL_PATH),
        "--num-jepa-tokens",
        "128",
        "--strict-highcap-jepa",
        "--precision",
        "bf16",
        "--device",
        "cuda",
        "--num-gpus",
        "1",
        "--workers",
        "1",
        "--allow-partial-final-chunk",
        "--log-every",
        "50",
    ]


def geometry_command(shard_index: int) -> list[str]:
    return [
        PYTHON_BIN,
        "scripts/build_last_vla_full_geometry_cache.py",
        "--chunk-cache-root",
        str(BASE_CHUNK_ROOT),
        "--chunk-name-pattern",
        TRAIN_PATTERN,
        "--output-cache-root",
        str(GEOMETRY_ROOT / f"full_geometry_overlay_shard_{shard_index}_of_{GEOMETRY_SHARDS}"),
        "--vggt-model-path",
        str(VGGT_MODEL_PATH),
        "--precision",
        "bf16",
        "--device",
        "cuda",
        "--num-geometry-tokens",
        "192",
        "--geometry-grid-rows",
        "12",
        "--geometry-grid-cols",
        "16",
        "--geometry-teacher-dim",
        "512",
        "--require-full-geometry",
        "--shard-index",
        str(shard_index),
        "--num-shards",
        str(GEOMETRY_SHARDS),
    ]


def run_parallel_jobs(jobs: list[tuple[str, list[str], Path]], stage: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    pending: set[Any] = set()
    job_iter = iter(enumerate(jobs))
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        while True:
            while len(pending) < MAX_PARALLEL:
                try:
                    idx, (name, cmd, log_path) = next(job_iter)
                except StopIteration:
                    break
                gpu = idx % max(1, GPU_COUNT)
                env = {"CUDA_VISIBLE_DEVICES": str(gpu)}
                future = executor.submit(run_command, cmd, log_path, env=env)
                future.job_info = {"stage": stage, "name": name, "gpu": gpu, "log": str(log_path), "command": cmd}  # type: ignore[attr-defined]
                pending.add(future)
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                info = dict(future.job_info)  # type: ignore[attr-defined]
                code = int(future.result())
                info["exit_code"] = code
                results.append(info)
                if code != 0:
                    raise RuntimeError(f"{stage} job failed: {info['name']} log={info['log']}")
    return results


def run_merge() -> dict[str, Any]:
    cmd = [
        PYTHON_BIN,
        "scripts/merge_last_vla_geometry_cache_into_chunks.py",
        "--base-chunk-root",
        str(BASE_CHUNK_ROOT),
        "--geometry-cache-root",
        str(GEOMETRY_ROOT),
        "--geometry-chunk-name-pattern",
        "*",
        "--jepa-cache-root",
        str(JEPA_ROOT),
        "--jepa-chunk-name-pattern",
        "*",
        "--output-chunk-root",
        str(FULL_ROOT),
        "--chunk-name-pattern",
        TRAIN_PATTERN,
        "--copy-mode",
        "hardlink",
        "--strict-coverage",
        "--min-coverage",
        "0.99",
        "--strict-jepa-coverage",
        "--min-jepa-coverage",
        "0.99",
        "--expected-jepa-tokens",
        "128",
        "--jepa-dim",
        "1024",
        "--num-geometry-tokens",
        "192",
        "--geometry-grid-rows",
        "12",
        "--geometry-grid-cols",
        "16",
        "--geometry-teacher-dim",
        "512",
    ]
    code = run_command(cmd, LOG_ROOT / "merge.log")
    if code != 0:
        raise RuntimeError(f"merge failed, see {LOG_ROOT / 'merge.log'}")
    return read_json(FULL_ROOT / "merge_summary.json")


def run_audit() -> dict[str, Any]:
    output = MANIFEST_DIR / "train_manifest.json"
    cmd = [
        PYTHON_BIN,
        "scripts/audit_last_vla_cache_manifest.py",
        "--cache-root",
        str(FULL_ROOT),
        "--chunk-name-pattern",
        TRAIN_PATTERN,
        "--strict-full-geometry",
        "--min-full-geometry-coverage",
        "0.99",
        "--expected-jepa-tokens",
        "128",
        "--expected-geometry-tokens",
        "192",
        "--geometry-teacher-dim",
        "512",
        "--strict-no-risk",
        "--output",
        str(output),
    ]
    code = run_command(cmd, LOG_ROOT / "audit.log")
    if code != 0:
        raise RuntimeError(f"audit failed, see {LOG_ROOT / 'audit.log'}")
    return read_json(output)


def run_readiness() -> Path:
    env = {
        "FULL_HIGHCAP_TRAIN_CHUNK_ROOT": str(FULL_ROOT),
        "A0_INIT_CHECKPOINT": str(A0_INIT_CHECKPOINT),
        "OUT_ROOT": str(OUT_ROOT),
        "PYTHON_BIN": PYTHON_BIN,
    }
    cmd = ["bash", "scripts/last_vla_v2/highcap_no_risk/run_final_readiness_gate.sh"]
    code = run_command(cmd, LOG_ROOT / "readiness.log", env=env)
    report = OUT_ROOT / "readiness" / "final_readiness_report.md"
    if code != 0:
        raise RuntimeError(f"readiness gate failed, see {report}")
    if "Status: READY" not in report.read_text(encoding="utf-8"):
        raise RuntimeError(f"readiness report is not READY: {report}")
    return report


def run_training_launch(readiness_report: Path) -> dict[str, Any]:
    env = {
        "RUN_TRAIN": "1",
        "PROJECT_ROOT": str(REPO_ROOT),
        "OUT_ROOT": str(OUT_ROOT),
        "READINESS_REPORT": str(readiness_report),
        "FULL_HIGHCAP_TRAIN_CHUNK_ROOT": str(FULL_ROOT),
        "A0_INIT_CHECKPOINT": str(A0_INIT_CHECKPOINT),
        "REMOTE_HOST": os.environ.get("REMOTE_HOST", "training-vla-zt-peer"),
        "REMOTE_PROJECT_ROOT": os.environ.get("REMOTE_PROJECT_ROOT", str(REPO_ROOT)),
        "REMOTE_OUT_ROOT": os.environ.get("REMOTE_OUT_ROOT", str(OUT_ROOT) + "_remote"),
        "REMOTE_FULL_HIGHCAP_TRAIN_CHUNK_ROOT": os.environ.get("REMOTE_FULL_HIGHCAP_TRAIN_CHUNK_ROOT", str(FULL_ROOT)),
        "REMOTE_A0_INIT_CHECKPOINT": os.environ.get("REMOTE_A0_INIT_CHECKPOINT", str(A0_INIT_CHECKPOINT)),
        "REMOTE_VLM_PATH": os.environ.get("REMOTE_VLM_PATH", str(VLM_PATH)),
        "REMOTE_NAVSIM_LOG_PATH": os.environ.get("REMOTE_NAVSIM_LOG_PATH", "/mnt/navsim/trainval_navsim_logs/trainval"),
        "REMOTE_SENSOR_BLOBS_PATH": os.environ.get("REMOTE_SENSOR_BLOBS_PATH", "/mnt/navsim/trainval_sensor_blobs/trainval"),
        "MASTER_PORT_LOCAL": os.environ.get("MASTER_PORT_LOCAL", "29531"),
        "MASTER_PORT_REMOTE": os.environ.get("MASTER_PORT_REMOTE", "29541"),
        "LORA_PRESET": os.environ.get("LORA_PRESET", "attention_mlp"),
        "LORA_SCOPE": os.environ.get("LORA_SCOPE", "llm"),
        "LORA_R": os.environ.get("LORA_R", "32"),
        "LORA_ALPHA": os.environ.get("LORA_ALPHA", "64"),
        "LORA_DROPOUT": os.environ.get("LORA_DROPOUT", "0.05"),
        "LORA_USE_RSLORA": os.environ.get("LORA_USE_RSLORA", "true"),
        "LORA_USE_DORA": os.environ.get("LORA_USE_DORA", "false"),
        "LORA_TARGET_MODULES": os.environ.get("LORA_TARGET_MODULES", ""),
        "LORA_HIDDEN_ANCHOR_EVERY_N_STEPS": os.environ.get("LORA_HIDDEN_ANCHOR_EVERY_N_STEPS", "4"),
        "PYTHON_BIN": PYTHON_BIN,
    }
    cmd = ["bash", "scripts/last_vla_v2/highcap_no_risk/launch_local_remote_full_training.sh"]
    code = run_command(cmd, LOG_ROOT / "launch_training.log", env=env)
    report = OUT_ROOT / "reports" / "last_vla_v2_highcap_training_launch_report.json"
    if code != 0:
        raise RuntimeError(f"training launch failed, see {LOG_ROOT / 'launch_training.log'}")
    return read_json(report)


def write_report(payload: dict[str, Any]) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    report: dict[str, Any] = {
        "started_at": now(),
        "repo_root": str(REPO_ROOT),
        "base_chunk_root": str(BASE_CHUNK_ROOT),
        "cache_root": str(CACHE_ROOT),
        "out_root": str(OUT_ROOT),
        "full_highcap_train_chunk_root": str(FULL_ROOT),
        "training_launched": False,
        "full_eval_launched": False,
        "baseline_pdms": 0.864891,
    }
    try:
        validate_inputs()
        plan = chunk_plan()
        report["chunk_count"] = len(plan)
        report["chunk_records"] = sum(int(item["records"]) for item in plan)
        write_report(report)

        jepa_jobs = [
            (item["name"], jepa_command(item), LOG_ROOT / "jepa128" / f"{item['name']}.log")
            for item in plan
        ]
        report["jepa_results"] = run_parallel_jobs(jepa_jobs, "jepa128")
        report["jepa_completed_at"] = now()
        write_report(report)

        geometry_jobs = [
            (f"geometry_shard_{idx}_of_{GEOMETRY_SHARDS}", geometry_command(idx), LOG_ROOT / "geometry192" / f"shard_{idx}_of_{GEOMETRY_SHARDS}.log")
            for idx in range(GEOMETRY_SHARDS)
        ]
        report["geometry_results"] = run_parallel_jobs(geometry_jobs, "geometry192")
        report["geometry_completed_at"] = now()
        write_report(report)

        report["merge_summary"] = run_merge()
        report["merge_completed_at"] = now()
        write_report(report)

        report["audit_manifest"] = run_audit()
        report["audit_completed_at"] = now()
        write_report(report)

        readiness = run_readiness()
        report["readiness_report"] = str(readiness)
        report["readiness_completed_at"] = now()
        write_report(report)

        report["training_launch_report"] = run_training_launch(readiness)
        report["training_launched"] = True
        report["training_launch_completed_at"] = now()
        write_report(report)
        return 0
    except Exception as exc:
        report["failed_at"] = now()
        report["error"] = repr(exc)
        write_report(report)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
