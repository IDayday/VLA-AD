#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

REPO_ROOT = Path(os.environ.get("PROJECT_ROOT", "/mnt/project/VLA-AD_last_vla_dev"))
PYTHON_BIN = os.environ.get("PYTHON_BIN", "/root/miniconda3/envs/navsim/bin/python")
REMOTE_HOST = os.environ.get("REMOTE_HOST", "training-vla-zt-peer")
REMOTE_PROJECT_ROOT = Path(os.environ.get("REMOTE_PROJECT_ROOT", str(REPO_ROOT)))
BASE_CHUNK_ROOT = Path(os.environ.get("BASE_CHUNK_ROOT", "/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1"))
CACHE_ROOT = Path(os.environ.get("HIGHCAP_CACHE_ROOT", "/mnt/project/VLA-AD/cache/last_vla_v2/highcap_no_risk"))
OUT_ROOT = Path(os.environ.get("HIGHCAP_OUT_ROOT", "/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_full_sft"))
REMOTE_OUT_ROOT = Path(os.environ.get("REMOTE_OUT_ROOT", "/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_full_sft_remote"))
A0_INIT_CHECKPOINT = Path(os.environ.get("A0_INIT_CHECKPOINT", "/mnt/project/VLA-AD/outputs/a0_stage2_repro_20260531_003029/a0_official_aligned_a0complete/step_00100000.ckpt"))
VGGT_MODEL_PATH = Path(os.environ.get("VGGT_MODEL_PATH", "/mnt/project/VLA-AD/checkpoints/teachers/VGGT-1B"))
VJEPA_MODEL_PATH = Path(os.environ.get("VJEPA_MODEL_PATH", "/mnt/project/VLA-AD/checkpoints/teachers/vjepa2-vitl-fpc64-256"))
VLM_PATH = Path(os.environ.get("VLM_PATH", "/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B"))
NAVSIM_LOG_PATH = os.environ.get("NAVSIM_LOG_PATH", "/mnt/navsim/trainval_navsim_logs/trainval")
SENSOR_BLOBS_PATH = os.environ.get("SENSOR_BLOBS_PATH", "/mnt/navsim/trainval_sensor_blobs/trainval")

TRAIN_PATTERN = "train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*"
LOCAL_GPUS = [int(x) for x in os.environ.get("LOCAL_GPUS", "0,1,2,3,4,5,6,7").split(",") if x.strip()]
REMOTE_GPUS = [int(x) for x in os.environ.get("REMOTE_GPUS", "0,1,2,3,4,5,6,7").split(",") if x.strip()]
GEOMETRY_SHARDS = int(os.environ.get("HIGHCAP_GEOMETRY_SHARDS", str(len(LOCAL_GPUS) + len(REMOTE_GPUS))))

JEPA_ROOT = CACHE_ROOT / "train_jepa128_overlay"
GEOMETRY_ROOT = CACHE_ROOT / "train_geometry192_overlay"
FULL_ROOT = CACHE_ROOT / "train_full_highcap_chunks"
MANIFEST_DIR = CACHE_ROOT / "manifests"
LOG_ROOT = OUT_ROOT / "cache_generation_logs_distributed"
REPORT_JSON = OUT_ROOT / "reports" / "last_vla_v2_highcap_distributed_cache_supervisor_report.json"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_local(cmd: list[str], log_path: Path, gpu: int, cwd: Path = REPO_ROOT) -> int:
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[{now()}] host=local gpu={gpu} cwd={cwd}\n")
        f.write(f"[{now()}] command={shell_join(cmd)}\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"[{now()}] exit_code={proc.returncode}\n")
        return int(proc.returncode)


def run_remote(cmd: list[str], log_path: Path, gpu: int, cwd: Path = REMOTE_PROJECT_ROOT) -> int:
    inner = (
        f"cd {shlex.quote(str(cwd))} && "
        f"env CUDA_VISIBLE_DEVICES={shlex.quote(str(gpu))} "
        "PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false "
        f"{shell_join(cmd)}"
    )
    ssh_cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", REMOTE_HOST, inner]
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[{now()}] host={REMOTE_HOST} gpu={gpu} cwd={cwd}\n")
        f.write(f"[{now()}] command={inner}\n")
        f.flush()
        proc = subprocess.run(ssh_cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT)
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
            chunks.append(
                {
                    "name": name,
                    "chunk_index": int(match.group(1)) if match else len(chunks),
                    "records": line_count(chunk_dir / "index.jsonl"),
                    "chunk_start": int(metadata["chunk_start"]),
                    "chunk_stop": int(metadata.get("chunk_end", metadata["chunk_start"] + metadata.get("raw_window_size", 0))),
                }
            )
    if not chunks:
        raise RuntimeError(f"No base chunks under {BASE_CHUNK_ROOT}")
    return chunks


def validate_inputs() -> None:
    required = [REPO_ROOT, BASE_CHUNK_ROOT, A0_INIT_CHECKPOINT, VGGT_MODEL_PATH, VJEPA_MODEL_PATH, VLM_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required path(s): " + ", ".join(missing))
    for path in (JEPA_ROOT, GEOMETRY_ROOT, FULL_ROOT):
        if path.exists() and any(path.iterdir()):
            if os.environ.get("OVERWRITE_HIGHCAP_CACHE", "0") != "1":
                raise RuntimeError(f"Output dir is non-empty; set OVERWRITE_HIGHCAP_CACHE=1 to replace: {path}")
            backup = path.with_name(path.name + ".bak_" + time.strftime("%Y%m%d_%H%M%S"))
            path.rename(backup)
    for path in (JEPA_ROOT, GEOMETRY_ROOT, FULL_ROOT, MANIFEST_DIR, LOG_ROOT, REPORT_JSON.parent):
        path.mkdir(parents=True, exist_ok=True)


def jepa_command(item: dict[str, Any]) -> list[str]:
    return [
        PYTHON_BIN, "scripts/build_recogdrive_chunk_cache.py",
        "--navsim-root", "/mnt/navsim",
        "--split", "navtrain",
        "--chunk-index", str(item["chunk_index"]),
        "--chunk-start", str(item["chunk_start"]),
        "--chunk-stop", str(item["chunk_stop"]),
        "--chunk-size", str(item["records"]),
        "--output-dir", str(JEPA_ROOT / item["name"]),
        "--build-jepa",
        "--jepa-model-path", str(VJEPA_MODEL_PATH),
        "--num-jepa-tokens", "128",
        "--strict-highcap-jepa",
        "--precision", "bf16",
        "--device", "cuda",
        "--num-gpus", "1",
        "--workers", "1",
        "--allow-partial-final-chunk",
        "--log-every", "50",
    ]


def geometry_command(shard_index: int) -> list[str]:
    return [
        PYTHON_BIN, "scripts/build_last_vla_full_geometry_cache.py",
        "--chunk-cache-root", str(BASE_CHUNK_ROOT),
        "--chunk-name-pattern", TRAIN_PATTERN,
        "--output-cache-root", str(GEOMETRY_ROOT / f"full_geometry_overlay_shard_{shard_index}_of_{GEOMETRY_SHARDS}"),
        "--vggt-model-path", str(VGGT_MODEL_PATH),
        "--precision", "bf16",
        "--device", "cuda",
        "--num-geometry-tokens", "192",
        "--geometry-grid-rows", "12",
        "--geometry-grid-cols", "16",
        "--geometry-teacher-dim", "512",
        "--require-full-geometry",
        "--shard-index", str(shard_index),
        "--num-shards", str(GEOMETRY_SHARDS),
    ]


def slots() -> list[tuple[str, int]]:
    return [("local", gpu) for gpu in LOCAL_GPUS] + [("remote", gpu) for gpu in REMOTE_GPUS]


def run_job(host: str, gpu: int, cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if host == "local":
        return run_local(cmd, log_path, gpu)
    return run_remote(cmd, log_path, gpu)


def run_parallel(jobs: list[tuple[str, list[str], Path]], stage: str) -> list[dict[str, Any]]:
    slot_list = slots()
    if not slot_list:
        raise RuntimeError("No local/remote GPU slots configured.")
    results: list[dict[str, Any]] = []
    pending: set[Any] = set()
    next_job = iter(enumerate(jobs))
    available = list(slot_list)
    with ThreadPoolExecutor(max_workers=len(slot_list)) as executor:
        while True:
            while available:
                try:
                    _, (name, cmd, log_path) = next(next_job)
                except StopIteration:
                    break
                host, gpu = available.pop(0)
                future = executor.submit(run_job, host, gpu, cmd, log_path)
                future.job_info = {"stage": stage, "name": name, "host": host, "gpu": gpu, "log": str(log_path), "command": cmd}  # type: ignore[attr-defined]
                pending.add(future)
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                info = dict(future.job_info)  # type: ignore[attr-defined]
                available.append((info["host"], int(info["gpu"])))
                code = int(future.result())
                info["exit_code"] = code
                results.append(info)
                write_report({"last_stage": stage, "partial_results": results})
                if code != 0:
                    raise RuntimeError(f"{stage} job failed: {info['name']} log={info['log']}")
    return results


def run_single(cmd: list[str], log_name: str, env: dict[str, str] | None = None) -> int:
    log_path = LOG_ROOT / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    merged_env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[{now()}] command={shell_join(cmd)}\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=merged_env, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"[{now()}] exit_code={proc.returncode}\n")
        return int(proc.returncode)


def merge_cache() -> dict[str, Any]:
    cmd = [
        PYTHON_BIN, "scripts/merge_last_vla_geometry_cache_into_chunks.py",
        "--base-chunk-root", str(BASE_CHUNK_ROOT),
        "--geometry-cache-root", str(GEOMETRY_ROOT),
        "--geometry-chunk-name-pattern", "*",
        "--jepa-cache-root", str(JEPA_ROOT),
        "--jepa-chunk-name-pattern", "*",
        "--output-chunk-root", str(FULL_ROOT),
        "--chunk-name-pattern", TRAIN_PATTERN,
        "--copy-mode", "hardlink",
        "--strict-coverage",
        "--min-coverage", "0.99",
        "--strict-jepa-coverage",
        "--min-jepa-coverage", "0.99",
        "--expected-jepa-tokens", "128",
        "--jepa-dim", "1024",
        "--num-geometry-tokens", "192",
        "--geometry-grid-rows", "12",
        "--geometry-grid-cols", "16",
        "--geometry-teacher-dim", "512",
    ]
    if run_single(cmd, "merge.log") != 0:
        raise RuntimeError(f"merge failed: {LOG_ROOT / 'merge.log'}")
    return read_json(FULL_ROOT / "merge_summary.json")


def audit_cache() -> dict[str, Any]:
    output = MANIFEST_DIR / "train_manifest.json"
    cmd = [
        PYTHON_BIN, "scripts/audit_last_vla_cache_manifest.py",
        "--cache-root", str(FULL_ROOT),
        "--chunk-name-pattern", TRAIN_PATTERN,
        "--strict-full-geometry",
        "--min-full-geometry-coverage", "0.99",
        "--expected-jepa-tokens", "128",
        "--expected-geometry-tokens", "192",
        "--geometry-teacher-dim", "512",
        "--strict-no-risk",
        "--output", str(output),
    ]
    if run_single(cmd, "audit.log") != 0:
        raise RuntimeError(f"audit failed: {LOG_ROOT / 'audit.log'}")
    return read_json(output)


def readiness() -> Path:
    report = OUT_ROOT / "readiness" / "final_readiness_report.md"
    env = {
        "FULL_HIGHCAP_TRAIN_CHUNK_ROOT": str(FULL_ROOT),
        "A0_INIT_CHECKPOINT": str(A0_INIT_CHECKPOINT),
        "OUT_ROOT": str(OUT_ROOT),
        "PYTHON_BIN": PYTHON_BIN,
    }
    if run_single(["bash", "scripts/last_vla_v2/highcap_no_risk/run_final_readiness_gate.sh"], "readiness.log", env=env) != 0:
        raise RuntimeError(f"readiness failed: {report}")
    if "Status: READY" not in report.read_text(encoding="utf-8"):
        raise RuntimeError(f"readiness report is not READY: {report}")
    return report


def launch_training(readiness_report: Path) -> dict[str, Any]:
    env = {
        "RUN_TRAIN": "1",
        "PROJECT_ROOT": str(REPO_ROOT),
        "OUT_ROOT": str(OUT_ROOT),
        "READINESS_REPORT": str(readiness_report),
        "FULL_HIGHCAP_TRAIN_CHUNK_ROOT": str(FULL_ROOT),
        "A0_INIT_CHECKPOINT": str(A0_INIT_CHECKPOINT),
        "REMOTE_HOST": REMOTE_HOST,
        "REMOTE_PROJECT_ROOT": str(REMOTE_PROJECT_ROOT),
        "REMOTE_OUT_ROOT": str(REMOTE_OUT_ROOT),
        "REMOTE_FULL_HIGHCAP_TRAIN_CHUNK_ROOT": str(FULL_ROOT),
        "REMOTE_A0_INIT_CHECKPOINT": str(A0_INIT_CHECKPOINT),
        "REMOTE_VLM_PATH": str(VLM_PATH),
        "REMOTE_NAVSIM_LOG_PATH": NAVSIM_LOG_PATH,
        "REMOTE_SENSOR_BLOBS_PATH": SENSOR_BLOBS_PATH,
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
    if run_single(["bash", "scripts/last_vla_v2/highcap_no_risk/launch_local_remote_full_training.sh"], "launch_training.log", env=env) != 0:
        raise RuntimeError(f"training launch failed: {LOG_ROOT / 'launch_training.log'}")
    return read_json(OUT_ROOT / "reports" / "last_vla_v2_highcap_training_launch_report.json")


REPORT: dict[str, Any] = {}


def write_report(extra: dict[str, Any] | None = None) -> None:
    if extra:
        REPORT.update(extra)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(REPORT, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    REPORT.update(
        {
            "started_at": now(),
            "repo_root": str(REPO_ROOT),
            "remote_host": REMOTE_HOST,
            "remote_project_root": str(REMOTE_PROJECT_ROOT),
            "slots": [{"host": h, "gpu": g} for h, g in slots()],
            "cache_root": str(CACHE_ROOT),
            "full_highcap_train_chunk_root": str(FULL_ROOT),
            "out_root": str(OUT_ROOT),
            "remote_out_root": str(REMOTE_OUT_ROOT),
            "training_launched": False,
            "full_eval_launched": False,
            "baseline_pdms": 0.864891,
        }
    )
    try:
        validate_inputs()
        plan = chunk_plan()
        write_report({"chunk_count": len(plan), "chunk_records": sum(int(x["records"]) for x in plan)})

        jepa_jobs = [(item["name"], jepa_command(item), LOG_ROOT / "jepa128" / f"{item['name']}.log") for item in plan]
        write_report({"jepa_started_at": now()})
        write_report({"jepa_results": run_parallel(jepa_jobs, "jepa128"), "jepa_completed_at": now()})

        geometry_jobs = [
            (f"geometry_shard_{idx}_of_{GEOMETRY_SHARDS}", geometry_command(idx), LOG_ROOT / "geometry192" / f"shard_{idx}_of_{GEOMETRY_SHARDS}.log")
            for idx in range(GEOMETRY_SHARDS)
        ]
        write_report({"geometry_started_at": now()})
        write_report({"geometry_results": run_parallel(geometry_jobs, "geometry192"), "geometry_completed_at": now()})

        write_report({"merge_summary": merge_cache(), "merge_completed_at": now()})
        write_report({"audit_manifest": audit_cache(), "audit_completed_at": now()})
        ready_report = readiness()
        write_report({"readiness_report": str(ready_report), "readiness_completed_at": now()})
        write_report({"training_launch_report": launch_training(ready_report), "training_launched": True, "training_launch_completed_at": now()})
        return 0
    except Exception as exc:
        write_report({"failed_at": now(), "error": repr(exc)})
        print(f"ERROR: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
