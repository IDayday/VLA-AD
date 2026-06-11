#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path("/mnt/project/VLA-AD_last_vla_dev")
PYTHON_BIN = Path("/root/miniconda3/envs/navsim/bin/python")
LAUNCHER = Path("/mnt/project/skill/stable-gpu-job-launch/scripts/stable_gpu_launcher.py")
TRAIN_CHUNK_ROOT = Path("/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/train_full_highcap_chunks")
TRAIN_METRIC_CACHE = Path("/mnt/project/VLA-AD/cache/metric_cache_train_full")
TRAIN_ANCHOR_CACHE = Path("/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/vlm_text_anchor_2bbase_train_20260609T052233Z")

NO_RESIDUAL_CONFIG = "configs/last_vla_v2/decoupled_highcap_no_risk/progressive_sft_eval_flat.yaml"
PURE_RESIDUAL_CONFIG = "configs/last_vla_v2/decoupled_highcap_no_risk/original_recogdrive_residual_anchor_eval_flat.yaml"
LASTVLA_RESIDUAL_CONFIG = "configs/last_vla_v2/decoupled_highcap_no_risk/progressive_sft_vlm_text_residual_eval_flat.yaml"

NO_RESIDUAL_ROOT = Path("/mnt/project/VLA-AD/outputs/last_vla_v2/no_residual_stage2_ab_top5val_20260609T021059Z/A_frozen_vlm_stage2_progressive")
PURE_RESIDUAL_ROOT = Path("/mnt/project/VLA-AD/outputs/recogdrive_stage2_residual_anchor_base2b_20260610T034055Z_setsid_freezecot")
LASTVLA_RESIDUAL_ROOT = Path("/mnt/project/VLA-AD/outputs/last_vla_v2/vlm_text_residual_anchor_stage2_A_remote_20260610T030302Z/A_frozen_vlm_stage2_progressive")

METRIC_KEYS = ("PDMS", "NC", "DAC", "TTC", "comfort", "EP", "DDC")
HOST_ALIASES = {
    "zt2": "local",
    "local": "local",
    "zt3": "training-vla-zt3",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_hash(text: str, seed: int) -> int:
    digest = hashlib.blake2b(f"{seed}:{text}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def safe_name(text: str) -> str:
    keep = []
    for char in text:
        if char.isalnum() or char in ("-", "_", "."):
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("._") or "item"


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def command_string(command: Sequence[Any]) -> str:
    return shlex.join(str(part) for part in command)


def default_probe_specs() -> List[Dict[str, Any]]:
    probes: List[Dict[str, Any]] = [
        {
            "name": "noresid_A_step_00050000",
            "family": "no_residual_A",
            "config": NO_RESIDUAL_CONFIG,
            "checkpoint": NO_RESIDUAL_ROOT / "step_00050000.ckpt",
            "navtest_pdms": 0.841452586394648,
            "navtest_l1": 0.2943435068860731,
        },
        {
            "name": "noresid_A_step_00060000",
            "family": "no_residual_A",
            "config": NO_RESIDUAL_CONFIG,
            "checkpoint": NO_RESIDUAL_ROOT / "step_00060000.ckpt",
            "navtest_pdms": 0.8478216694334099,
            "navtest_l1": 0.3678235472752729,
        },
        {
            "name": "noresid_A_step_00080000",
            "family": "no_residual_A",
            "config": NO_RESIDUAL_CONFIG,
            "checkpoint": NO_RESIDUAL_ROOT / "step_00080000.ckpt",
            "navtest_pdms": 0.8569099693101081,
            "navtest_l1": 0.2740614752036901,
        },
    ]
    pure_scores = {
        "00050000": (0.8427448503741305, 0.2952873244967202),
        "00060000": (0.8481858313519718, 0.3065953213689616),
        "00080000": (0.8428058004518755, 0.28232862890892624),
        "00100000": (0.8459827106773717, 0.28400464088916866),
        "00120000": (0.8472651883047604, 0.28193270540005677),
    }
    for step, (pdms, l1) in pure_scores.items():
        probes.append(
            {
                "name": f"pure_residual_step_{step}",
                "family": "pure_recogdrive_residual",
                "config": PURE_RESIDUAL_CONFIG,
                "checkpoint": PURE_RESIDUAL_ROOT / f"step_{step}.ckpt",
                "navtest_pdms": pdms,
                "navtest_l1": l1,
            }
        )
    lastvla_scores = {
        "00050000": (0.8341622539165908, 0.3044469989393021),
        "00060000": (0.8414276359216591, 0.31117710609034394),
        "00080000": (0.8432956925432473, 0.27802522500868654),
        "00100000": (0.8444044151001739, 0.28304506976632215),
        "00120000": (0.8450680938224321, 0.28005417320513404),
    }
    for step, (pdms, l1) in lastvla_scores.items():
        probes.append(
            {
                "name": f"lastvla_residual_step_{step}",
                "family": "lastvla_A_residual",
                "config": LASTVLA_RESIDUAL_CONFIG,
                "checkpoint": LASTVLA_RESIDUAL_ROOT / f"step_{step}.ckpt",
                "navtest_pdms": pdms,
                "navtest_l1": l1,
            }
        )
    return probes


def scan_chunk_records(chunk_root: Path, pattern: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    seen_tokens: set[str] = set()
    for chunk_dir in sorted(path for path in chunk_root.glob(pattern) if path.is_dir()):
        index = chunk_dir / "index.jsonl"
        if not index.is_file():
            continue
        with index.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                item = json.loads(raw)
                token = str(item.get("sample_token") or Path(str(item["path"])).stem)
                if token in seen_tokens:
                    raise RuntimeError(f"Duplicate sample_token in chunk cache: {token}")
                seen_tokens.add(token)
                records.append(
                    {
                        "sample_token": token,
                        "scene_token": str(item.get("scene_token", "")),
                        "log_name": str(item.get("log_name", "")),
                        "chunk": chunk_dir.name,
                        "path": str(item.get("path", "")),
                    }
                )
    if not records:
        raise RuntimeError(f"No records found under {chunk_root} with pattern {pattern!r}")
    return records


def choose_candidate_records(records: List[Dict[str, str]], size: int, seed: int) -> List[Dict[str, str]]:
    if size <= 0 or size >= len(records):
        return sorted(records, key=lambda row: row["sample_token"])
    by_log: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in records:
        by_log[row["log_name"]].append(row)
    selected: List[Dict[str, str]] = []
    quotas: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []
    for log_name, items in by_log.items():
        exact = size * len(items) / len(records)
        quota = int(math.floor(exact))
        quotas[log_name] = quota
        remainders.append((exact - quota, log_name))
    remaining = size - sum(quotas.values())
    for _, log_name in sorted(remainders, reverse=True)[:remaining]:
        quotas[log_name] += 1
    for log_name, items in by_log.items():
        ordered = sorted(items, key=lambda row: stable_hash(row["sample_token"], seed))
        selected.extend(ordered[: quotas[log_name]])
    return sorted(selected, key=lambda row: row["sample_token"])


def write_tokens(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row["sample_token"] + "\n")


def prepare(args: argparse.Namespace) -> int:
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    records = scan_chunk_records(args.chunk_root, args.chunk_pattern)
    candidates = choose_candidate_records(records, args.candidate_size, args.seed)
    write_csv(
        out_root / "candidate_manifest.csv",
        ["sample_token", "scene_token", "log_name", "chunk", "path"],
        candidates,
    )
    write_tokens(out_root / "candidate_tokens.txt", candidates)

    hosts = [item.strip() for item in args.hosts.split(",") if item.strip()]
    if not hosts:
        raise ValueError("--hosts must contain at least one host key")
    probes = []
    for idx, probe in enumerate(default_probe_specs()):
        row = dict(probe)
        row["checkpoint"] = str(row["checkpoint"])
        row["host_key"] = hosts[idx % len(hosts)]
        row["enabled"] = "1"
        probes.append(row)
    for probe in probes:
        checkpoint = Path(str(probe["checkpoint"]))
        config = PROJECT_ROOT / str(probe["config"])
        if not checkpoint.is_file():
            raise FileNotFoundError(str(checkpoint))
        if not config.is_file():
            raise FileNotFoundError(str(config))
    write_csv(
        out_root / "probe_manifest.csv",
        ["name", "family", "config", "checkpoint", "navtest_pdms", "navtest_l1", "host_key", "enabled"],
        probes,
    )
    summary = {
        "created_at": utc_now(),
        "chunk_root": str(args.chunk_root),
        "chunk_pattern": args.chunk_pattern,
        "num_train_records": len(records),
        "candidate_size": len(candidates),
        "candidate_token_file": str(out_root / "candidate_tokens.txt"),
        "probe_manifest": str(out_root / "probe_manifest.csv"),
        "hosts": hosts,
        "selection_size": args.selection_size,
        "strategy": (
            "Evaluate probe checkpoints on the train candidate pool, then select tokens whose "
            "per-sample PDMS response patterns preserve navtest checkpoint ranking while keeping "
            "coverage across logs, chunks, and difficulty bins."
        ),
    }
    write_json(out_root / "prepare_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def metric_mean(items: List[Dict[str, Any]], key: str, weight_key: str) -> Optional[float]:
    total = 0.0
    weight = 0
    for item in items:
        value = item.get(key)
        w = int(item.get(weight_key) or 0)
        if value is None or w <= 0:
            continue
        total += float(value) * w
        weight += w
    return total / weight if weight else None


def aggregate_probe(run_dir: Path, probe: Dict[str, str], num_shards: int) -> Dict[str, Any]:
    shard_metrics: List[Dict[str, Any]] = []
    missing = []
    for shard in range(num_shards):
        metrics_path = run_dir / f"shard_{shard:02d}" / "metrics.json"
        if metrics_path.is_file():
            shard_metrics.append(json.loads(metrics_path.read_text(encoding="utf-8")))
        else:
            missing.append(str(metrics_path))
    if missing:
        raise FileNotFoundError(f"Missing shard metrics for {probe['name']}: {missing[:3]} ({len(missing)} total)")
    payload: Dict[str, Any] = {
        "probe": probe["name"],
        "family": probe["family"],
        "config": probe["config"],
        "checkpoint": probe["checkpoint"],
        "navtest_pdms": float(probe["navtest_pdms"]),
        "navtest_l1": float(probe["navtest_l1"]),
        "num_shards": num_shards,
        "num_samples": sum(int(item.get("num_samples") or 0) for item in shard_metrics),
        "num_pdm_valid": sum(int(item.get("num_pdm_valid") or 0) for item in shard_metrics),
        "num_pdm_missing_metric_cache": sum(int(item.get("num_pdm_missing_metric_cache") or 0) for item in shard_metrics),
        "num_pdm_failed": sum(int(item.get("num_pdm_failed") or 0) for item in shard_metrics),
        "trajectory_l1": metric_mean(shard_metrics, "trajectory_l1", "num_samples"),
        "shards": [str(run_dir / f"shard_{idx:02d}") for idx in range(num_shards)],
        "completed_at": utc_now(),
    }
    for key in METRIC_KEYS:
        payload[key] = metric_mean(shard_metrics, key, "num_pdm_valid")
    payload["pdm_score"] = payload["PDMS"]
    write_json(run_dir / "metrics.json", payload)
    return payload


def probe_done(run_dir: Path, num_shards: int) -> bool:
    if not (run_dir / "metrics.json").is_file():
        return False
    return all((run_dir / f"shard_{idx:02d}" / "metrics.json").is_file() for idx in range(num_shards))


def write_probe_summary(out_root: Path) -> None:
    rows = []
    for metrics_path in sorted((out_root / "eval").glob("*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({**payload, "metrics_path": str(metrics_path)})
    write_csv(
        out_root / "summary" / "probe_summary.csv",
        [
            "probe",
            "family",
            "checkpoint",
            "navtest_pdms",
            "PDMS",
            "trajectory_l1",
            "num_samples",
            "num_pdm_valid",
            "num_pdm_missing_metric_cache",
            "num_pdm_failed",
            "completed_at",
            "metrics_path",
        ],
        rows,
    )


def build_probe_jobs(args: argparse.Namespace, probe: Dict[str, str], run_dir: Path, checkpoint_snapshot: Path) -> Path:
    jobs_path = run_dir / "jobs" / "jobs.tsv"
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    gpus = [item.strip() for item in args.gpus_csv.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus-csv must not be empty")
    config_path = PROJECT_ROOT / probe["config"]
    with jobs_path.open("w", encoding="utf-8") as f:
        for shard in range(args.num_shards):
            shard_dir = run_dir / f"shard_{shard:02d}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                str(args.python_bin),
                "scripts/eval_recogdrive_expert_pdm.py",
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint_snapshot),
                "--split",
                "navtrain_pdms_probe",
                "--chunk-cache-root",
                str(args.chunk_root),
                "--chunk-name-pattern",
                args.chunk_pattern,
                "--metric-cache-dir",
                str(args.metric_cache_dir),
                "--sample-token-file",
                str(args.sample_token_file),
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
            f.write(f"{probe['name']}_shard_{shard:02d}\t{gpus[shard % len(gpus)]}\t{command_string(cmd)}\n")
    return jobs_path


def run_host(args: argparse.Namespace) -> int:
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    host_log = out_root / "logs" / f"run_host_{args.host_key}.log"
    host_log.parent.mkdir(parents=True, exist_ok=True)
    probe_rows = [
        row
        for row in read_csv(args.probe_manifest)
        if row.get("enabled", "1") not in {"0", "false", "False"} and row.get("host_key") == args.host_key
    ]
    if not probe_rows:
        print(f"No probes assigned to host_key={args.host_key}")
        return 0
    state_path = out_root / f"state_{args.host_key}.json"
    write_json(state_path, {"host_key": args.host_key, "started_at": utc_now(), "probes": [row["name"] for row in probe_rows]})
    failures = 0
    for probe in probe_rows:
        run_dir = out_root / "eval" / safe_name(probe["name"])
        if probe_done(run_dir, args.num_shards):
            with host_log.open("a", encoding="utf-8") as f:
                f.write(f"[{utc_now()}] skip completed {probe['name']}\n")
            continue
        checkpoint = Path(probe["checkpoint"])
        snapshot = out_root / "checkpoint_snapshots" / safe_name(probe["name"]) / checkpoint.name
        hardlink_or_copy(checkpoint, snapshot)
        jobs_tsv = build_probe_jobs(args, probe, run_dir, snapshot)
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
            str(args.poll_seconds),
        ]
        with (out_root / "commands.log").open("a", encoding="utf-8") as f:
            f.write(f"[{utc_now()}] host={args.host_key} probe={probe['name']} {command_string(command)}\n")
        with host_log.open("a", encoding="utf-8") as f:
            f.write(f"[{utc_now()}] start probe={probe['name']} checkpoint={checkpoint}\n")
        proc = subprocess.run(command, cwd=args.project_root)
        if proc.returncode != 0:
            failures += 1
            with host_log.open("a", encoding="utf-8") as f:
                f.write(f"[{utc_now()}] failed launcher probe={probe['name']} returncode={proc.returncode}\n")
            continue
        try:
            metrics = aggregate_probe(run_dir, probe, args.num_shards)
        except Exception as exc:
            failures += 1
            with host_log.open("a", encoding="utf-8") as f:
                f.write(f"[{utc_now()}] failed aggregate probe={probe['name']} error={exc!r}\n")
            continue
        with host_log.open("a", encoding="utf-8") as f:
            f.write(f"[{utc_now()}] done probe={probe['name']} PDMS={metrics.get('PDMS')}\n")
        write_probe_summary(out_root)
    write_json(state_path, {"host_key": args.host_key, "finished_at": utc_now(), "failures": failures})
    return 1 if failures else 0


def pids_for_gpu_stress() -> List[int]:
    try:
        out = subprocess.check_output(["ps", "-eo", "pid=,ppid=,cmd="], text=True)
    except Exception:
        return []
    pids = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid = int(parts[0])
        cmd = parts[2]
        if "/mnt/project/gpu_stress.py" in cmd or cmd.endswith(" gpu_stress.py") or "gpu_stress.py" in cmd:
            pids.append(pid)
    return pids


def high_memory_stress_orphans() -> List[int]:
    try:
        gpu_out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True,
        )
        ps_out = subprocess.check_output(["ps", "-eo", "pid=,cmd="], text=True)
    except Exception:
        return []
    cmds: Dict[int, str] = {}
    for line in ps_out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            cmds[int(parts[0])] = parts[1]
    pids: List[int] = []
    for line in gpu_out.splitlines():
        if not line.strip():
            continue
        pid_raw, mem_raw = [part.strip() for part in line.split(",", 1)]
        pid = int(pid_raw)
        mem = int(mem_raw)
        cmd = cmds.get(pid, "")
        if mem > 50000 and "spawn_main" in cmd and "eval_recogdrive_expert_pdm.py" not in cmd:
            pids.append(pid)
    return pids


def kill_gpu_stress_local() -> List[int]:
    pids = sorted(set(pids_for_gpu_stress()) | set(high_memory_stress_orphans()))
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if pids:
        time.sleep(5)
    remaining = set(pids_for_gpu_stress()) | set(high_memory_stress_orphans())
    for pid in sorted(remaining):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return pids


def remote_shell(host: str, shell: str) -> List[str]:
    return ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=no", host, shell]


def launch(args: argparse.Namespace) -> int:
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.probe_manifest)
    host_keys = sorted({row["host_key"] for row in rows if row.get("enabled", "1") not in {"0", "false", "False"}})
    launch_rows = []
    for host_key in host_keys:
        host = HOST_ALIASES.get(host_key, host_key)
        if args.kill_gpu_stress:
            if host == "local":
                killed = kill_gpu_stress_local()
                launch_rows.append({"host_key": host_key, "host": host, "killed_gpu_stress_pids": killed})
            else:
                kill_cmd = (
                    "/root/miniconda3/envs/navsim/bin/python - <<'PY'\n"
                    "import os, signal, subprocess, time\n"
                    "targets=set()\n"
                    "try:\n"
                    "    ps=subprocess.check_output(['ps','-eo','pid=,cmd='], text=True)\n"
                    "    cmds={}\n"
                    "    for line in ps.splitlines():\n"
                    "        parts=line.strip().split(None,1)\n"
                    "        if len(parts)==2:\n"
                    "            pid=int(parts[0]); cmds[pid]=parts[1]\n"
                    "            if 'gpu_stress.py' in parts[1]: targets.add(pid)\n"
                    "    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'], text=True)\n"
                    "    for line in gpu.splitlines():\n"
                    "        if not line.strip(): continue\n"
                    "        pid_s, mem_s=[x.strip() for x in line.split(',',1)]\n"
                    "        pid=int(pid_s); mem=int(mem_s); cmd=cmds.get(pid,'')\n"
                    "        if mem > 50000 and 'spawn_main' in cmd and 'eval_recogdrive_expert_pdm.py' not in cmd: targets.add(pid)\n"
                    "except Exception as exc:\n"
                    "    print('gpu_stress_scan_error', repr(exc))\n"
                    "print('killed_gpu_stress', sorted(targets))\n"
                    "for pid in sorted(targets):\n"
                    "    try: os.kill(pid, signal.SIGTERM)\n"
                    "    except ProcessLookupError: pass\n"
                    "time.sleep(5)\n"
                    "for pid in sorted(targets):\n"
                    "    try: os.kill(pid, 0)\n"
                    "    except ProcessLookupError: continue\n"
                    "    try: os.kill(pid, signal.SIGKILL)\n"
                    "    except ProcessLookupError: pass\n"
                    "PY"
                )
                subprocess.run(remote_shell(host, kill_cmd), check=False)
        log_path = out_root / "logs" / f"launcher_{host_key}.outer.log"
        pid_path = out_root / f"launcher_{host_key}.pid"
        cmd = [
            str(args.python_bin),
            str(Path(__file__).resolve()),
            "run-host",
            "--out-root",
            str(out_root),
            "--host-key",
            host_key,
            "--probe-manifest",
            str(args.probe_manifest),
            "--sample-token-file",
            str(args.sample_token_file),
            "--project-root",
            str(args.project_root),
            "--python-bin",
            str(args.python_bin),
            "--launcher",
            str(args.launcher),
            "--chunk-root",
            str(args.chunk_root),
            "--chunk-pattern",
            args.chunk_pattern,
            "--metric-cache-dir",
            str(args.metric_cache_dir),
            "--vlm-text-anchor-cache-root",
            str(args.vlm_text_anchor_cache_root),
            "--num-shards",
            str(args.num_shards),
            "--gpus-csv",
            args.gpus_csv,
            "--precision",
            args.precision,
            "--trajectory-output-key",
            args.trajectory_output_key,
            "--stagger-seconds",
            str(args.stagger_seconds),
            "--poll-seconds",
            str(args.poll_seconds),
        ]
        shell = (
            f"cd {shlex.quote(str(args.project_root))} && "
            f"setsid {command_string(cmd)} > {shlex.quote(str(log_path))} 2>&1 < /dev/null & "
            f"echo $! > {shlex.quote(str(pid_path))}; exit 0"
        )
        if host == "local":
            subprocess.run(["bash", "-lc", shell], check=True)
        else:
            subprocess.run(remote_shell(host, shell), check=True)
        launch_rows.append({"host_key": host_key, "host": host, "pid_file": str(pid_path), "log": str(log_path)})
    write_json(out_root / "launch_summary.json", {"launched_at": utc_now(), "launches": launch_rows})
    print(json.dumps({"out_root": str(out_root), "launches": launch_rows}, indent=2, sort_keys=True))
    return 0


def status(args: argparse.Namespace) -> int:
    rows = read_csv(args.probe_manifest)
    payload_rows = []
    for row in rows:
        run_dir = args.out_root / "eval" / safe_name(row["name"])
        metrics_path = run_dir / "metrics.json"
        item = {
            "probe": row["name"],
            "family": row["family"],
            "host_key": row.get("host_key"),
            "navtest_pdms": row.get("navtest_pdms"),
            "state": "done" if probe_done(run_dir, args.num_shards) else "pending",
            "metrics_path": str(metrics_path),
        }
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            item.update(
                {
                    "PDMS": metrics.get("PDMS"),
                    "num_samples": metrics.get("num_samples"),
                    "num_pdm_valid": metrics.get("num_pdm_valid"),
                    "num_pdm_failed": metrics.get("num_pdm_failed"),
                }
            )
        payload_rows.append(item)
    counts = Counter(row["state"] for row in payload_rows)
    write_csv(
        args.out_root / "summary" / "status.csv",
        ["probe", "family", "host_key", "state", "navtest_pdms", "PDMS", "num_samples", "num_pdm_valid", "num_pdm_failed", "metrics_path"],
        payload_rows,
    )
    print(json.dumps({"counts": dict(counts), "status_csv": str(args.out_root / "summary" / "status.csv")}, indent=2, sort_keys=True))
    return 0


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 0 or vb <= 0:
        return 0.0
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return cov / math.sqrt(va * vb)


def rankdata(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(order):
        end = idx + 1
        while end < len(order) and values[order[end]] == values[order[idx]]:
            end += 1
        rank = (idx + end - 1) / 2.0
        for pos in range(idx, end):
            ranks[order[pos]] = rank
        idx = end
    return ranks


def load_probe_sample_scores(out_root: Path, probes: List[Dict[str, str]]) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    by_probe: Dict[str, Dict[str, float]] = {}
    token_sets = []
    for probe in probes:
        probe_name = probe["name"]
        run_dir = out_root / "eval" / safe_name(probe_name)
        scores: Dict[str, float] = {}
        for csv_path in sorted(run_dir.glob("shard_*/pdm_results.csv")):
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    token = row.get("sample_token", "")
                    if not token or token == "average":
                        continue
                    if str(row.get("valid")).lower() not in {"true", "1"}:
                        continue
                    value = row.get("score")
                    if value in {None, ""}:
                        continue
                    scores[token] = float(value)
        if not scores:
            raise RuntimeError(f"No per-sample scores found for probe {probe_name} under {run_dir}")
        by_probe[probe_name] = scores
        token_sets.append(set(scores))
    complete = sorted(set.intersection(*token_sets))
    if not complete:
        raise RuntimeError("No tokens have complete scores across all probes")
    return complete, by_probe


def load_candidate_meta(path: Path) -> Dict[str, Dict[str, str]]:
    return {row["sample_token"]: row for row in read_csv(path)}


def select(args: argparse.Namespace) -> int:
    probes = [row for row in read_csv(args.probe_manifest) if row.get("enabled", "1") not in {"0", "false", "False"}]
    incomplete = [row["name"] for row in probes if not probe_done(args.out_root / "eval" / safe_name(row["name"]), args.num_shards)]
    if incomplete and not args.allow_incomplete:
        raise RuntimeError(f"Cannot select until all probes are complete. Incomplete: {incomplete}")
    if incomplete:
        probes = [row for row in probes if row["name"] not in set(incomplete)]
    tokens, by_probe = load_probe_sample_scores(args.out_root, probes)
    meta = load_candidate_meta(args.candidate_manifest)
    target = [float(row["navtest_pdms"]) for row in probes]
    target_rank = rankdata(target)
    families = [row["family"] for row in probes]
    family_names = sorted(set(families))
    family_indices = {family: [idx for idx, value in enumerate(families) if value == family] for family in family_names}
    vectors: Dict[str, List[float]] = {token: [by_probe[row["name"]][token] for row in probes] for token in tokens}
    centered_target = [x - sum(target) / len(target) for x in target]
    rows = []
    for token in tokens:
        values = vectors[token]
        mean = sum(values) / len(values)
        std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
        influence = max(0.0, pearson(values, target)) * std
        m = meta.get(token, {})
        rows.append(
            {
                "sample_token": token,
                "mean_pdms": mean,
                "std_pdms": std,
                "influence": influence,
                "difficulty_bin": None,
                "log_name": m.get("log_name", ""),
                "chunk": m.get("chunk", ""),
                "scene_token": m.get("scene_token", ""),
            }
        )
    rows.sort(key=lambda row: row["mean_pdms"])
    bins = max(args.difficulty_bins, 1)
    for idx, row in enumerate(rows):
        row["difficulty_bin"] = min(bins - 1, int(idx * bins / len(rows)))
    by_bin: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bin[int(row["difficulty_bin"])].append(row)

    selected_tokens: List[str] = []
    quota_base = args.selection_size // bins
    remainder = args.selection_size % bins
    for bin_idx in range(bins):
        quota = quota_base + (1 if bin_idx < remainder else 0)
        candidates = sorted(
            by_bin[bin_idx],
            key=lambda row: (row["influence"], row["std_pdms"], stable_hash(row["sample_token"], args.seed)),
            reverse=True,
        )
        selected_tokens.extend(row["sample_token"] for row in candidates[:quota])
    selected = set(selected_tokens)

    rng = random.Random(args.seed)
    all_tokens = [row["sample_token"] for row in rows]

    def subset_mean(token_list: Sequence[str]) -> List[float]:
        sums = [0.0] * len(probes)
        for token in token_list:
            values = vectors[token]
            for idx, value in enumerate(values):
                sums[idx] += value
        return [value / len(token_list) for value in sums]

    def indexed(values: Sequence[float], indices: Sequence[int]) -> List[float]:
        return [values[idx] for idx in indices]

    def valid_group_corr(means: Sequence[float], indices: Sequence[int], *, rank: bool = False) -> Optional[float]:
        if len(indices) < 3:
            return None
        left = indexed(means, indices)
        right = indexed(target, indices)
        if max(left) - min(left) <= 1e-12 or max(right) - min(right) <= 1e-12:
            return None
        if rank:
            left = rankdata(left)
            right = rankdata(right)
        return pearson(left, right)

    def family_corrs(means: Sequence[float]) -> Dict[str, Dict[str, Optional[float]]]:
        payload: Dict[str, Dict[str, Optional[float]]] = {}
        for family, indices in family_indices.items():
            payload[family] = {
                "pearson": valid_group_corr(means, indices),
                "spearman": valid_group_corr(means, indices, rank=True),
                "num_probes": float(len(indices)),
            }
        return payload

    def leave_one_family_out_corrs(means: Sequence[float]) -> Dict[str, Dict[str, Optional[float]]]:
        payload: Dict[str, Dict[str, Optional[float]]] = {}
        all_indices = list(range(len(probes)))
        for family, drop_indices in family_indices.items():
            keep = [idx for idx in all_indices if idx not in set(drop_indices)]
            payload[family] = {
                "pearson": valid_group_corr(means, keep),
                "spearman": valid_group_corr(means, keep, rank=True),
                "num_probes": float(len(keep)),
            }
        return payload

    def objective(token_list: Sequence[str]) -> float:
        means = subset_mean(token_list)
        corr = pearson(means, target)
        rank_corr = pearson(rankdata(means), target_rank)
        fam = family_corrs(means)
        fam_values = [item["pearson"] for item in fam.values() if item["pearson"] is not None]
        lofo = leave_one_family_out_corrs(means)
        lofo_values = [item["pearson"] for item in lofo.values() if item["pearson"] is not None]
        family_term = sum(fam_values) / len(fam_values) if fam_values else 0.0
        lofo_term = min(lofo_values) if lofo_values else 0.0
        mean_center = [x - sum(means) / len(means) for x in means]
        target_center = centered_target
        scale = math.sqrt(sum(x * x for x in target_center) / len(target_center)) or 1.0
        rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(mean_center, target_center)) / len(target_center)) / scale
        return corr + 0.5 * rank_corr + 0.25 * family_term + 0.25 * lofo_term - 0.15 * rmse

    selected_list = sorted(selected)
    best_score = objective(selected_list)
    if args.swap_iters > 0:
        selected_set = set(selected_list)
        outside = [token for token in all_tokens if token not in selected_set]
        selected_list = list(selected_set)
        for _ in range(args.swap_iters):
            if not outside or not selected_list:
                break
            remove_idx = rng.randrange(len(selected_list))
            add_idx = rng.randrange(len(outside))
            removed = selected_list[remove_idx]
            added = outside[add_idx]
            trial = selected_list.copy()
            trial[remove_idx] = added
            score = objective(trial)
            if score > best_score:
                selected_list = trial
                selected_set.remove(removed)
                selected_set.add(added)
                outside[add_idx] = removed
                best_score = score

    selected_list = sorted(selected_list)
    means = subset_mean(selected_list)
    final_corr = pearson(means, target)
    final_rank_corr = pearson(rankdata(means), target_rank)
    full_means = subset_mean(tokens)
    final_family_corrs = family_corrs(means)
    final_lofo_corrs = leave_one_family_out_corrs(means)
    full_family_corrs = family_corrs(full_means)
    full_lofo_corrs = leave_one_family_out_corrs(full_means)
    final_dir = args.out_root / "selected_valset"
    final_dir.mkdir(parents=True, exist_ok=True)
    with (final_dir / "navtrain_pdms_val6000_tokens.txt").open("w", encoding="utf-8") as f:
        for token in selected_list:
            f.write(token + "\n")
    selected_rows = []
    for token in selected_list:
        m = meta.get(token, {})
        selected_rows.append(
            {
                "sample_token": token,
                "scene_token": m.get("scene_token", ""),
                "log_name": m.get("log_name", ""),
                "chunk": m.get("chunk", ""),
                "mean_pdms": sum(vectors[token]) / len(probes),
                "std_pdms": math.sqrt(sum((x - sum(vectors[token]) / len(probes)) ** 2 for x in vectors[token]) / len(probes)),
            }
        )
    write_csv(final_dir / "navtrain_pdms_val6000_manifest.csv", ["sample_token", "scene_token", "log_name", "chunk", "mean_pdms", "std_pdms"], selected_rows)
    validation_rows = []
    for idx, probe in enumerate(probes):
        validation_rows.append(
            {
                "probe": probe["name"],
                "family": probe["family"],
                "navtest_pdms": target[idx],
                "selected_pdms": means[idx],
                "full_candidate_pdms": full_means[idx],
                "checkpoint": probe["checkpoint"],
            }
        )
    write_csv(final_dir / "probe_correlation_report.csv", ["probe", "family", "navtest_pdms", "selected_pdms", "full_candidate_pdms", "checkpoint"], validation_rows)
    family_rows = []
    for family in family_names:
        family_rows.append(
            {
                "scope": "family",
                "family": family,
                "num_probes": int(final_family_corrs[family]["num_probes"] or 0),
                "pearson_selected_vs_navtest": final_family_corrs[family]["pearson"],
                "spearman_selected_vs_navtest": final_family_corrs[family]["spearman"],
                "pearson_full_candidate_vs_navtest": full_family_corrs[family]["pearson"],
                "spearman_full_candidate_vs_navtest": full_family_corrs[family]["spearman"],
            }
        )
    for family in family_names:
        family_rows.append(
            {
                "scope": "leave_one_family_out",
                "family": family,
                "num_probes": int(final_lofo_corrs[family]["num_probes"] or 0),
                "pearson_selected_vs_navtest": final_lofo_corrs[family]["pearson"],
                "spearman_selected_vs_navtest": final_lofo_corrs[family]["spearman"],
                "pearson_full_candidate_vs_navtest": full_lofo_corrs[family]["pearson"],
                "spearman_full_candidate_vs_navtest": full_lofo_corrs[family]["spearman"],
            }
        )
    write_csv(
        final_dir / "family_correlation_report.csv",
        [
            "scope",
            "family",
            "num_probes",
            "pearson_selected_vs_navtest",
            "spearman_selected_vs_navtest",
            "pearson_full_candidate_vs_navtest",
            "spearman_full_candidate_vs_navtest",
        ],
        family_rows,
    )
    report = {
        "created_at": utc_now(),
        "selection_size": len(selected_list),
        "num_complete_candidate_tokens": len(tokens),
        "num_probes": len(probes),
        "pearson_selected_vs_navtest": final_corr,
        "spearman_selected_vs_navtest": final_rank_corr,
        "pearson_full_candidate_vs_navtest": pearson(full_means, target),
        "spearman_full_candidate_vs_navtest": pearson(rankdata(full_means), target_rank),
        "family_correlations_selected": final_family_corrs,
        "leave_one_family_out_correlations_selected": final_lofo_corrs,
        "objective": best_score,
        "token_file": str(final_dir / "navtrain_pdms_val6000_tokens.txt"),
        "manifest": str(final_dir / "navtrain_pdms_val6000_manifest.csv"),
        "probe_report": str(final_dir / "probe_correlation_report.csv"),
        "family_report": str(final_dir / "family_correlation_report.csv"),
        "strategy": {
            "candidate_pool": "all complete train chunk samples unless --candidate-size was used in prepare",
            "initial_selection": "difficulty-decile quotas, ranked by positive per-sample correlation with navtest probe ordering times probe-response std",
            "refinement": f"random swap hill-climb for {args.swap_iters} iterations using overall, rank, family, and leave-one-family-out correlations",
        },
    }
    write_json(final_dir / "selection_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--python-bin", type=Path, default=PYTHON_BIN)
    parser.add_argument("--launcher", type=Path, default=LAUNCHER)
    parser.add_argument("--chunk-root", type=Path, default=TRAIN_CHUNK_ROOT)
    parser.add_argument("--chunk-pattern", default="train_*")
    parser.add_argument("--metric-cache-dir", type=Path, default=TRAIN_METRIC_CACHE)
    parser.add_argument("--vlm-text-anchor-cache-root", type=Path, default=TRAIN_ANCHOR_CACHE)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--gpus-csv", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="fp32")
    parser.add_argument("--trajectory-output-key", choices=("pred_traj", "pred_coarse_traj"), default="pred_traj")
    parser.add_argument("--stagger-seconds", type=float, default=8.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and validate a 6000-sample navtrain PDMS validation set.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prepare = sub.add_parser("prepare")
    p_prepare.add_argument("--out-root", type=Path, required=True)
    p_prepare.add_argument("--chunk-root", type=Path, default=TRAIN_CHUNK_ROOT)
    p_prepare.add_argument("--chunk-pattern", default="train_*")
    p_prepare.add_argument("--candidate-size", type=int, default=0, help="0 means all train chunk samples.")
    p_prepare.add_argument("--selection-size", type=int, default=6000)
    p_prepare.add_argument("--seed", type=int, default=20260611)
    p_prepare.add_argument("--hosts", default="zt2,zt3")
    p_prepare.set_defaults(func=prepare)

    p_run = sub.add_parser("run-host")
    add_common_eval_args(p_run)
    p_run.add_argument("--host-key", required=True)
    p_run.add_argument("--probe-manifest", type=Path, required=True)
    p_run.add_argument("--sample-token-file", type=Path, required=True)
    p_run.set_defaults(func=run_host)

    p_launch = sub.add_parser("launch")
    add_common_eval_args(p_launch)
    p_launch.add_argument("--probe-manifest", type=Path, required=True)
    p_launch.add_argument("--sample-token-file", type=Path, required=True)
    p_launch.add_argument("--kill-gpu-stress", action="store_true")
    p_launch.set_defaults(func=launch)

    p_status = sub.add_parser("status")
    p_status.add_argument("--out-root", type=Path, required=True)
    p_status.add_argument("--probe-manifest", type=Path, required=True)
    p_status.add_argument("--num-shards", type=int, default=8)
    p_status.set_defaults(func=status)

    p_select = sub.add_parser("select")
    p_select.add_argument("--out-root", type=Path, required=True)
    p_select.add_argument("--probe-manifest", type=Path, required=True)
    p_select.add_argument("--candidate-manifest", type=Path, required=True)
    p_select.add_argument("--selection-size", type=int, default=6000)
    p_select.add_argument("--difficulty-bins", type=int, default=10)
    p_select.add_argument("--swap-iters", type=int, default=20000)
    p_select.add_argument("--seed", type=int, default=20260611)
    p_select.add_argument("--num-shards", type=int, default=8)
    p_select.add_argument("--allow-incomplete", action="store_true")
    p_select.set_defaults(func=select)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
