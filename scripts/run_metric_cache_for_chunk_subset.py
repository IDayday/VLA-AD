#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NAVSIM metric cache for tokens listed in a chunk-cache index.")
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/project/VLA-AD"))
    parser.add_argument("--chunk-cache-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--split", default="navtest")
    parser.add_argument("--navsim-log-path", type=Path, default=Path("/mnt/navsim/test_navsim_logs/test"))
    parser.add_argument("--openscene-root", type=Path, default=Path("/mnt/navsim"))
    parser.add_argument("--maps-root", type=Path, default=Path("/mnt/navsim/maps"))
    parser.add_argument("--worker", default="single_machine_thread_pool")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--python", default="/root/miniconda3/envs/navsim/bin/python")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_records(chunk_cache_dir: Path, max_samples: int) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    index_path = chunk_cache_dir / "index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if "sample_token" not in record or "log_name" not in record:
                raise KeyError(f"Index record missing sample_token/log_name: {record}")
            records.append(record)
            if len(records) >= max_samples:
                break
    if not records:
        raise RuntimeError(f"No records found in {index_path}")
    return records


def hydra_list(values: List[str]) -> str:
    return "[" + ",".join(values) + "]"


def main() -> int:
    args = parse_args()
    records = load_records(args.chunk_cache_dir, args.max_samples)
    tokens = [record["sample_token"] for record in records]
    log_names = sorted({record["log_name"] for record in records})
    args.output_cache.mkdir(parents=True, exist_ok=True)
    summary = {
        "chunk_cache_dir": str(args.chunk_cache_dir),
        "max_samples": args.max_samples,
        "num_tokens": len(tokens),
        "num_logs": len(log_names),
        "log_names": log_names,
        "output_cache": str(args.output_cache),
    }
    (args.output_cache / "subset_request.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cmd = [
        args.python,
        "navsim/planning/script/run_metric_caching.py",
        f"train_test_split={args.split}",
        f"navsim_log_path={args.navsim_log_path}",
        f"cache.cache_path={args.output_cache}",
        f"worker={args.worker}",
        f"worker.max_workers={args.max_workers}",
        f"train_test_split.scene_filter.log_names={hydra_list(log_names)}",
        f"train_test_split.scene_filter.tokens={hydra_list(tokens)}",
    ]
    (args.output_cache / "command.json").write_text(json.dumps(cmd, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    print("RUN", " ".join(cmd), flush=True)
    if args.dry_run:
        return 0
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{args.project_root}:{env.get('PYTHONPATH', '')}"
    env["OPENSCENE_DATA_ROOT"] = str(args.openscene_root)
    env["NUPLAN_MAPS_ROOT"] = str(args.maps_root)
    env["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"
    env.setdefault("NAVSIM_EXP_ROOT", str(args.project_root / "experiments/recogdrive_expert"))
    subprocess.run(cmd, cwd=args.project_root, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
