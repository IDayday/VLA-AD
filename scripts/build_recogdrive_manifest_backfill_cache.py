#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_recogdrive_chunk_cache import (  # noqa: E402
    chunk_start_value,
    frame_camera_path,
    get_cached_loader,
    normalize_path_args,
    process_records,
    write_built_chunk_outputs,
)
from scripts.run_full_cache_generation import PersistentWorkerPool  # noqa: E402


DEFAULT_MANIFEST = (
    REPO_ROOT
    / "experiments/recogdrive_expert/cache_generation/local_union/data_completeness/"
    / "navtrain_recoverable_backfill_manifest.jsonl"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "cache/recogdrive_expert_chunks/full_v1"
DEFAULT_RECOGDRIVE_VLM = REPO_ROOT / "checkpoints/recogdrive/ReCogDrive-VLM-2B"
DEFAULT_JEPA = REPO_ROOT / "checkpoints/teachers/vjepa2-vitl-fpc64-256"
DEFAULT_VGGT = REPO_ROOT / "checkpoints/teachers/VGGT-1B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ReCogDrive backfill chunks from a local-union recovery manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--navsim-root", "--data-root", dest="data_root", type=Path, default=Path("/mnt/navsim"))
    parser.add_argument("--split", choices=("navtrain", "navtest"), default="navtrain")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--chunk-prefix", default="train_backfill_chunk")
    parser.add_argument("--partition-start", type=int, default=0)
    parser.add_argument("--partition-end", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--start-output-index", type=int, default=0)
    parser.add_argument("--max-backfill-chunks", type=int, default=None)
    parser.add_argument("--recogdrive-vlm-path", type=Path, default=DEFAULT_RECOGDRIVE_VLM)
    parser.add_argument("--jepa-model-path", type=Path, default=DEFAULT_JEPA)
    parser.add_argument("--vggt-model-path", type=Path, default=DEFAULT_VGGT)
    parser.add_argument("--build-vlm-hidden", dest="build_vlm_hidden", action="store_true", default=True)
    parser.add_argument("--no-build-vlm-hidden", dest="build_vlm_hidden", action="store_false")
    parser.add_argument("--build-jepa", dest="build_jepa", action="store_true", default=True)
    parser.add_argument("--no-build-jepa", dest="build_jepa", action="store_false")
    parser.add_argument("--build-vggt", dest="build_vggt", action="store_true", default=True)
    parser.add_argument("--no-build-vggt", dest="build_vggt", action="store_false")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.set_defaults(persistent_workers=True)
    parser.add_argument("--skip-if-present-anywhere", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def output_dir(args: argparse.Namespace, local_chunk_index: int) -> Path:
    return args.output_root / f"{args.chunk_prefix}_{args.start_output_index + local_chunk_index:06d}"


def read_manifest(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with args.manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("split") != args.split:
                continue
            raw_index = int(row["raw_index"])
            if raw_index < args.partition_start:
                continue
            if args.partition_end is not None and raw_index >= args.partition_end:
                continue
            if not row.get("old_root_skipped_but_local_recoverable", row.get("union_valid", False)):
                continue
            rows.append(row)
    rows.sort(key=lambda item: (int(item["raw_index"]), str(item["sample_token"])))
    return rows


def existing_sample_tokens(root: Path, *, exclude_dir_prefix: str) -> set[str]:
    tokens: set[str] = set()
    if not root.is_dir():
        return tokens
    for samples_dir in root.glob("*/samples"):
        if samples_dir.parent.name.startswith(exclude_dir_prefix):
            continue
        for path in samples_dir.glob("*.pt"):
            tokens.add(path.stem)
    return tokens


def make_builder_args(args: argparse.Namespace, local_chunk_index: int, rows: List[Dict[str, Any]]) -> argparse.Namespace:
    raw_indices = [int(record["raw_index"]) for record in rows]
    raw_span = (max(raw_indices) - min(raw_indices) + 1) if raw_indices else 1
    return normalize_path_args(
        argparse.Namespace(
            data_root=args.data_root,
            project_root=REPO_ROOT,
            split=args.split,
            chunk_index=args.start_output_index + local_chunk_index,
            chunk_size=raw_span,
            chunk_start=min(raw_indices) if raw_indices else args.partition_start,
            chunk_stop=(max(raw_indices) + 1) if raw_indices else args.partition_end,
            allow_partial_final_chunk=True,
            strict_token_window=False,
            output_dir=output_dir(args, local_chunk_index),
            build_vlm_hidden=args.build_vlm_hidden,
            build_jepa=args.build_jepa,
            build_vggt=args.build_vggt,
            recogdrive_vlm_path=args.recogdrive_vlm_path,
            jepa_model_path=args.jepa_model_path,
            vggt_model_path=args.vggt_model_path,
            precision=args.precision,
            device=args.device,
            num_gpus=args.num_gpus,
            workers=args.workers,
            workers_per_gpu=args.workers_per_gpu,
            max_samples=(max(raw_indices) + 1) if raw_indices else None,
            overwrite=args.overwrite,
            skip_existing=False,
            resume=True,
            allow_missing_future_frames=False,
            log_every=args.log_every,
            finalize_existing=False,
        )
    )


def records_from_manifest(
    builder_args: argparse.Namespace,
    rows: List[Dict[str, Any]],
    *,
    globally_existing: Optional[set[str]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cached = get_cached_loader(builder_args)
    loader = cached["loader"]
    scene_filter = cached["scene_filter"]
    blobs = cached["blobs"]
    h = scene_filter.num_history_frames
    records: List[Dict[str, Any]] = []
    skipped_existing = 0
    missing_image_records = 0
    duplicate_records = 0
    seen: set[str] = set()
    for row in rows:
        token = row["loader_token"]
        frames = loader.scene_frames_dicts[token]
        history_frames = frames[:h]
        future_frames = frames[h : h + 4]
        history_paths = [frame_camera_path(frame, blobs) for frame in history_frames[-4:]]
        future_paths = [frame_camera_path(frame, blobs) for frame in future_frames]
        required_paths: List[str] = []
        if builder_args.build_vlm_hidden or builder_args.build_jepa or builder_args.build_vggt:
            required_paths.extend(history_paths[-4:])
        if builder_args.build_jepa and not builder_args.allow_missing_future_frames:
            required_paths.extend(future_paths[:4])
        if builder_args.build_vggt and history_paths:
            required_paths.append(history_paths[-1])
        missing_paths = [path for path in required_paths if not Path(path).is_file()]
        if missing_paths:
            missing_image_records += 1
            if builder_args.log_every and missing_image_records <= 20:
                print(f"skip backfill token={token} missing required image(s): {missing_paths[:3]}")
            continue
        sample_token = frames[h - 1]["token"]
        if sample_token in seen:
            duplicate_records += 1
            continue
        if globally_existing is not None and sample_token in globally_existing:
            skipped_existing += 1
            continue
        seen.add(sample_token)
        records.append(
            {
                "scene_token": frames[h - 1].get("scene_token", token),
                "sample_token": sample_token,
                "log_name": frames[h - 1].get("log_name"),
                "history_cam_f0": history_paths,
                "future_cam_f0": future_paths,
                "loader_token": token,
                "raw_index": int(row["raw_index"]),
                "source_manifest_status": row.get("status"),
            }
        )
    raw_indices = [int(row["raw_index"]) for row in rows]
    info = {
        "NAVSIM_DATA_ROOT": str(cached["data_root"]),
        "OPENSCENE_DATA_ROOT": str(cached["openscene"]),
        "NUPLAN_MAPS_ROOT": str(cached["maps"]),
        "navsim_log_path": str(cached["logs"]),
        "sensor_blobs_path": str(cached["blobs"]),
        "scene_filter": str(cached["scene_filter_path"]),
        "num_available": len(loader.tokens),
        "chunk_start": chunk_start_value(builder_args),
        "chunk_stop": builder_args.chunk_stop,
        "chunk_end": max(raw_indices) + 1 if raw_indices else builder_args.chunk_stop,
        "num_missing_image_records": missing_image_records,
        "num_duplicate_records": duplicate_records,
        "num_skipped_existing_global": skipped_existing,
        "strict_token_window": False,
        "raw_window_size": len(rows),
        "scene_filter_config": asdict(scene_filter),
        "backfill_from_manifest": True,
        "backfill_manifest": str(rows[0].get("_manifest_path", "")) if rows else "",
        "backfill_manifest_rows": len(rows),
        "backfill_raw_index_min": min(raw_indices) if raw_indices else None,
        "backfill_raw_index_max": max(raw_indices) if raw_indices else None,
    }
    print(json.dumps({k: v for k, v in info.items() if k != "scene_filter_config"}, indent=2, sort_keys=True))
    return records, info


def process_one_chunk(
    args: argparse.Namespace,
    local_chunk_index: int,
    rows: List[Dict[str, Any]],
    *,
    pool: Optional[PersistentWorkerPool],
    globally_existing: Optional[set[str]],
) -> PersistentWorkerPool | None:
    for row in rows:
        row["_manifest_path"] = str(args.manifest)
    builder_args = make_builder_args(args, local_chunk_index, rows)
    records, info = records_from_manifest(builder_args, rows, globally_existing=globally_existing)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "output_dir": str(builder_args.output_dir),
                    "manifest_rows": len(rows),
                    "records_to_build": len(records),
                    "chunk_index": builder_args.chunk_index,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return pool
    if not records:
        print(f"No records selected for {builder_args.output_dir}; skipping.")
        return pool
    builder_args.output_dir.mkdir(parents=True, exist_ok=True)
    (builder_args.output_dir / "samples").mkdir(parents=True, exist_ok=True)
    if args.persistent_workers:
        if pool is None:
            from scripts.build_recogdrive_chunk_cache import choose_worker_count

            worker_count = choose_worker_count(builder_args, max(1, args.chunk_size))
            print(f"Starting {worker_count} persistent backfill worker(s).", flush=True)
            pool = PersistentWorkerPool(worker_count, vars(builder_args).copy())
        chunk_key = f"{args.split}:backfill:{builder_args.chunk_index}:{chunk_start_value(builder_args)}"
        index_records, worker_reports = pool.process_chunk(chunk_key, vars(builder_args).copy(), records)
    else:
        index_records, worker_reports = process_records(builder_args, records)
    write_built_chunk_outputs(builder_args, info, index_records, worker_reports)
    return pool


def main() -> int:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    rows = read_manifest(args)
    if args.max_backfill_chunks is not None:
        max_rows = args.max_backfill_chunks * args.chunk_size
        rows = rows[:max_rows]
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "selected_rows": len(rows),
                "partition_start": args.partition_start,
                "partition_end": args.partition_end,
                "output_root": str(args.output_root),
                "chunk_prefix": args.chunk_prefix,
                "chunk_size": args.chunk_size,
                "dry_run": args.dry_run,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    globally_existing = (
        existing_sample_tokens(args.output_root, exclude_dir_prefix=args.chunk_prefix)
        if args.skip_if_present_anywhere
        else None
    )
    if globally_existing is not None:
        print(f"Loaded {len(globally_existing)} existing sample token(s) from {args.output_root}", flush=True)
    chunks = [rows[i : i + args.chunk_size] for i in range(0, len(rows), args.chunk_size)]
    pool: Optional[PersistentWorkerPool] = None
    try:
        for local_chunk_index, chunk_rows in enumerate(chunks):
            pool = process_one_chunk(
                args,
                local_chunk_index,
                chunk_rows,
                pool=pool,
                globally_existing=globally_existing,
            )
    finally:
        if pool is not None:
            pool.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
