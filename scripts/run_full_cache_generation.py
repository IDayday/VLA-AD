#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue as queue_lib
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CACHE_ROOT = REPO_ROOT / "cache/recogdrive_expert_chunks/full_v1"
DEFAULT_REPORT_ROOT = REPO_ROOT / "experiments/recogdrive_expert/cache_generation"
DEFAULT_NAVSIM_ROOT = Path(os.environ.get("NAVSIM_DATA_ROOT", "/mnt/navsim"))
DEFAULT_RECOGDRIVE_VLM = REPO_ROOT / "checkpoints/recogdrive/ReCogDrive-VLM-2B"
DEFAULT_JEPA = REPO_ROOT / "checkpoints/teachers/vjepa2-vitl-fpc64-256"
DEFAULT_VGGT = REPO_ROOT / "checkpoints/teachers/VGGT-1B"

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate full ReCogDrive expert-token chunk caches with resumable per-chunk execution.")
    parser.add_argument("--navsim-root", type=Path, default=DEFAULT_NAVSIM_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--splits", default="navtrain,navtest")
    parser.add_argument("--train-chunk-size", type=int, default=4096)
    parser.add_argument("--eval-chunk-size", type=int, default=4096)
    parser.add_argument("--window-mode", choices=("valid-fill", "strict"), default="valid-fill", help="valid-fill makes each chunk contain up to chunk-size valid samples and advances by builder chunk_end; strict partitions raw tokens directly.")
    parser.add_argument("--assume-navtrain-samples", type=int, default=None)
    parser.add_argument("--assume-navtest-samples", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--partition-count", type=int, default=1, help="Split each selected NAVSIM split into this many raw-token partitions.")
    parser.add_argument("--partition-index", type=int, default=0, help="Zero-based raw-token partition index to process on this node.")
    parser.add_argument("--recogdrive-vlm-path", type=Path, default=DEFAULT_RECOGDRIVE_VLM)
    parser.add_argument("--jepa-model-path", type=Path, default=DEFAULT_JEPA)
    parser.add_argument("--vggt-model-path", type=Path, default=DEFAULT_VGGT)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--workers", type=int, default=None, help="Total chunk-cache workers per builder process. Overrides --workers-per-gpu.")
    parser.add_argument("--workers-per-gpu", type=int, default=1, help="CUDA worker processes per requested GPU.")
    parser.add_argument("--persistent-workers", dest="persistent_workers", action="store_true", default=True, help="Keep chunk-cache workers alive across chunks so model weights are loaded once per worker.")
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false", help="Use the legacy one-builder-subprocess-per-chunk execution path.")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=int, default=30)
    parser.add_argument("--stop-after-first-failure", action="store_true")
    parser.add_argument("--stop-after-chunks", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cuda-visible-devices", default=None)
    return parser.parse_args()

def selected_splits(raw: str) -> List[str]:
    splits = [item.strip() for item in raw.split(",") if item.strip()]
    bad = [item for item in splits if item not in {"navtrain", "navtest"}]
    if bad:
        raise ValueError(f"Unknown split(s): {bad}")
    return splits

def count_split_tokens(navsim_root: Path, split: str) -> int:
    from scripts.build_recogdrive_chunk_cache import autodetect_data_paths, cam_f0_sensor_config, load_scene_filter, split_paths
    from navsim.common.dataloader import SceneLoader
    _, openscene, _ = autodetect_data_paths(navsim_root)
    logs, blobs = split_paths(openscene, split)
    scene_filter, _ = load_scene_filter(split, None)
    loader = SceneLoader(data_path=logs, sensor_blobs_path=blobs, scene_filter=scene_filter, sensor_config=cam_f0_sensor_config(), load_image_path=True)
    return len(loader.tokens)

def split_total(args: argparse.Namespace, split: str) -> int:
    if split == "navtrain":
        total = args.assume_navtrain_samples if args.assume_navtrain_samples is not None else count_split_tokens(args.navsim_root, split)
        return min(total, args.max_train_samples) if args.max_train_samples is not None else total
    total = args.assume_navtest_samples if args.assume_navtest_samples is not None else count_split_tokens(args.navsim_root, split)
    return min(total, args.max_eval_samples) if args.max_eval_samples is not None else total

def split_chunk_size(args: argparse.Namespace, split: str) -> int:
    return args.train_chunk_size if split == "navtrain" else args.eval_chunk_size

def partition_range(args: argparse.Namespace, split: str) -> tuple[int, int]:
    total = split_total(args, split)
    start = (total * args.partition_index) // args.partition_count
    end = (total * (args.partition_index + 1)) // args.partition_count
    return start, end

def partition_base_index(args: argparse.Namespace, split: str, start: int) -> int:
    if start <= 0:
        return 0
    return math.ceil(start / max(1, split_chunk_size(args, split)))

def split_prefix(split: str) -> str:
    return "train_full_chunk" if split == "navtrain" else "navtest_full_chunk"

def chunk_dir(args: argparse.Namespace, split: str, index: int) -> Path:
    return args.output_root / f"{split_prefix(split)}_{index:06d}"

def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

def make_item(args: argparse.Namespace, split: str, index: int, start: int, split_total_value: int, range_start: int, range_end: int) -> Dict[str, Any]:
    raw_size = min(split_chunk_size(args, split), max(0, range_end - start))
    return {
        "split": split,
        "index": index,
        "start": start,
        "size": raw_size,
        "total": split_total_value,
        "partition_start": range_start,
        "partition_end": range_end,
        "output_dir": str(chunk_dir(args, split, index)),
        "is_final_start": start + raw_size >= range_end,
    }

def chunk_status(args: argparse.Namespace, item: Dict[str, Any]) -> Dict[str, Any]:
    out = Path(item["output_dir"])
    samples_dir = out / "samples"
    metadata_path = out / "metadata.json"
    index_path = out / "index.jsonl"
    sample_count = len(list(samples_dir.glob("*.pt"))) if samples_dir.is_dir() else 0
    index_count = line_count(index_path)
    metadata: Dict[str, Any] = {}
    metadata_error: Optional[str] = None
    if metadata_path.is_file():
        try:
            metadata = read_json(metadata_path)
        except Exception as exc:
            metadata_error = str(exc)
    metadata_records = metadata.get("num_records")
    expected = int(metadata_records) if isinstance(metadata_records, int) else int(item["size"])
    flags_ok = (
        metadata.get("is_dummy") is False
        and metadata.get("contains_vlm_hidden") is True
        and metadata.get("contains_jepa") is True
        and metadata.get("contains_vggt") is True
    )
    mode_ok = True
    if args.window_mode == "strict":
        mode_ok = metadata.get("strict_token_window") is True
    else:
        mode_ok = metadata.get("strict_token_window") is not True
    start_ok = metadata.get("chunk_start") in (None, item["start"]) if not metadata else metadata.get("chunk_start") == item["start"]
    complete = metadata_path.is_file() and flags_ok and mode_ok and start_ok and sample_count >= expected and index_count >= expected
    return {
        "output_dir": str(out),
        "expected_records": expected,
        "sample_files": sample_count,
        "index_records": index_count,
        "metadata_exists": metadata_path.is_file(),
        "metadata_error": metadata_error,
        "metadata_records": metadata.get("num_records"),
        "metadata_chunk_start": metadata.get("chunk_start"),
        "metadata_chunk_end": metadata.get("chunk_end"),
        "metadata_chunk_size": metadata.get("chunk_size"),
        "strict_token_window": metadata.get("strict_token_window"),
        "contains_vlm_hidden": metadata.get("contains_vlm_hidden"),
        "contains_jepa": metadata.get("contains_jepa"),
        "contains_vggt": metadata.get("contains_vggt"),
        "complete": complete,
    }

def discover_split(args: argparse.Namespace, split: str) -> List[Dict[str, Any]]:
    total = split_total(args, split)
    range_start, range_end = partition_range(args, split)
    items: List[Dict[str, Any]] = []
    start = range_start
    index = partition_base_index(args, split, range_start)
    max_chunks = math.ceil(max(0, range_end - range_start) / max(1, split_chunk_size(args, split))) + 512
    while start < range_end and len(items) < max_chunks:
        item = make_item(args, split, index, start, total, range_start, range_end)
        status = chunk_status(args, item)
        merged = {**item, **status}
        items.append(merged)
        if not status["complete"]:
            break
        if args.window_mode == "strict":
            next_start = start + item["size"]
        else:
            next_start = status.get("metadata_chunk_end")
            if not isinstance(next_start, int) or next_start <= start:
                next_start = start + item["size"]
        start = min(next_start, range_end)
        index += 1
    return items

def discover_all(args: argparse.Namespace) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for split in selected_splits(args.splits):
        items.extend(discover_split(args, split))
    return items

def runner_command(args: argparse.Namespace) -> List[str]:
    cmd = [args.python, "scripts/run_full_cache_generation.py", "--output-root", str(args.output_root), "--report-root", str(args.report_root), "--navsim-root", str(args.navsim_root), "--splits", args.splits, "--train-chunk-size", str(args.train_chunk_size), "--eval-chunk-size", str(args.eval_chunk_size), "--window-mode", args.window_mode, "--partition-count", str(args.partition_count), "--partition-index", str(args.partition_index), "--num-gpus", str(args.num_gpus), "--workers-per-gpu", str(args.workers_per_gpu), "--precision", args.precision, "--retries", str(args.retries)]
    if args.workers is not None:
        cmd.extend(["--workers", str(args.workers)])
    if args.assume_navtrain_samples is not None:
        cmd.extend(["--assume-navtrain-samples", str(args.assume_navtrain_samples)])
    if args.assume_navtest_samples is not None:
        cmd.extend(["--assume-navtest-samples", str(args.assume_navtest_samples)])
    if args.cuda_visible_devices:
        cmd.extend(["--cuda-visible-devices", args.cuda_visible_devices])
    if not args.persistent_workers:
        cmd.append("--no-persistent-workers")
    return cmd

def builder_namespace(args: argparse.Namespace, item: Dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        data_root=args.navsim_root,
        project_root=REPO_ROOT,
        split=item["split"],
        chunk_index=int(item["index"]),
        chunk_size=int(item["size"]),
        chunk_start=int(item["start"]),
        chunk_stop=int(item["partition_end"]),
        allow_partial_final_chunk=True,
        strict_token_window=args.window_mode == "strict",
        output_dir=Path(item["output_dir"]),
        build_vlm_hidden=True,
        build_jepa=True,
        build_vggt=True,
        recogdrive_vlm_path=args.recogdrive_vlm_path,
        jepa_model_path=args.jepa_model_path,
        vggt_model_path=args.vggt_model_path,
        precision=args.precision,
        device=args.device,
        num_gpus=args.num_gpus,
        workers=args.workers,
        workers_per_gpu=args.workers_per_gpu,
        max_samples=None,
        overwrite=args.overwrite,
        skip_existing=False,
        resume=True,
        allow_missing_future_frames=False,
        log_every=args.log_every,
        finalize_existing=False,
    )


def builder_command(args: argparse.Namespace, item: Dict[str, Any]) -> List[str]:
    cmd = [
        args.python, "scripts/build_recogdrive_chunk_cache.py",
        "--navsim-root", str(args.navsim_root),
        "--split", item["split"],
        "--chunk-index", str(item["index"]),
        "--chunk-start", str(item["start"]),
        "--chunk-size", str(item["size"]),
        "--chunk-stop", str(item["partition_end"]),
        "--output-dir", item["output_dir"],
        "--build-vlm-hidden", "--build-jepa", "--build-vggt",
        "--recogdrive-vlm-path", str(args.recogdrive_vlm_path),
        "--jepa-model-path", str(args.jepa_model_path),
        "--vggt-model-path", str(args.vggt_model_path),
        "--precision", args.precision,
        "--device", args.device,
        "--num-gpus", str(args.num_gpus),
        "--workers-per-gpu", str(args.workers_per_gpu),
        "--log-every", str(args.log_every),
        "--allow-partial-final-chunk",
    ]
    if args.workers is not None:
        cmd.extend(["--workers", str(args.workers)])
    if args.window_mode == "strict":
        cmd.append("--strict-token-window")
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd

def shell_join(cmd: Sequence[str]) -> str:
    import shlex
    return " ".join(shlex.quote(str(part)) for part in cmd)

def write_reports(args: argparse.Namespace, failures: List[Dict[str, Any]], launched: int, started_at: str) -> None:
    args.report_root.mkdir(parents=True, exist_ok=True)
    items = discover_all(args)
    complete = [item for item in items if item["complete"]]
    incomplete = [item for item in items if not item["complete"]]
    totals = {split: split_total(args, split) for split in selected_splits(args.splits)}
    ranges = {split: partition_range(args, split) for split in selected_splits(args.splits)}
    estimates = {split: math.ceil(max(0, ranges[split][1] - ranges[split][0]) / split_chunk_size(args, split)) for split in totals}
    report = {
        "started_at": started_at,
        "updated_at": now_iso(),
        "output_root": str(args.output_root),
        "navsim_root": str(args.navsim_root),
        "window_mode": args.window_mode,
        "precision": args.precision,
        "num_gpus": args.num_gpus,
        "workers": args.workers,
        "workers_per_gpu": args.workers_per_gpu,
        "persistent_workers": args.persistent_workers,
        "execution_mode": "persistent-workers" if args.persistent_workers else "subprocess-per-chunk",
        "partition_count": args.partition_count,
        "partition_index": args.partition_index,
        "partition_ranges": {key: {"start": value[0], "end": value[1]} for key, value in ranges.items()},
        "split_totals": totals,
        "minimum_chunk_estimates": estimates,
        "known_chunks": len(items),
        "complete_chunks": len(complete),
        "incomplete_known_chunks": len(incomplete),
        "launched_this_run": launched,
        "failures": failures,
        "chunks": items,
    }
    (args.report_root / "full_cache_generation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Full Cache Generation Report", "",
        f"Updated: {report['updated_at']}",
        f"Output root: `{args.output_root}`",
        f"Window mode: `{args.window_mode}`",
        f"Partition: {args.partition_index}/{args.partition_count}",
        f"Worker config: workers={args.workers}, workers_per_gpu={args.workers_per_gpu}, num_gpus={args.num_gpus}",
        f"Execution mode: `{report['execution_mode']}`",
        f"Known chunks: {len(complete)} complete / {len(items)} discovered",
        f"Launched this run: {launched}",
        f"Failures: {len(failures)}", "",
        "| Split | Chunk | Partition Start | Partition End | Start | Raw Size | Valid/Expected | Complete | Samples | Index | Chunk End | Dir |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for item in items:
        lines.append(f"| {item['split']} | {item['index']} | {item['partition_start']} | {item['partition_end']} | {item['start']} | {item['size']} | {item['expected_records']} | {item['complete']} | {item['sample_files']} | {item['index_records']} | {item.get('metadata_chunk_end')} | `{item['output_dir']}` |")
    lines.extend(["", "## Retry Command", "", "```bash", shell_join(runner_command(args)), "```"])
    if incomplete:
        lines.extend(["", "## Next Incomplete Chunk", "", "```bash", shell_join(builder_command(args, incomplete[0])), "```"])
    (args.report_root / "full_cache_generation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_one(args: argparse.Namespace, item: Dict[str, Any], attempt: int) -> int:
    log_dir = args.report_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{item['split']}_{item['index']:06d}_attempt{attempt}.log"
    cmd = builder_command(args, item)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    with log_path.open("a", encoding="utf-8") as log_fp:
        log_fp.write(f"\n[{now_iso()}] attempt={attempt} command={shell_join(cmd)}\n")
        log_fp.flush()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=log_fp, stderr=subprocess.STDOUT)
        log_fp.write(f"[{now_iso()}] exit_code={proc.returncode}\n")
        return proc.returncode

def persistent_worker_entry(worker_id: int, args_dict: Dict[str, Any], task_queue: Any, result_queue: Any) -> None:
    try:
        from scripts.build_recogdrive_chunk_cache import ChunkCacheWorkerRuntime
        runtime = ChunkCacheWorkerRuntime(worker_id, args_dict)
        result_queue.put({"event": "ready", "worker_id": worker_id, "device": runtime.actual_device})
    except Exception:
        result_queue.put({"event": "init_failed", "worker_id": worker_id, "traceback": traceback.format_exc()})
        return

    while True:
        task = task_queue.get()
        if task is None:
            result_queue.put({"event": "stopped", "worker_id": worker_id})
            return
        try:
            result = runtime.process(task["args_dict"], task["records"])
            result_queue.put({
                "event": "result",
                "ok": True,
                "chunk_key": task["chunk_key"],
                "worker_id": worker_id,
                "result": result,
            })
        except Exception:
            result_queue.put({
                "event": "result",
                "ok": False,
                "chunk_key": task.get("chunk_key"),
                "worker_id": worker_id,
                "traceback": traceback.format_exc(),
            })

class PersistentWorkerPool:
    def __init__(self, worker_count: int, base_args_dict: Dict[str, Any]) -> None:
        if worker_count <= 0:
            raise ValueError("persistent worker_count must be positive")
        self.worker_count = worker_count
        self.ctx = mp.get_context("spawn")
        self.result_queue = self.ctx.Queue()
        self.task_queues: Dict[int, Any] = {}
        self.processes: Dict[int, Any] = {}
        for worker_id in range(worker_count):
            task_queue = self.ctx.Queue()
            process = self.ctx.Process(
                target=persistent_worker_entry,
                args=(worker_id, base_args_dict, task_queue, self.result_queue),
                name=f"persistent-chunk-cache-worker-{worker_id}",
            )
            process.start()
            self.task_queues[worker_id] = task_queue
            self.processes[worker_id] = process
        self._wait_ready()

    def _wait_ready(self) -> None:
        ready: Dict[int, Dict[str, Any]] = {}
        failures: List[Dict[str, Any]] = []
        while len(ready) + len(failures) < self.worker_count:
            try:
                message = self.result_queue.get(timeout=10)
            except queue_lib.Empty:
                dead = [
                    (worker_id, process)
                    for worker_id, process in self.processes.items()
                    if process.exitcode not in (0, None) and worker_id not in ready
                ]
                if not dead:
                    continue
                for worker_id, process in dead:
                    failures.append({
                        "worker_id": worker_id,
                        "traceback": f"process exited with code {process.exitcode} before reporting ready",
                    })
                break
            event = message.get("event")
            if event == "ready":
                ready[int(message["worker_id"])] = message
            elif event == "init_failed":
                failures.append(message)
            else:
                failures.append({"worker_id": message.get("worker_id"), "traceback": f"unexpected startup message: {message}"})
        if failures:
            self.close(terminate=True)
            raise RuntimeError("persistent worker startup failed: " + json.dumps(failures, default=str))
        devices = ", ".join(str(ready[idx].get("device")) for idx in sorted(ready))
        print(f"Persistent chunk-cache workers ready: {self.worker_count} ({devices})", flush=True)

    def _assert_alive(self, worker_ids: Optional[Sequence[int]] = None) -> None:
        ids = list(worker_ids) if worker_ids is not None else list(self.processes)
        dead = [
            {"worker_id": worker_id, "exit_code": self.processes[worker_id].exitcode}
            for worker_id in ids
            if self.processes[worker_id].exitcode not in (0, None)
        ]
        if dead:
            raise RuntimeError("persistent worker process died: " + json.dumps(dead, sort_keys=True))

    def process_chunk(
        self,
        chunk_key: str,
        args_dict: Dict[str, Any],
        records: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        from scripts.build_recogdrive_chunk_cache import merge_worker_reports

        assigned: List[int] = []
        self._assert_alive()
        for worker_id in range(self.worker_count):
            shard = records[worker_id::self.worker_count]
            if not shard:
                continue
            assigned.append(worker_id)
            self.task_queues[worker_id].put({
                "chunk_key": chunk_key,
                "args_dict": args_dict,
                "records": shard,
            })
        if not assigned:
            return [], []

        results: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []
        pending = set(assigned)
        while len(results) + len(failures) < len(assigned):
            try:
                message = self.result_queue.get(timeout=10)
            except queue_lib.Empty:
                dead = [
                    worker_id for worker_id in pending
                    if self.processes[worker_id].exitcode not in (0, None)
                ]
                if not dead:
                    continue
                for worker_id in dead:
                    failures.append({
                        "worker_id": worker_id,
                        "traceback": f"process exited with code {self.processes[worker_id].exitcode} before reporting result",
                    })
                    pending.discard(worker_id)
                break
            if message.get("event") != "result":
                continue
            if message.get("chunk_key") != chunk_key:
                failures.append({
                    "worker_id": message.get("worker_id"),
                    "traceback": f"unexpected result for chunk {message.get('chunk_key')}",
                })
                continue
            worker_id = int(message.get("worker_id"))
            pending.discard(worker_id)
            if message.get("ok"):
                results.append(message["result"])
            else:
                failures.append(message)
        if failures:
            raise RuntimeError("persistent worker chunk failed: " + json.dumps(failures, default=str))
        return merge_worker_reports(records, results)

    def close(self, *, terminate: bool = False) -> None:
        if terminate:
            for process in self.processes.values():
                if process.is_alive():
                    process.terminate()
        else:
            for task_queue in self.task_queues.values():
                task_queue.put(None)
        for process in self.processes.values():
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)

def persistent_worker_count(args: argparse.Namespace, builder_args: argparse.Namespace) -> int:
    from scripts.build_recogdrive_chunk_cache import choose_worker_count
    capacity = max(1, args.train_chunk_size, args.eval_chunk_size)
    return choose_worker_count(builder_args, capacity)

def run_one_persistent(args: argparse.Namespace, item: Dict[str, Any], pool: Optional[PersistentWorkerPool]) -> tuple[PersistentWorkerPool, Dict[str, Any]]:
    from scripts.build_recogdrive_chunk_cache import build_records, write_built_chunk_outputs

    builder_args = builder_namespace(args, item)
    records, info = build_records(builder_args)
    if not records:
        raise RuntimeError("No records selected for this chunk.")
    if pool is None:
        worker_count = persistent_worker_count(args, builder_args)
        print(f"Starting {worker_count} persistent chunk-cache worker(s).", flush=True)
        pool = PersistentWorkerPool(worker_count, vars(builder_args).copy())
    chunk_key = f"{item['split']}:{item['index']}:{item['start']}"
    index_records, worker_reports = pool.process_chunk(chunk_key, vars(builder_args).copy(), records)
    write_built_chunk_outputs(builder_args, info, index_records, worker_reports)
    return pool, {"num_records": len(index_records), "worker_reports": worker_reports}

def run_persistent(args: argparse.Namespace, failures: List[Dict[str, Any]], launched: int, started_at: str) -> int:
    pool: Optional[PersistentWorkerPool] = None
    try:
        while True:
            item = next_incomplete(args)
            if item is None:
                write_reports(args, failures, launched, started_at)
                print("Cache generation complete", flush=True)
                return 0 if not failures else 1
            if args.stop_after_chunks is not None and launched >= args.stop_after_chunks:
                write_reports(args, failures, launched, started_at)
                print(f"stop-after-chunks reached after launching {launched} chunk(s)", flush=True)
                return 0
            print(f"build {item['split']} chunk {item['index']} start={item['start']} size={item['size']} -> {item['output_dir']}", flush=True)
            ok = False
            for attempt in range(1, args.retries + 1):
                try:
                    pool, _ = run_one_persistent(args, item, pool)
                    refreshed = {**item, **chunk_status(args, item)}
                    if refreshed["complete"]:
                        ok = True
                        break
                    failure = {**refreshed, "attempt": attempt, "exit_code": 1, "time": now_iso(), "mode": "persistent-workers", "error": "chunk output incomplete after worker success"}
                    failures.append(failure)
                except Exception as exc:
                    refreshed = {**item, **chunk_status(args, item)}
                    failure = {
                        **refreshed,
                        "attempt": attempt,
                        "exit_code": 1,
                        "time": now_iso(),
                        "mode": "persistent-workers",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                    failures.append(failure)
                    if pool is not None:
                        pool.close(terminate=True)
                        pool = None
                write_reports(args, failures, launched + 1, started_at)
                if ok:
                    break
                print(f"chunk failed attempt={attempt}: {item['output_dir']}", flush=True)
                if attempt < args.retries:
                    time.sleep(args.retry_sleep)
            launched += 1
            write_reports(args, failures, launched, started_at)
            if not ok and args.stop_after_first_failure:
                return 1
    finally:
        if pool is not None:
            pool.close()


def next_incomplete(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    for split in selected_splits(args.splits):
        for item in discover_split(args, split):
            if not item["complete"]:
                return item
    return None

def main() -> int:
    args = parse_args()
    if args.train_chunk_size <= 0 or args.eval_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if args.partition_count <= 0:
        raise ValueError("--partition-count must be positive")
    if args.partition_index < 0 or args.partition_index >= args.partition_count:
        raise ValueError("--partition-index must satisfy 0 <= index < partition_count")
    if args.partition_count > 1 and args.report_root.name != f"partition_{args.partition_index:02d}_of_{args.partition_count:02d}":
        args.report_root = args.report_root / f"partition_{args.partition_index:02d}_of_{args.partition_count:02d}"
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    started_at = now_iso()
    failures: List[Dict[str, Any]] = []
    launched = 0
    write_reports(args, failures, launched, started_at)
    if args.dry_run or args.status_only:
        print(f"Discovered {len(discover_all(args))} chunk entry/entries under {args.output_root}")
        print(f"Report: {args.report_root / 'full_cache_generation_report.md'}")
        return 0

    if args.persistent_workers:
        return run_persistent(args, failures, launched, started_at)

    while True:
        item = next_incomplete(args)
        if item is None:
            write_reports(args, failures, launched, started_at)
            print("Cache generation complete", flush=True)
            return 0 if not failures else 1
        if args.stop_after_chunks is not None and launched >= args.stop_after_chunks:
            write_reports(args, failures, launched, started_at)
            print(f"stop-after-chunks reached after launching {launched} chunk(s)", flush=True)
            return 0
        print(f"build {item['split']} chunk {item['index']} start={item['start']} size={item['size']} -> {item['output_dir']}", flush=True)
        ok = False
        for attempt in range(1, args.retries + 1):
            code = run_one(args, item, attempt)
            refreshed = {**item, **chunk_status(args, item)}
            if code == 0 and refreshed["complete"]:
                ok = True
                break
            failure = {**refreshed, "attempt": attempt, "exit_code": code, "time": now_iso()}
            failures.append(failure)
            write_reports(args, failures, launched + 1, started_at)
            print(f"chunk failed attempt={attempt} exit={code}: {item['output_dir']}", flush=True)
            if attempt < args.retries:
                time.sleep(args.retry_sleep)
        launched += 1
        write_reports(args, failures, launched, started_at)
        if not ok and args.stop_after_first_failure:
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
