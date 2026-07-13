#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


_SAMPLE_TOKENS: Set[str] = set()
_LOG_ROOT: Path
_BLOB_ROOT: Path
_NUM_HISTORY_FRAMES = 4
_NUM_FRAMES = 14
_FRAME_INTERVAL = 1
_HAS_ROUTE = True
_VERIFY_IMAGES = "current-cam-f0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit NAVSIM strict-cache online-training data for missing/corrupt sensor files."
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--train-args", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--progress-jsonl", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 1))
    parser.add_argument(
        "--verify-images",
        choices=("none", "current-cam-f0", "all-history"),
        default="current-cam-f0",
        help="Which JPEG files to open with PIL.Image.verify(). Existence is always checked for all history cameras.",
    )
    parser.add_argument(
        "--alternate-root",
        action="append",
        type=Path,
        default=[],
        help="Alternate sensor blob root to probe for good replacements.",
    )
    parser.add_argument("--max-logs", type=int, default=None)
    return parser.parse_args()


def iter_cache_index(cache_root: Path) -> Iterable[Dict[str, Any]]:
    if (cache_root / "index.jsonl").is_file():
        index_paths = [cache_root / "index.jsonl"]
    else:
        index_paths = sorted(cache_root.glob("*/index.jsonl"))
    for index_path in index_paths:
        with index_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def load_cache_sets(cache_root: Path) -> Tuple[Set[str], Set[str], int]:
    sample_tokens: Set[str] = set()
    log_names: Set[str] = set()
    rows = 0
    for record in iter_cache_index(cache_root):
        rows += 1
        token = record.get("sample_token")
        if token is not None:
            sample_tokens.add(str(token))
        log_name = record.get("log_name")
        if log_name is not None:
            log_names.add(str(log_name))
    return sample_tokens, log_names, rows


def init_worker(
    sample_tokens: Set[str],
    log_root: str,
    blob_root: str,
    num_history_frames: int,
    num_frames: int,
    frame_interval: int,
    has_route: bool,
    verify_images: str,
) -> None:
    global _SAMPLE_TOKENS, _LOG_ROOT, _BLOB_ROOT, _NUM_HISTORY_FRAMES, _NUM_FRAMES
    global _FRAME_INTERVAL, _HAS_ROUTE, _VERIFY_IMAGES
    _SAMPLE_TOKENS = sample_tokens
    _LOG_ROOT = Path(log_root)
    _BLOB_ROOT = Path(blob_root)
    _NUM_HISTORY_FRAMES = num_history_frames
    _NUM_FRAMES = num_frames
    _FRAME_INTERVAL = frame_interval
    _HAS_ROUTE = has_route
    _VERIFY_IMAGES = verify_images


def split_windows(frames: List[Dict[str, Any]]) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(frames), _FRAME_INTERVAL):
        window = frames[start : start + _NUM_FRAMES]
        if len(window) == _NUM_FRAMES:
            yield window


def cam_rel_path(frame: Dict[str, Any], camera_key: str) -> Optional[str]:
    for raw_key, camera in frame.get("cams", {}).items():
        if raw_key.lower() == camera_key:
            rel_path = camera.get("data_path")
            return str(rel_path) if rel_path is not None else None
    return None


def all_camera_rel_paths(frame: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    for raw_key, camera in frame.get("cams", {}).items():
        rel_path = camera.get("data_path")
        if rel_path is not None:
            yield raw_key.lower(), str(rel_path)


def check_image(path: Path, *, verify: bool) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return {"reason": "missing_file", "path": str(path)}
    try:
        size = path.stat().st_size
    except OSError as exc:
        return {"reason": type(exc).__name__, "path": str(path), "error": str(exc)}
    if size <= 0:
        return {"reason": "empty_file", "path": str(path), "file_size": size}
    if verify:
        try:
            from PIL import Image

            with Image.open(path) as image:
                image.verify()
        except Exception as exc:  # PIL raises several concrete image exceptions.
            return {
                "reason": "image_verify_failed",
                "path": str(path),
                "file_size": size,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            }
    return None


def parse_pcd_header(path: Path) -> Dict[str, Any]:
    try:
        file_size = path.stat().st_size
    except OSError as exc:
        return {"ok": False, "reason": type(exc).__name__, "path": str(path), "error": str(exc)}

    with path.open("rb") as f:
        header_bytes = 0
        header_lines: List[str] = []
        data_type = ""
        while True:
            line = f.readline()
            if not line:
                return {
                    "ok": False,
                    "reason": "missing_DATA_header",
                    "path": str(path),
                    "file_size": file_size,
                    "header_bytes": header_bytes,
                }
            header_bytes += len(line)
            text = line.decode("latin1", errors="replace").strip()
            header_lines.append(text)
            if text.startswith("DATA"):
                parts = text.split(maxsplit=1)
                data_type = parts[1] if len(parts) > 1 else ""
                break

    meta: Dict[str, List[str]] = {}
    for text in header_lines:
        parts = text.split()
        if parts:
            meta[parts[0].upper()] = parts[1:]

    if data_type != "binary":
        return {
            "ok": True,
            "reason": f"unchecked_DATA_{data_type}",
            "path": str(path),
            "file_size": file_size,
            "header_bytes": header_bytes,
        }

    try:
        sizes = [int(item) for item in meta.get("SIZE", [])]
        counts = [int(item) for item in meta.get("COUNT", [])] or [1] * len(sizes)
        points = int((meta.get("POINTS") or [0])[0])
        if points <= 0 and "WIDTH" in meta:
            width = int(meta["WIDTH"][0])
            height = int((meta.get("HEIGHT") or [1])[0])
            points = width * height
    except (TypeError, ValueError) as exc:
        return {
            "ok": False,
            "reason": "bad_header",
            "path": str(path),
            "file_size": file_size,
            "header_bytes": header_bytes,
            "error": str(exc),
        }

    if not sizes or not counts or points <= 0:
        return {
            "ok": False,
            "reason": "bad_header",
            "path": str(path),
            "file_size": file_size,
            "header_bytes": header_bytes,
        }

    point_step = sum(size * count for size, count in zip(sizes, counts))
    expected_data_bytes = points * point_step
    actual_data_bytes = max(0, file_size - header_bytes)
    ok = actual_data_bytes >= expected_data_bytes
    return {
        "ok": ok,
        "reason": "ok" if ok else "incomplete_stream",
        "path": str(path),
        "file_size": file_size,
        "header_bytes": header_bytes,
        "expected_data_bytes": expected_data_bytes,
        "actual_data_bytes": actual_data_bytes,
        "points": points,
        "point_step": point_step,
    }


def check_lidar(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return {"reason": "missing_file", "path": str(path)}
    result = parse_pcd_header(path)
    if result.get("ok"):
        return None
    result.pop("ok", None)
    return result


def audit_log(log_name: str) -> Dict[str, Any]:
    started = time.time()
    log_path = _LOG_ROOT / f"{log_name}.pkl"
    result: Dict[str, Any] = {
        "log_name": log_name,
        "tokens": 0,
        "history_camera_files": 0,
        "verified_image_files": 0,
        "history_lidar_files": 0,
        "bad_images": [],
        "bad_lidar": [],
        "bad_log": None,
        "elapsed_sec": None,
    }
    if not log_path.is_file():
        result["bad_log"] = {"reason": "missing_pkl", "path": str(log_path)}
        result["elapsed_sec"] = round(time.time() - started, 3)
        return result

    try:
        with log_path.open("rb") as f:
            frames = pickle.load(f)
    except Exception as exc:
        result["bad_log"] = {
            "reason": "pkl_load_failed",
            "path": str(log_path),
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }
        result["elapsed_sec"] = round(time.time() - started, 3)
        return result

    history_camera_paths: Dict[str, Set[str]] = {}
    current_cam_f0_paths: Set[str] = set()
    history_lidar_paths: Set[str] = set()

    for window in split_windows(frames):
        current = window[_NUM_HISTORY_FRAMES - 1]
        if _HAS_ROUTE and len(current.get("roadblock_ids") or []) == 0:
            continue
        token = str(current.get("token"))
        if token not in _SAMPLE_TOKENS:
            continue
        result["tokens"] += 1
        current_cam_f0 = cam_rel_path(current, "cam_f0")
        if current_cam_f0 is not None:
            current_cam_f0_paths.add(current_cam_f0)

        for frame in window[:_NUM_HISTORY_FRAMES]:
            for camera_key, rel_path in all_camera_rel_paths(frame):
                history_camera_paths.setdefault(rel_path, set()).add(camera_key)
            lidar_path = frame.get("lidar_path")
            if lidar_path is not None:
                history_lidar_paths.add(str(lidar_path))

    verify_all_images = _VERIFY_IMAGES == "all-history"
    verify_current_images = _VERIFY_IMAGES == "current-cam-f0"
    for rel_path, camera_keys in sorted(history_camera_paths.items()):
        full_path = _BLOB_ROOT / rel_path
        should_verify = verify_all_images or (verify_current_images and rel_path in current_cam_f0_paths)
        issue = check_image(full_path, verify=should_verify)
        result["history_camera_files"] += 1
        if should_verify:
            result["verified_image_files"] += 1
        if issue is not None:
            issue.update({"rel_path": rel_path, "camera_keys": sorted(camera_keys)})
            result["bad_images"].append(issue)

    for rel_path in sorted(history_lidar_paths):
        full_path = _BLOB_ROOT / rel_path
        issue = check_lidar(full_path)
        result["history_lidar_files"] += 1
        if issue is not None:
            issue.update({"rel_path": rel_path})
            result["bad_lidar"].append(issue)

    result["elapsed_sec"] = round(time.time() - started, 3)
    return result


def alternate_path(root: Path, rel_path: str) -> Iterable[Path]:
    rel = Path(rel_path)
    yield root / rel
    yield root / "trainval" / rel


def annotate_alternates(issue: Dict[str, Any], alternate_roots: List[Path], kind: str) -> None:
    candidates: List[Dict[str, Any]] = []
    current_path = issue.get("path")
    rel_path = issue.get("rel_path")
    if not rel_path:
        issue["alternate_candidates"] = candidates
        return
    seen: Set[str] = set()
    for root in alternate_roots:
        for candidate in alternate_path(root, rel_path):
            key = str(candidate)
            if key in seen or key == current_path:
                continue
            seen.add(key)
            if not candidate.is_file():
                continue
            if kind == "lidar":
                check = parse_pcd_header(candidate)
                ok = bool(check.pop("ok", False))
                candidates.append({"path": key, "ok": ok, **check})
            else:
                check = check_image(candidate, verify=True)
                candidates.append({"path": key, "ok": check is None, **(check or {"reason": "ok"})})
    issue["alternate_candidates"] = candidates


def main() -> int:
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    if args.progress_jsonl is not None:
        args.progress_jsonl.parent.mkdir(parents=True, exist_ok=True)

    train_args = json.load(args.train_args.open("r", encoding="utf-8"))
    scene_filter = train_args["train_test_split"]["scene_filter"]
    num_history_frames = int(scene_filter.get("num_history_frames", 4))
    num_frames = num_history_frames + int(scene_filter.get("num_future_frames", 10))
    frame_interval = int(scene_filter.get("frame_interval") or num_frames)
    has_route = bool(scene_filter.get("has_route", True))
    log_root = Path(train_args["navsim_log_path"])
    blob_root = Path(train_args["sensor_blobs_path"])

    sample_tokens, cache_logs, cache_index_rows = load_cache_sets(args.cache_root)
    cfg_logs = set(scene_filter.get("log_names") or [])
    train_logs = set(train_args.get("train_logs") or [])
    val_logs = set(train_args.get("val_logs") or [])
    scan_logs = sorted(cache_logs)
    if args.max_logs is not None:
        scan_logs = scan_logs[: args.max_logs]

    actual_train_logs = (cfg_logs & train_logs & cache_logs) if cfg_logs else (train_logs & cache_logs)
    actual_val_logs = (cfg_logs & val_logs & cache_logs) if cfg_logs else (val_logs & cache_logs)

    workers = min(max(1, args.workers), len(scan_logs) or 1)
    completed = 0
    bad_logs: List[Dict[str, Any]] = []
    bad_images: List[Dict[str, Any]] = []
    bad_lidar: List[Dict[str, Any]] = []
    total_tokens = 0
    total_history_camera_files = 0
    total_verified_image_files = 0
    total_history_lidar_files = 0
    started = time.time()

    progress_file = args.progress_jsonl.open("w", encoding="utf-8") if args.progress_jsonl else None
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_worker,
            initargs=(
                sample_tokens,
                str(log_root),
                str(blob_root),
                num_history_frames,
                num_frames,
                frame_interval,
                has_route,
                args.verify_images,
            ),
        ) as executor:
            futures = {executor.submit(audit_log, log_name): log_name for log_name in scan_logs}
            for future in as_completed(futures):
                item = future.result()
                completed += 1
                split = "train" if item["log_name"] in actual_train_logs else (
                    "val" if item["log_name"] in actual_val_logs else "cache_only"
                )
                item["split"] = split
                total_tokens += int(item.get("tokens") or 0)
                total_history_camera_files += int(item.get("history_camera_files") or 0)
                total_verified_image_files += int(item.get("verified_image_files") or 0)
                total_history_lidar_files += int(item.get("history_lidar_files") or 0)
                if item.get("bad_log"):
                    bad = {"log_name": item["log_name"], "split": split, **item["bad_log"]}
                    bad_logs.append(bad)
                for issue in item.get("bad_images") or []:
                    bad_images.append({"log_name": item["log_name"], "split": split, **issue})
                for issue in item.get("bad_lidar") or []:
                    bad_lidar.append({"log_name": item["log_name"], "split": split, **issue})

                if progress_file is not None:
                    progress_file.write(
                        json.dumps(
                            {
                                "completed": completed,
                                "total": len(scan_logs),
                                "log_name": item["log_name"],
                                "split": split,
                                "tokens": item.get("tokens", 0),
                                "bad_log": bool(item.get("bad_log")),
                                "bad_images": len(item.get("bad_images") or []),
                                "bad_lidar": len(item.get("bad_lidar") or []),
                                "elapsed_sec": item.get("elapsed_sec"),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    progress_file.flush()
                if completed % 50 == 0 or item.get("bad_log") or item.get("bad_images") or item.get("bad_lidar"):
                    elapsed = time.time() - started
                    print(
                        f"progress {completed}/{len(scan_logs)} elapsed={elapsed:.1f}s "
                        f"bad_logs={len(bad_logs)} bad_images={len(bad_images)} bad_lidar={len(bad_lidar)}",
                        flush=True,
                    )
    finally:
        if progress_file is not None:
            progress_file.close()

    for issue in bad_images:
        annotate_alternates(issue, args.alternate_root, "image")
    for issue in bad_lidar:
        annotate_alternates(issue, args.alternate_root, "lidar")

    report = {
        "cache_root": str(args.cache_root),
        "train_args": str(args.train_args),
        "navsim_log_path": str(log_root),
        "sensor_blobs_path": str(blob_root),
        "num_history_frames": num_history_frames,
        "num_frames": num_frames,
        "frame_interval": frame_interval,
        "has_route": has_route,
        "verify_images": args.verify_images,
        "workers": workers,
        "cache_index_rows": cache_index_rows,
        "cache_sample_tokens": len(sample_tokens),
        "cache_log_count": len(cache_logs),
        "audited_log_count": len(scan_logs),
        "actual_train_log_count": len(actual_train_logs),
        "actual_val_log_count": len(actual_val_logs),
        "tokens_in_cache_after_log_scan": total_tokens,
        "history_camera_files_checked": total_history_camera_files,
        "history_camera_images_verified": total_verified_image_files,
        "history_lidar_pcd_checked": total_history_lidar_files,
        "bad_log_count": len(bad_logs),
        "bad_image_count": len(bad_images),
        "bad_lidar_count": len(bad_lidar),
        "bad_log_by_reason": dict(Counter(item.get("reason") for item in bad_logs)),
        "bad_image_by_reason": dict(Counter(item.get("reason") for item in bad_images)),
        "bad_lidar_by_reason": dict(Counter(item.get("reason") for item in bad_lidar)),
        "bad_logs": bad_logs,
        "bad_images": bad_images,
        "bad_lidar": bad_lidar,
        "elapsed_sec": round(time.time() - started, 3),
    }
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "audited_log_count",
        "tokens_in_cache_after_log_scan",
        "history_camera_files_checked",
        "history_camera_images_verified",
        "history_lidar_pcd_checked",
        "bad_log_count",
        "bad_image_count",
        "bad_lidar_count",
        "bad_log_by_reason",
        "bad_image_by_reason",
        "bad_lidar_by_reason",
        "elapsed_sec",
    )}, indent=2, sort_keys=True))
    print(f"OUT_JSON {args.out_json}")
    return 0 if not (bad_logs or bad_images or bad_lidar) else 1


if __name__ == "__main__":
    raise SystemExit(main())
