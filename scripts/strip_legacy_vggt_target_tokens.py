#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import iter_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove legacy VGGT target-token fields from chunk-cache samples.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--chunk-name-pattern", default=None)
    parser.add_argument("--key", default="vggt_target_tokens")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--update-metadata", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def chunk_dirs(cache_root: Path, pattern: Optional[str]) -> List[Path]:
    if (cache_root / "index.jsonl").is_file():
        return [cache_root]
    if pattern:
        dirs: List[Path] = []
        seen = set()
        for item in pattern.split(","):
            item = item.strip()
            if not item:
                continue
            for path in sorted(cache_root.glob(item)):
                if path.is_dir() and (path / "index.jsonl").is_file() and path not in seen:
                    dirs.append(path)
                    seen.add(path)
        return dirs
    return sorted(path for path in cache_root.iterdir() if path.is_dir() and (path / "index.jsonl").is_file())


def resolve_sample_path(chunk_dir: Path, record: Dict[str, Any]) -> Path:
    raw = Path(record["path"])
    if raw.is_file():
        return raw
    if not raw.is_absolute():
        candidate = chunk_dir / raw
        if candidate.is_file():
            return candidate
    return raw


def collect_sample_paths(cache_root: Path, pattern: Optional[str]) -> List[Path]:
    paths: List[Path] = []
    for chunk_dir in chunk_dirs(cache_root, pattern):
        for record in iter_index(chunk_dir):
            paths.append(resolve_sample_path(chunk_dir, record))
    return paths


def strip_key(path: Path, key: str, dry_run: bool) -> Tuple[str, bool, str]:
    try:
        try:
            sample = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            sample = torch.load(path, map_location="cpu")
        if not isinstance(sample, dict) or key not in sample:
            return str(path), False, ""
        if dry_run:
            return str(path), True, ""
        del sample[key]
        tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        torch.save(sample, tmp)
        os.replace(tmp, path)
        return str(path), True, ""
    except Exception as exc:
        try:
            tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return str(path), False, f"{type(exc).__name__}: {exc}"


def update_metadata(cache_root: Path, pattern: Optional[str], key: str, removed_count: int, dry_run: bool) -> None:
    if dry_run:
        return
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for chunk_dir in chunk_dirs(cache_root, pattern):
        metadata_path = chunk_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        metadata[f"{key}_removed"] = True
        metadata[f"{key}_removed_at"] = now
        metadata[f"{key}_removed_total_for_cache"] = int(removed_count)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    paths = collect_sample_paths(args.cache_root, args.chunk_name_pattern)
    if not paths:
        raise FileNotFoundError(f"No samples found under {args.cache_root}")

    removed = 0
    errors: List[str] = []
    path_iter = iter(paths)
    max_pending = max(args.workers * 4, args.workers)
    pending: Dict[Future[Tuple[str, bool, str]], Path] = {}

    def submit_next(executor: ProcessPoolExecutor) -> bool:
        try:
            path = next(path_iter)
        except StopIteration:
            return False
        future = executor.submit(strip_key, path, args.key, bool(args.dry_run))
        pending[future] = path
        return True

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for _ in range(min(max_pending, len(paths))):
            submit_next(executor)

        idx = 0
        next_report = 1000
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                path_hint = pending.pop(future)
                idx += 1
                try:
                    path, did_remove, error = future.result()
                except Exception as exc:
                    path = str(path_hint)
                    did_remove = False
                    error = f"{type(exc).__name__}: {exc}"
                if did_remove:
                    removed += 1
                if error:
                    errors.append(f"{path}:{error}")
                submit_next(executor)
            if idx >= next_report or idx == len(paths):
                print(f"processed={idx}/{len(paths)} removed={removed} errors={len(errors)}", flush=True)
                while next_report <= idx:
                    next_report += 1000

    if args.update_metadata:
        update_metadata(args.cache_root, args.chunk_name_pattern, args.key, removed, bool(args.dry_run))

    report = {
        "cache_root": str(args.cache_root),
        "chunk_name_pattern": args.chunk_name_pattern,
        "dry_run": bool(args.dry_run),
        "key": args.key,
        "num_errors": len(errors),
        "processed": len(paths),
        "removed": removed,
        "errors": errors[:200],
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
