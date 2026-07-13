#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import atomic_torch_save, iter_index, load_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Last-VLA teacher trajectory cache into chunk samples.")
    parser.add_argument("--base-chunk-root", type=Path, required=True)
    parser.add_argument("--teacher-cache-root", type=Path, required=True)
    parser.add_argument("--output-chunk-root", type=Path, required=True)
    parser.add_argument("--chunk-name-pattern", default="train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*")
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--in-place", action="store_true")
    return parser.parse_args()


def load_teacher_index(root: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    index_path = root / "index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"Teacher cache index not found: {index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            path = Path(record["path"])
            mapping[str(record["sample_token"])] = path if path.is_absolute() else root / path
    return mapping


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def chunk_dirs(root: Path, pattern: str) -> list[Path]:
    if (root / "index.jsonl").is_file():
        return [root]
    dirs: list[Path] = []
    seen = set()
    for item in str(pattern).split(","):
        item = item.strip()
        if not item:
            continue
        for path in sorted(root.glob(item)):
            if path.is_dir() and (path / "index.jsonl").is_file() and path not in seen:
                dirs.append(path)
                seen.add(path)
    if not dirs:
        raise FileNotFoundError(f"No chunk directories matching {pattern!r} under {root}")
    return dirs


def main() -> int:
    args = parse_args()
    if args.in_place:
        args.output_chunk_root = args.base_chunk_root
    elif args.output_chunk_root.resolve() == args.base_chunk_root.resolve():
        raise ValueError("Refusing in-place merge without --in-place.")
    teacher_index = load_teacher_index(args.teacher_cache_root)
    merged = missing = 0
    for chunk_dir in chunk_dirs(args.base_chunk_root, args.chunk_name_pattern):
        out_chunk = args.output_chunk_root / chunk_dir.name
        out_chunk.mkdir(parents=True, exist_ok=True)
        index_records = []
        for record in iter_index(chunk_dir):
            src_path = Path(record["path"])
            if not src_path.is_absolute():
                src_path = chunk_dir / src_path
            token = str(record.get("sample_token") or src_path.stem)
            rel = Path(record["path"]) if not Path(record["path"]).is_absolute() else Path("samples") / src_path.name
            dst_path = out_chunk / rel
            if token in teacher_index:
                sample = load_sample(src_path)
                teacher = load_sample(teacher_index[token])
                updated = dict(sample)
                for key in (
                    "teacher_trajectory",
                    "teacher_trajectory_norm",
                    "teacher_score",
                    "gt_score",
                    "oracle_best_of_k_score",
                    "candidate_scores",
                    "candidate_count",
                    "teacher_source",
                    "score_mode",
                    "pdm_components",
                    "gt_pdm_components",
                    "candidate_pdm_components",
                    "candidate_trajectories",
                ):
                    if key in teacher:
                        updated[key] = teacher[key]
                atomic_torch_save(updated, dst_path)
                merged += 1
            else:
                copy_or_link(src_path, dst_path, args.copy_mode)
                missing += 1
            out_record = dict(record)
            out_record["path"] = str(rel)
            index_records.append(out_record)
        with (out_chunk / "index.jsonl").open("w", encoding="utf-8") as f:
            for item in index_records:
                f.write(json.dumps(item, sort_keys=True) + "\n")
    args.output_chunk_root.mkdir(parents=True, exist_ok=True)
    (args.output_chunk_root / "last_vla_teacher_merge_summary.json").write_text(
        json.dumps({"merged": merged, "missing": missing}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"merged": merged, "missing": missing}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
