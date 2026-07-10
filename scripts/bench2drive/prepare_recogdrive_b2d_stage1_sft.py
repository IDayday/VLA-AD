#!/usr/bin/env python3
"""Prepare contract-aligned Bench2Drive Stage-1 trajectory SFT data.

The released Bench2Drive trajectory JSONL uses six cameras, no pose history,
and six targets in a different coordinate convention. Closed-loop inference in
this repository consumes one front image, four relative ego poses, a 3-way
command, and predicts eight relative poses. This builder derives SFT examples
from the raw annotations with exactly that online contract.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.recogdrive_backbone import (  # noqa: E402
    build_recogdrive_planning_question,
    format_recogdrive_trajectory_answer,
    resolve_recogdrive_system_message,
)
from scripts.build_bench2drive_recogdrive_chunk_cache import (  # noqa: E402
    annotation_paths,
    clip_dirs,
    command_one_hot,
    pose_of,
    read_annotation,
    relative_poses,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/data/Bench2Drive-Base"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--history-frames", type=int, default=4)
    parser.add_argument("--future-frames", type=int, default=8)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16, help="Parallel clip readers; output order remains deterministic.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def stable_clip_split(
    clip_names: Iterable[str],
    *,
    validation_fraction: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Return deterministic, exactly-sized train/validation clip partitions."""
    names = sorted(set(clip_names))
    if len(names) < 2:
        raise ValueError("At least two Bench2Drive clips are required for a train/validation split")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be strictly between 0 and 1")
    validation_count = min(len(names) - 1, max(1, round(len(names) * validation_fraction)))
    ranked = sorted(
        names,
        key=lambda name: (hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest(), name),
    )
    validation = set(ranked[:validation_count])
    return [name for name in names if name not in validation], sorted(validation)


def _sample_row(
    *,
    clip_dir: Path,
    annotations: Mapping[int, Dict[str, Any]],
    current_idx: int,
    frame_step: int,
    history_frames: int,
    future_frames: int,
    data_root: Path,
    system_message: str,
) -> Dict[str, Any]:
    history_indices = [current_idx - frame_step * offset for offset in reversed(range(history_frames))]
    future_indices = [current_idx + frame_step * offset for offset in range(1, future_frames + 1)]
    origin = pose_of(annotations[current_idx])
    history = relative_poses(origin, np.stack([pose_of(annotations[index]) for index in history_indices]))
    future = relative_poses(origin, np.stack([pose_of(annotations[index]) for index in future_indices]))
    command = command_one_hot(annotations[current_idx])
    frame_name = f"{current_idx:05d}"
    image_path = clip_dir / "camera" / "rgb_front" / f"{frame_name}.jpg"
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing Bench2Drive front image: {image_path}")
    try:
        relative_image_path = image_path.resolve().relative_to(data_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Image {image_path} is outside data root {data_root}") from exc
    sample_token = f"{clip_dir.name}_{frame_name}"
    return {
        "id": sample_token,
        "image": relative_image_path,
        "conversations": [
            {"from": "system", "value": system_message},
            {"from": "human", "value": build_recogdrive_planning_question(history, command)},
            {"from": "gpt", "value": format_recogdrive_trajectory_answer(future)},
        ],
        "sample_token": sample_token,
        "clip": clip_dir.name,
        "frame_index": current_idx,
        "coordinate_policy": "global_se2_to_current_ego_relative",
    }


def collect_clip_rows(
    clip_dir: Path,
    *,
    data_root: Path,
    frame_step: int,
    sample_stride: int,
    history_frames: int,
    future_frames: int,
    system_message: str,
) -> Tuple[List[Dict[str, Any]], int, int]:
    paths = annotation_paths(clip_dir)
    min_current = frame_step * (history_frames - 1)
    max_current = len(paths) - frame_step * future_frames - 1
    current_indices = list(range(min_current, max_current + 1, sample_stride))
    needed_indices = set()
    for current_idx in current_indices:
        needed_indices.update(current_idx - frame_step * offset for offset in reversed(range(history_frames)))
        needed_indices.update(current_idx + frame_step * offset for offset in range(1, future_frames + 1))
    annotations: Dict[int, Dict[str, Any]] = {}
    invalid_annotations = 0
    for index in sorted(needed_indices):
        try:
            annotations[index] = read_annotation(paths[index])
        except (OSError, EOFError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            invalid_annotations += 1
    rows: List[Dict[str, Any]] = []
    skipped_windows = 0
    for current_idx in current_indices:
        try:
            rows.append(
                _sample_row(
                    clip_dir=clip_dir,
                    annotations=annotations,
                    current_idx=current_idx,
                    frame_step=frame_step,
                    history_frames=history_frames,
                    future_frames=future_frames,
                    data_root=data_root,
                    system_message=system_message,
                )
            )
        except (FileNotFoundError, IndexError, KeyError, TypeError, ValueError):
            skipped_windows += 1
    return rows, skipped_windows, invalid_annotations


def _write_split(
    path: Path,
    *,
    clips: List[Path],
    data_root: Path,
    frame_step: int,
    sample_stride: int,
    history_frames: int,
    future_frames: int,
    system_message: str,
    max_samples: Optional[int],
    workers: int,
) -> Tuple[int, int, int]:
    count = 0
    skipped_windows = 0
    invalid_annotations = 0
    temporary = path.with_suffix(path.suffix + ".tmp")
    def read_clip(clip: Path) -> Tuple[List[Dict[str, Any]], int, int]:
        return collect_clip_rows(
            clip,
            data_root=data_root,
            frame_step=frame_step,
            sample_stride=sample_stride,
            history_frames=history_frames,
            future_frames=future_frames,
            system_message=system_message,
        )

    executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
    if workers > 1:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        clip_rows = executor.map(read_clip, clips)
    else:
        clip_rows = map(read_clip, clips)
    with temporary.open("w", encoding="utf-8") as output:
        for rows, clip_skipped_windows, clip_invalid_annotations in clip_rows:
            skipped_windows += clip_skipped_windows
            invalid_annotations += clip_invalid_annotations
            for row in rows:
                if max_samples is not None and count >= max_samples:
                    break
                output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
            if max_samples is not None and count >= max_samples:
                break
        output.flush()
        os.fsync(output.fileno())
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)
    temporary.replace(path)
    return count, skipped_windows, invalid_annotations


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def prepare_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    if args.frame_step <= 0 or args.sample_stride <= 0:
        raise ValueError("--frame-step and --sample-stride must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.history_frames != 4 or args.future_frames != 8:
        raise ValueError("The online ReCogDrive contract requires 4 history and 8 future frames")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_clips = clip_dirs(args.data_root, None, args.max_clips)
    train_names, val_names = stable_clip_split(
        (path.name for path in all_clips),
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    by_name = {path.name: path for path in all_clips}
    train_clips = [by_name[name] for name in train_names]
    val_clips = [by_name[name] for name in val_names]
    system_message = resolve_recogdrive_system_message("bench2drive")

    train_path = args.output_dir / "train.jsonl"
    val_path = args.output_dir / "val.jsonl"
    train_count, train_skipped, train_invalid_annotations = _write_split(
        train_path,
        clips=train_clips,
        data_root=args.data_root,
        frame_step=args.frame_step,
        sample_stride=args.sample_stride,
        history_frames=args.history_frames,
        future_frames=args.future_frames,
        system_message=system_message,
        max_samples=args.max_train_samples,
        workers=args.workers,
    )
    val_count, val_skipped, val_invalid_annotations = _write_split(
        val_path,
        clips=val_clips,
        data_root=args.data_root,
        frame_step=args.frame_step,
        sample_stride=args.sample_stride,
        history_frames=args.history_frames,
        future_frames=args.future_frames,
        system_message=system_message,
        max_samples=args.max_val_samples,
        workers=args.workers,
    )
    if train_count == 0 or val_count == 0:
        raise RuntimeError(f"Prepared an empty split: train={train_count}, val={val_count}")

    _write_text(args.output_dir / "train_clips.txt", "\n".join(train_names) + "\n")
    _write_text(args.output_dir / "val_clips.txt", "\n".join(val_names) + "\n")
    root = str(args.data_root.resolve())
    train_meta = {
        "Bench2Drive_Runtime_Trajectory_Train": {
            "root": root,
            "annotation": str(train_path.resolve()),
            "data_augment": True,
            "repeat_time": 1,
            "length": train_count,
        }
    }
    val_meta = {
        "Bench2Drive_Runtime_Trajectory_Val": {
            "root": root,
            "annotation": str(val_path.resolve()),
            "data_augment": False,
            "repeat_time": 1,
            "length": val_count,
        }
    }
    _write_text(args.output_dir / "train_meta.json", json.dumps(train_meta, indent=2, sort_keys=True) + "\n")
    _write_text(args.output_dir / "val_meta.json", json.dumps(val_meta, indent=2, sort_keys=True) + "\n")

    manifest = {
        "dataset": "Bench2Drive",
        "recipe": "runtime_contract_front_camera_trajectory_sft_v1",
        "data_root": root,
        "system_prompt_profile": "bench2drive",
        "split_unit": "clip",
        "split_algorithm": "sha256(seed:clip), exact validation count",
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "num_clips": len(all_clips),
        "num_train_clips": len(train_names),
        "num_val_clips": len(val_names),
        "num_train_samples": train_count,
        "num_val_samples": val_count,
        "num_skipped_windows": train_skipped + val_skipped,
        "num_train_skipped_windows": train_skipped,
        "num_val_skipped_windows": val_skipped,
        "num_invalid_annotations": train_invalid_annotations + val_invalid_annotations,
        "frame_step": args.frame_step,
        "sample_stride": args.sample_stride,
        "history_frames": args.history_frames,
        "future_frames": args.future_frames,
        "workers": args.workers,
        "coordinate_policy": "global_se2_to_current_ego_relative",
        "train_jsonl_sha256": _sha256(train_path),
        "val_jsonl_sha256": _sha256(val_path),
    }
    _write_text(args.output_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    manifest = prepare_dataset(parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
