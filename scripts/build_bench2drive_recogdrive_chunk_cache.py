#!/usr/bin/env python3
"""Build the closest-public ReCogDrive Stage2 cache from raw Bench2Drive clips."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.bench2drive_contract import (  # noqa: E402
    BENCH2DRIVE_CAMERA_ORDER,
    BENCH2DRIVE_CONTRACT_ID,
    bench2drive_camera_paths,
    build_bench2drive_stage1_question,
    relative_lidar_poses,
)
from navsim.agents.recogdrive.expert_cache import (  # noqa: E402
    CHUNK_VERSION,
    atomic_torch_save,
    write_index,
    write_json,
)


class InvalidPoseWindow(ValueError):
    """A raw window whose required pose/status values are not finite."""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/data/Bench2Drive-Base"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--clip-start", type=int, default=0)
    parser.add_argument("--clip-stop", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--clip-name", action="append", default=None)
    parser.add_argument("--clip-list", type=Path, default=None)
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--history-frames", type=int, default=4)
    parser.add_argument("--future-frames", type=int, default=6)
    parser.add_argument(
        "--hidden-source",
        choices=("recogdrive-vlm",),
        default="recogdrive-vlm",
        help="Synthetic/pre-Stage1 hidden states are intentionally unsupported.",
    )
    parser.add_argument(
        "--recogdrive-vlm-path",
        type=Path,
        default=Path("checkpoints/recogdrive/ReCogDrive-VLM-2B"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--system-prompt-profile",
        choices=("bench2drive",),
        default="bench2drive",
    )
    parser.add_argument(
        "--cache-hidden-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Limit PyTorch intra-op threads for multi-worker-per-GPU cache generation.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args(argv)


def clip_dirs(
    data_root: Path,
    clip_names: Optional[List[str]],
    max_clips: Optional[int],
    clip_start: int = 0,
    clip_stop: Optional[int] = None,
    clip_list: Optional[Path] = None,
) -> List[Path]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Bench2Drive data root does not exist: {data_root}")
    if clip_start < 0:
        raise ValueError("--clip-start must be non-negative")
    if clip_stop is not None and clip_stop < clip_start:
        raise ValueError("--clip-stop must be >= --clip-start")
    if clip_names and clip_list is not None:
        raise ValueError("Use only one of --clip-name or --clip-list")

    requested = set(clip_names or [])
    if clip_list is not None:
        if not clip_list.is_file():
            raise FileNotFoundError(f"Bench2Drive clip list does not exist: {clip_list}")
        requested = {
            line.strip()
            for line in clip_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        if not requested:
            raise ValueError(f"Bench2Drive clip list is empty: {clip_list}")

    directories = [
        child
        for child in sorted(data_root.iterdir())
        if child.is_dir()
        and child.name != "maps"
        and (child / "anno").is_dir()
        and (not requested or child.name in requested)
    ]
    if requested:
        missing = sorted(requested - {path.name for path in directories})
        if missing:
            raise FileNotFoundError(f"Requested Bench2Drive clips not found: {missing}")
    directories = directories[clip_start:clip_stop]
    if max_clips is not None:
        directories = directories[:max_clips]
    if not directories:
        raise FileNotFoundError(f"No Bench2Drive clip directories found under {data_root}")
    return directories


def annotation_paths(clip_dir: Path) -> List[Path]:
    return sorted((clip_dir / "anno").glob("*.json.gz"))


def read_annotation(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Annotation is not an object: {path}")
    return value


# Compatibility helpers for the explicitly retired front-only Stage1 data
# builder.  Formal Stage2 cache generation never calls these global-x/y pose
# functions; it uses relative_lidar_poses above.
def pose_of(annotation: Dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [float(annotation["x"]), float(annotation["y"]), float(annotation["theta"])],
        dtype=np.float64,
    )


def relative_poses(origin: np.ndarray, poses: np.ndarray) -> np.ndarray:
    theta = -float(origin[2])
    rotation = np.asarray(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=np.float64,
    )
    result = np.asarray(poses, dtype=np.float64).copy()
    result -= np.asarray(origin, dtype=np.float64).reshape(1, 3)
    result[:, :2] = result[:, :2] @ rotation.T
    result[:, 2] = np.arctan2(np.sin(result[:, 2]), np.cos(result[:, 2]))
    return result.astype(np.float32)


def command_value(annotation: Dict[str, Any]) -> int:
    return int(
        annotation.get(
            "command_near",
            annotation.get("next_command", annotation.get("command_far", 4)),
        )
    )


def command_one_hot(annotation: Dict[str, Any]) -> np.ndarray:
    command = command_value(annotation)
    one_hot = np.zeros(3, dtype=np.float32)
    if command in {1, 5}:
        one_hot[0] = 1.0
    elif command in {2, 6}:
        one_hot[2] = 1.0
    else:
        one_hot[1] = 1.0
    return one_hot


def status_feature(annotation: Dict[str, Any], high_command: np.ndarray) -> np.ndarray:
    speed = float(annotation.get("speed", 0.0))
    acceleration_raw = annotation.get("acceleration") or [0.0, 0.0, 0.0]
    acceleration = np.asarray(acceleration_raw[:3], dtype=np.float32)
    if acceleration.shape != (3,):
        acceleration = np.pad(acceleration.reshape(-1)[:3], (0, max(0, 3 - acceleration.size)))
    # Keep the raw planner-state x/y axes; the released text prompt applies its
    # separate y-sign conversion when it is built below.  The z component is
    # gravity-dominated and is neutralized for planner state.
    acceleration[2] = 0.0
    status = np.concatenate(
        [high_command.astype(np.float32), np.array([speed, 0.0], dtype=np.float32), acceleration]
    )
    if status.shape != (8,) or not np.isfinite(status).all():
        raise InvalidPoseWindow("non-finite or malformed ego status")
    return status.astype(np.float32)


class Bench2DriveVLMEncoder:
    """Small cache-only encoder that does not import the NAVSIM/nuPlan stack."""

    def __init__(self, checkpoint_path: Path, device: str) -> None:
        from navsim.agents.recogdrive.recogdrive_backbone import RecogDriveBackbone

        self.device = torch.device(device)
        self.backbone = RecogDriveBackbone(
            model_type="internvl",
            checkpoint_path=str(checkpoint_path),
            device=device,
            system_prompt_profile="bench2drive",
        )

    def compute_bench2drive_multiview_features(self, **kwargs: Any) -> Dict[str, torch.Tensor]:
        from navsim.agents.recogdrive.utils.internvl_preprocess import load_image

        image_paths = [Path(path) for path in kwargs["image_paths"]]
        if len(image_paths) != len(BENCH2DRIVE_CAMERA_ORDER):
            raise ValueError(f"Expected six image paths, got {len(image_paths)}")
        patch_tensors = []
        patch_counts = []
        for camera_name, image_path in zip(BENCH2DRIVE_CAMERA_ORDER, image_paths):
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing {camera_name} image: {image_path}")
            patches = load_image(str(image_path), max_num=2)
            patch_tensors.append(patches)
            patch_counts.append(int(patches.shape[0]))
        question = build_bench2drive_stage1_question(
            speed=float(kwargs["speed"]),
            acceleration_xy=kwargs["acceleration_xy"],
            command=kwargs["command"],
        )
        with torch.inference_mode():
            outputs = self.backbone(
                torch.cat(patch_tensors, dim=0).to(self.device),
                [question],
                num_patches_list=[patch_counts],
            )
            hidden = self.backbone.active_hidden_state(outputs).detach().cpu()
        return {
            "last_hidden_state": hidden,
            "image_patch_counts": torch.tensor(patch_counts, dtype=torch.int16),
        }


def build_vlm_feature_builder(args: argparse.Namespace):
    if not args.recogdrive_vlm_path.exists():
        raise FileNotFoundError(f"ReCogDrive VLM path not found: {args.recogdrive_vlm_path}")
    return Bench2DriveVLMEncoder(args.recogdrive_vlm_path, args.device)


def _cache_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def sample_payload(
    *,
    clip_dir: Path,
    annotations: List[Dict[str, Any]],
    current_idx: int,
    frame_step: int,
    history_frames: int,
    future_frames: int,
    cache_hidden_dtype: str,
    vlm_feature_builder: Any,
) -> Dict[str, Any]:
    history_indices = [current_idx - frame_step * offset for offset in reversed(range(history_frames))]
    future_indices = [current_idx + frame_step * offset for offset in range(1, future_frames + 1)]
    current = annotations[current_idx]
    try:
        history = relative_lidar_poses(current, [annotations[index] for index in history_indices])
        future = relative_lidar_poses(current, [annotations[index] for index in future_indices])
    except (KeyError, ValueError, np.linalg.LinAlgError) as exc:
        raise InvalidPoseWindow(str(exc)) from exc

    high_command = command_one_hot(current)
    status = status_feature(current, high_command)
    image_paths = bench2drive_camera_paths(clip_dir, current_idx)
    missing_images = [str(path) for path in image_paths if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing six-view images: {missing_images}")

    features = vlm_feature_builder.compute_bench2drive_multiview_features(
        image_paths=image_paths,
        history_trajectory=torch.from_numpy(history),
        high_command_one_hot=torch.from_numpy(high_command),
        status_feature=torch.from_numpy(status),
        speed=float(current.get("speed", 0.0)),
        acceleration_xy=[
            float((current.get("acceleration") or [0.0, 0.0])[0]),
            -float((current.get("acceleration") or [0.0, 0.0])[1]),
        ],
        command=command_value(current),
    )
    last_hidden_state = features["last_hidden_state"].detach().to(dtype=_cache_dtype(cache_hidden_dtype)).cpu()
    patch_counts = features.get("image_patch_counts")
    sample_token = f"{clip_dir.name}_{current_idx:05d}"
    payload = {
        "scene_token": clip_dir.name,
        "sample_token": sample_token,
        "log_name": clip_dir.name,
        "history_trajectory": torch.from_numpy(history).float(),
        "high_command_one_hot": torch.from_numpy(high_command).float(),
        "status_feature": torch.from_numpy(status).float(),
        "trajectory": torch.from_numpy(future).float(),
        "last_hidden_state": last_hidden_state,
        "meta": {
            "dataset": "Bench2Drive",
            "contract_id": BENCH2DRIVE_CONTRACT_ID,
            "clip": clip_dir.name,
            "frame_index": current_idx,
            "frame_step": frame_step,
            "camera_order": list(BENCH2DRIVE_CAMERA_ORDER),
            "image_paths": [str(path) for path in image_paths],
            "image_patch_counts": patch_counts.tolist() if isinstance(patch_counts, torch.Tensor) else None,
            "raw_command_near": current.get("command_near"),
            "raw_command_far": current.get("command_far"),
            "next_command": current.get("next_command"),
            "hidden_source": "recogdrive_vlm",
            "hidden_tokens": int(last_hidden_state.shape[0]),
            "vlm_feature_dim": int(last_hidden_state.shape[-1]),
            "coordinate_policy": "current_world2lidar_x_forward_y_lateral_relative_yaw",
            "trajectory_interval_seconds": 0.5,
            "trajectory_horizon_seconds": 3.0,
        },
    }
    for key in (
        "history_trajectory",
        "high_command_one_hot",
        "status_feature",
        "trajectory",
        "last_hidden_state",
    ):
        value = payload[key]
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value.float()).all():
            raise RuntimeError(f"Generated non-finite cache tensor {key!r} for {sample_token}")
    return payload


def build_cache(args: argparse.Namespace, *, vlm_feature_builder: Any = None) -> Dict[str, Any]:
    if args.frame_step <= 0 or args.sample_stride <= 0:
        raise ValueError("--frame-step and --sample-stride must be positive")
    if args.history_frames != 4:
        raise ValueError("The closest-public ReCogDrive contract requires four history poses")
    if args.future_frames != 6:
        raise ValueError("The released Bench2Drive trajectory contract requires six future poses")
    if args.frame_step != 5 or args.sample_stride != 1:
        raise ValueError(
            "The formal contract requires consecutive 10 Hz anchors and 0.5-second history/future spacing"
        )
    if args.cpu_threads is not None and args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive when supplied")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    if vlm_feature_builder is None:
        vlm_feature_builder = build_vlm_feature_builder(args)

    records: List[Dict[str, Any]] = []
    skipped = 0
    scanned = 0
    hidden_token_counts: List[int] = []
    hidden_feature_dims: List[int] = []
    written_by_clip: Dict[str, int] = {}
    selected_clip_dirs = clip_dirs(
        args.data_root,
        args.clip_name,
        args.max_clips,
        args.clip_start,
        args.clip_stop,
        clip_list=args.clip_list,
    )
    min_current = args.frame_step * (args.history_frames - 1)
    for clip_dir in selected_clip_dirs:
        paths = annotation_paths(clip_dir)
        if not paths:
            continue
        annotations = [read_annotation(path) for path in paths]
        max_current = len(annotations) - args.frame_step * args.future_frames - 1
        if max_current < min_current:
            continue
        for current_idx in range(min_current, max_current + 1, args.sample_stride):
            if args.max_samples is not None and len(records) >= args.max_samples:
                break
            scanned += 1
            try:
                payload = sample_payload(
                    clip_dir=clip_dir,
                    annotations=annotations,
                    current_idx=current_idx,
                    frame_step=args.frame_step,
                    history_frames=args.history_frames,
                    future_frames=args.future_frames,
                    cache_hidden_dtype=args.cache_hidden_dtype,
                    vlm_feature_builder=vlm_feature_builder,
                )
            except InvalidPoseWindow as exc:
                skipped += 1
                if args.log_every and skipped <= 20:
                    print(f"skip invalid pose window clip={clip_dir.name} idx={current_idx}: {exc}", flush=True)
                continue

            sample_path = samples_dir / f"{payload['sample_token']}.pt"
            atomic_torch_save(payload, sample_path)
            records.append(
                {
                    "path": str(sample_path.relative_to(args.output_dir)),
                    "sample_token": payload["sample_token"],
                    "scene_token": payload["scene_token"],
                    "log_name": payload["log_name"],
                }
            )
            hidden_token_counts.append(int(payload["last_hidden_state"].shape[0]))
            hidden_feature_dims.append(int(payload["last_hidden_state"].shape[1]))
            written_by_clip[clip_dir.name] = written_by_clip.get(clip_dir.name, 0) + 1
            if args.log_every and len(records) % args.log_every == 0:
                print(f"written={len(records)} scanned={scanned} skipped={skipped}", flush=True)
        if args.max_samples is not None and len(records) >= args.max_samples:
            break
    if not records:
        raise RuntimeError("No Bench2Drive samples were written")
    if len(set(hidden_feature_dims)) != 1:
        raise RuntimeError(f"VLM hidden feature dimensions changed within one shard: {sorted(set(hidden_feature_dims))}")

    write_index(args.output_dir, records)
    effective_stop = args.clip_stop if args.clip_stop is not None else args.clip_start + len(selected_clip_dirs)
    metadata = {
        "version": CHUNK_VERSION,
        "dataset": "Bench2Drive",
        "contract_id": BENCH2DRIVE_CONTRACT_ID,
        "classification": "closest-public",
        "is_dummy": False,
        "contains_vlm_hidden": True,
        "hidden_source": "recogdrive_vlm",
        "contains_jepa": False,
        "contains_vggt": False,
        "target_tokens_are_train_only": True,
        "data_root": str(args.data_root.resolve()),
        "num_records": len(records),
        "num_selected_clips": len(selected_clip_dirs),
        "clip_start": args.clip_start,
        "clip_stop": effective_stop,
        "clip_list": str(args.clip_list.resolve()) if args.clip_list else None,
        "num_scanned_windows": scanned,
        "num_skipped_windows": skipped,
        "frame_step": args.frame_step,
        "sample_stride": args.sample_stride,
        "raw_frequency_hz": 10,
        "training_anchor_frequency_hz": 10,
        "target_waypoint_frequency_hz": 2,
        "history_frames": args.history_frames,
        "future_frames": args.future_frames,
        "future_horizon_seconds": 3.0,
        "camera_order": list(BENCH2DRIVE_CAMERA_ORDER),
        "hidden_tokens_min": min(hidden_token_counts),
        "hidden_tokens_max": max(hidden_token_counts),
        "vlm_feature_dim": hidden_feature_dims[0],
        "hidden_dtype": args.cache_hidden_dtype,
        "cpu_threads": args.cpu_threads,
        "recogdrive_vlm_path": str(args.recogdrive_vlm_path.resolve()),
        "system_prompt_profile": "bench2drive",
        "coordinate_policy": "current_world2lidar_x_forward_y_lateral_relative_yaw",
        "written_by_clip": written_by_clip,
    }
    write_json(args.output_dir / "metadata.json", metadata)
    return metadata


def main() -> int:
    args = parse_args()
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(min(2, args.cpu_threads))
    metadata = build_cache(args)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
