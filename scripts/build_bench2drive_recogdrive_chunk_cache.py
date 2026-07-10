#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import (  # noqa: E402
    CHUNK_VERSION,
    atomic_torch_save,
    write_index,
    write_json,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a ReCogDrive planner chunk cache from Bench2Drive raw clips. "
            "The default hidden state is deterministic and motion-conditioned so the "
            "planner/data/optimizer path can run before real VLM hidden-state caching is available."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/data/Bench2Drive-Base"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--clip-start", type=int, default=0, help="Start index in sorted Bench2Drive clip list, after optional --clip-name filtering.")
    parser.add_argument("--clip-stop", type=int, default=None, help="Exclusive stop index in sorted Bench2Drive clip list, after optional --clip-name filtering.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--clip-name", action="append", default=None, help="Limit to one or more exact clip directory names.")
    parser.add_argument("--frame-step", type=int, default=5, help="Frame spacing for 0.5s-like history/future samples in 10Hz Bench2Drive clips.")
    parser.add_argument("--sample-stride", type=int, default=5, help="Stride between current-frame anchors.")
    parser.add_argument("--history-frames", type=int, default=4)
    parser.add_argument("--future-frames", type=int, default=8)
    parser.add_argument("--hidden-tokens", type=int, default=32)
    parser.add_argument("--vlm-feature-dim", type=int, default=1536)
    parser.add_argument("--hidden-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument(
        "--hidden-source",
        choices=("deterministic", "recogdrive-vlm"),
        default="deterministic",
        help="Use deterministic motion-conditioned hidden states, or run ReCogDrive VLM to cache real hidden states.",
    )
    parser.add_argument("--recogdrive-vlm-path", type=Path, default=Path("checkpoints/recogdrive/ReCogDrive-VLM-2B"))
    parser.add_argument("--device", default="cuda", help="Device for --hidden-source recogdrive-vlm.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args(argv)


def normalize_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def clip_dirs(
    data_root: Path,
    clip_names: Optional[List[str]],
    max_clips: Optional[int],
    clip_start: int = 0,
    clip_stop: Optional[int] = None,
) -> List[Path]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Bench2Drive data root does not exist: {data_root}")
    if clip_start < 0:
        raise ValueError("--clip-start must be non-negative")
    if clip_stop is not None and clip_stop < clip_start:
        raise ValueError("--clip-stop must be >= --clip-start")
    requested = set(clip_names or [])
    dirs = [
        child
        for child in sorted(data_root.iterdir())
        if child.is_dir()
        and child.name != "maps"
        and (child / "anno").is_dir()
        and (not requested or child.name in requested)
    ]
    if requested:
        found = {path.name for path in dirs}
        missing = sorted(requested - found)
        if missing:
            raise FileNotFoundError(f"Requested Bench2Drive clip(s) not found under {data_root}: {missing}")
    dirs = dirs[clip_start:clip_stop]
    if max_clips is not None:
        dirs = dirs[:max_clips]
    if not dirs:
        raise FileNotFoundError(f"No Bench2Drive clip directories with anno/ found under {data_root}")
    return dirs


def annotation_paths(clip_dir: Path) -> List[Path]:
    return sorted((clip_dir / "anno").glob("*.json.gz"))


def read_annotation(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("x", "y", "theta"):
        if key not in data:
            raise KeyError(f"{path} missing required key '{key}'")
    return data


def pose_of(annotation: Dict[str, Any]) -> np.ndarray:
    return np.array(
        [float(annotation["x"]), float(annotation["y"]), float(annotation["theta"])],
        dtype=np.float64,
    )


def relative_poses(origin: np.ndarray, poses: np.ndarray) -> np.ndarray:
    theta = -float(origin[2])
    rotation = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=np.float64,
    )
    rel = poses.astype(np.float64, copy=True)
    rel -= origin.reshape(1, 3)
    rel[:, :2] = rel[:, :2] @ rotation.T
    rel[:, 2] = normalize_angle(rel[:, 2])
    return rel.astype(np.float32)


def command_one_hot(annotation: Dict[str, Any]) -> np.ndarray:
    command = int(annotation.get("command_near", annotation.get("next_command", annotation.get("command_far", 4))))
    one_hot = np.zeros(3, dtype=np.float32)
    if command in {1, 5}:  # left / change lane left
        one_hot[0] = 1.0
    elif command in {2, 6}:  # right / change lane right
        one_hot[2] = 1.0
    else:  # straight / lane follow / unknown
        one_hot[1] = 1.0
    return one_hot


def local_xy(origin_heading: float, xy: Iterable[float]) -> np.ndarray:
    theta = -float(origin_heading)
    rotation = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=np.float64,
    )
    return np.asarray(list(xy), dtype=np.float64) @ rotation.T


def status_feature(annotation: Dict[str, Any], high_command: np.ndarray) -> np.ndarray:
    speed = float(annotation.get("speed", 0.0))
    velocity = np.array([speed, 0.0], dtype=np.float32)
    acceleration_raw = annotation.get("acceleration") or [0.0, 0.0, 0.0]
    acceleration_xy = local_xy(float(annotation["theta"]), acceleration_raw[:2]).astype(np.float32)
    # CARLA's z acceleration is dominated by gravity here; keep the third slot neutral.
    acceleration = np.array([acceleration_xy[0], acceleration_xy[1], 0.0], dtype=np.float32)
    return np.concatenate([high_command.astype(np.float32), velocity, acceleration], axis=0)


def stable_phase(sample_token: str) -> float:
    digest = hashlib.sha256(sample_token.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "little")
    return (value % 1_000_000) / 1_000_000.0 * 2.0 * math.pi


def deterministic_hidden_state(
    *,
    sample_token: str,
    history_trajectory: np.ndarray,
    high_command: np.ndarray,
    status: np.ndarray,
    hidden_tokens: int,
    feature_dim: int,
    dtype: str,
) -> torch.Tensor:
    base = np.concatenate(
        [
            history_trajectory.reshape(-1),
            high_command.reshape(-1),
            status.reshape(-1),
        ]
    ).astype(np.float32)
    if not np.isfinite(base).all():
        raise ValueError(f"Non-finite motion features for {sample_token}")
    dim_idx = np.arange(feature_dim, dtype=np.float32)
    phase = stable_phase(sample_token)
    rows: List[np.ndarray] = []
    for token_idx in range(hidden_tokens):
        source = base[(dim_idx.astype(np.int64) + token_idx) % len(base)]
        freq = 0.013 * ((dim_idx % 37.0) + 1.0)
        row = np.sin(source * freq + phase + token_idx * 0.17) + 0.5 * np.cos(freq * (token_idx + 1.0))
        rows.append(row.astype(np.float32))
    tensor = torch.from_numpy(np.stack(rows, axis=0))
    return tensor.half() if dtype == "float16" else tensor.float()


def build_vlm_feature_builder(args: argparse.Namespace):
    if args.hidden_source != "recogdrive-vlm":
        return None
    if not args.recogdrive_vlm_path.exists():
        raise FileNotFoundError(f"ReCogDrive VLM path not found: {args.recogdrive_vlm_path}")
    from navsim.agents.recogdrive.recogdrive_features import ReCogDriveFeatureBuilder

    return ReCogDriveFeatureBuilder(
        cache_hidden_state=True,
        model_type="internvl",
        checkpoint_path=str(args.recogdrive_vlm_path),
        device=args.device,
        cache_mode=True,
        use_expert_features=False,
    )


def build_agent_input(
    *,
    clip_name: str,
    sample_token: str,
    image_path: Path,
    history_trajectory: np.ndarray,
    high_command: np.ndarray,
    status: np.ndarray,
):
    from navsim.common.dataclasses import AgentInput, Camera, Cameras, EgoStatus, Lidar

    empty_camera = Camera()
    cameras = Cameras(
        cam_f0=Camera(image=image_path),
        cam_l0=empty_camera,
        cam_l1=empty_camera,
        cam_l2=empty_camera,
        cam_r0=empty_camera,
        cam_r1=empty_camera,
        cam_r2=empty_camera,
        cam_b0=empty_camera,
    )
    velocity = status[3:5].astype(np.float32)
    acceleration = status[5:8].astype(np.float32)
    ego_statuses = [
        EgoStatus(
            ego_pose=row.astype(np.float32),
            ego_velocity=velocity,
            ego_acceleration=acceleration,
            driving_command=high_command.astype(np.float32),
        )
        for row in history_trajectory
    ]
    return AgentInput(
        ego_statuses=ego_statuses,
        cameras=[cameras for _ in range(len(ego_statuses))],
        lidars=[Lidar() for _ in range(len(ego_statuses))],
        token=sample_token,
        log_name=clip_name,
        scene_token=clip_name,
    )


def sample_payload(
    *,
    clip_dir: Path,
    annotations: List[Dict[str, Any]],
    current_idx: int,
    frame_step: int,
    history_frames: int,
    future_frames: int,
    hidden_tokens: int,
    feature_dim: int,
    hidden_dtype: str,
    hidden_source: str,
    vlm_feature_builder: Any = None,
) -> Dict[str, Any]:
    history_indices = [current_idx - frame_step * offset for offset in reversed(range(history_frames))]
    future_indices = [current_idx + frame_step * offset for offset in range(1, future_frames + 1)]
    current = annotations[current_idx]
    origin = pose_of(current)
    history_global = np.stack([pose_of(annotations[idx]) for idx in history_indices], axis=0)
    future_global = np.stack([pose_of(annotations[idx]) for idx in future_indices], axis=0)
    history = relative_poses(origin, history_global)
    future = relative_poses(origin, future_global)
    high_command = command_one_hot(current)
    status = status_feature(current, high_command)
    frame_name = f"{current_idx:05d}"
    sample_token = f"{clip_dir.name}_{frame_name}"
    image_path = clip_dir / "camera" / "rgb_front" / f"{frame_name}.jpg"
    if hidden_source == "recogdrive-vlm":
        if vlm_feature_builder is None:
            raise ValueError("hidden_source='recogdrive-vlm' requires a VLM feature builder.")
        if not image_path.is_file():
            raise FileNotFoundError(f"Front image for VLM hidden state not found: {image_path}")
        agent_input = build_agent_input(
            clip_name=clip_dir.name,
            sample_token=sample_token,
            image_path=image_path,
            history_trajectory=history,
            high_command=high_command,
            status=status,
        )
        vlm_features = vlm_feature_builder.compute_features(agent_input)
        last_hidden_state = vlm_features["last_hidden_state"].float()
    else:
        last_hidden_state = deterministic_hidden_state(
            sample_token=sample_token,
            history_trajectory=history,
            high_command=high_command,
            status=status,
            hidden_tokens=hidden_tokens,
            feature_dim=feature_dim,
            dtype=hidden_dtype,
        )
    return {
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
            "clip": clip_dir.name,
            "frame_index": current_idx,
            "frame_step": frame_step,
            "sample_stride": None,
            "front_image_path": str(image_path),
            "front_image_exists": image_path.is_file(),
            "raw_command_near": current.get("command_near"),
            "raw_command_far": current.get("command_far"),
            "next_command": current.get("next_command"),
            "hidden_source": "recogdrive_vlm" if hidden_source == "recogdrive-vlm" else "motion_conditioned_deterministic",
            "hidden_tokens": int(last_hidden_state.shape[0]),
            "vlm_feature_dim": int(last_hidden_state.shape[-1]),
            "coordinate_policy": "global_se2_to_current_ego_relative",
            "command_policy": "carla_1_5_left__2_6_right__3_4_straight",
        },
    }


def finite_required(payload: Dict[str, Any]) -> bool:
    for key in ("history_trajectory", "high_command_one_hot", "status_feature", "trajectory", "last_hidden_state"):
        value = payload[key]
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value.float()).all():
            return False
    return True


def build_cache(args: argparse.Namespace) -> Dict[str, Any]:
    if args.frame_step <= 0 or args.sample_stride <= 0:
        raise ValueError("--frame-step and --sample-stride must be positive")
    if args.history_frames != 4:
        raise ValueError("ReCogDrive currently expects --history-frames 4")
    if args.future_frames != 8:
        raise ValueError("ReCogDrive currently expects --future-frames 8")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}. Pass --overwrite to replace samples/index.")
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    vlm_feature_builder = build_vlm_feature_builder(args)

    records: List[Dict[str, Any]] = []
    skipped = 0
    scanned = 0
    min_current = args.frame_step * (args.history_frames - 1)
    written_by_clip: Dict[str, int] = {}
    selected_clip_dirs = clip_dirs(args.data_root, args.clip_name, args.max_clips, args.clip_start, args.clip_stop)
    for clip_dir in selected_clip_dirs:
        paths = annotation_paths(clip_dir)
        if not paths:
            continue
        annotations = [read_annotation(path) for path in paths]
        max_current = len(annotations) - args.frame_step * args.future_frames - 1
        if max_current < min_current:
            skipped += len(annotations)
            continue
        for current_idx in range(min_current, max_current + 1, args.sample_stride):
            scanned += 1
            if args.max_samples is not None and len(records) >= args.max_samples:
                break
            try:
                payload = sample_payload(
                    clip_dir=clip_dir,
                    annotations=annotations,
                    current_idx=current_idx,
                    frame_step=args.frame_step,
                    history_frames=args.history_frames,
                    future_frames=args.future_frames,
                    hidden_tokens=args.hidden_tokens,
                    feature_dim=args.vlm_feature_dim,
                    hidden_dtype=args.hidden_dtype,
                    hidden_source=args.hidden_source,
                    vlm_feature_builder=vlm_feature_builder,
                )
                payload["meta"]["sample_stride"] = args.sample_stride
                if not finite_required(payload):
                    skipped += 1
                    continue
            except Exception as exc:
                skipped += 1
                if args.log_every and skipped <= 20:
                    print(f"skip clip={clip_dir.name} idx={current_idx}: {exc}", flush=True)
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
            written_by_clip[clip_dir.name] = written_by_clip.get(clip_dir.name, 0) + 1
            if args.log_every and len(records) % args.log_every == 0:
                print(f"written={len(records)} scanned={scanned} skipped={skipped}", flush=True)
        if args.max_samples is not None and len(records) >= args.max_samples:
            break
    if not records:
        raise RuntimeError("No Bench2Drive samples were written.")

    write_index(args.output_dir, records)
    metadata = {
        "version": CHUNK_VERSION,
        "dataset": "Bench2Drive",
        "is_dummy": False,
        "contains_vlm_hidden": True,
        "hidden_source": "recogdrive_vlm" if args.hidden_source == "recogdrive-vlm" else "motion_conditioned_deterministic",
        "contains_jepa": False,
        "contains_vggt": False,
        "target_tokens_are_train_only": True,
        "data_root": str(args.data_root),
        "num_records": len(records),
        "num_selected_clips": len(selected_clip_dirs),
        "clip_start": args.clip_start,
        "clip_stop": args.clip_stop,
        "num_scanned_windows": scanned,
        "num_skipped_windows": skipped,
        "frame_step": args.frame_step,
        "sample_stride": args.sample_stride,
        "history_frames": args.history_frames,
        "future_frames": args.future_frames,
        "hidden_tokens": args.hidden_tokens if args.hidden_source == "deterministic" else "from_vlm_output",
        "vlm_feature_dim": args.vlm_feature_dim if args.hidden_source == "deterministic" else "from_vlm_output",
        "hidden_dtype": args.hidden_dtype,
        "recogdrive_vlm_path": str(args.recogdrive_vlm_path) if args.hidden_source == "recogdrive-vlm" else None,
        "written_by_clip": written_by_clip,
    }
    write_json(args.output_dir / "metadata.json", metadata)
    return metadata


def main() -> int:
    args = parse_args()
    metadata = build_cache(args)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
