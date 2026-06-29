#!/usr/bin/env python3
"""Bridge OneVL AR Answer hidden states into the ReCogDrive DiT planner.

This is an interface smoke/cache utility, not a training launcher. It extracts
Qwen3-VL final-layer hidden states from AR Answer prompts, builds the planner
state tensors expected by ReCogDrive, and optionally runs a DiT forward loss.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion


DEFAULT_MODEL_PATH = (
    "/mnt/project/onevl_navsim_exp/answer_bs64_20260625_082451/"
    "swift_output/v0-20260625-082510/checkpoint-3228"
)
DEFAULT_DATA_JSONL = "/mnt/project/onevl_navsim_data/navsim_answer_official_paths.jsonl"
DEFAULT_IMAGE_BASE = "/mnt/project/OneVL_training"
DEFAULT_OUTPUT_DIR = "/mnt/project/onevl_navsim_exp/ar_answer_dit_bridge_smoke"
DEFAULT_VLA_AD_ROOT = "/mnt/project/VLA-AD"
DEFAULT_NAVSIM_LOG_PATH = "/mnt/navsim/trainval_navsim_logs"
DEFAULT_STAGE2_TARGET_INDEX = (
    "/mnt/project/VLA-AD/outputs/psi_drive/support_index/"
    "stage2_pareto_support_clean_full_gt_supplemented_20260625T162351Z.pt"
)
OFFICIAL_SENSOR_PREFIX = "navsim_v1.1_all/dataset/sensor_blobs/trainval/"
OFFICIAL_TEST_SENSOR_PREFIX = "navsim_v1.1_all/dataset/sensor_blobs/test/"
LOCAL_SENSOR_PREFIX = "trainval_sensor_blobs/trainval/"
LOCAL_TEST_SENSOR_PREFIX = "test_sensor_blobs/test/"


def official_sensor_prefixes() -> tuple[str, ...]:
    return (OFFICIAL_SENSOR_PREFIX, OFFICIAL_TEST_SENSOR_PREFIX)


def official_sensor_prefix_for_split(split: str) -> str:
    return OFFICIAL_TEST_SENSOR_PREFIX if split == "test" else OFFICIAL_SENSOR_PREFIX


COMMAND_TEXT_FROM_ONE_HOT = {
    0: "TURN LEFT",
    1: "MOVE FORWARD",
    2: "TURN RIGHT",
}
COMMAND_ONE_HOT_FROM_TEXT = {
    "TURN LEFT": [1.0, 0.0, 0.0],
    "MOVE FORWARD": [0.0, 1.0, 0.0],
    "GO STRAIGHT": [0.0, 1.0, 0.0],
    "TURN RIGHT": [0.0, 0.0, 1.0],
}
PROMPT_PARSE_RE = re.compile(
    r"Command:\s*(?P<command>[^.]+)\.\s*"
    r"Velocity:\s*(?P<velocity>\[[^\]]+\])\.\s*"
    r"Acceleration:\s*(?P<acceleration>\[[^\]]+\])\.\s*"
    r"Historical trajectory:\s*(?P<history>.*?)\.\s*"
    r"(?:Predict|Output)",
    re.IGNORECASE | re.DOTALL,
)
POINT_RE = re.compile(r"\[[^\]]+\]")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-jsonl", default=DEFAULT_DATA_JSONL)
    parser.add_argument("--image-base-path", default=DEFAULT_IMAGE_BASE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vla-ad-root", default=DEFAULT_VLA_AD_ROOT)
    parser.add_argument("--navsim-log-path", default=DEFAULT_NAVSIM_LOG_PATH)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dit-device", default="cpu")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-image-size", type=int, default=1792)
    parser.add_argument("--assistant-prefix", default="")
    parser.add_argument("--hidden-padding", choices=("none", "max_length"), default="none")
    parser.add_argument("--hidden-max-length", type=int, default=2800)
    parser.add_argument("--hidden-padding-side", choices=("left", "right"), default="left")
    parser.add_argument("--hidden-truncation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--current-image-policy",
        choices=("first", "last", "single_or_last", "strict_single"),
        default="first",
        help=(
            "Select which row image is treated as the NAVSIM current frame. "
            "Default preserves the historical bridge behavior."
        ),
    )
    parser.add_argument(
        "--prompt-source",
        choices=("row", "scene", "row_strict_scene_check"),
        default="row",
        help=(
            "Use row prompt text, scene-derived prompt text, or row prompt with strict "
            "row-vs-scene tensor alignment checks."
        ),
    )
    parser.add_argument(
        "--include-control-convention",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add ego-frame/control convention text to scene-derived prompts only.",
    )
    parser.add_argument("--skip-vlm", action="store_true")
    parser.add_argument("--dummy-seq-len", type=int, default=64)
    parser.add_argument("--qwen-hidden-dim", type=int, default=2560)
    parser.add_argument("--dit-type", choices=("small", "large"), default="small")
    parser.add_argument("--sampling-method", choices=("ddim", "ddpm", "flow"), default="ddim")
    parser.add_argument("--num-history-frames", type=int, default=4)
    parser.add_argument("--num-future-frames", type=int, default=10)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument(
        "--use-answer-target",
        action="store_true",
        help="Use the rounded AR Answer text as target. Default uses NAVSIM future trajectory.",
    )
    parser.add_argument(
        "--target-index-path",
        default="",
        help=(
            "Optional PSI/ReCogDrive stage2 support target index. If set, the selected support trajectory "
            "replaces the NAVSIM GT future target unless --use-answer-target is used."
        ),
    )
    parser.add_argument(
        "--target-selection",
        choices=("max_weight", "best_score", "first_valid"),
        default="max_weight",
    )
    parser.add_argument(
        "--stage2-support-mode",
        choices=("single_target", "preserve_support"),
        default="single_target",
        help=(
            "single_target keeps the legacy behavior and replaces trajectory with one selected support. "
            "preserve_support keeps GT trajectory and writes APSD support fields for weighted training-time sampling."
        ),
    )
    parser.add_argument("--cache-format", choices=("flat", "nested"), default="flat")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-dit-forward", action="store_true")
    parser.add_argument("--save-cache", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in [0, num_shards)")
    if args.end_index is not None and args.end_index <= args.start_index:
        raise ValueError("--end-index must be greater than --start-index")
    if args.hidden_max_length <= 0:
        raise ValueError("--hidden-max-length must be positive")
    return args


def torch_dtype(name: str, device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def load_rows(
    path: Path,
    max_samples: int,
    *,
    start_index: int = 0,
    end_index: int | None = None,
    num_shards: int = 1,
    shard_id: int = 0,
) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]]

    def keep(line_index: int) -> bool:
        if line_index < start_index:
            return False
        if end_index is not None and line_index >= end_index:
            return False
        if num_shards > 1 and line_index % num_shards != shard_id:
            return False
        return True

    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line_index, line in enumerate(f):
                if line.strip() and keep(line_index):
                    rows.append((line_index, json.loads(line)))
                if 0 < max_samples <= len(rows):
                    break
    else:
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, list):
            raise ValueError(f"expected list in {path}")
        rows = []
        for line_index, row in enumerate(loaded):
            if keep(line_index):
                rows.append((line_index, row))
            if 0 < max_samples <= len(rows):
                break
    return rows


def resolve_image_path(path: str, image_base_path: Path) -> Path:
    normalized = normalize_official_trainval_image(path)
    candidates = [
        Path(path),
        image_base_path / path,
        Path("/mnt/project") / path,
        Path("/mnt/project/OneVL_training") / path,
    ]
    if normalized != path:
        candidates.extend(
            [
                image_base_path / normalized,
                Path("/mnt/project/OneVL_training") / normalized,
                Path("/mnt/project") / normalized,
            ]
        )
    if normalized.startswith(OFFICIAL_SENSOR_PREFIX):
        rel = normalized[len(OFFICIAL_SENSOR_PREFIX):]
        candidates.append(Path("/mnt/navsim/trainval_sensor_blobs/trainval") / rel)
    if normalized.startswith(OFFICIAL_TEST_SENSOR_PREFIX):
        rel = normalized[len(OFFICIAL_TEST_SENSOR_PREFIX):]
        candidates.append(Path("/mnt/navsim/test_sensor_blobs/test") / rel)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"image not found: {path}")


def normalize_official_trainval_image(path: str, default_split: str = "trainval") -> str:
    path = path.replace("file://", "")
    for prefix in official_sensor_prefixes():
        if prefix in path:
            return prefix + path.split(prefix, 1)[1]
    split_specs = (
        ("trainval", OFFICIAL_SENSOR_PREFIX, LOCAL_SENSOR_PREFIX),
        ("test", OFFICIAL_TEST_SENSOR_PREFIX, LOCAL_TEST_SENSOR_PREFIX),
    )
    for split, official_prefix, local_prefix in split_specs:
        for marker in (f"/sensor_blobs/{split}/", f"/{local_prefix}"):
            if marker in path:
                return official_prefix + path.split(marker, 1)[1]
        if path.startswith(local_prefix):
            return official_prefix + path[len(local_prefix):]
        if path.startswith(f"{split}/"):
            return official_prefix + path[len(f"{split}/"):]

    parts = Path(path).parts
    if len(parts) >= 3 and parts[-2].startswith("CAM_"):
        return official_sensor_prefix_for_split(default_split) + "/".join(parts[-3:])
    return path


def select_current_image(row: dict[str, Any], policy: str, default_split: str = "trainval") -> str:
    images = row.get("images") or []
    if not images:
        raise ValueError("row has no images")
    if policy == "strict_single" and len(images) != 1:
        raise ValueError(f"--current-image-policy=strict_single requires one image, got {len(images)}")
    if policy == "first":
        selected = images[0]
    elif policy in {"last", "single_or_last"}:
        selected = images[-1]
    elif policy == "strict_single":
        selected = images[0]
    else:
        raise ValueError(f"unknown current image policy: {policy}")
    return normalize_official_trainval_image(str(selected), default_split)


def current_image_meta(row: dict[str, Any], policy: str, selected: str, default_split: str = "trainval") -> dict[str, Any]:
    images = [normalize_official_trainval_image(str(image), default_split) for image in (row.get("images") or [])]
    first = images[0] if images else ""
    last = images[-1] if images else ""
    return {
        "row_image_count": len(images),
        "row_image_first": first,
        "row_image_last": last,
        "current_image_policy": policy,
        "current_image_selected": selected,
        "stage1_submission_image_policy": "last",
        "current_image_first_eq_last": bool(first and first == last),
    }


def command_text_from_one_hot(one_hot: torch.Tensor) -> str:
    idx = int(torch.argmax(one_hot.float()).item())
    return COMMAND_TEXT_FROM_ONE_HOT.get(idx, "MOVE FORWARD")


def format_float(value: float) -> str:
    value = float(value)
    if abs(value) < 0.005:
        value = 0.0
    return f"{value:.2f}"


def format_point(point: Any) -> str:
    values = [float(x) for x in point]
    if len(values) != 3:
        raise ValueError(f"expected point length 3, got {len(values)}")
    return f"[{format_float(values[0])}, {format_float(values[1])}, {format_float(values[2])}]"


def split_status_feature_values(
    status_feature: torch.Tensor,
    high_command_one_hot: torch.Tensor,
) -> tuple[list[float], list[float], str]:
    status = [float(x) for x in status_feature.flatten().tolist()]
    high_command = [float(x) for x in high_command_one_hot.flatten().tolist()]
    if len(status) != 8:
        raise ValueError(f"expected status_feature length 8, got {len(status)}")
    if all(abs(status[i] - high_command[i]) < 1e-5 for i in range(3)):
        return status[3:5], status[5:8], "command3_velocity2_acceleration3"
    return status[4:6], status[6:8], "command4_velocity2_acceleration2"


def build_qwen_prompt_from_scene_tensors(
    history_trajectory: torch.Tensor,
    high_command_one_hot: torch.Tensor,
    status_feature: torch.Tensor,
    *,
    include_control_convention: bool = False,
) -> str:
    command = command_text_from_one_hot(high_command_one_hot)
    velocity, acceleration, _ = split_status_feature_values(status_feature, high_command_one_hot)
    history = ", ".join(format_point(point) for point in history_trajectory.tolist())
    prompt = (
        f"<image> is the front view. Command: {command}. "
        f"Velocity: [{format_float(velocity[0])}, {format_float(velocity[1])}]. "
        f"Acceleration: [{', '.join(format_float(value) for value in acceleration)}]. "
        f"Historical trajectory: {history}. "
    )
    if include_control_convention:
        prompt += (
            "Control convention: coordinates are ego-local, x is forward, y is lateral, "
            "heading is in radians, and the output horizon is 8 future waypoints. "
        )
    prompt += (
        "Predict the future trajectory in <answer></answer>. "
        "For the content in <answer></answer>, only generate 8 future waypoints in pure text format: "
        "[x_1, y_1, heading_1], [x_2, y_2, heading_2], ..., [x_8, y_8, heading_8]. "
        "Each waypoint must be [x, y, heading] with exactly 2 digits after the decimal point. "
        "Separate waypoints with commas. "
        "Do not include reasoning, extra text, an extra outer list, or invalid values."
    )
    return prompt


def parse_prompt_fields(prompt: str) -> dict[str, Any]:
    match = PROMPT_PARSE_RE.search(prompt)
    if match is None:
        raise ValueError(f"unrecognized AR Answer prompt: {prompt[:240]}")
    command = match.group("command").strip().upper()
    velocity = ast.literal_eval(match.group("velocity"))
    acceleration = ast.literal_eval(match.group("acceleration"))
    history_points = []
    for point_text in POINT_RE.findall(match.group("history")):
        parsed = ast.literal_eval(point_text)
        if len(parsed) != 3:
            raise ValueError(f"invalid history point: {point_text}")
        history_points.append([float(x) for x in parsed])
    return {
        "command": command,
        "velocity": [float(x) for x in velocity],
        "acceleration": [float(x) for x in acceleration],
        "history": history_points,
    }


def prompt_scene_alignment(
    row_prompt: str,
    history_trajectory: torch.Tensor,
    high_command_one_hot: torch.Tensor,
    status_feature: torch.Tensor,
    *,
    atol: float = 0.06,
) -> dict[str, Any]:
    scene_command = command_text_from_one_hot(high_command_one_hot)
    scene_history = [[float(x) for x in point] for point in history_trajectory.tolist()]
    scene_velocity, scene_acceleration, _ = split_status_feature_values(status_feature, high_command_one_hot)
    result: dict[str, Any] = {
        "prompt_command": None,
        "scene_command": scene_command,
        "prompt_history_points": None,
        "scene_history_points": len(scene_history),
        "prompt_scene_alignment_pass": False,
        "prompt_scene_alignment_error": "",
    }
    try:
        fields = parse_prompt_fields(row_prompt)
        result["prompt_command"] = fields["command"]
        result["prompt_history_points"] = len(fields["history"])
        errors: list[str] = []
        if COMMAND_ONE_HOT_FROM_TEXT.get(fields["command"]) != [float(x) for x in high_command_one_hot.tolist()]:
            errors.append(f"command prompt={fields['command']} scene={scene_command}")
        if len(fields["history"]) != len(scene_history):
            errors.append(f"history_count prompt={len(fields['history'])} scene={len(scene_history)}")
        else:
            max_history_err = max(
                abs(float(a) - float(b))
                for prompt_point, scene_point in zip(fields["history"], scene_history)
                for a, b in zip(prompt_point, scene_point)
            )
            if max_history_err > atol:
                errors.append(f"history_max_abs_error={max_history_err:.4f}")
        if len(fields["velocity"]) == len(scene_velocity):
            max_vel_err = max(abs(float(a) - float(b)) for a, b in zip(fields["velocity"], scene_velocity))
            if max_vel_err > atol:
                errors.append(f"velocity_max_abs_error={max_vel_err:.4f}")
        else:
            errors.append(f"velocity_len prompt={len(fields['velocity'])} scene={len(scene_velocity)}")
        if len(fields["acceleration"]) == len(scene_acceleration):
            max_acc_err = max(abs(float(a) - float(b)) for a, b in zip(fields["acceleration"], scene_acceleration))
            if max_acc_err > atol:
                errors.append(f"acceleration_max_abs_error={max_acc_err:.4f}")
        else:
            errors.append(f"acceleration_len prompt={len(fields['acceleration'])} scene={len(scene_acceleration)}")
        result["prompt_scene_alignment_pass"] = not errors
        result["prompt_scene_alignment_error"] = "; ".join(errors)
    except Exception as exc:  # noqa: BLE001 - audit metadata must preserve parse failures.
        result["prompt_scene_alignment_error"] = str(exc)
    return result


def image_log_name(official_image: str) -> str:
    rel = official_image
    for prefix in official_sensor_prefixes():
        if prefix in official_image:
            rel = official_image.split(prefix, 1)[-1]
            break
    parts = rel.split("/")
    if len(parts) < 3:
        raise ValueError(f"cannot parse log name from image path: {official_image}")
    return parts[0]


def resolve_navsim_log_path(path: Path) -> Path:
    if any(path.glob("*.pkl")):
        return path
    for child_name in ("trainval", "test", "mini"):
        child_path = path / child_name
        if child_path.is_dir() and any(child_path.glob("*.pkl")):
            return child_path
    for child_path in sorted(child for child in path.iterdir() if child.is_dir()):
        if any(child_path.glob("*.pkl")):
            return child_path
    raise FileNotFoundError(f"no NAVSIM .pkl logs found under {path} or its immediate subdirectories")


def frame_to_global_pose(frame: dict[str, Any]) -> np.ndarray:
    ego_translation = frame["ego2global_translation"]
    ego_quaternion = Quaternion(*frame["ego2global_rotation"])
    return np.array(
        [ego_translation[0], ego_translation[1], ego_quaternion.yaw_pitch_roll[0]],
        dtype=np.float64,
    )


def _relative_poses(global_poses: list[np.ndarray], origin_index: int, state_se2_cls: Any) -> torch.Tensor:
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
        convert_absolute_to_relative_se2_array,
    )

    origin = state_se2_cls(*global_poses[origin_index])
    local = convert_absolute_to_relative_se2_array(origin, np.array(global_poses, dtype=np.float64))
    return torch.tensor(local, dtype=torch.float32)


def relative_poses_from_frames(frames: list[dict[str, Any]], origin_index: int) -> torch.Tensor:
    from nuplan.common.actor_state.state_representation import StateSE2

    global_poses = [frame_to_global_pose(frame) for frame in frames]
    return _relative_poses(global_poses, origin_index, StateSE2)


def build_scene_mapping(
    rows: list[dict[str, Any]],
    navsim_log_path: Path,
    num_history_frames: int,
    num_future_frames: int,
    current_image_policy: str = "first",
) -> dict[str, dict[str, Any]]:
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter, SensorConfig

    resolved_log_path = resolve_navsim_log_path(navsim_log_path)
    default_split = "test" if resolved_log_path.name == "test" else "trainval"
    current_images = [select_current_image(row, current_image_policy, default_split) for row in rows]
    log_names = sorted({image_log_name(image) for image in current_images})
    wanted_images = set(current_images)
    scene_filter = SceneFilter(
        num_history_frames=num_history_frames,
        num_future_frames=num_future_frames,
        frame_interval=1,
        has_route=True,
        log_names=log_names,
    )
    scene_loader = SceneLoader(
        data_path=resolved_log_path,
        sensor_blobs_path=None,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )

    current_index = num_history_frames - 1
    mapping: dict[str, dict[str, Any]] = {}
    for token, frames in scene_loader.scene_frames_dicts.items():
        current = normalize_official_trainval_image(
            frames[current_index]["cams"]["CAM_F0"]["data_path"],
            default_split,
        )
        if current not in wanted_images:
            continue
        if current in mapping:
            raise RuntimeError(f"duplicate SceneLoader mapping for {current}")
        mapping[current] = {
            "token": token,
            "log_name": frames[current_index]["log_name"],
            "scene_token": frames[current_index].get("scene_token"),
            "frames": frames,
        }

    missing = sorted(wanted_images - set(mapping))
    if missing:
        raise RuntimeError(f"missing NAVSIM mapping for {len(missing)} images; first={missing[:3]}")
    return mapping


def infer_high_command_from_future(trajectory: torch.Tensor, lateral_threshold: float = 1.0) -> torch.Tensor:
    if trajectory.numel() == 0:
        return torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    idx = min(3, trajectory.shape[0] - 1)
    lateral = float(trajectory[idx, 1].item())
    if lateral > lateral_threshold:
        return torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    if lateral < -lateral_threshold:
        return torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    return torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)


def navsim_command_three(raw_command: Any, fallback_trajectory: torch.Tensor | None = None) -> tuple[torch.Tensor, str]:
    command = torch.as_tensor(raw_command, dtype=torch.float32).flatten()
    if command.numel() == 4:
        command3 = command[:3]
    elif command.numel() == 3:
        command3 = command
    else:
        raise ValueError(f"expected driving_command length 3 or 4, got {command.numel()}")
    if float(command3.sum().item()) > 0.0:
        return command3, "raw_command_first3"
    if fallback_trajectory is not None:
        return infer_high_command_from_future(fallback_trajectory), "future_lateral_fallback"
    return torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32), "straight_fallback"


def parse_waypoints(answer: str) -> torch.Tensor:
    text = answer.replace("<answer>", "").replace("</answer>", "").strip()
    if not text:
        raise ValueError("empty answer trajectory")
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        value = ast.literal_eval("[" + text + "]")
    if value and isinstance(value[0], (int, float)):
        value = [value]
    trajectory = torch.tensor(value, dtype=torch.float32)
    if trajectory.shape != (8, 3):
        raise ValueError(f"expected trajectory shape (8, 3), got {tuple(trajectory.shape)}")
    return trajectory


def extract_answer_target(row: dict[str, Any]) -> dict[str, torch.Tensor]:
    messages = row.get("messages") or []
    if len(messages) < 2:
        raise ValueError("row must contain user and assistant messages")
    answer = str(messages[1]["content"])
    return {"trajectory": parse_waypoints(answer)}


def load_stage2_target_index(path: str | Path) -> dict[str, Any] | None:
    if not path:
        return None
    index_path = Path(path)
    if not index_path.is_file():
        raise FileNotFoundError(f"stage2 target index not found: {index_path}")
    obj = torch.load(index_path, map_location="cpu")
    required = ("token_to_row", "support_trajectories", "support_mask", "support_weights")
    missing = [key for key in required if key not in obj]
    if missing:
        raise KeyError(f"stage2 target index {index_path} missing keys: {missing}")
    if tuple(obj["support_trajectories"].shape[-2:]) != (8, 3):
        raise ValueError(
            f"stage2 target trajectories must end with shape (8, 3), got "
            f"{tuple(obj['support_trajectories'].shape)}"
        )
    obj["_path"] = str(index_path)
    return obj


def select_stage2_target(
    target_index: dict[str, Any],
    token: str,
    selection: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    token_to_row = target_index["token_to_row"]
    if token not in token_to_row:
        raise KeyError(f"token {token} not found in stage2 target index {target_index.get('_path')}")
    row = int(token_to_row[token])
    mask = target_index["support_mask"][row].bool()
    if not bool(mask.any().item()):
        raise ValueError(f"token {token} has no valid support targets in {target_index.get('_path')}")

    if selection == "max_weight":
        scores = target_index["support_weights"][row].float().clone()
    elif selection == "best_score":
        scores = target_index.get("support_scores", target_index["support_weights"])[row].float().clone()
    elif selection == "first_valid":
        scores = torch.arange(mask.numel(), 0, -1, dtype=torch.float32)
    else:
        raise ValueError(f"unknown target selection: {selection}")
    scores[~mask] = -torch.inf
    slot = int(torch.argmax(scores).item())
    trajectory = target_index["support_trajectories"][row, slot].float().clone()
    if tuple(trajectory.shape) != (8, 3):
        raise ValueError(f"selected support target for {token} has shape {tuple(trajectory.shape)}")
    if not torch.isfinite(trajectory).all():
        raise ValueError(f"selected support target for {token} contains non-finite values")

    sources = target_index.get("support_sources")
    source = ""
    if isinstance(sources, list) and row < len(sources):
        row_sources = sources[row]
        if isinstance(row_sources, (list, tuple)) and slot < len(row_sources):
            source = str(row_sources[slot])
        else:
            source = str(row_sources)
    meta = {
        "stage2_target_index": str(target_index.get("_path", "")),
        "stage2_target_selection": selection,
        "stage2_target_row": row,
        "stage2_target_slot": slot,
        "stage2_target_source": source,
        "stage2_target_weight": float(target_index["support_weights"][row, slot].item()),
    }
    for key in ("support_scores", "support_pdms", "support_core"):
        if key in target_index:
            meta[f"stage2_target_{key.removeprefix('support_')}"] = float(
                target_index[key][row, slot].item()
            )
    return trajectory, meta


def stage2_support_sources(target_index: dict[str, Any], row: int) -> list[str]:
    sources = target_index.get("support_sources")
    if isinstance(sources, list) and row < len(sources):
        row_sources = sources[row]
        if isinstance(row_sources, (list, tuple)):
            return [str(source) for source in row_sources]
        return [str(row_sources)]
    return []


def preserve_stage2_support_targets(
    target_index: dict[str, Any],
    token: str,
    fallback_trajectory: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    token_to_row = target_index["token_to_row"]
    row_obj = token_to_row.get(token)
    missing = row_obj is None
    if missing:
        support_trajectories = torch.zeros((3, 8, 3), dtype=fallback_trajectory.dtype)
        support_trajectories[0] = fallback_trajectory
        support_mask = torch.tensor([True, False, False], dtype=torch.bool)
        support_weights = torch.tensor([1.0, 0.0, 0.0], dtype=fallback_trajectory.dtype)
        support_scores = torch.zeros(3, dtype=fallback_trajectory.dtype)
        row = -1
        sources: list[str] = ["gt_fallback"]
    else:
        row = int(row_obj)
        support_trajectories = target_index["support_trajectories"][row].float().clone()
        support_mask = target_index["support_mask"][row].bool().clone()
        support_weights = target_index["support_weights"][row].float().clone()
        support_scores = target_index.get("support_scores", torch.zeros_like(target_index["support_weights"]))[
            row
        ].float().clone()
        sources = stage2_support_sources(target_index, row)

    if tuple(support_trajectories.shape) != (3, 8, 3):
        raise ValueError(f"support trajectories for {token} have shape {tuple(support_trajectories.shape)}")
    if tuple(support_mask.shape) != (3,) or tuple(support_weights.shape) != (3,):
        raise ValueError(
            f"support mask/weights for {token} have shapes {tuple(support_mask.shape)} and {tuple(support_weights.shape)}"
        )
    if not bool(support_mask.any().item()):
        raise ValueError(f"support targets for {token} contain no valid entries")
    finite_support = torch.isfinite(support_trajectories).flatten(1).all(dim=1)
    finite_weights = torch.isfinite(support_weights)
    valid_mask = support_mask & finite_support & finite_weights
    if not bool(valid_mask.any().item()):
        raise ValueError(f"support targets for {token} have no finite valid entries")
    weight_sum = float((support_weights.clamp(min=0.0) * valid_mask.float()).sum().item())
    if weight_sum <= 0.0:
        support_weights = valid_mask.float() / valid_mask.float().sum().clamp(min=1.0)
    elif abs(weight_sum - 1.0) > 1e-4:
        support_weights = support_weights.clamp(min=0.0) * valid_mask.float() / weight_sum

    support = {
        "support_trajectories": support_trajectories,
        "support_mask": valid_mask,
        "support_weights": support_weights,
        "support_scores": support_scores,
        "support_missing_mask": torch.tensor(bool(missing), dtype=torch.bool),
    }
    meta = {
        "stage2_target_index": str(target_index.get("_path", "")),
        "stage2_target_mode": "preserve_support",
        "stage2_target_row": row,
        "stage2_support_count": int(valid_mask.sum().item()),
        "stage2_support_weight_sum": float((support_weights * valid_mask.float()).sum().item()),
        "stage2_support_sources": sources,
        "stage2_support_missing": bool(missing),
    }
    return support, meta


def extract_navsim_planner_tensors(
    row: dict[str, Any],
    scene_mapping: dict[str, dict[str, Any]],
    num_history_frames: int,
    action_horizon: int,
    use_answer_target: bool,
    current_image_policy: str = "first",
    target_index: dict[str, Any] | None = None,
    target_selection: str = "max_weight",
    stage2_support_mode: str = "single_target",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    current_image = select_current_image(row, current_image_policy)
    item = scene_mapping[current_image]
    frames = item["frames"]
    current_index = num_history_frames - 1
    if len(frames) < current_index + action_horizon + 1:
        raise ValueError(
            f"NAVSIM frames too short for {current_image}: "
            f"need {current_index + action_horizon + 1}, got {len(frames)}"
        )

    history_frames = frames[:num_history_frames]
    history = relative_poses_from_frames(history_frames, origin_index=current_index)
    if tuple(history.shape) != (num_history_frames, 3):
        raise ValueError(f"expected history shape ({num_history_frames}, 3), got {tuple(history.shape)}")

    future_frames = frames[current_index: current_index + action_horizon + 1]
    future_all = relative_poses_from_frames(future_frames, origin_index=0)
    future_trajectory = future_all[1:]
    if tuple(future_trajectory.shape) != (action_horizon, 3):
        raise ValueError(f"expected trajectory shape ({action_horizon}, 3), got {tuple(future_trajectory.shape)}")

    current_frame = frames[current_index]
    raw_command = torch.tensor(current_frame["driving_command"], dtype=torch.float32).flatten()
    high_command, command_policy = navsim_command_three(raw_command, fallback_trajectory=future_trajectory)
    ego_dynamic_state = torch.tensor(current_frame["ego_dynamic_state"], dtype=torch.float32).flatten()
    velocity = ego_dynamic_state[:2]
    acceleration = ego_dynamic_state[2:]
    if raw_command.numel() == 4 and acceleration.numel() == 2:
        status = torch.cat([raw_command, velocity, acceleration], dim=0)
        status_policy = "navsim_command4_velocity2_acceleration2"
    elif raw_command.numel() in {3, 4} and acceleration.numel() == 3:
        status = torch.cat([high_command, velocity, acceleration], dim=0)
        status_policy = "navsim_command3_velocity2_acceleration3"
    else:
        raise ValueError(
            "cannot build status_feature=8 from NAVSIM fields: "
            f"driving_command={raw_command.numel()} velocity={velocity.numel()} "
            f"acceleration={acceleration.numel()}"
        )
    if tuple(status.shape) != (8,):
        raise ValueError(f"expected status_feature shape (8,), got {tuple(status.shape)}")

    stage2_target_meta: dict[str, Any] = {}
    if use_answer_target:
        targets = extract_answer_target(row)
        target_source = "ar_answer_text"
    elif target_index is not None:
        if stage2_support_mode == "preserve_support":
            support_targets, stage2_target_meta = preserve_stage2_support_targets(
                target_index,
                str(item["token"]),
                future_trajectory,
            )
            targets = {"trajectory": future_trajectory, **support_targets}
            target_source = "stage2_pareto_support"
        else:
            trajectory, stage2_target_meta = select_stage2_target(target_index, str(item["token"]), target_selection)
            targets = {"trajectory": trajectory}
            target_source = "stage2_support_index"
    else:
        targets = {"trajectory": future_trajectory}
        target_source = "navsim_future_trajectory"

    features = {
        "history_trajectory": history,
        "high_command_one_hot": high_command,
        "status_feature": status,
    }
    meta = {
        "planner_state_source": "navsim_scene_frames",
        "target_source": target_source,
        "current_image": current_image,
        **current_image_meta(row, current_image_policy, current_image),
        "token": item["token"],
        "scene_loader_token": item["token"],
        "support_index_token": item["token"],
        "log_name": item["log_name"],
        "scene_token": item["scene_token"],
        "raw_driving_command": list(map(int, raw_command.flatten().tolist())),
        "raw_command_shape": list(raw_command.shape),
        "raw_command_values": [float(x) for x in raw_command.flatten().tolist()],
        "converted_high_command_one_hot": [float(x) for x in high_command.tolist()],
        "high_command_policy": command_policy,
        "high_command_values": [float(x) for x in high_command.tolist()],
        "velocity_values": [float(x) for x in velocity.flatten().tolist()],
        "acceleration_values": [float(x) for x in acceleration.flatten().tolist()],
        "status_feature_values": [float(x) for x in status.flatten().tolist()],
        "history_policy": "navsim_num_history_frames_4_relative_to_current",
        "status_policy": status_policy,
        **stage2_target_meta,
    }
    return features, targets, meta


def load_qwen(model_path: str, device: str, dtype: torch.dtype, max_image_size: int):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device).eval()

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.image_processor.max_pixels = max_image_size * max_image_size
    processor.image_processor.size["longest_edge"] = max_image_size * max_image_size
    return model, processor


def extract_hidden_state(
    row: dict[str, Any],
    image_base_path: Path,
    model: Any,
    processor: Any,
    model_path: str,
    device: str,
    assistant_prefix: str,
    hidden_padding: str,
    hidden_max_length: int,
    hidden_padding_side: str,
    hidden_truncation: bool,
    prompt_text: str | None = None,
    prompt_source: str = "row",
) -> tuple[torch.Tensor, dict[str, Any]]:
    prompt = str(prompt_text if prompt_text is not None else row["messages"][0]["content"]).replace("<image>", "")
    image_paths = [resolve_image_path(str(p), image_base_path) for p in row["images"]]
    messages = [{"role": "user", "content": []}]
    for image_path in image_paths:
        messages[0]["content"].append({"type": "image", "image": str(image_path)})
    messages[0]["content"].append({"type": "text", "text": prompt})

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text += assistant_prefix
    images = [Image.open(path).convert("RGB") for path in image_paths]
    old_padding_side = getattr(getattr(processor, "tokenizer", None), "padding_side", None)
    if old_padding_side is not None:
        processor.tokenizer.padding_side = hidden_padding_side
    processor_kwargs = {
        "text": [text],
        "images": images,
        "return_tensors": "pt",
        "padding": "max_length" if hidden_padding == "max_length" else False,
    }
    if hidden_padding == "max_length" or hidden_truncation:
        processor_kwargs["max_length"] = hidden_max_length
        processor_kwargs["truncation"] = hidden_truncation
    try:
        inputs = processor(**processor_kwargs).to(device)
    finally:
        if old_padding_side is not None:
            processor.tokenizer.padding_side = old_padding_side
    if hidden_padding == "max_length" and int(inputs["input_ids"].shape[1]) != hidden_max_length:
        raise RuntimeError(
            "fixed hidden padding requested but processor returned "
            f"seq_len={int(inputs['input_ids'].shape[1])}, expected {hidden_max_length}. "
            "Increase --hidden-max-length or enable --hidden-truncation explicitly."
        )
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hidden = outputs.hidden_states[-1].squeeze(0).float().cpu()
    meta = {
        "input_ids_shape": list(inputs["input_ids"].shape),
        "input_token_count": int(inputs["input_ids"].shape[1]),
        "valid_hidden_length": int(inputs["attention_mask"].sum().item()) if "attention_mask" in inputs else int(inputs["input_ids"].shape[1]),
        "hidden_shape": list(hidden.shape),
        "attention_mask_sum": int(inputs["attention_mask"].sum().item()) if "attention_mask" in inputs else None,
        "images": [str(path) for path in image_paths],
        "assistant_prefix": assistant_prefix,
        "hidden_source": "onevl_ar_answer_qwen3_vl",
        "hidden_layer": "final",
        "hidden_extraction_mode": "processor_chat_template_add_generation_prompt_final_hidden_state",
        "model_checkpoint": model_path,
        "processor_checkpoint": model_path,
        "prompt_template_hash": sha256_text(text),
        "prompt_text_hash": sha256_text(prompt),
        "chat_template_text_hash": sha256_text(text),
        "prompt_source": prompt_source,
        "hidden_padding": hidden_padding,
        "hidden_max_length": hidden_max_length,
        "hidden_padding_side": hidden_padding_side,
        "hidden_truncation": hidden_truncation,
    }
    return hidden, meta


def make_dummy_hidden(seq_len: int, hidden_dim: int, sample_index: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(20260626 + sample_index)
    return torch.randn(seq_len, hidden_dim, generator=generator, dtype=torch.float32) * 0.02


def make_dit_config(dit_type: str, sampling_method: str, vlm_feature_dim: int):
    sys.path.insert(0, DEFAULT_VLA_AD_ROOT)
    from navsim.agents.recogdrive.recogdrive_diffusion_planner import (  # noqa: PLC0415
        ReCogDriveDiffusionPlannerConfig,
    )

    if dit_type == "small":
        diffusion_model_cfg = {"num_heads": 8, "head_dim": 48, "num_layers": 16, "output_dim": 512}
        input_embedding_dim = 384
    else:
        diffusion_model_cfg = {"num_heads": 32, "head_dim": 48, "num_layers": 16, "output_dim": 1536}
        input_embedding_dim = 1536
    diffusion_model_cfg.update(
        {
            "dropout": 0.0,
            "attention_bias": True,
            "norm_eps": 1e-5,
            "interleave_attention": True,
        }
    )
    return ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg=diffusion_model_cfg,
        input_embedding_dim=input_embedding_dim,
        action_dim=3,
        action_horizon=8,
        sampling_method=sampling_method,
        num_inference_steps=5,
        grpo=False,
        model_dtype="float32",
        vlm_size="small",
        vlm_feature_dim=vlm_feature_dim,
        planner_dim=input_embedding_dim,
    )


def run_dit_forward(
    features: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    hidden: torch.Tensor,
    dit_type: str,
    sampling_method: str,
    dit_device: str,
) -> dict[str, Any]:
    sys.path.insert(0, DEFAULT_VLA_AD_ROOT)
    from transformers.feature_extraction_utils import BatchFeature  # noqa: PLC0415
    from navsim.agents.recogdrive.recogdrive_diffusion_planner import (  # noqa: PLC0415
        ReCogDriveDiffusionPlanner,
    )

    config = make_dit_config(dit_type, sampling_method, hidden.shape[-1])
    planner = ReCogDriveDiffusionPlanner(config).to(dit_device).train()
    vl_features = hidden.unsqueeze(0).to(dit_device)
    action_input = BatchFeature(
        data={
            "his_traj": features["history_trajectory"].reshape(1, -1).to(dit_device),
            "history_trajectory": features["history_trajectory"].unsqueeze(0).to(dit_device),
            "status_feature": features["status_feature"].unsqueeze(0).to(dit_device),
            "high_command_one_hot": features["high_command_one_hot"].unsqueeze(0).to(dit_device),
            "action": targets["trajectory"].unsqueeze(0).to(dit_device),
        }
    )
    with torch.no_grad():
        out = planner(vl_features, action_input)
    return {
        "loss": float(out["loss"].detach().cpu()),
        "diffusion_loss": float(out["diffusion_loss"].detach().cpu()),
        "vlm_feature_dim": int(hidden.shape[-1]),
        "vlm_seq_len": int(hidden.shape[0]),
        "dit_type": dit_type,
        "sampling_method": sampling_method,
        "planner_dim": int(config.input_embedding_dim),
    }


def write_cache_metadata(output_dir: Path, args: argparse.Namespace, target_index: dict[str, Any] | None) -> None:
    metadata = {
        "version": "onevl_ar_answer_recogdrive_stage2_cache_v1",
        "cache_schema": "flat_recogdrive_chunk" if args.cache_format == "flat" else "nested_bridge_debug",
        "contains_vlm_hidden": not bool(args.skip_vlm),
        "vlm_feature_dim": int(args.qwen_hidden_dim),
        "model_path": args.model_path,
        "data_jsonl": args.data_jsonl,
        "target_index_path": str(target_index.get("_path", "")) if target_index is not None else "",
        "target_selection": args.target_selection,
        "use_answer_target": bool(args.use_answer_target),
        "stage2_support_mode": args.stage2_support_mode,
        "num_history_frames": int(args.num_history_frames),
        "action_horizon": int(args.action_horizon),
        "hidden_padding": args.hidden_padding,
        "hidden_max_length": int(args.hidden_max_length),
        "hidden_padding_side": args.hidden_padding_side,
        "hidden_truncation": bool(args.hidden_truncation),
        "current_image_policy": args.current_image_policy,
        "stage1_submission_image_policy": "last",
        "prompt_source_mode": args.prompt_source,
        "include_control_convention": bool(args.include_control_convention),
        "hidden_source": "dummy" if args.skip_vlm else "onevl_ar_answer_qwen3_vl",
        "hidden_layer": "final",
        "hidden_extraction_mode": "dummy_hidden" if args.skip_vlm else "processor_chat_template_add_generation_prompt_final_hidden_state",
        "model_checkpoint": args.model_path,
        "processor_checkpoint": args.model_path,
        "prompt_source": "row.messages[0].content with <image> removed; processor.apply_chat_template(add_generation_prompt=True)",
        "range": {"start_index": args.start_index, "end_index": args.end_index},
        "shard": {"num_shards": args.num_shards, "shard_id": args.shard_id},
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, sort_keys=True)


def sample_cache_path(output_dir: Path, line_index: int, token: str) -> Path:
    safe_token = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in token)
    return output_dir / "samples" / f"{line_index:06d}_{safe_token}.pt"


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_jsonl)
    image_base_path = Path(args.image_base_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_cache and args.cache_format == "flat":
        (output_dir / "samples").mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, "/mnt")
    if args.vla_ad_root != DEFAULT_VLA_AD_ROOT:
        sys.path.insert(0, args.vla_ad_root)
    else:
        sys.path.insert(0, DEFAULT_VLA_AD_ROOT)

    rows = load_rows(
        data_path,
        args.max_samples,
        start_index=args.start_index,
        end_index=args.end_index,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    )
    if not rows:
        raise RuntimeError(
            f"no rows selected from {data_path} with start={args.start_index} end={args.end_index} "
            f"num_shards={args.num_shards} shard_id={args.shard_id} max_samples={args.max_samples}"
        )
    target_index = load_stage2_target_index(args.target_index_path)
    if args.save_cache:
        write_cache_metadata(output_dir, args, target_index)
    scene_mapping = build_scene_mapping(
        [row for _, row in rows],
        Path(args.navsim_log_path),
        num_history_frames=args.num_history_frames,
        num_future_frames=args.num_future_frames,
        current_image_policy=args.current_image_policy,
    )
    dtype = torch_dtype(args.dtype, args.device)
    model = processor = None
    if not args.skip_vlm:
        model, processor = load_qwen(args.model_path, args.device, dtype, args.max_image_size)

    summaries: list[dict[str, Any]] = []
    index_fp = None
    if args.save_cache and args.cache_format == "flat":
        index_fp = (output_dir / "index.jsonl").open("a", encoding="utf-8")
    try:
        for sample_index, (line_index, row) in enumerate(rows):
            current_image = select_current_image(row, args.current_image_policy)
            scene_item = scene_mapping[current_image]
            token = str(scene_item["token"])
            cache_path = (
                sample_cache_path(output_dir, line_index, token)
                if args.cache_format == "flat"
                else output_dir / f"sample_{sample_index:06d}.pt"
            )
            if args.save_cache and args.skip_existing and cache_path.is_file():
                result = {
                    "sample_index": sample_index,
                    "line_index": line_index,
                    "sample_token": token,
                    "cache_path": str(cache_path),
                    "skipped_existing": True,
                }
                summaries.append(result)
                print(json.dumps(result, ensure_ascii=False))
                continue

            features, targets, meta = extract_navsim_planner_tensors(
                row,
                scene_mapping,
                num_history_frames=args.num_history_frames,
                action_horizon=args.action_horizon,
                use_answer_target=args.use_answer_target,
                current_image_policy=args.current_image_policy,
                target_index=target_index,
                target_selection=args.target_selection,
                stage2_support_mode=args.stage2_support_mode,
            )
            row_prompt = str(row["messages"][0]["content"])
            scene_prompt = build_qwen_prompt_from_scene_tensors(
                features["history_trajectory"],
                features["high_command_one_hot"],
                features["status_feature"],
                include_control_convention=args.include_control_convention,
            )
            alignment_meta = prompt_scene_alignment(
                row_prompt,
                features["history_trajectory"],
                features["high_command_one_hot"],
                features["status_feature"],
            )
            if args.prompt_source == "row_strict_scene_check" and not alignment_meta["prompt_scene_alignment_pass"]:
                raise RuntimeError(
                    f"row prompt does not align with SceneLoader tensors at line {line_index}: "
                    f"{alignment_meta['prompt_scene_alignment_error']}"
                )
            prompt_for_hidden = scene_prompt if args.prompt_source == "scene" else row_prompt
            effective_prompt_source = "scene" if args.prompt_source == "scene" else "row"
            if args.skip_vlm:
                hidden = make_dummy_hidden(args.dummy_seq_len, args.qwen_hidden_dim, line_index)
                hidden_meta = {
                    "hidden_shape": list(hidden.shape),
                    "source": "dummy",
                    "hidden_source": "dummy",
                    "hidden_layer": "final",
                    "hidden_extraction_mode": "dummy_hidden",
                    "model_checkpoint": args.model_path,
                    "processor_checkpoint": args.model_path,
                    "prompt_template_hash": "",
                    "input_token_count": int(hidden.shape[0]),
                    "valid_hidden_length": int(hidden.shape[0]),
                    "prompt_source": effective_prompt_source,
                    "prompt_text_hash": sha256_text(prompt_for_hidden.replace("<image>", "")),
                }
            else:
                hidden, hidden_meta = extract_hidden_state(
                    row,
                    image_base_path,
                    model,
                    processor,
                    args.model_path,
                    args.device,
                    args.assistant_prefix,
                    args.hidden_padding,
                    args.hidden_max_length,
                    args.hidden_padding_side,
                    args.hidden_truncation,
                    prompt_text=prompt_for_hidden,
                    prompt_source=effective_prompt_source,
                )
                hidden_meta["source"] = "qwen3_vl_ar_answer"

            features_with_hidden = {**features, "last_hidden_state": hidden}
            result: dict[str, Any] = {
                "sample_index": sample_index,
                "line_index": line_index,
                "sample_token": token,
                "feature_shapes": {k: list(v.shape) for k, v in features_with_hidden.items()},
                "target_shapes": {k: list(v.shape) for k, v in targets.items()},
                "meta": {
                    **meta,
                    **alignment_meta,
                    **hidden_meta,
                    "prompt_source_mode": args.prompt_source,
                    "scene_prompt_hash": sha256_text(scene_prompt.replace("<image>", "")),
                    "row_prompt_hash": sha256_text(row_prompt.replace("<image>", "")),
                    "include_control_convention": bool(args.include_control_convention),
                },
            }
            if not args.no_dit_forward:
                result["dit_forward"] = run_dit_forward(
                    features,
                    targets,
                    hidden,
                    args.dit_type,
                    args.sampling_method,
                    args.dit_device,
                )

            if args.save_cache:
                if args.cache_format == "flat":
                    payload = {
                        **features_with_hidden,
                        **targets,
                        "meta": result["meta"],
                    }
                    torch.save(payload, cache_path)
                    record = {
                        "path": str(cache_path.relative_to(output_dir)),
                        "sample_token": token,
                        "log_name": meta["log_name"],
                        "scene_token": meta["scene_token"],
                        "line_index": line_index,
                        "current_image": meta["current_image"],
                        "target_source": meta["target_source"],
                    }
                    if index_fp is None:
                        raise RuntimeError("flat cache index file is not open")
                    index_fp.write(json.dumps(record, sort_keys=True) + "\n")
                    index_fp.flush()
                else:
                    torch.save(
                        {
                            "features": features_with_hidden,
                            "targets": targets,
                            "meta": result["meta"],
                        },
                        cache_path,
                    )
                result["cache_path"] = str(cache_path)

            summaries.append(result)
            print(json.dumps(result, ensure_ascii=False))
    finally:
        if index_fp is not None:
            index_fp.close()

    summary = {
        "model_path": args.model_path,
        "data_jsonl": str(data_path),
        "output_dir": str(output_dir),
        "skip_vlm": bool(args.skip_vlm),
        "dit_type": args.dit_type,
        "sampling_method": args.sampling_method,
        "planner_state_source": "navsim_scene_frames",
        "target_source": (
            "ar_answer_text"
            if args.use_answer_target
            else "stage2_support_index" if target_index is not None else "navsim_future_trajectory"
        ),
        "target_index_path": str(target_index.get("_path", "")) if target_index is not None else "",
        "target_selection": args.target_selection,
        "cache_format": args.cache_format,
        "hidden_padding": args.hidden_padding,
        "hidden_max_length": int(args.hidden_max_length),
        "hidden_padding_side": args.hidden_padding_side,
        "hidden_truncation": bool(args.hidden_truncation),
        "current_image_policy": args.current_image_policy,
        "stage1_submission_image_policy": "last",
        "prompt_source_mode": args.prompt_source,
        "include_control_convention": bool(args.include_control_convention),
        "selected_rows": len(rows),
        "samples": summaries,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
