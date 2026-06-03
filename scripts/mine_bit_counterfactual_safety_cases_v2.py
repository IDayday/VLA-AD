#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.bit_safety_selector import trajectory_delta_features  # noqa: E402
from scripts.build_bit_counterfactual_dataset import (  # noqa: E402
    compact_metrics,
    resolve_checkpoint_path,
    score_prediction,
    split_chunk_pattern,
)
from scripts.eval_bit_drive_pdm import (  # noqa: E402
    build_metric_cache_loader,
    build_pdm_tools,
    build_planner,
    chunk_dirs,
    dtype_from_precision,
    load_checkpoint,
    load_sample,
    load_yaml,
    make_batch,
    sample_paths,
    seed_everything,
)


METRIC_EPS = 1e-9
DEFAULT_ISOLATED_ROOT = Path(os.environ.get("BIT_WORK_ROOT", "/mnt/project/bit_drive_left_tail"))
DEFAULT_EXP_ROOT = Path(os.environ.get("BIT_EXP_ROOT", DEFAULT_ISOLATED_ROOT / "experiments/bit_drive"))
DEFAULT_SHARED_ROOT = Path(os.environ.get("VLA_AD_SHARED_ROOT", "/mnt/project/VLA-AD"))
SHARED_EXPERIMENT_ROOT = DEFAULT_SHARED_ROOT / "experiments"
SUPPORTED_CANDIDATES = {"base_det", "bit_det", "bit_stochastic_seed0", "bit_stochastic_seed1", "bit_stochastic_seed2"}


class CombinedMetricCacheLoader:
    def __init__(self, loaders: List[Any]) -> None:
        self.loaders = loaders
        self.metric_cache_paths: Dict[str, Any] = {}
        for loader in loaders:
            self.metric_cache_paths.update(getattr(loader, "metric_cache_paths", {}) or {})

    def get_from_token(self, token: str) -> Any:
        for loader in self.loaders:
            if token in getattr(loader, "metric_cache_paths", {}):
                return loader.get_from_token(token)
        raise KeyError(token)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D5-M2 multi-candidate interaction-aware BiT safety mining.")
    parser.add_argument("--navsim-root", type=Path, default=Path("/mnt/navsim"))
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--bit-config", type=Path, required=True)
    parser.add_argument("--bit-checkpoint", type=Path, default=None)
    parser.add_argument("--split", default="navtrain")
    parser.add_argument("--chunk-cache-root", type=Path, default=DEFAULT_SHARED_ROOT / "cache/recogdrive_expert_chunks/full_v1")
    parser.add_argument("--chunk-name-pattern", default=None)
    parser.add_argument("--metric-cache-dir", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-total-scenes", type=int, default=30000)
    parser.add_argument("--sampling-pool-multiplier", type=float, default=4.0)
    parser.add_argument("--max-sampling-pool-scenes", type=int, default=60000)
    parser.add_argument("--target-hard-nc", type=int, default=50)
    parser.add_argument("--target-hard-ttc", type=int, default=150)
    parser.add_argument("--target-soft-safety", type=int, default=500)
    parser.add_argument("--target-dac-fix", type=int, default=200)
    parser.add_argument("--candidate-modes", default="base_det,bit_det,bit_stochastic_seed0,bit_stochastic_seed1,bit_stochastic_seed2")
    parser.add_argument(
        "--sampling-mode",
        choices=(
            "random",
            "stratified_by_command",
            "high_speed",
            "high_ego_speed",
            "high_curvature",
            "high_history_curvature",
            "dense_agents",
            "pedestrian_present",
            "vehicle_present",
            "vehicle_count_high",
            "moving_vehicle_present",
            "turn_left",
            "turn_right",
            "endpoint_delta_high",
            "mixed",
        ),
        default="mixed",
    )
    parser.add_argument("--margin-ttc", type=float, default=0.2)
    parser.add_argument("--soft-early-step-threshold", type=float, default=0.75)
    parser.add_argument("--soft-early-x-threshold", type=float, default=1.0)
    parser.add_argument("--soft-heading-threshold", type=float, default=0.35)
    parser.add_argument("--instability-l2-threshold", type=float, default=1.0)
    parser.add_argument("--instability-heading-threshold", type=float, default=0.35)
    parser.add_argument("--neutral-sample-count", type=int, default=1000)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260601)
    return parser.parse_args()


def configure_navsim_env(navsim_root: Path) -> None:
    root = navsim_root.resolve()
    os.environ["NAVSIM_DATA_ROOT"] = str(root)
    os.environ.setdefault("OPENSCENE_DATA_ROOT", str(root / "openscene"))
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(root / "maps"))


def assert_not_shared_experiment_output(path: Path) -> None:
    resolved = path.resolve()
    shared = SHARED_EXPERIMENT_ROOT.resolve()
    if resolved == shared or shared in resolved.parents:
        raise RuntimeError(f"Refusing to write mining outputs under shared experiment root: {shared}")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def parse_modes(value: str) -> Tuple[List[str], List[str]]:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    modes = [mode for mode in requested if mode in SUPPORTED_CANDIDATES]
    warnings = [f"candidate_mode_skipped:{mode}" for mode in requested if mode not in SUPPORTED_CANDIDATES]
    if "base_det" not in modes:
        modes.insert(0, "base_det")
    if "bit_det" not in modes:
        modes.append("bit_det")
    return modes, warnings


def discover_bit_checkpoint(exp_root: Path, explicit: Optional[Path]) -> Path:
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            exp_root / "v3/C1_lateral_terminal/best.ckpt",
            exp_root / "v2/B5_terminal_only_context/best.ckpt",
            DEFAULT_EXP_ROOT / "v3/C1_lateral_terminal/best.ckpt",
            DEFAULT_EXP_ROOT / "v2/B5_terminal_only_context/best.ckpt",
            DEFAULT_SHARED_ROOT / "experiments/bit_drive/v3/C1_lateral_terminal/best.ckpt",
            DEFAULT_SHARED_ROOT / "experiments/bit_drive/v2/B5_terminal_only_context/best.ckpt",
        ]
    )
    for path in candidates:
        if path is not None and path.exists():
            return resolve_checkpoint_path(path)
    raise FileNotFoundError("No C1/B5 BiT checkpoint found. Pass --bit-checkpoint explicitly.")


def auto_metric_cache_dirs(split: str, output_dir: Path) -> List[Path]:
    terms = {split.lower(), split.lower().replace("nav", "")}
    roots = [
        output_dir,
        DEFAULT_EXP_ROOT,
        DEFAULT_ISOLATED_ROOT / "cache",
        DEFAULT_SHARED_ROOT / "experiments/bit_drive/select",
        DEFAULT_SHARED_ROOT / "cache",
    ]
    paths: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("metric_cache*"):
            if not path.is_dir():
                continue
            name = str(path).lower()
            if "navtest" in name or "/test" in name:
                continue
            if any(term and term in name for term in terms):
                paths.append(path)
    return sorted(set(paths))


def metric_loader_from_dirs(paths: List[Path]) -> Optional[CombinedMetricCacheLoader]:
    loaders = []
    for path in paths:
        try:
            loaders.append(build_metric_cache_loader(path))
        except Exception as exc:
            print(f"warning: metric cache skipped {path}: {exc!r}", flush=True)
    return CombinedMetricCacheLoader(loaders) if loaders else None


def tensor_list(value: Any) -> List[float]:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        out: List[float] = []
        for item in value:
            out.extend(tensor_list(item) if isinstance(item, (list, tuple)) else [float(item)] if isinstance(item, (int, float)) else [])
        return out
    return []


def command_bucket(sample: Optional[Dict[str, Any]], record: Dict[str, Any]) -> str:
    if sample is not None and isinstance(sample.get("high_command_one_hot"), torch.Tensor):
        command = sample["high_command_one_hot"].detach().float().reshape(-1)
        if command.numel() and torch.isfinite(command).all():
            return f"cmd_{int(torch.argmax(command).item())}"
    for key in ("command", "high_command", "log_name"):
        value = (sample or {}).get(key, record.get(key))
        if value is not None:
            return str(value)
    return "unknown"


def history_xy(sample: Dict[str, Any]) -> Optional[torch.Tensor]:
    hist = sample.get("history_trajectory")
    if not isinstance(hist, torch.Tensor):
        return None
    hist = hist.detach().float()
    if hist.ndim != 2 or hist.shape[0] < 2 or hist.shape[1] < 2:
        return None
    return hist[:, :2]


def history_speed(sample: Dict[str, Any]) -> float:
    hist = history_xy(sample)
    if hist is not None:
        steps = torch.linalg.vector_norm(hist[1:] - hist[:-1], dim=-1)
        if steps.numel() and torch.isfinite(steps).all():
            return float(steps.mean().item())
    status = tensor_list(sample.get("status_feature"))
    if len(status) >= 2 and all(math.isfinite(value) for value in status[:2]):
        return float(math.hypot(status[0], status[1]))
    return 0.0


def history_curvature(sample: Dict[str, Any]) -> float:
    hist = history_xy(sample)
    if hist is None or hist.shape[0] < 3:
        return 0.0
    delta = hist[1:] - hist[:-1]
    headings = torch.atan2(delta[:, 1], delta[:, 0])
    turn = torch.diff(headings)
    turn = torch.atan2(torch.sin(turn), torch.cos(turn)).abs()
    return float(turn.mean().item()) if turn.numel() and torch.isfinite(turn).all() else 0.0


def endpoint_proxy(sample: Dict[str, Any]) -> float:
    traj = sample.get("trajectory")
    if isinstance(traj, torch.Tensor) and traj.ndim == 2 and traj.shape[0] > 0 and traj.shape[1] >= 2:
        endpoint = traj.detach().float()[-1, :2]
        if torch.isfinite(endpoint).all():
            return float(torch.linalg.vector_norm(endpoint).item())
    return 0.0


def annotation_histogram(sample: Dict[str, Any]) -> Counter[str]:
    hist: Counter[str] = Counter()
    for key in ("annotations", "agent_annotations", "objects", "tracked_objects"):
        value = sample.get(key)
        if isinstance(value, dict):
            iterable = value.values()
        elif isinstance(value, (list, tuple)):
            iterable = value
        else:
            continue
        for item in iterable:
            if isinstance(item, dict):
                name = item.get("name") or item.get("category") or item.get("type") or item.get("label")
            else:
                name = getattr(item, "name", None) or getattr(item, "category", None) or getattr(item, "tracked_object_type", None)
            if name is not None:
                hist[str(name).lower()] += 1
    return hist


def command_is_left(command: str) -> bool:
    value = command.lower()
    return "left" in value or value in {"cmd_0", "0"}


def command_is_right(command: str) -> bool:
    value = command.lower()
    return "right" in value or value in {"cmd_2", "2"}


def scene_metadata(sample: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    hist = annotation_histogram(sample)
    vehicles = sum(count for name, count in hist.items() if "vehicle" in name or "car" in name or "bus" in name or "truck" in name)
    pedestrians = sum(count for name, count in hist.items() if "pedestrian" in name or "person" in name)
    moving_vehicles = sum(
        count
        for name, count in hist.items()
        if ("vehicle" in name or "car" in name or "bus" in name or "truck" in name) and ("moving" in name or "dynamic" in name)
    )
    command = command_bucket(sample, record)
    return {
        "num_vehicles": int(vehicles),
        "num_pedestrians": int(pedestrians),
        "num_agents": int(sum(hist.values())),
        "moving_vehicle_present": bool(moving_vehicles > 0),
        "ego_speed": history_speed(sample),
        "history_curvature": history_curvature(sample),
        "endpoint_delta_proxy": endpoint_proxy(sample),
        "command_class": command,
        "turn_left": command_is_left(command),
        "turn_right": command_is_right(command),
        "front_camera_available": bool(sample.get("image_path_tensor") is not None or sample.get("front_camera") is not None),
        "future_frames_available": bool(sample.get("trajectory") is not None),
        "annotation_names_histogram": dict(hist),
        "annotation_access": "available" if hist else "unavailable",
    }


def scene_strata(meta: Dict[str, Any]) -> List[str]:
    strata = ["random"]
    if int(meta.get("num_pedestrians", 0)) > 0:
        strata.append("pedestrian_present")
    if int(meta.get("num_vehicles", 0)) > 0:
        strata.append("vehicle_present")
    if int(meta.get("num_vehicles", 0)) >= 5:
        strata.append("vehicle_count_high")
    if bool(meta.get("moving_vehicle_present", False)):
        strata.append("moving_vehicle_present")
    if bool(meta.get("turn_left", False)):
        strata.append("turn_left")
    if bool(meta.get("turn_right", False)):
        strata.append("turn_right")
    if float(meta.get("ego_speed", 0.0)) > 0.5:
        strata.append("high_ego_speed")
    if float(meta.get("history_curvature", 0.0)) > 0.1:
        strata.append("high_history_curvature")
    if int(meta.get("num_agents", 0)) >= 8:
        strata.append("dense_agents")
    return strata


def primary_stratum(meta: Dict[str, Any]) -> str:
    priority = (
        "pedestrian_present",
        "moving_vehicle_present",
        "vehicle_count_high",
        "dense_agents",
        "turn_left",
        "turn_right",
        "high_ego_speed",
        "high_history_curvature",
        "vehicle_present",
    )
    strata = set(scene_strata(meta))
    for name in priority:
        if name in strata:
            return name
    return "random"


def candidate_chunk_dirs(args: argparse.Namespace, local_args: argparse.Namespace, *, infer_from_metric_caches: bool) -> List[Path]:
    if not infer_from_metric_caches:
        return chunk_dirs(local_args)
    inferred: List[Path] = []
    for metric_dir in args.metric_cache_dir or []:
        match = re.search(r"chunk0*([0-9]+)", metric_dir.name)
        if not match:
            continue
        chunk_name = f"train_full_chunk_{int(match.group(1)):06d}"
        chunk_path = args.chunk_cache_root / chunk_name
        if chunk_path.is_dir():
            inferred.append(chunk_path)
    if inferred:
        return sorted(set(inferred))
    return chunk_dirs(local_args)


def candidate_items(args: argparse.Namespace, allowed_tokens: Optional[set[str]] = None) -> List[Tuple[Path, Path, Dict[str, Any], Dict[str, Any]]]:
    infer_from_metric_caches = args.chunk_name_pattern is None
    chunk_name_pattern = args.chunk_name_pattern or split_chunk_pattern(args.split)
    local_args = argparse.Namespace(
        chunk_cache_dir=None,
        chunk_cache_root=args.chunk_cache_root,
        chunk_name_pattern=chunk_name_pattern,
        max_samples=None,
    )
    items: List[Tuple[Path, Path, Dict[str, Any], Dict[str, Any]]] = []
    pool_limit = max(
        int(args.max_total_scenes),
        min(int(args.max_sampling_pool_scenes), int(math.ceil(float(args.max_total_scenes) * float(args.sampling_pool_multiplier)))),
    )
    for chunk, path, record in sample_paths(candidate_chunk_dirs(args, local_args, infer_from_metric_caches=infer_from_metric_caches), None):
        record_token = str(record.get("sample_token") or path.stem)
        if allowed_tokens is not None and record_token not in allowed_tokens:
            continue
        try:
            sample = load_sample(path)
            meta = scene_metadata(sample, record)
        except Exception as exc:
            meta = {
                "num_vehicles": 0,
                "num_pedestrians": 0,
                "num_agents": 0,
                "moving_vehicle_present": False,
                "ego_speed": 0.0,
                "history_curvature": 0.0,
                "endpoint_delta_proxy": 0.0,
                "command_class": command_bucket(None, record),
                "turn_left": False,
                "turn_right": False,
                "front_camera_available": False,
                "future_frames_available": False,
                "annotation_names_histogram": {},
                "annotation_access": f"failed:{exc!r}",
            }
        items.append((chunk, path, record, meta))
        if len(items) >= pool_limit:
            break
    rng = random.Random(args.seed)
    rng.shuffle(items)
    return sample_order(items, args)[: args.max_total_scenes]


def round_robin(items: List[Tuple[Path, Path, Dict[str, Any], Dict[str, Any]]], key: str) -> List[Tuple[Path, Path, Dict[str, Any], Dict[str, Any]]]:
    groups: Dict[str, List[Tuple[Path, Path, Dict[str, Any], Dict[str, Any]]]] = defaultdict(list)
    for item in items:
        groups[str(item[3].get(key, "unknown"))].append(item)
    out: List[Tuple[Path, Path, Dict[str, Any], Dict[str, Any]]] = []
    while groups:
        for group_key in sorted(list(groups)):
            group = groups[group_key]
            if group:
                out.append(group.pop(0))
            if not group:
                groups.pop(group_key, None)
    return out


def unique_items(items: Iterable[Tuple[Path, Path, Dict[str, Any], Dict[str, Any]]]) -> List[Tuple[Path, Path, Dict[str, Any], Dict[str, Any]]]:
    seen: set[str] = set()
    out = []
    for item in items:
        token = str(item[2].get("sample_token") or item[1])
        if token in seen:
            continue
        seen.add(token)
        out.append(item)
    return out


def sample_order(items: List[Tuple[Path, Path, Dict[str, Any], Dict[str, Any]]], args: argparse.Namespace) -> List[Tuple[Path, Path, Dict[str, Any], Dict[str, Any]]]:
    if args.sampling_mode == "random":
        return items
    if args.sampling_mode == "stratified_by_command":
        return round_robin(items, "command_class")
    if args.sampling_mode in {"high_speed", "high_ego_speed"}:
        return sorted(items, key=lambda item: float(item[3].get("ego_speed", 0.0)), reverse=True)
    if args.sampling_mode in {"high_curvature", "high_history_curvature"}:
        return sorted(items, key=lambda item: float(item[3].get("history_curvature", 0.0)), reverse=True)
    if args.sampling_mode == "dense_agents":
        return sorted(items, key=lambda item: int(item[3].get("num_agents", 0)), reverse=True)
    if args.sampling_mode == "pedestrian_present":
        return sorted(items, key=lambda item: int(item[3].get("num_pedestrians", 0)), reverse=True)
    if args.sampling_mode in {"vehicle_present", "vehicle_count_high"}:
        return sorted(items, key=lambda item: int(item[3].get("num_vehicles", 0)), reverse=True)
    if args.sampling_mode == "moving_vehicle_present":
        return sorted(items, key=lambda item: bool(item[3].get("moving_vehicle_present", False)), reverse=True)
    if args.sampling_mode == "turn_left":
        return sorted(items, key=lambda item: bool(item[3].get("turn_left", False)), reverse=True)
    if args.sampling_mode == "turn_right":
        return sorted(items, key=lambda item: bool(item[3].get("turn_right", False)), reverse=True)
    if args.sampling_mode == "endpoint_delta_high":
        return sorted(items, key=lambda item: float(item[3].get("endpoint_delta_proxy", 0.0)), reverse=True)

    n = len(items)
    quotas = {
        "random": int(0.30 * n),
        "command": int(0.20 * n),
        "speed_curvature": int(0.20 * n),
        "interaction": int(0.30 * n),
    }
    random_pool = list(items)
    command_pool = round_robin(items, "command_class")
    speed_curv_pool = unique_items(
        sorted(items, key=lambda item: float(item[3].get("ego_speed", 0.0)), reverse=True)
        + sorted(items, key=lambda item: float(item[3].get("history_curvature", 0.0)), reverse=True)
    )
    interaction_pool = unique_items(
        sorted(items, key=lambda item: int(item[3].get("num_agents", 0)), reverse=True)
        + sorted(items, key=lambda item: int(item[3].get("num_pedestrians", 0)), reverse=True)
        + sorted(items, key=lambda item: int(item[3].get("num_vehicles", 0)), reverse=True)
        + sorted(items, key=lambda item: bool(item[3].get("moving_vehicle_present", False)), reverse=True)
    )
    ordered: List[Tuple[Path, Path, Dict[str, Any], Dict[str, Any]]] = []
    ordered.extend(random_pool[: quotas["random"]])
    ordered.extend(command_pool[: quotas["command"]])
    ordered.extend(speed_curv_pool[: quotas["speed_curvature"]])
    ordered.extend(interaction_pool[: quotas["interaction"]])
    ordered.extend(items)
    return unique_items(ordered)


def compact_metric_delta(base_metrics: Optional[Dict[str, Any]], candidate_metrics: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for key in ("pdm", "dac", "nc", "ttc", "ego", "comfort"):
        base = None if not base_metrics or base_metrics.get(key) is None else float(base_metrics[key])
        candidate = None if not candidate_metrics or candidate_metrics.get(key) is None else float(candidate_metrics[key])
        out[key] = None if base is None or candidate is None else candidate - base
    return out


def proxy_features(base_traj: torch.Tensor, candidate_traj: torch.Tensor) -> Dict[str, float]:
    features = trajectory_delta_features(base_traj, candidate_traj)
    features["endpoint_x_delta"] = float(candidate_traj[-1, 0].item() - base_traj[-1, 0].item())
    features["endpoint_y_delta"] = float(candidate_traj[-1, 1].item() - base_traj[-1, 1].item())
    features["heading_delta_mean"] = float((candidate_traj[:, 2] - base_traj[:, 2]).mean().item())
    features["max_lateral_delta"] = float(torch.abs(candidate_traj[:, 1] - base_traj[:, 1]).max().item())
    features["max_forward_delta"] = float(torch.max(candidate_traj[:, 0] - base_traj[:, 0]).item())
    return features


def aggressive_proxy(features: Dict[str, float], *, early_step_threshold: float, early_x_threshold: float, heading_threshold: float) -> bool:
    return (
        float(features.get("early_x_delta_mean", 0.0)) > float(early_x_threshold)
        or float(features.get("early_step_length_delta", 0.0)) > float(early_step_threshold)
        or abs(float(features.get("endpoint_y_delta", 0.0))) > 1.5
        or abs(float(features.get("heading_delta_mean", 0.0))) > float(heading_threshold)
        or abs(float(features.get("max_lateral_delta", 0.0))) > 2.0
    )


def tag_candidate(
    base_metrics: Optional[Dict[str, Any]],
    candidate_metrics: Optional[Dict[str, Any]],
    features: Dict[str, float],
    *,
    margin_ttc: float,
    early_step_threshold: float,
    early_x_threshold: float,
    heading_threshold: float,
) -> Tuple[List[str], Dict[str, Any]]:
    tags: List[str] = []
    base_nc = None if not base_metrics or base_metrics.get("nc") is None else float(base_metrics["nc"])
    bit_nc = None if not candidate_metrics or candidate_metrics.get("nc") is None else float(candidate_metrics["nc"])
    base_ttc = None if not base_metrics or base_metrics.get("ttc") is None else float(base_metrics["ttc"])
    bit_ttc = None if not candidate_metrics or candidate_metrics.get("ttc") is None else float(candidate_metrics["ttc"])
    base_dac = None if not base_metrics or base_metrics.get("dac") is None else float(base_metrics["dac"])
    bit_dac = None if not candidate_metrics or candidate_metrics.get("dac") is None else float(candidate_metrics["dac"])
    base_pdm = None if not base_metrics or base_metrics.get("pdm") is None else float(base_metrics["pdm"])
    bit_pdm = None if not candidate_metrics or candidate_metrics.get("pdm") is None else float(candidate_metrics["pdm"])

    hard_nc = base_nc is not None and bit_nc is not None and base_nc > METRIC_EPS and bit_nc <= METRIC_EPS
    hard_ttc = base_ttc is not None and bit_ttc is not None and base_ttc > METRIC_EPS and bit_ttc <= METRIC_EPS
    ttc_score_drop = (
        base_ttc is not None
        and bit_ttc is not None
        and bit_ttc > METRIC_EPS
        and bit_ttc < base_ttc - float(margin_ttc)
    )
    ttc_zero_or_drop = hard_ttc or ttc_score_drop
    aggressive_motion = (
        float(features.get("early_step_length_delta", 0.0)) > float(early_step_threshold)
        or float(features.get("early_x_delta_mean", 0.0)) > float(early_x_threshold)
    )
    large_heading = abs(float(features.get("heading_delta_mean", 0.0))) > float(heading_threshold)
    dac_fix = (
        base_dac is not None
        and bit_dac is not None
        and ((base_dac <= METRIC_EPS and bit_dac > METRIC_EPS) or bit_dac > base_dac + 0.2)
    )
    zero_fix = base_pdm is not None and bit_pdm is not None and base_pdm <= METRIC_EPS and bit_pdm > METRIC_EPS
    bit_bad = base_pdm is not None and bit_pdm is not None and bit_pdm + METRIC_EPS < base_pdm
    proxy_safety = (candidate_metrics is None or base_metrics is None) and aggressive_proxy(
        features,
        early_step_threshold=early_step_threshold,
        early_x_threshold=early_x_threshold,
        heading_threshold=heading_threshold,
    )

    if hard_nc:
        tags.append("hard_nc_regression")
    if hard_ttc:
        tags.append("hard_ttc_regression")
    if ttc_score_drop:
        tags.append("ttc_score_drop")
        tags.append("soft_ttc_regression")
    if ttc_zero_or_drop:
        tags.append("ttc_zero_or_drop")
    if aggressive_motion:
        tags.append("aggressive_early_motion")
    if large_heading:
        tags.append("large_heading_change")
    bit_safety_worse = hard_nc or hard_ttc or ttc_score_drop or aggressive_motion or large_heading or proxy_safety
    if bit_safety_worse:
        tags.append("bit_safety_worse")
        tags.append("soft_safety_regression")
    if dac_fix:
        tags.append("dac_fix")
    if zero_fix:
        tags.append("zero_fix")
    if bit_bad:
        tags.append("bit_bad")
    tags = sorted(set(tags)) or ["neutral"]
    if hard_nc or hard_ttc:
        safety_weight = 1.0
    elif ttc_score_drop:
        safety_weight = 0.5
    elif aggressive_motion or large_heading or proxy_safety:
        safety_weight = 0.25
    else:
        safety_weight = 0.0
    return tags, {
        "hard_safety_mask": bool(hard_nc or hard_ttc),
        "soft_safety_mask": bool(bit_safety_worse and not (hard_nc or hard_ttc)),
        "safety_weight": safety_weight,
        "dac_fix_mask": bool(dac_fix),
        "dac_weight": 1.0 if dac_fix else 0.0,
        "soft_label_flags": {
            "ttc_score_drop": bool(ttc_score_drop),
            "ttc_zero_or_drop": bool(ttc_zero_or_drop),
            "aggressive_early_motion": bool(aggressive_motion),
            "large_heading_change": bool(large_heading),
            "candidate_instability": False,
            "bit_safety_worse": bool(bit_safety_worse),
        },
    }


def candidate_prediction(
    mode: str,
    base_planner: Any,
    bit_planner: Any,
    vl_features: torch.Tensor,
    action_input: Any,
    *,
    seed: int,
) -> torch.Tensor:
    if mode == "base_det":
        output = base_planner.get_action(vl_features, action_input, deterministic=True)
    elif mode == "bit_det":
        output = bit_planner.get_action(vl_features, action_input, deterministic=True)
    elif mode.startswith("bit_stochastic_seed"):
        offset = int(mode.replace("bit_stochastic_seed", ""))
        torch.manual_seed(seed + offset)
        torch.cuda.manual_seed_all(seed + offset)
        output = bit_planner.get_action(vl_features, action_input, deterministic=False)
    else:
        raise ValueError(f"Unsupported candidate mode: {mode}")
    pred = output["pred_traj"].detach().float().cpu().squeeze(0)
    if not torch.isfinite(pred).all():
        raise RuntimeError(f"Non-finite prediction for candidate mode {mode}")
    return pred


def risk_labels_from_metrics(metrics: Optional[Dict[str, Any]]) -> Optional[List[float]]:
    if not metrics:
        return None
    return [
        1.0 if float(metrics.get("pdm") or 0.0) <= METRIC_EPS else 0.0,
        1.0 if float(metrics.get("dac") or 0.0) <= METRIC_EPS else 0.0,
        1.0 if float(metrics.get("nc") or 0.0) <= METRIC_EPS else 0.0,
        1.0 if float(metrics.get("ttc") or 0.0) <= METRIC_EPS else 0.0,
    ]


def select_training_rows(candidate_rows: List[Dict[str, Any]], neutral_count: int, seed: int) -> List[Dict[str, Any]]:
    safety = [row for row in candidate_rows if bool(row.get("hard_safety_mask")) or bool(row.get("soft_safety_mask"))]
    dac = [row for row in candidate_rows if bool(row.get("dac_fix_mask")) and row not in safety]
    neutral = [row for row in candidate_rows if row.get("tags") == ["neutral"] and row.get("candidate_mode") != "base_det"]
    rng = random.Random(seed)
    rng.shuffle(neutral)
    selected = safety + dac + neutral[: max(0, int(neutral_count))]
    return selected


def row_has(row: Dict[str, Any], tag: str) -> bool:
    return tag in set(row.get("tags") or [])


def add_candidate_instability(
    mode_rows: List[Dict[str, Any]],
    *,
    l2_threshold: float,
    heading_threshold: float,
) -> bool:
    stochastic = [row for row in mode_rows if str(row.get("candidate_mode", "")).startswith("bit_stochastic_seed")]
    if len(stochastic) < 2:
        return False
    trajs = [torch.tensor(row["candidate_pred_traj"], dtype=torch.float32) for row in stochastic]
    max_l2 = 0.0
    max_heading = 0.0
    for left in range(len(trajs)):
        for right in range(left + 1, len(trajs)):
            delta = trajs[left] - trajs[right]
            max_l2 = max(max_l2, float(torch.linalg.vector_norm(delta[:, :2], dim=-1).mean().item()))
            max_heading = max(max_heading, float(torch.abs(delta[:, 2]).mean().item()))
    unstable = max_l2 > float(l2_threshold) or max_heading > float(heading_threshold)
    if not unstable:
        return False
    for row in stochastic:
        tags = set(row.get("tags") or [])
        tags.update({"candidate_instability", "bit_safety_worse", "soft_safety_regression"})
        row["tags"] = sorted(tags)
        row["bit_counterfactual_tags"] = row["tags"]
        flags = row.setdefault("soft_label_flags", {})
        flags["candidate_instability"] = True
        flags["bit_safety_worse"] = True
        if not bool(row.get("hard_safety_mask", False)):
            row["soft_safety_mask"] = True
            row["safety_weight"] = max(float(row.get("safety_weight", 0.0)), 0.25)
        row["bit_safety_regression_mask"] = bool(row.get("hard_safety_mask", False) or row.get("soft_safety_mask", False))
        features = row.setdefault("features", {})
        features["candidate_instability_l2"] = max_l2
        features["candidate_instability_heading"] = max_heading
        row["trajectory_delta_features"] = features
    return True


def counts_for(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "candidate_rows": len(rows),
        "hard_nc_regressions": sum(1 for row in rows if "hard_nc_regression" in row.get("tags", [])),
        "hard_ttc_regressions": sum(1 for row in rows if "hard_ttc_regression" in row.get("tags", [])),
        "soft_safety_regressions": sum(1 for row in rows if "soft_safety_regression" in row.get("tags", [])),
        "dac_fixes": sum(1 for row in rows if "dac_fix" in row.get("tags", [])),
        "zero_fixes": sum(1 for row in rows if "zero_fix" in row.get("tags", [])),
    }


def label_histograms(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(row.get("tags") or [])
        flags = row.get("soft_label_flags")
        if isinstance(flags, dict):
            for name, enabled in flags.items():
                if enabled:
                    counter[f"flag:{name}"] += 1
    return dict(counter)


def yield_table(rows: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, int]]:
    table: Dict[str, Dict[str, int]] = {}
    for row in rows:
        values = row.get(key)
        if values is None:
            values = ["missing"]
        elif not isinstance(values, list):
            values = [values]
        for value in values:
            name = str(value)
            bucket = table.setdefault(
                name,
                {
                    "candidate_rows": 0,
                    "hard_nc": 0,
                    "hard_ttc": 0,
                    "soft_safety": 0,
                    "dac_fixes": 0,
                    "zero_fixes": 0,
                },
            )
            bucket["candidate_rows"] += 1
            bucket["hard_nc"] += int(row_has(row, "hard_nc_regression"))
            bucket["hard_ttc"] += int(row_has(row, "hard_ttc_regression"))
            bucket["soft_safety"] += int(row_has(row, "soft_safety_regression"))
            bucket["dac_fixes"] += int(row_has(row, "dac_fix"))
            bucket["zero_fixes"] += int(row_has(row, "zero_fix"))
    return table


def stratum_table(
    rows: List[Dict[str, Any]],
    attempted: Counter[str],
    valid: Counter[str],
) -> Dict[str, Dict[str, int]]:
    table = yield_table(rows, "sampling_strata")
    for name in set(table) | set(attempted) | set(valid):
        bucket = table.setdefault(
            name,
            {
                "candidate_rows": 0,
                "hard_nc": 0,
                "hard_ttc": 0,
                "soft_safety": 0,
                "dac_fixes": 0,
                "zero_fixes": 0,
            },
        )
        bucket["scenes_attempted"] = int(attempted.get(name, 0))
        bucket["scenes_valid"] = int(valid.get(name, 0))
    return table


def gate_status(summary: Dict[str, Any]) -> Dict[str, Any]:
    hard_nc = int(summary.get("hard_nc_regressions", 0))
    hard_ttc = int(summary.get("hard_ttc_regressions", 0))
    soft = int(summary.get("soft_safety_regressions", 0))
    dac = int(summary.get("dac_fixes", 0))
    ideal = hard_nc >= 50 and hard_ttc >= 150 and dac >= 200
    acceptable = hard_nc >= 30 and hard_ttc >= 100 and soft >= 500 and dac >= 200
    return {"ideal_gate_passed": ideal, "acceptable_gate_passed": acceptable, "gate_passed": ideal or acceptable}


def write_summary_markdown(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# D5-M2 Interaction-Aware Safety Mining Summary",
        "",
        f"Training allowed: `{summary['training_allowed']}`",
        f"Gate passed: `{summary['gate_passed']}`",
        f"Scenes processed: `{summary['scenes_processed']}`",
        f"Candidate rows: `{summary['candidate_rows']}`",
        "",
        "| Label | Count | Target |",
        "| --- | ---: | ---: |",
        f"| hard NC regressions | {summary['hard_nc_regressions']} | {summary['target_hard_nc']} |",
        f"| hard TTC regressions | {summary['hard_ttc_regressions']} | {summary['target_hard_ttc']} |",
        f"| soft safety regressions | {summary['soft_safety_regressions']} | {summary['target_soft_safety']} |",
        f"| DAC fixes | {summary['dac_fixes']} | {summary['target_dac_fix']} |",
        "",
        "## Candidate Modes",
        "",
        "```json",
        json.dumps(summary["candidate_modes"], indent=2, sort_keys=True),
        "```",
        "",
        "## Sampling Distribution",
        "",
        "```json",
        json.dumps(summary["sampling_distribution"], indent=2, sort_keys=True),
        "```",
        "",
        "## Candidate Mode Yields",
        "",
        "```json",
        json.dumps(summary["candidate_mode_yields"], indent=2, sort_keys=True),
        "```",
        "",
        "## Sampling Stratum Yields",
        "",
        "```json",
        json.dumps(summary["sampling_stratum_yields"], indent=2, sort_keys=True),
        "```",
        "",
        "## Label Histograms",
        "",
        "```json",
        json.dumps(summary["label_histograms"], indent=2, sort_keys=True),
        "```",
        "",
        "## Warnings",
        "",
        "```json",
        json.dumps(summary["warnings"], indent=2, sort_keys=True),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    assert_not_shared_experiment_output(args.output_dir)
    configure_navsim_env(args.navsim_root)
    seed_everything(args.seed)
    split_lower = args.split.lower()
    training_allowed = "test" not in split_lower and not args.analysis_only
    if "test" in split_lower and not args.analysis_only:
        raise RuntimeError("Refusing to mine training labels from navtest/test split. Use --analysis-only for analysis output.")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dtype = dtype_from_precision(args.precision, device)
    modes, warnings = parse_modes(args.candidate_modes)
    if args.num_gpus > 1:
        warnings.append(f"num_gpus_requested:{args.num_gpus}; single-process miner uses {device}")

    metric_dirs = args.metric_cache_dir or auto_metric_cache_dirs(args.split, args.output_dir)
    metric_loader = metric_loader_from_dirs(metric_dirs)
    if metric_loader is None:
        raise RuntimeError("No non-test metric cache found. Pass --metric-cache-dir for the requested split.")
    pdm_tools = build_pdm_tools()

    bit_checkpoint = discover_bit_checkpoint(DEFAULT_EXP_ROOT, args.bit_checkpoint)
    base_planner = build_planner(load_yaml(args.base_config)).to(device)
    bit_planner = build_planner(load_yaml(args.bit_config)).to(device)
    if dtype != torch.float32:
        base_planner = base_planner.to(dtype=dtype)
        bit_planner = bit_planner.to(dtype=dtype)
    load_checkpoint(base_planner, resolve_checkpoint_path(args.base_checkpoint))
    load_checkpoint(bit_planner, bit_checkpoint)
    base_planner.eval()
    bit_planner.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows: List[Dict[str, Any]] = []
    scene_rows: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    sampling_distribution: Counter[str] = Counter()
    stratum_attempted: Counter[str] = Counter()
    stratum_valid: Counter[str] = Counter()
    start = time.time()

    items = candidate_items(args, set(metric_loader.metric_cache_paths))
    for index, (chunk, path, record, meta) in enumerate(items, start=1):
        sample = load_sample(path)
        sample_token = str(sample.get("sample_token", record.get("sample_token", path.stem)))
        scene_token = str(sample.get("scene_token", record.get("scene_token", path.stem)))
        strata = scene_strata(meta)
        bucket = primary_stratum(meta)
        sampling_distribution[bucket] += 1
        for stratum in strata:
            stratum_attempted[stratum] += 1
        if sample_token not in metric_loader.metric_cache_paths:
            rejected.append({"sample_token": sample_token, "scene_token": scene_token, "error": "missing_metric_cache", "split": args.split})
            continue
        vl_features, action_input = make_batch(sample, device, dtype)
        try:
            with torch.no_grad():
                base_traj = candidate_prediction("base_det", base_planner, bit_planner, vl_features, action_input, seed=args.seed + index)
        except Exception as exc:
            rejected.append({"sample_token": sample_token, "scene_token": scene_token, "error": repr(exc), "split": args.split})
            continue
        base_metrics = compact_metrics(score_prediction(sample_token=sample_token, pred=base_traj, metric_cache_loader=metric_loader, pdm_tools=pdm_tools))
        for stratum in strata:
            stratum_valid[stratum] += 1
        scene_counts = Counter()
        mode_rows = []
        for mode in modes:
            try:
                with torch.no_grad():
                    pred = base_traj if mode == "base_det" else candidate_prediction(mode, base_planner, bit_planner, vl_features, action_input, seed=args.seed + index * 100)
                candidate_metrics = base_metrics if mode == "base_det" else compact_metrics(
                    score_prediction(sample_token=sample_token, pred=pred, metric_cache_loader=metric_loader, pdm_tools=pdm_tools)
                )
                features = proxy_features(base_traj, pred)
                tags, masks = tag_candidate(
                    base_metrics,
                    candidate_metrics,
                    features,
                    margin_ttc=args.margin_ttc,
                    early_step_threshold=args.soft_early_step_threshold,
                    early_x_threshold=args.soft_early_x_threshold,
                    heading_threshold=args.soft_heading_threshold,
                )
                delta = compact_metric_delta(base_metrics, candidate_metrics)
                row = {
                    "scene_token": scene_token,
                    "sample_token": sample_token,
                    "split": args.split,
                    "chunk": chunk.name,
                    "candidate_mode": mode,
                    "counterfactual_candidate_mode": mode,
                    "base_pred_traj": base_traj.tolist(),
                    "bit_pred_traj": pred.tolist(),
                    "candidate_pred_traj": pred.tolist(),
                    "base_metrics": base_metrics,
                    "bit_metrics": candidate_metrics,
                    "candidate_metrics": candidate_metrics,
                    "delta": delta,
                    "delta_metrics": delta,
                    "tags": tags,
                    "features": features,
                    "trajectory_delta_features": features,
                    "scene_metadata": meta,
                    "sampling_bucket": bucket,
                    "sampling_stratum": bucket,
                    "sampling_strata": strata,
                    "bit_counterfactual_tags": tags,
                    "bit_risk_labels": risk_labels_from_metrics(candidate_metrics),
                    **masks,
                    "bit_safety_regression_mask": bool(masks["hard_safety_mask"] or masks["soft_safety_mask"]),
                    "bit_nc_regression_mask": "hard_nc_regression" in tags,
                    "bit_ttc_regression_mask": "hard_ttc_regression" in tags,
                    "bit_dac_fix_mask": bool(masks["dac_fix_mask"]),
                }
                mode_rows.append(row)
                candidate_rows.append(row)
                for tag in tags:
                    scene_counts[tag] += 1
            except Exception as exc:
                rejected.append({"sample_token": sample_token, "scene_token": scene_token, "candidate_mode": mode, "error": repr(exc), "split": args.split})
        if add_candidate_instability(
            mode_rows,
            l2_threshold=args.instability_l2_threshold,
            heading_threshold=args.instability_heading_threshold,
        ):
            pass
        scene_counts = Counter(tag for row in mode_rows for tag in row.get("tags", []))
        scene_rows.append(
            {
                "scene_token": scene_token,
                "sample_token": sample_token,
                "split": args.split,
                "chunk": chunk.name,
                "candidate_count": len(mode_rows),
                "tag_counts": dict(scene_counts),
                "scene_metadata": meta,
                "sampling_bucket": bucket,
                "sampling_stratum": bucket,
                "sampling_strata": strata,
            }
        )
        summary_counts = counts_for(candidate_rows)
        if args.log_every and index % args.log_every == 0:
            progress = {
                "scenes_processed": index,
                "elapsed_sec": round(time.time() - start, 2),
                **summary_counts,
            }
            (args.output_dir / "mining_progress.json").write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(progress, sort_keys=True), flush=True)
        target_gate = {
            "hard_nc_regressions": summary_counts["hard_nc_regressions"],
            "hard_ttc_regressions": summary_counts["hard_ttc_regressions"],
            "soft_safety_regressions": summary_counts["soft_safety_regressions"],
            "dac_fixes": summary_counts["dac_fixes"],
        }
        ideal_done = (
            target_gate["hard_nc_regressions"] >= args.target_hard_nc
            and target_gate["hard_ttc_regressions"] >= args.target_hard_ttc
            and target_gate["dac_fixes"] >= args.target_dac_fix
        )
        acceptable_done = (
            target_gate["hard_nc_regressions"] >= 30
            and target_gate["hard_ttc_regressions"] >= 100
            and target_gate["soft_safety_regressions"] >= args.target_soft_safety
            and target_gate["dac_fixes"] >= args.target_dac_fix
        )
        if ideal_done or acceptable_done:
            break

    selected = select_training_rows(candidate_rows, args.neutral_sample_count, args.seed)
    candidate_yields = yield_table(candidate_rows, "candidate_mode")
    sampling_stratum_yields = stratum_table(candidate_rows, stratum_attempted, stratum_valid)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/mine_bit_counterfactual_safety_cases_v2.py",
        "training_allowed": bool(training_allowed),
        "analysis_only": bool(args.analysis_only),
        "split": args.split,
        "scenes_processed": len(scene_rows),
        "total_rows": len(selected),
        "selected_training_rows": len(selected),
        "rejected_rows": len(rejected),
        "candidate_modes": modes,
        "warnings": warnings,
        "sampling_mode": args.sampling_mode,
        "sampling_pool_multiplier": args.sampling_pool_multiplier,
        "max_sampling_pool_scenes": args.max_sampling_pool_scenes,
        "sampling_distribution": dict(sampling_distribution),
        "sampling_stratum_yields": sampling_stratum_yields,
        "metric_cache_dirs": [str(path) for path in metric_dirs],
        "base_config": str(args.base_config),
        "base_checkpoint": str(args.base_checkpoint),
        "bit_config": str(args.bit_config),
        "bit_checkpoint": str(bit_checkpoint),
        "target_hard_nc": args.target_hard_nc,
        "target_hard_ttc": args.target_hard_ttc,
        "target_soft_safety": args.target_soft_safety,
        "target_dac_fix": args.target_dac_fix,
        "soft_label_thresholds": {
            "margin_ttc": args.margin_ttc,
            "soft_early_step_threshold": args.soft_early_step_threshold,
            "soft_early_x_threshold": args.soft_early_x_threshold,
            "soft_heading_threshold": args.soft_heading_threshold,
            "instability_l2_threshold": args.instability_l2_threshold,
            "instability_heading_threshold": args.instability_heading_threshold,
        },
        "label_histograms": label_histograms(candidate_rows),
        "candidate_mode_yields": candidate_yields,
        **counts_for(candidate_rows),
    }
    summary.update(gate_status(summary))
    metadata = {
        **summary,
        "exact_splits": [args.split],
        "scene_tokens": sorted({row["scene_token"] for row in scene_rows if row.get("scene_token")}),
        "sample_tokens": sorted({row["sample_token"] for row in scene_rows if row.get("sample_token")}),
    }

    write_jsonl(args.output_dir / "counterfactual_candidates.jsonl", candidate_rows)
    write_jsonl(args.output_dir / "counterfactual_scene_summary.jsonl", scene_rows)
    write_jsonl(args.output_dir / "selected_training_labels.jsonl", selected)
    write_jsonl(args.output_dir / "rejected_or_unusable.jsonl", rejected)
    (args.output_dir / "mining_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "mining_progress.json").write_text(json.dumps({"done": True, **summary}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary_markdown(args.output_dir / "mining_summary.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
