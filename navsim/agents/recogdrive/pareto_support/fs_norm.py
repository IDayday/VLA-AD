from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class FSNormStats:
    mean: np.ndarray
    std: np.ndarray
    median: Optional[np.ndarray] = None
    mad: Optional[np.ndarray] = None
    delta_min: Optional[np.ndarray] = None
    delta_max: Optional[np.ndarray] = None
    clip_lower: Optional[np.ndarray] = None
    clip_upper: Optional[np.ndarray] = None
    use_robust: bool = False
    clip: float = 5.0


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def trajectory_to_delta(traj: np.ndarray) -> np.ndarray:
    arr = np.asarray(traj, dtype=np.float32)
    if arr.ndim < 2 or arr.shape[-1] != 3:
        raise ValueError(f"trajectory must end with [H, 3], got {arr.shape}.")
    delta = arr.copy()
    if arr.shape[-2] > 1:
        delta[..., 1:, :2] = arr[..., 1:, :2] - arr[..., :-1, :2]
        delta[..., 1:, 2] = wrap_angle(arr[..., 1:, 2] - arr[..., :-1, 2])
    delta[..., 0, 2] = wrap_angle(delta[..., 0, 2])
    return delta


def delta_to_trajectory(delta: np.ndarray) -> np.ndarray:
    arr = np.asarray(delta, dtype=np.float32)
    if arr.ndim < 2 or arr.shape[-1] != 3:
        raise ValueError(f"delta must end with [H, 3], got {arr.shape}.")
    out = arr.copy()
    out[..., :2] = np.cumsum(arr[..., :2], axis=-2)
    out[..., 2] = wrap_angle(np.cumsum(arr[..., 2], axis=-1))
    return out


def fit_fs_norm_stats(trajectories: np.ndarray, robust: bool = True, clip: float = 5.0, eps: float = 1e-6) -> FSNormStats:
    trajs = np.asarray(trajectories, dtype=np.float32)
    if trajs.ndim != 3 or trajs.shape[-1] != 3:
        raise ValueError(f"trajectories must have shape [N, H, 3], got {trajs.shape}.")
    delta = trajectory_to_delta(trajs)
    mean = delta.mean(axis=0).astype(np.float32)
    std = delta.std(axis=0).astype(np.float32)
    std = np.maximum(std, eps).astype(np.float32)
    delta_min = delta.min(axis=0).astype(np.float32)
    delta_max = delta.max(axis=0).astype(np.float32)
    if robust:
        median = np.median(delta, axis=0).astype(np.float32)
        mad = np.median(np.abs(delta - median[None, ...]), axis=0).astype(np.float32)
        mad = np.maximum(mad, eps).astype(np.float32)
        center = median
        scale = np.maximum(1.4826 * mad, eps).astype(np.float32)
    else:
        median = None
        mad = None
        center = mean
        scale = std
    normalized = (delta - center[None, ...]) / scale[None, ...]
    clip_lower = np.quantile(normalized, 0.001, axis=0).astype(np.float32)
    clip_upper = np.quantile(normalized, 0.999, axis=0).astype(np.float32)
    return FSNormStats(
        mean=mean,
        std=std,
        median=median,
        mad=mad,
        delta_min=delta_min,
        delta_max=delta_max,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
        use_robust=bool(robust),
        clip=float(clip),
    )


def _center_scale(stats: FSNormStats, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if stats.use_robust and stats.median is not None and stats.mad is not None:
        center = stats.median.astype(ref.dtype, copy=False)
        scale = (1.4826 * stats.mad).astype(ref.dtype, copy=False)
    else:
        center = stats.mean.astype(ref.dtype, copy=False)
        scale = stats.std.astype(ref.dtype, copy=False)
    return center, np.maximum(scale, 1e-6)


def normalize_delta(delta: np.ndarray, stats: FSNormStats) -> np.ndarray:
    arr = np.asarray(delta, dtype=np.float32)
    center, scale = _center_scale(stats, arr)
    out = (arr - center) / scale
    if stats.clip > 0:
        out = np.clip(out, -float(stats.clip), float(stats.clip))
    return out.astype(np.float32)


def denormalize_delta(norm_delta: np.ndarray, stats: FSNormStats) -> np.ndarray:
    arr = np.asarray(norm_delta, dtype=np.float32)
    center, scale = _center_scale(stats, arr)
    return (arr * scale + center).astype(np.float32)


def normalize_trajectory(traj: np.ndarray, stats: FSNormStats) -> np.ndarray:
    return normalize_delta(trajectory_to_delta(traj), stats)


def denormalize_trajectory(norm_traj: np.ndarray, stats: FSNormStats) -> np.ndarray:
    return delta_to_trajectory(denormalize_delta(norm_traj, stats))


def save_fs_norm_stats(path: str | Path, stats: FSNormStats) -> None:
    payload = {
        "mean": stats.mean,
        "std": stats.std,
        "median": stats.median if stats.median is not None else np.asarray([], dtype=np.float32),
        "mad": stats.mad if stats.mad is not None else np.asarray([], dtype=np.float32),
        "delta_min": stats.delta_min if stats.delta_min is not None else np.asarray([], dtype=np.float32),
        "delta_max": stats.delta_max if stats.delta_max is not None else np.asarray([], dtype=np.float32),
        "clip_lower": stats.clip_lower if stats.clip_lower is not None else np.asarray([], dtype=np.float32),
        "clip_upper": stats.clip_upper if stats.clip_upper is not None else np.asarray([], dtype=np.float32),
        "use_robust": np.asarray(int(stats.use_robust), dtype=np.int64),
        "clip": np.asarray(float(stats.clip), dtype=np.float32),
    }
    np.savez(path, **payload)


def load_fs_norm_stats(path: str | Path) -> FSNormStats:
    data = np.load(path, allow_pickle=True)
    median = data["median"] if data["median"].size else None
    mad = data["mad"] if data["mad"].size else None
    delta_min = data["delta_min"] if "delta_min" in data.files and data["delta_min"].size else None
    delta_max = data["delta_max"] if "delta_max" in data.files and data["delta_max"].size else None
    clip_lower = data["clip_lower"] if "clip_lower" in data.files and data["clip_lower"].size else None
    clip_upper = data["clip_upper"] if "clip_upper" in data.files and data["clip_upper"].size else None
    return FSNormStats(
        mean=data["mean"].astype(np.float32),
        std=data["std"].astype(np.float32),
        median=None if median is None else median.astype(np.float32),
        mad=None if mad is None else mad.astype(np.float32),
        delta_min=None if delta_min is None else delta_min.astype(np.float32),
        delta_max=None if delta_max is None else delta_max.astype(np.float32),
        clip_lower=None if clip_lower is None else clip_lower.astype(np.float32),
        clip_upper=None if clip_upper is None else clip_upper.astype(np.float32),
        use_robust=bool(int(data["use_robust"])),
        clip=float(data["clip"]),
    )
