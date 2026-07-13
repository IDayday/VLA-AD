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
    version: int = 1
    representation: str = "ego_step_delta"
    p0: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scene_balanced: bool = False
    heading_center_zero: bool = False
    num_scenes: int = 0
    num_supports: int = 0
    archive_path: str = ""
    archive_fingerprint: str = ""


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


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> np.ndarray:
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}.")
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.shape[0] != weights.shape[0]:
        raise ValueError("values and weights must have matching leading dimensions.")
    flat = values.reshape(values.shape[0], -1)
    result = np.empty((flat.shape[1],), dtype=np.float64)
    for column in range(flat.shape[1]):
        order = np.argsort(flat[:, column], kind="mergesort")
        sorted_values = flat[order, column]
        sorted_weights = weights[order]
        cumulative = np.cumsum(sorted_weights)
        threshold = float(quantile) * float(sorted_weights.sum())
        index = int(np.searchsorted(cumulative, threshold, side="left"))
        result[column] = sorted_values[min(index, sorted_values.shape[0] - 1)]
    return result.reshape(values.shape[1:]).astype(np.float32)


def _scene_balanced_weights(scene_ids: Optional[np.ndarray], count: int) -> tuple[np.ndarray, int]:
    if scene_ids is None:
        inverse = np.arange(count, dtype=np.int64)
        counts = np.ones((count,), dtype=np.int64)
    else:
        scene_arr = np.asarray(scene_ids)
        if scene_arr.shape != (count,):
            raise ValueError(f"scene_ids must have shape [{count}], got {scene_arr.shape}.")
        _, inverse, counts = np.unique(scene_arr.astype(str), return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    weights /= weights.sum()
    return weights, int(counts.shape[0])


def fit_fs_norm_stats(
    trajectories: np.ndarray,
    robust: bool = True,
    clip: float = 5.0,
    eps: float = 1e-6,
    *,
    scene_ids: Optional[np.ndarray] = None,
    lower_quantile: float = 0.001,
    upper_quantile: float = 0.999,
    archive_path: str = "",
    archive_fingerprint: str = "",
) -> FSNormStats:
    trajs = np.asarray(trajectories, dtype=np.float32)
    if trajs.ndim != 3 or trajs.shape[-1] != 3:
        raise ValueError(f"trajectories must have shape [N, H, 3], got {trajs.shape}.")
    if trajs.shape[1:] != (8, 3):
        raise ValueError(f"FS-Norm v2 requires trajectory shape [N, 8, 3], got {trajs.shape}.")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("Expected 0 <= lower_quantile < upper_quantile <= 1.")
    delta = trajectory_to_delta(trajs)
    weights, num_scenes = _scene_balanced_weights(scene_ids, trajs.shape[0])
    expanded_weights = weights[:, None, None]
    mean = (delta.astype(np.float64) * expanded_weights).sum(axis=0).astype(np.float32)
    mean[..., 2] = 0.0
    centered = delta.astype(np.float64) - mean[None, ...].astype(np.float64)
    std = np.sqrt(((centered ** 2) * expanded_weights).sum(axis=0)).astype(np.float32)
    std[..., 2] = np.sqrt(
        ((delta[..., 2].astype(np.float64) ** 2) * weights[:, None]).sum(axis=0)
    ).astype(np.float32)
    std = np.maximum(std, eps).astype(np.float32)
    delta_min = delta.min(axis=0).astype(np.float32)
    delta_max = delta.max(axis=0).astype(np.float32)
    if robust:
        median = _weighted_quantile(delta, weights, 0.5)
        median[..., 2] = 0.0
        mad = _weighted_quantile(np.abs(delta - median[None, ...]), weights, 0.5)
        mad[..., 2] = _weighted_quantile(np.abs(delta[..., 2]), weights, 0.5)
        mad = np.maximum(mad, eps).astype(np.float32)
        center = median
        scale = np.maximum(1.4826 * mad, eps).astype(np.float32)
        scale[..., 2] = np.maximum(mad[..., 2], eps)
    else:
        median = None
        mad = None
        center = mean
        scale = std
    normalized = (delta - center[None, ...]) / scale[None, ...]
    clip_lower = _weighted_quantile(normalized, weights, lower_quantile)
    clip_upper = _weighted_quantile(normalized, weights, upper_quantile)
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
        version=2,
        representation="ego_step_delta",
        p0=(0.0, 0.0, 0.0),
        scene_balanced=True,
        heading_center_zero=True,
        num_scenes=num_scenes,
        num_supports=int(trajs.shape[0]),
        archive_path=str(archive_path),
        archive_fingerprint=str(archive_fingerprint),
    )


def _center_scale(stats: FSNormStats, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if stats.use_robust and stats.median is not None and stats.mad is not None:
        center = stats.median.astype(ref.dtype, copy=False)
        scale = (1.4826 * stats.mad).astype(ref.dtype, copy=True)
        if stats.version >= 2 and stats.heading_center_zero:
            center = center.copy()
            center[..., 2] = 0.0
            scale[..., 2] = stats.mad[..., 2]
    else:
        center = stats.mean.astype(ref.dtype, copy=False)
        scale = stats.std.astype(ref.dtype, copy=False)
    return center, np.maximum(scale, 1e-6)


def normalize_delta(
    delta: np.ndarray,
    stats: FSNormStats,
    *,
    apply_clip: bool = False,
    clip_value: Optional[float] = None,
    use_stats_bounds: bool = False,
) -> np.ndarray:
    arr = np.asarray(delta, dtype=np.float32)
    center, scale = _center_scale(stats, arr)
    out = (arr - center) / scale
    if not apply_clip:
        if clip_value is not None or use_stats_bounds:
            raise ValueError("clip_value/use_stats_bounds require apply_clip=True.")
        return out.astype(np.float32)
    if use_stats_bounds:
        if clip_value is not None:
            raise ValueError("Choose either stats bounds or scalar clip_value, not both.")
        if stats.clip_lower is None or stats.clip_upper is None:
            raise ValueError("FS-Norm stats bounds are unavailable.")
        out = np.minimum(np.maximum(out, stats.clip_lower), stats.clip_upper)
    elif clip_value is not None and float(clip_value) > 0.0:
        out = np.clip(out, -float(clip_value), float(clip_value))
    else:
        raise ValueError("apply_clip=True requires a positive clip_value or use_stats_bounds=True.")
    return out.astype(np.float32)


def denormalize_delta(norm_delta: np.ndarray, stats: FSNormStats) -> np.ndarray:
    arr = np.asarray(norm_delta, dtype=np.float32)
    center, scale = _center_scale(stats, arr)
    return (arr * scale + center).astype(np.float32)


def normalize_trajectory(
    traj: np.ndarray,
    stats: FSNormStats,
    *,
    apply_clip: bool = False,
    clip_value: Optional[float] = None,
    use_stats_bounds: bool = False,
) -> np.ndarray:
    return normalize_delta(
        trajectory_to_delta(traj),
        stats,
        apply_clip=apply_clip,
        clip_value=clip_value,
        use_stats_bounds=use_stats_bounds,
    )


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
        "version": np.asarray(int(stats.version), dtype=np.int64),
        "representation": np.asarray(str(stats.representation)),
        "p0": np.asarray(stats.p0, dtype=np.float32),
        "scene_balanced": np.asarray(int(stats.scene_balanced), dtype=np.int64),
        "heading_center_zero": np.asarray(int(stats.heading_center_zero), dtype=np.int64),
        "num_scenes": np.asarray(int(stats.num_scenes), dtype=np.int64),
        "num_supports": np.asarray(int(stats.num_supports), dtype=np.int64),
        "archive_path": np.asarray(str(stats.archive_path)),
        "archive_fingerprint": np.asarray(str(stats.archive_fingerprint)),
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
        version=int(data["version"]) if "version" in data.files else 1,
        representation=str(data["representation"].item()) if "representation" in data.files else "ego_step_delta",
        p0=tuple(float(item) for item in data["p0"]) if "p0" in data.files else (0.0, 0.0, 0.0),
        scene_balanced=bool(int(data["scene_balanced"])) if "scene_balanced" in data.files else False,
        heading_center_zero=bool(int(data["heading_center_zero"])) if "heading_center_zero" in data.files else False,
        num_scenes=int(data["num_scenes"]) if "num_scenes" in data.files else 0,
        num_supports=int(data["num_supports"]) if "num_supports" in data.files else 0,
        archive_path=str(data["archive_path"].item()) if "archive_path" in data.files else "",
        archive_fingerprint=(
            str(data["archive_fingerprint"].item()) if "archive_fingerprint" in data.files else ""
        ),
    )
