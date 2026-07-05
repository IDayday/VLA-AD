from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import cfg_value, compute_feasibility_metrics


def _wrap_angle_np(x: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(x), np.cos(x))


def compute_cheap_metrics(traj: np.ndarray, ref_traj: np.ndarray | None = None, cfg: Any = None) -> dict[str, float]:
    arr = np.asarray(traj, dtype=np.float32)
    metrics = {
        "non_finite": float(not np.isfinite(arr).all()),
        "horizon_shape_ok": float(arr.ndim == 2 and arr.shape[-1] == 3),
    }
    if arr.ndim != 2 or arr.shape[-1] != 3 or arr.shape[0] == 0:
        metrics.update(
            {
                "final_progress": 0.0,
                "tail_reverse_cost": 1e6,
                "total_reverse_cost": 1e6,
                "early_kink_cost": 1e6,
                "max_heading_jump": 1e6,
                "curvature_proxy": 1e6,
                "jerk_proxy": 1e6,
                "endpoint_distance_to_ref": 1e6,
                "mean_distance_to_ref": 1e6,
                "duplicate_distance_to_archive": 1e6,
                "route_heading_proxy": 0.0,
            }
        )
        return metrics
    feas = compute_feasibility_metrics(arr, cfg)
    metrics.update(feas)
    delta = np.diff(arr[:, :2], axis=0)
    dist = np.linalg.norm(delta, axis=-1)
    heading_delta = np.abs(_wrap_angle_np(np.diff(arr[:, 2], axis=0)))
    curvature = heading_delta / np.maximum(dist, 1e-4)
    jerk = np.diff(delta, n=1, axis=0)
    metrics["final_progress"] = float(arr[-1, 0])
    metrics["max_heading_jump"] = float(heading_delta.max(initial=0.0))
    metrics["curvature_proxy"] = float(curvature.max(initial=0.0))
    metrics["jerk_proxy"] = float(np.linalg.norm(jerk, axis=-1).max(initial=0.0)) if jerk.size else 0.0
    metrics["route_heading_proxy"] = float(np.cos(arr[-1, 2])) if arr.size else 0.0
    if ref_traj is not None:
        ref = np.asarray(ref_traj, dtype=np.float32)
        n = min(arr.shape[0], ref.shape[0])
        if n > 0 and ref.ndim == 2 and ref.shape[-1] == 3:
            metrics["endpoint_distance_to_ref"] = float(np.linalg.norm(arr[n - 1, :2] - ref[n - 1, :2]))
            metrics["mean_distance_to_ref"] = float(np.linalg.norm(arr[:n, :2] - ref[:n, :2], axis=-1).mean())
        else:
            metrics["endpoint_distance_to_ref"] = 1e6
            metrics["mean_distance_to_ref"] = 1e6
    else:
        metrics["endpoint_distance_to_ref"] = 0.0
        metrics["mean_distance_to_ref"] = 0.0
    metrics["duplicate_distance_to_archive"] = metrics["mean_distance_to_ref"]
    return {k: float(v) for k, v in metrics.items()}


def cheap_filter(candidate: Any, ref_metrics: dict | None = None, cfg: Any = None) -> bool:
    traj = getattr(candidate, "trajectory", candidate)
    ref_traj = None
    if isinstance(ref_metrics, dict) and "ref_trajectory" in ref_metrics:
        ref_traj = np.asarray(ref_metrics["ref_trajectory"], dtype=np.float32)
    metrics = compute_cheap_metrics(np.asarray(traj, dtype=np.float32), ref_traj=ref_traj, cfg=cfg)
    if hasattr(candidate, "cheap_metrics"):
        candidate.cheap_metrics = metrics
    if metrics["non_finite"] > 0.0 or metrics["horizon_shape_ok"] < 1.0:
        return False
    source = str(getattr(candidate, "source", "")).lower()
    allow_stop = "yield" in source or "stop" in source
    if not allow_stop and metrics["final_progress"] < float(cfg_value(cfg, "min_final_progress", 0.1)):
        return False
    if metrics["early_kink_cost"] > float(cfg_value(cfg, "cheap_max_early_kink_cost", 0.75)):
        return False
    if metrics["tail_reverse_cost"] > float(cfg_value(cfg, "cheap_max_tail_reverse_cost", 0.35)):
        return False
    if metrics["max_heading_jump"] > float(cfg_value(cfg, "cheap_max_heading_jump", 1.2)):
        return False
    if metrics["curvature_proxy"] > float(cfg_value(cfg, "cheap_max_curvature_proxy", 2.5)):
        return False
    trust_region = float(cfg_value(cfg, "cheap_trust_region_max_m", 12.0))
    if "gt" not in source and "stage3" not in source and metrics["mean_distance_to_ref"] > trust_region:
        return False
    return True
