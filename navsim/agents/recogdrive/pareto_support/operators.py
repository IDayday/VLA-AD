from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .cheap_filter import cheap_filter, compute_cheap_metrics
from .dataclasses import TrajectoryCandidate
from .metrics import cfg_value, metric_value


def _wrap_angle_np(x: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(x), np.cos(x))


def diagnose_scene_or_anchor(candidate: TrajectoryCandidate, ref_metrics: dict | None = None, cfg: Any = None) -> set[str]:
    metrics = candidate.true_metrics or candidate.cheap_metrics or compute_cheap_metrics(candidate.trajectory, cfg=cfg)
    ref_metrics = ref_metrics or {}
    tags: set[str] = set()
    if metric_value(metrics, "ego_progress", "ep") < metric_value(ref_metrics, "ego_progress", "ep") + float(cfg_value(cfg, "ep_low_margin", 0.02)):
        if metric_value(metrics, "time_to_collision_within_bound", "ttc", default=1.0) >= metric_value(ref_metrics, "time_to_collision_within_bound", "ttc", default=1.0) - 0.03:
            tags.add("ep_low_safety_ok")
    if metric_value(metrics, "time_to_collision_within_bound", "ttc", default=1.0) < metric_value(ref_metrics, "time_to_collision_within_bound", "ttc", default=1.0) - 0.03:
        tags.add("ttc_risky")
    if metric_value(metrics, "no_at_fault_collisions", "nc", default=1.0) < 1.0:
        tags.add("nc_risky")
    if metric_value(metrics, "driving_direction_compliance", "ddc", default=1.0) < float(cfg_value(cfg, "ddc_low_threshold", 0.95)):
        tags.add("ddc_low")
    if metric_value(metrics, "feas_cost", default=0.0) > float(cfg_value(cfg, "geometry_bad_feas_cost", 0.10)):
        tags.add("geometry_bad")
    if not tags:
        tags.add("valid_high_quality")
    return tags


def _make_candidate(parent: TrajectoryCandidate, traj: np.ndarray, source: str, operator: str, idx: int) -> TrajectoryCandidate:
    return TrajectoryCandidate(
        scene_token=parent.scene_token,
        candidate_id=f"{parent.scene_token}:{operator}:{parent.candidate_id.split(':')[-1]}:{idx}",
        source=source,
        parent_ids=[parent.candidate_id],
        operator=operator,
        trajectory=traj.astype(np.float32),
    )


def reference_relative_progress_tune(anchor: TrajectoryCandidate, cfg: Any = None) -> list[TrajectoryCandidate]:
    traj = anchor.trajectory.astype(np.float32)
    x = np.linspace(0.0, 1.0, traj.shape[0], dtype=np.float32)
    basis = [x, x * x, np.sin(0.5 * np.pi * x)]
    deltas = cfg_value(cfg, "progress_tune_deltas_m", (0.3, 0.5, 1.0))
    out = []
    idx = 0
    for delta in deltas:
        for b in basis:
            new = traj.copy()
            new[:, 0] += float(delta) * b
            out.append(_make_candidate(anchor, new, "progress_tune", "progress_tune", idx))
            idx += 1
    return out


def yield_delay_recovery(anchor: TrajectoryCandidate, cfg: Any = None) -> list[TrajectoryCandidate]:
    traj = anchor.trajectory.astype(np.float32)
    lam = np.linspace(0.0, 1.0, traj.shape[0], dtype=np.float32)
    curves = [
        lam**1.3,
        np.maximum(0.0, (lam - 0.15) / 0.85),
        0.35 * lam + 0.65 * lam**2,
        np.minimum(lam * 0.35, 0.25),
    ]
    out = []
    for idx, curve in enumerate(curves):
        new = traj.copy()
        for dim in range(3):
            new[:, dim] = np.interp(curve, lam, traj[:, dim]).astype(np.float32)
        new[:, 2] = _wrap_angle_np(new[:, 2])
        out.append(_make_candidate(anchor, new, "yield_delay", "yield_delay", idx))
    return out


def ddc_route_repair(anchor: TrajectoryCandidate, best_ddc_anchor: TrajectoryCandidate, cfg: Any = None) -> list[TrajectoryCandidate]:
    base = anchor.trajectory.astype(np.float32)
    target = best_ddc_anchor.trajectory.astype(np.float32)
    n = min(base.shape[0], target.shape[0])
    u = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out = []
    for idx, alpha_max in enumerate(cfg_value(cfg, "ddc_repair_alpha", (0.25, 0.5))):
        alpha = float(alpha_max) * u * u
        new = base.copy()
        new[:n, :2] = (1.0 - alpha[:, None]) * base[:n, :2] + alpha[:, None] * target[:n, :2]
        new[:n, 2] = _wrap_angle_np((1.0 - alpha) * base[:n, 2] + alpha * target[:n, 2])
        out.append(_make_candidate(anchor, new, "ddc_repair", "ddc_repair", idx))
    return out


def feasibility_repair(anchor: TrajectoryCandidate, cfg: Any = None) -> list[TrajectoryCandidate]:
    traj = anchor.trajectory.astype(np.float32)
    out = []
    kernel = np.asarray([0.25, 0.5, 0.25], dtype=np.float32)
    smooth = traj.copy()
    for dim in range(2):
        smooth[1:-1, dim] = np.convolve(traj[:, dim], kernel, mode="same")[1:-1]
    smooth[:, 0] = np.maximum.accumulate(smooth[:, 0])
    step_heading = np.diff(smooth[:, 2], prepend=smooth[0, 2])
    max_step = float(cfg_value(cfg, "feas_repair_max_heading_step", 0.25))
    smooth[:, 2] = _wrap_angle_np(np.cumsum(np.clip(_wrap_angle_np(step_heading), -max_step, max_step)))
    cand = _make_candidate(anchor, smooth, "feas_repair", "feas_repair", 0)
    cand.repair_distance = float(np.linalg.norm(cand.trajectory[:, :2] - traj[:, :2], axis=-1).mean())
    out.append(cand)
    return out


def trust_region_control_expansion(anchor: TrajectoryCandidate, cfg: Any = None) -> list[TrajectoryCandidate]:
    traj = anchor.trajectory.astype(np.float32)
    out = []
    accel_deltas = cfg_value(cfg, "control_expand_accel_deltas", (-0.05, 0.05))
    yaw_deltas = cfg_value(cfg, "control_expand_yaw_deltas", (-0.02, 0.02))
    dt = float(cfg_value(cfg, "control_expand_dt", 0.5))
    idx = 0
    for accel in accel_deltas:
        for yaw_delta in yaw_deltas:
            new = traj.copy()
            xy_step = np.diff(np.vstack([np.zeros((1, 2), dtype=np.float32), traj[:, :2]]), axis=0)
            speed = np.linalg.norm(xy_step, axis=-1)
            speed = np.maximum(0.0, speed + float(accel) * dt * np.arange(traj.shape[0], dtype=np.float32))
            heading = _wrap_angle_np(traj[:, 2] + float(yaw_delta) * np.arange(traj.shape[0], dtype=np.float32))
            steps = np.stack([np.cos(heading) * speed, np.sin(heading) * speed], axis=-1)
            new[:, :2] = np.cumsum(steps, axis=0)
            new[:, 2] = heading
            cand = _make_candidate(anchor, new, "control_expand", "control_expand", idx)
            idx += 1
            if float(np.linalg.norm(cand.trajectory[:, :2] - traj[:, :2], axis=-1).mean()) <= float(cfg_value(cfg, "max_control_expand_distance", 2.0)):
                out.append(cand)
    return out


def failure_conditioned_operators(
    anchors: list[TrajectoryCandidate],
    ref_metrics: dict | None = None,
    cfg: Any = None,
) -> tuple[list[TrajectoryCandidate], dict[str, dict[str, float]]]:
    stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    best_ddc = max(anchors, key=lambda c: metric_value(c.true_metrics or {}, "driving_direction_compliance", "ddc", default=0.0)) if anchors else None
    generated: list[TrajectoryCandidate] = []
    for anchor in anchors:
        tags = diagnose_scene_or_anchor(anchor, ref_metrics, cfg)
        pools: list[TrajectoryCandidate] = []
        if "ep_low_safety_ok" in tags:
            pools.extend(reference_relative_progress_tune(anchor, cfg))
        if "ttc_risky" in tags or "nc_risky" in tags:
            pools.extend(yield_delay_recovery(anchor, cfg))
        if "ddc_low" in tags and best_ddc is not None:
            pools.extend(ddc_route_repair(anchor, best_ddc, cfg))
        if "geometry_bad" in tags:
            pools.extend(feasibility_repair(anchor, cfg))
        if "valid_high_quality" in tags:
            pools.extend(trust_region_control_expansion(anchor, cfg))
        for cand in pools:
            stats[cand.operator]["generated_count"] += 1
            if cheap_filter(cand, {"ref_trajectory": anchor.trajectory}, cfg):
                stats[cand.operator]["cheap_pass_count"] += 1
                generated.append(cand)
    return generated, {k: dict(v) for k, v in stats.items()}
