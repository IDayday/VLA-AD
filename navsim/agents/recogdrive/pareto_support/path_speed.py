from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .dataclasses import TrajectoryCandidate
from .metrics import cfg_value


def _wrap_angle_np(x: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(x), np.cos(x))


@dataclass
class PathAtom:
    path_atom_id: str
    parent_candidate_id: str
    source: str
    path_points: np.ndarray
    headings: np.ndarray
    arc_lengths: np.ndarray
    total_length: float
    path_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class SpeedAtom:
    speed_atom_id: str
    parent_candidate_id: str
    source: str
    lambda_curve: np.ndarray
    terminal_progress: float
    progress_per_step: np.ndarray
    speed_metrics: dict[str, float] = field(default_factory=dict)


def _arc_length(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if points.shape[0] == 1:
        return np.zeros((1,), dtype=np.float32)
    step = np.linalg.norm(np.diff(points, axis=0), axis=-1)
    return np.concatenate([np.zeros((1,), dtype=np.float32), np.cumsum(step).astype(np.float32)])


def _interp_by_arc(points: np.ndarray, arc: np.ndarray, query: np.ndarray) -> np.ndarray:
    if points.shape[0] == 1:
        return np.repeat(points, query.shape[0], axis=0)
    out = np.stack([np.interp(query, arc, points[:, dim]) for dim in range(points.shape[1])], axis=-1)
    return out.astype(np.float32)


def extract_path_atom(candidate: TrajectoryCandidate, cfg: Any = None) -> PathAtom:
    traj = candidate.trajectory.astype(np.float32)
    points = traj[:, :2]
    arc = _arc_length(points)
    total_length = float(arc[-1]) if arc.size else 0.0
    min_len = float(cfg_value(cfg, "path_min_total_length", 0.05))
    if total_length < min_len:
        raise ValueError(f"Candidate {candidate.candidate_id} path too short: {total_length:.3f}m.")
    n_path = int(cfg_value(cfg, "path_atom_points", 64))
    query = np.linspace(0.0, total_length, n_path, dtype=np.float32)
    path_points = _interp_by_arc(points, arc, query)
    tangents = np.gradient(path_points, axis=0)
    headings = np.arctan2(tangents[:, 1], tangents[:, 0]).astype(np.float32)
    heading_delta = np.abs(_wrap_angle_np(np.diff(headings)))
    curvature_cost = float(heading_delta.mean()) if heading_delta.size else 0.0
    return PathAtom(
        path_atom_id=f"{candidate.candidate_id}:path",
        parent_candidate_id=candidate.candidate_id,
        source=candidate.source,
        path_points=path_points,
        headings=headings,
        arc_lengths=query,
        total_length=total_length,
        path_metrics={
            "ddc": float((candidate.true_metrics or {}).get("driving_direction_compliance", 0.0)),
            "dac": float((candidate.true_metrics or {}).get("drivable_area_compliance", 0.0)),
            "curvature_cost": curvature_cost,
            "route_heading_proxy": float(np.cos(headings[-1])),
            "early_kink_cost": float((candidate.true_metrics or {}).get("early_kink_cost", 0.0)),
            "tail_reverse_cost": float((candidate.true_metrics or {}).get("tail_reverse_cost", 0.0)),
            "endpoint_x": float(path_points[-1, 0]),
            "endpoint_y": float(path_points[-1, 1]),
        },
    )


def extract_speed_atom(candidate: TrajectoryCandidate, cfg: Any = None) -> SpeedAtom:
    traj = candidate.trajectory.astype(np.float32)
    arc = _arc_length(traj[:, :2])
    monotonic_arc = np.maximum.accumulate(arc)
    terminal = float(max(monotonic_arc[-1], 1e-6))
    lambda_curve = np.clip(monotonic_arc / terminal, 0.0, 1.0).astype(np.float32)
    progress = np.diff(monotonic_arc, prepend=0.0).astype(np.float32)
    return SpeedAtom(
        speed_atom_id=f"{candidate.candidate_id}:speed",
        parent_candidate_id=candidate.candidate_id,
        source=candidate.source,
        lambda_curve=lambda_curve,
        terminal_progress=terminal,
        progress_per_step=progress,
        speed_metrics={
            "ep": float((candidate.true_metrics or {}).get("ego_progress", 0.0)),
            "ttc": float((candidate.true_metrics or {}).get("time_to_collision_within_bound", 0.0)),
            "comfort": float((candidate.true_metrics or {}).get("history_comfort", 0.0)),
            "monotonicity": float(np.mean(np.diff(lambda_curve) >= -1e-5)) if lambda_curve.shape[0] > 1 else 1.0,
            "jerk_cost": float((candidate.true_metrics or {}).get("jerk_cost", 0.0)),
            "delay_type": 0.0,
        },
    )


def select_path_bank(seed_candidates: list[TrajectoryCandidate], cfg: Any = None) -> list[PathAtom]:
    atoms: list[PathAtom] = []
    for candidate in seed_candidates:
        try:
            atoms.append(extract_path_atom(candidate, cfg))
        except ValueError:
            continue
    atoms.sort(
        key=lambda a: (
            a.path_metrics.get("ddc", 0.0),
            a.path_metrics.get("dac", 0.0),
            -a.path_metrics.get("curvature_cost", 0.0),
            a.total_length,
        ),
        reverse=True,
    )
    keep = int(cfg_value(cfg, "path_bank_size", 4))
    selected: list[PathAtom] = []
    for atom in atoms:
        endpoint = atom.path_points[-1]
        if selected:
            min_dist = min(float(np.linalg.norm(endpoint - prev.path_points[-1])) for prev in selected)
            if min_dist < float(cfg_value(cfg, "path_bank_endpoint_nms_m", 0.5)) and len(selected) >= 2:
                continue
        selected.append(atom)
        if len(selected) >= keep:
            break
    return selected


def select_speed_bank(seed_candidates: list[TrajectoryCandidate], cfg: Any = None) -> list[SpeedAtom]:
    atoms = [extract_speed_atom(candidate, cfg) for candidate in seed_candidates]
    atoms.sort(
        key=lambda a: (
            a.speed_metrics.get("ttc", 0.0),
            a.speed_metrics.get("ep", 0.0),
            -a.speed_metrics.get("jerk_cost", 0.0),
            a.terminal_progress,
        ),
        reverse=True,
    )
    keep = int(cfg_value(cfg, "speed_bank_size", 4))
    return atoms[:keep]


def recombine_path_speed(path_atom: PathAtom, speed_atom: SpeedAtom, cfg: Any = None) -> TrajectoryCandidate:
    horizon = int(cfg_value(cfg, "trajectory_horizon", speed_atom.lambda_curve.shape[0]))
    lambda_curve = speed_atom.lambda_curve
    if lambda_curve.shape[0] != horizon:
        src = np.linspace(0.0, 1.0, lambda_curve.shape[0], dtype=np.float32)
        dst = np.linspace(0.0, 1.0, horizon, dtype=np.float32)
        lambda_curve = np.interp(dst, src, lambda_curve).astype(np.float32)
    max_s = min(speed_atom.terminal_progress, path_atom.total_length + float(cfg_value(cfg, "path_speed_length_margin_m", 0.2)))
    if speed_atom.terminal_progress > path_atom.total_length + float(cfg_value(cfg, "path_speed_length_margin_m", 0.2)):
        raise ValueError("Speed atom terminal progress exceeds path length trust region.")
    s = np.clip(lambda_curve * max_s, 0.0, path_atom.total_length)
    xy = _interp_by_arc(path_atom.path_points, path_atom.arc_lengths, s)
    heading = np.interp(s, path_atom.arc_lengths, path_atom.headings).astype(np.float32)
    traj = np.concatenate([xy, heading[:, None]], axis=-1).astype(np.float32)
    scene_token = path_atom.path_atom_id.split(":", 1)[0]
    return TrajectoryCandidate(
        scene_token=scene_token,
        candidate_id=f"{scene_token}:path_speed:{path_atom.path_atom_id.split(':')[-2]}:{speed_atom.speed_atom_id.split(':')[-2]}",
        source="path_speed_recombine",
        parent_ids=[path_atom.parent_candidate_id, speed_atom.parent_candidate_id],
        operator="path_speed_recombination",
        trajectory=traj,
        path_atom_id=path_atom.path_atom_id,
        speed_atom_id=speed_atom.speed_atom_id,
    )
