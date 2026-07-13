from __future__ import annotations

import numpy as np

from navsim.agents.recogdrive.pareto_support.dataclasses import TrajectoryCandidate
from navsim.agents.recogdrive.pareto_support.path_speed import extract_path_atom, extract_speed_atom, recombine_path_speed


def _candidate(source: str = "gt") -> TrajectoryCandidate:
    x = np.linspace(0.2, 5.0, 8, dtype=np.float32)
    y = 0.2 * np.sin(x).astype(np.float32)
    heading = np.arctan2(np.gradient(y), np.gradient(x)).astype(np.float32)
    traj = np.stack([x, y, heading], axis=-1)
    return TrajectoryCandidate(
        scene_token="scene",
        candidate_id=f"scene:{source}:0",
        source=source,
        trajectory=traj,
        true_metrics={"driving_direction_compliance": 1.0, "drivable_area_compliance": 1.0, "ego_progress": 1.0, "time_to_collision_within_bound": 1.0},
    )


def test_path_atom_arc_length_monotonic() -> None:
    atom = extract_path_atom(_candidate())
    assert np.all(np.diff(atom.arc_lengths) >= -1e-6)
    assert atom.total_length > 0.0


def test_speed_atom_lambda_monotonic() -> None:
    atom = extract_speed_atom(_candidate())
    assert np.all(atom.lambda_curve >= 0.0)
    assert np.all(atom.lambda_curve <= 1.0)
    assert np.all(np.diff(atom.lambda_curve) >= -1e-6)


def test_recombination_returns_finite_horizon() -> None:
    cand = _candidate()
    path = extract_path_atom(cand)
    speed = extract_speed_atom(cand)
    out = recombine_path_speed(path, speed, {"trajectory_horizon": 8})
    assert out.trajectory.shape == (8, 3)
    assert np.isfinite(out.trajectory).all()
