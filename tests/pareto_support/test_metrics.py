from __future__ import annotations

import numpy as np

from navsim.agents.recogdrive.pareto_support.cheap_filter import cheap_filter, compute_cheap_metrics
from navsim.agents.recogdrive.pareto_support.metrics import is_valid_metrics


def _straight() -> np.ndarray:
    x = np.linspace(0.5, 4.0, 8, dtype=np.float32)
    return np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=-1)


def test_non_finite_trajectory_rejected() -> None:
    traj = _straight()
    traj[2, 0] = np.nan
    assert not cheap_filter(traj, {}, {})


def test_tail_reverse_cost_positive_on_reverse_trajectory() -> None:
    traj = _straight()
    traj[-3:, 0] = traj[-3:, 0][::-1]
    metrics = compute_cheap_metrics(traj)
    assert metrics["tail_reverse_cost"] > 0.0


def test_early_kink_cost_positive_on_kink_trajectory() -> None:
    traj = _straight()
    traj[1, 2] = 1.0
    metrics = compute_cheap_metrics(traj)
    assert metrics["early_kink_cost"] > 0.0


def test_valid_gate_respects_ddc_and_feas_cost() -> None:
    metrics = {
        "no_at_fault_collisions": 1.0,
        "drivable_area_compliance": 1.0,
        "driving_direction_compliance": 0.90,
        "history_comfort": 1.0,
        "feas_cost": 0.0,
    }
    assert not is_valid_metrics(metrics, {}, {"fpv3_ddc_min_absolute": 0.95})
    metrics["driving_direction_compliance"] = 1.0
    metrics["feas_cost"] = 0.2
    assert not is_valid_metrics(metrics, {}, {"fpv3_feas_cost_max": 0.1})
    metrics["feas_cost"] = 0.0
    assert is_valid_metrics(metrics, {}, {})
