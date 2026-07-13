from __future__ import annotations

import numpy as np

from navsim.agents.recogdrive.pareto_support.acquisition import select_candidates_for_evaluation
from navsim.agents.recogdrive.pareto_support.dataclasses import TrajectoryCandidate


def _cand(idx: int, utility: float, operator: str = "progress_tune") -> TrajectoryCandidate:
    x = np.linspace(0.2, 4.0 + idx, 8, dtype=np.float32)
    traj = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=-1)
    return TrajectoryCandidate(
        scene_token="scene",
        candidate_id=f"scene:c:{idx}",
        source="progress_tune",
        operator=operator,
        trajectory=traj,
        true_metrics={"utility": utility, "feas_cost": 0.0},
    )


def test_duplicate_candidate_penalized_and_high_utility_selected() -> None:
    archive = [_cand(0, 0.0)]
    duplicate = _cand(0, 100.0)
    high = _cand(2, 2.0)
    selected = select_candidates_for_evaluation([duplicate, high], archive, None, {"true_eval_budget": 1, "acquisition_duplicate_distance_m": 0.2})
    assert selected == [high]
