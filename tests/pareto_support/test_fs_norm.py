from __future__ import annotations

import numpy as np

from navsim.agents.recogdrive.pareto_support.fs_norm import (
    delta_to_trajectory,
    denormalize_trajectory,
    fit_fs_norm_stats,
    normalize_trajectory,
    trajectory_to_delta,
)


def test_delta_roundtrip() -> None:
    x = np.linspace(0.2, 4.0, 8, dtype=np.float32)
    traj = np.stack([x, 0.1 * x, 0.02 * x], axis=-1)
    recon = delta_to_trajectory(trajectory_to_delta(traj))
    assert np.max(np.abs(recon - traj)) < 1e-5


def test_normalize_denormalize_roundtrip() -> None:
    trajs = []
    for idx in range(4):
        x = np.linspace(0.2, 4.0 + idx * 0.1, 8, dtype=np.float32)
        trajs.append(np.stack([x, 0.1 * x, 0.02 * x], axis=-1))
    arr = np.stack(trajs, axis=0)
    stats = fit_fs_norm_stats(arr, robust=True, clip=100.0)
    recon = denormalize_trajectory(normalize_trajectory(arr, stats), stats)
    assert np.max(np.abs(recon - arr)) < 1e-4
