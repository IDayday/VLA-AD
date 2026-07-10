from __future__ import annotations

import numpy as np

from navsim.agents.recogdrive.pareto_support.fs_norm import (
    fit_fs_norm_stats,
    load_fs_norm_stats,
    save_fs_norm_stats,
)
from scripts.tools.build_fs_norm_stats import _collect_trajs


def _constant_delta_trajectory(dx: float, dheading: float) -> np.ndarray:
    steps = np.arange(1, 9, dtype=np.float32)
    return np.stack((steps * dx, np.zeros_like(steps), steps * dheading), axis=-1)


def test_fs_norm_v2_is_scene_balanced_and_heading_centered(tmp_path) -> None:
    crowded_scene = [_constant_delta_trajectory(10.0, 0.4) for _ in range(9)]
    sparse_scene = [_constant_delta_trajectory(0.0, -0.2)]
    trajectories = np.stack(crowded_scene + sparse_scene)
    scene_ids = np.asarray(["crowded"] * 9 + ["sparse"])

    stats = fit_fs_norm_stats(
        trajectories,
        robust=False,
        scene_ids=scene_ids,
        lower_quantile=0.0,
        upper_quantile=1.0,
        archive_path="synthetic/support",
        archive_fingerprint="abc123",
    )

    assert stats.mean.shape == (8, 3)
    assert np.allclose(stats.mean[:, 0], 5.0, atol=1e-6)
    assert np.all(stats.mean[:, 2] == 0.0)
    assert stats.version == 2
    assert stats.scene_balanced is True
    assert stats.heading_center_zero is True
    assert stats.num_scenes == 2
    assert stats.num_supports == 10
    assert stats.archive_path == "synthetic/support"
    assert stats.archive_fingerprint == "abc123"

    path = tmp_path / "fs_stats_v2.npz"
    save_fs_norm_stats(path, stats)
    loaded = load_fs_norm_stats(path)
    assert loaded.version == 2
    assert loaded.scene_balanced is True
    assert loaded.heading_center_zero is True

    robust_stats = fit_fs_norm_stats(trajectories, robust=True, scene_ids=scene_ids)
    assert robust_stats.median is not None
    assert np.all(robust_stats.median[:, 2] == 0.0)
    assert np.all(robust_stats.mean[:, 2] == 0.0)


def test_stats_builder_uses_only_selected_support_indices() -> None:
    candidates = np.stack(
        [
            _constant_delta_trajectory(1.0, 0.0),
            _constant_delta_trajectory(99.0, 0.0),
            _constant_delta_trajectory(2.0, 0.0),
        ]
    )
    selected = _collect_trajs({"candidates": candidates, "support_indices": [0, 2], "hard_negative_indices": [1]})
    assert len(selected) == 2
    assert all(float(item[:, 0].max()) < 50.0 for item in selected)
