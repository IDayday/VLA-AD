from __future__ import annotations

import numpy as np
import torch

from navsim.agents.recogdrive.fs_norm import (
    FSNormStats,
    load_fs_norm_stats,
    save_fs_norm_stats,
)


def _stats() -> FSNormStats:
    values = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    return FSNormStats(
        mean=values,
        std=values + 1.0,
        delta_min=values - 2.0,
        delta_max=values + 2.0,
        clip_lower=torch.full_like(values, -5.0),
        clip_upper=torch.full_like(values, 5.0),
        use_robust=False,
        clip=5.0,
        version=2,
        scene_balanced=True,
        heading_center_zero=True,
        num_scenes=103288,
        num_supports=264813,
        archive_path="/tmp/support_v6",
        archive_fingerprint="archive-sha",
    )


def test_fs_norm_npz_round_trip_uses_numpy_container(tmp_path) -> None:
    path = tmp_path / "stats.npz"

    save_fs_norm_stats(str(path), _stats())
    with np.load(path, allow_pickle=True) as payload:
        assert "mean" in payload.files
        assert "archive_fingerprint" in payload.files
    restored = load_fs_norm_stats(str(path))

    torch.testing.assert_close(restored.mean, _stats().mean)
    torch.testing.assert_close(restored.std, _stats().std)
    assert restored.version == 2
    assert restored.num_scenes == 103288
    assert restored.num_supports == 264813
    assert restored.archive_path == "/tmp/support_v6"
    assert restored.archive_fingerprint == "archive-sha"


def test_fs_norm_pt_round_trip_remains_supported(tmp_path) -> None:
    path = tmp_path / "stats.pt"

    save_fs_norm_stats(str(path), _stats())
    restored = load_fs_norm_stats(str(path))

    torch.testing.assert_close(restored.mean, _stats().mean)
    assert restored.archive_fingerprint == "archive-sha"
