from __future__ import annotations

import lzma
import pickle

import numpy as np

from scripts.stage3.build_lfp_diversity_capacity_cache import _process_one


def _write_archive(path, *, include_mode: bool) -> None:
    gt = np.zeros((8, 3), dtype=np.float32)
    mode = gt.copy()
    mode[:, 0] = np.linspace(0.0, 4.0, 8)
    candidates = np.stack((gt, mode)) if include_mode else gt[None]
    count = candidates.shape[0]
    payload = {
        "version": 4,
        "token": "scene",
        "candidates": candidates,
        "support_indices": np.arange(count, dtype=np.int64),
        "support_tags": ["gt", "mode"][:count],
        "rewards": np.ones(count, dtype=np.float32),
        "mode_ids": np.arange(count, dtype=np.int64),
        "candidate_funnel": {
            "supervision_type": "frontier_modes" if include_mode else "gt_only",
            "gt_only_reason": "" if include_mode else "no_distinct_mode",
        },
        "build_metadata": {
            "candidate_capacity_contract": "observed_funnel_distinct_before_pareto_no_quota_v2"
        },
    }
    with lzma.open(path, "wb") as stream:
        pickle.dump(payload, stream)


def test_capacity_builder_preserves_observed_no_quota_semantics(tmp_path) -> None:
    path = tmp_path / "scene.pkl.xz"
    _write_archive(path, include_mode=True)

    token, record = _process_one((str(path), {}))

    assert token == "scene"
    assert record["selected_non_gt_count"] == 1
    assert record["reference_mode_count"] == 2
    assert record["supervision_type"] == "frontier_modes"
    assert record["gt_only_reason"] == ""
    assert record["candidate_capacity_contract"].endswith("no_quota_v2")


def test_capacity_builder_accepts_gt_only_scene(tmp_path) -> None:
    path = tmp_path / "scene.pkl.xz"
    _write_archive(path, include_mode=False)

    _, record = _process_one((str(path), {}))

    assert record["selected_non_gt_count"] == 0
    assert record["reference_mode_count"] == 1
    assert record["support_pairwise_ade_m"] == 0.0
    assert record["supervision_type"] == "gt_only"
    assert record["gt_only_reason"] == "no_distinct_mode"
