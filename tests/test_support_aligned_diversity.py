import lzma
import pickle

import numpy as np

from navsim.agents.recogdrive.support_aligned_diversity import (
    SupportAlignedDiversityConfig,
    compute_support_aligned_diversity,
    density_balanced_support_weights,
    load_support_reference,
    trajectory_distance_matrix,
)


def _trajectory(final_x: float, lateral: float = 0.0, heading: float = 0.0) -> np.ndarray:
    progress = np.linspace(final_x / 8.0, final_x, 8, dtype=np.float64)
    return np.stack(
        [progress, np.full(8, lateral, dtype=np.float64), np.full(8, heading, dtype=np.float64)],
        axis=-1,
    )


def test_single_gt_reference_is_well_defined_for_deterministic_policy():
    gt = _trajectory(16.0)
    predictions = np.repeat(gt[None], 16, axis=0)

    metrics = compute_support_aligned_diversity(predictions, gt[None])

    assert metrics["support_precision_auc"] == 1.0
    assert metrics["density_corrected_support_recall_auc"] == 1.0
    assert metrics["relative_dispersion_calibration"] == 1.0
    assert metrics["kernel_effective_mode_coverage_auc"] == 1.0
    assert metrics["snsad"] == 1.0
    assert np.isfinite(list(metrics.values())).all()


def test_multimodal_gt_collapse_has_fidelity_but_low_coverage_and_dispersion():
    support = np.stack([_trajectory(10.0), _trajectory(16.0), _trajectory(22.0)])
    predictions = np.repeat(support[1:2], 32, axis=0)

    metrics = compute_support_aligned_diversity(predictions, support)

    assert metrics["support_precision_auc"] > 0.99
    assert metrics["density_corrected_support_recall_auc"] < 0.75
    assert metrics["relative_dispersion_ratio"] < 0.25
    assert metrics["normalized_effective_mode_coverage"] < 1.0
    assert metrics["kernel_effective_mode_coverage_auc"] < 0.75


def test_covering_support_beats_collapsed_and_off_support_policies():
    support = np.stack([_trajectory(10.0), _trajectory(16.0), _trajectory(22.0)])
    covered = np.repeat(support, 8, axis=0)
    collapsed = np.repeat(support[1:2], 24, axis=0)
    off_support = np.stack([_trajectory(value, lateral=8.0) for value in np.linspace(4.0, 28.0, 24)])

    covered_metrics = compute_support_aligned_diversity(covered, support)
    collapsed_metrics = compute_support_aligned_diversity(collapsed, support)
    off_support_metrics = compute_support_aligned_diversity(off_support, support)

    assert covered_metrics["snsad"] > collapsed_metrics["snsad"]
    assert covered_metrics["snsad"] > off_support_metrics["snsad"]
    assert off_support_metrics["support_precision_auc"] < 0.2


def test_density_balancing_removes_duplicate_support_vote_advantage():
    support = np.stack(
        [
            _trajectory(10.0),
            _trajectory(10.0),
            _trajectory(10.0),
            _trajectory(22.0),
        ]
    )
    distance = trajectory_distance_matrix(support, support)
    weights = density_balanced_support_weights(distance, bandwidth=0.4)

    assert np.isclose(weights[:3].sum(), weights[3], atol=0.08)
    assert np.isclose(weights.sum(), 1.0)


def test_heading_distance_wraps_at_pi():
    left = _trajectory(12.0, heading=np.pi - 0.01)
    right = _trajectory(12.0, heading=-np.pi + 0.01)

    distance = trajectory_distance_matrix(left[None], right[None])

    assert 0.0 < distance.item() < 0.02


def test_invalid_configuration_fails_fast():
    try:
        SupportAlignedDiversityConfig(bandwidths=())
    except ValueError as exc:
        assert "bandwidths" in str(exc)
    else:
        raise AssertionError("Expected empty bandwidths to fail.")


def test_legacy_v3_loader_uses_selected_positive_support_only(tmp_path):
    trajectories = np.stack([_trajectory(10.0), _trajectory(16.0), _trajectory(22.0)])
    payload = {
        "version": 3,
        "token": "scene-a",
        "candidates": trajectories,
        "support_indices": [0, 2],
        "support_tags": ["gt_anchor", "hard_negative", "diversity_max"],
        "rewards": np.asarray([0.8, 0.1, 0.9]),
    }
    archive_path = tmp_path / "archive.pkl.xz"
    with lzma.open(archive_path, "wb") as stream:
        pickle.dump(payload, stream)

    reference = load_support_reference(archive_path)

    assert reference.archive_version == 3
    assert reference.token == "scene-a"
    assert reference.tags == ("gt_anchor", "diversity_max")
    np.testing.assert_allclose(reference.trajectories, trajectories[[0, 2]])
