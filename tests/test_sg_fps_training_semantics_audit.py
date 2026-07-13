from __future__ import annotations

import numpy as np

from scripts.tools.audit_sg_fps_training_semantics import analyze_record


def _record() -> dict:
    gt = np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    duplicate = gt.copy()
    external = np.asarray([[1.0, 0.5, 0.0], [2.0, 1.0, 0.0]], dtype=np.float32)
    count = 3
    return {
        "candidates": np.stack((gt, duplicate, external)),
        "sources": ["gt", "progress_speed", "ddv2"],
        "support_indices": [0, 1, 2],
        "support_tags": ["gt_anchor", "diversity_max", "best_pdms"],
        "rewards": np.ones(count, dtype=np.float32),
        "components": {
            "no_at_fault_collisions": np.ones(count, dtype=np.float32),
            "drivable_area_compliance": np.ones(count, dtype=np.float32),
            "time_to_collision_within_bound": np.ones(count, dtype=np.float32),
            "ego_progress": np.asarray([0.8, 0.8, 1.0], dtype=np.float32),
            "history_comfort": np.ones(count, dtype=np.float32),
            "driving_direction_compliance": np.ones(count, dtype=np.float32),
        },
    }


def test_training_semantics_audit_exposes_source_and_duplicate_structure() -> None:
    result = analyze_record(_record())

    assert result["candidate_buckets"]["external"] == 1
    assert result["selected_buckets"]["gt"] == 1
    assert result["all_selected_reward_one"] == 1
    assert result["near_duplicate_pair_count"] >= 1
    assert result["mode_count_snsad_040"] < result["selected_count"]


def test_training_semantics_audit_rejects_missing_gt() -> None:
    record = _record()
    record["sources"][0] = "progress_speed"

    try:
        analyze_record(record)
    except ValueError as exc:
        assert "exactly one GT" in str(exc)
    else:
        raise AssertionError("Expected missing GT to fail fast.")
