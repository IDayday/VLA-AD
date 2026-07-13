from __future__ import annotations

import numpy as np

from navsim.agents.recogdrive.pareto_support.full_training_v6 import (
    FullTrainingV6Config,
    full_training_v6_selection,
    promote_v5_record_to_v6,
)


def _trajectory(*, x_scale: float = 1.0, lateral: float = 0.0) -> np.ndarray:
    x = np.linspace(0.25, 4.0, 8, dtype=np.float32) * float(x_scale)
    y = np.full_like(x, float(lateral))
    return np.stack((x, y, np.zeros_like(x)), axis=-1)


def _record() -> dict:
    candidates = np.stack(
        (
            _trajectory(),
            _trajectory(lateral=1.0),
            _trajectory(lateral=-1.0),
            _trajectory(x_scale=1.35),
            _trajectory(lateral=1.8),
        ),
        axis=0,
    )
    count = candidates.shape[0]
    ones = np.ones((count,), dtype=np.float32)
    return {
        "version": 5,
        "token": "scene",
        "candidates": candidates,
        "sources": ["gt", "policy:0", "ddv2", "progress_endpoint", "lateral_offset"],
        "rewards": np.asarray([0.90, 0.95, 0.92, 0.91, 0.89], dtype=np.float32),
        "valid_mask": np.ones((count,), dtype=np.bool_),
        "support_quality_mask": np.ones((count,), dtype=np.bool_),
        "gt_relative_ade": np.asarray([0.0, 1.0, 1.0, 0.75, 1.4], dtype=np.float32),
        "gt_relative_fde": np.asarray([0.0, 1.0, 1.0, 1.4, 1.8], dtype=np.float32),
        "components": {
            "no_at_fault_collisions": ones.copy(),
            "drivable_area_compliance": ones.copy(),
            "time_to_collision_within_bound": ones.copy(),
            "driving_direction_compliance": ones.copy(),
            "ego_progress": np.asarray([0.80, 0.90, 0.82, 0.95, 0.78], dtype=np.float32),
        },
        "build_metadata": {"raw_internal_candidates": True},
    }


def test_full_training_v6_excludes_previous_policy_from_random_init_teacher_set() -> None:
    record = _record()

    selection = full_training_v6_selection(record)
    selected_sources = [record["sources"][index] for index in selection["support_indices"]]

    assert selected_sources[0] == "gt"
    assert "policy:0" not in selected_sources
    assert len(selected_sources) >= 2
    assert selection["candidate_funnel"]["policy_candidate_count"] == 1
    assert selection["candidate_funnel"]["supervision_type"] == "full_training_modes"
    assert selection["teacher_eligible_mask"][0]
    assert not selection["teacher_eligible_mask"][1]
    assert np.all(selection["gt_mode_distance_snsad"][selection["support_indices"]][1:] >= 0.40)


def test_full_training_v6_preserves_evaluator_tradeoffs_and_support_cap() -> None:
    record = _record()
    cfg = FullTrainingV6Config(support_top_m=3)

    selection = full_training_v6_selection(record, cfg)
    non_gt = selection["support_indices"][1:]

    assert len(selection["support_indices"]) <= 3
    assert non_gt
    assert all(selection["teacher_eligible_mask"][index] for index in non_gt)
    assert all(selection["pareto_front_mask"][index] for index in non_gt)
    assert selection["full_training_objectives"].shape == (5, 4)
    assert sorted(selection["mode_ids"][selection["support_indices"]].tolist()) == list(
        range(len(selection["support_indices"]))
    )


def test_promote_v5_record_to_v6_persists_policy_independent_contract() -> None:
    promoted = promote_v5_record_to_v6(_record())

    assert promoted["version"] == 6
    assert promoted["support_tags"][promoted["support_indices"][0]] == "gt_anchor"
    assert promoted["build_metadata"]["selection_strategy"] == "mode_full_training_v6"
    assert promoted["build_metadata"]["policy_candidates_excluded"] is True
    assert promoted["build_metadata"]["policy_fields_used_for_admission"] is False
    assert promoted["build_metadata"]["policy_fields_used_for_ranking"] is False
    assert promoted["candidate_funnel"]["selected_non_gt_count"] == len(
        promoted["support_indices"]
    ) - 1


def test_full_training_v6_rejects_invalid_and_gt_reward_outlier() -> None:
    record = _record()
    record["valid_mask"][2] = False
    record["rewards"][3] = 0.80
    record["support_quality_mask"][4] = False

    selection = full_training_v6_selection(record)

    assert selection["support_indices"] == [0]
    assert selection["candidate_funnel"]["supervision_type"] == "gt_only"
    assert selection["candidate_funnel"]["gt_only_reason"]
