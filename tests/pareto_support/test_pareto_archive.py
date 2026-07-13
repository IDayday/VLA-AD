from __future__ import annotations

import numpy as np
import pytest

from navsim.agents.recogdrive.pareto_support.dataclasses import CandidateRecord, TrajectoryCandidate
from navsim.agents.recogdrive.pareto_support.pareto_archive import (
    build_archive_record,
    build_support_set,
    pareto_front_mask,
    select_feasible_pareto_support,
)
from scripts.tools.reselect_sg_fps_support_archive import reselect_record


def _traj(offset: float) -> np.ndarray:
    x = np.linspace(0.2, 4.0 + offset, 8, dtype=np.float32)
    return np.stack([x, np.zeros_like(x) + offset * 0.1, np.zeros_like(x)], axis=-1)


def _cand(source: str, pdms: float, ep: float, ddc: float, feas: float, idx: int) -> TrajectoryCandidate:
    return TrajectoryCandidate(
        scene_token="scene",
        candidate_id=f"scene:{source}:{idx}",
        source=source,
        trajectory=_traj(float(idx) * 0.1),
        true_metrics={
            "pdms": pdms,
            "no_at_fault_collisions": 1.0,
            "drivable_area_compliance": 1.0,
            "time_to_collision_within_bound": 1.0,
            "ego_progress": ep,
            "history_comfort": 1.0,
            "lane_keeping": 1.0,
            "driving_direction_compliance": ddc,
            "traffic_light_compliance": 1.0,
            "feas_cost": feas,
        },
    )


def test_dominated_candidates_excluded_from_pareto_front() -> None:
    values = np.asarray([[1.0, 1.0, 1.0, 0.0], [0.5, 0.5, 0.5, -0.5]], dtype=np.float32)
    mask = pareto_front_mask(values, np.asarray([True, True]))
    assert mask.tolist() == [True, False]


def test_pareto_front_eps_preserves_near_equal_tradeoff() -> None:
    values = np.asarray([[1.0, 1.0], [0.995, 1.0]], dtype=np.float32)

    exact = pareto_front_mask(values, np.asarray([True, True]))
    tolerant = pareto_front_mask(values, np.asarray([True, True]), eps=0.01)

    assert exact.tolist() == [True, False]
    assert tolerant.tolist() == [True, True]


def test_quota_fills_gt_il_and_support_limit() -> None:
    candidates = [
        _cand("gt", 0.6, 0.5, 1.0, 0.0, 0),
        _cand("recogdrive_stage3", 0.7, 0.6, 1.0, 0.0, 1),
        _cand("ddv2", 0.8, 0.8, 1.0, 0.0, 2),
        _cand("driveor", 0.75, 0.7, 1.0, 0.0, 3),
        _cand("ddc_repair", 0.65, 0.6, 1.0, 0.0, 4),
    ]
    archive = build_support_set(candidates, candidates[1], {"support_top_m": 4})
    tags = {c.support_category for c in archive.support_set}
    assert "gt_anchor" in tags
    assert "il_anchor" in tags
    assert len(archive.support_set) <= 4


def _legacy_record(source: str, idx: int, *, valid: bool, reward: float) -> CandidateRecord:
    components = {
        "pdms": reward,
        "no_at_fault_collisions": 1.0 if valid else 0.0,
        "drivable_area_compliance": 1.0,
        "time_to_collision_within_bound": 1.0,
        "ego_progress": 0.5 + 0.1 * idx,
        "history_comfort": 1.0,
        "lane_keeping": 1.0,
        "driving_direction_compliance": 1.0,
        "traffic_light_compliance": 1.0,
    }
    return CandidateRecord(
        trajectory=_traj(float(idx) * 0.1),
        source=source,
        token="scene",
        components=components,
        reward=reward,
        feas={"feas_cost": 0.0},
        selection_score=reward,
    )


def test_legacy_support_does_not_tag_invalid_non_anchor_candidates() -> None:
    candidates = [
        _legacy_record("gt", 0, valid=False, reward=0.9),
        _legacy_record("il", 1, valid=False, reward=0.8),
        _legacy_record("progress_endpoint", 2, valid=False, reward=1.0),
        _legacy_record("progress_speed", 3, valid=True, reward=0.7),
    ]
    selected = select_feasible_pareto_support(candidates, candidates[0].components, {"support_top_m": 8})
    tagged = {candidate.source: candidate.support_tag for candidate in selected}

    assert tagged["gt"] == "gt_anchor"
    assert tagged["progress_speed"]
    assert "il" not in tagged
    assert "progress_endpoint" not in tagged


def test_archive_anchor_distance_uses_gt_even_when_gt_is_not_first() -> None:
    candidates = [
        _legacy_record("progress_endpoint", 4, valid=True, reward=0.8),
        _legacy_record("gt", 1, valid=True, reward=0.6),
        _legacy_record("ddv2", 2, valid=True, reward=0.7),
    ]

    record = build_archive_record(
        "scene",
        candidates,
        candidates,
        ref=candidates[1].components,
        cfg={"support_top_m": 4},
    )

    gt_xy = candidates[1].trajectory[..., :2]
    expected = [
        float(np.linalg.norm(candidate.trajectory[..., :2] - gt_xy, axis=-1).mean())
        for candidate in candidates
    ]
    assert record["anchor_distance_reference"] == "gt"
    assert record["anchor_distance"] == pytest.approx(expected)
    assert record["anchor_distance"][1] == 0.0


def test_mode_pareto_v4_is_gt_anchored_and_excludes_derived_external() -> None:
    candidates = [
        _legacy_record("gt", 0, valid=True, reward=0.90),
        _legacy_record("policy", 1, valid=True, reward=0.93),
        _legacy_record("ddv2", 3, valid=True, reward=0.96),
        _legacy_record("ddv2:failure_expand_0", 4, valid=True, reward=1.00),
        _legacy_record("progress_endpoint", 2, valid=True, reward=0.95),
        _legacy_record("driveor", 5, valid=True, reward=0.94),
    ]
    cfg = {
        "support_archive_version": 4,
        "support_selection_strategy": "mode_pareto_v4",
        "support_top_m": 4,
        "support_v4_max_gt_ade_m": 1.5,
        "support_v4_max_gt_fde_m": 4.0,
        "support_v4_mode_distance_threshold": 0.05,
        "support_v4_exclude_derived_external": True,
        "support_v4_require_policy_reachability": True,
        "support_v4_max_policy_snsad": 0.75,
        "support_v4_min_policy_neighbors": 1,
    }

    selected = select_feasible_pareto_support(candidates, candidates[0].components, cfg)
    record = build_archive_record("scene", candidates, selected, ref=candidates[0].components, cfg=cfg)

    selected_sources = [record["sources"][index] for index in record["support_indices"]]
    assert selected_sources[0] == "gt"
    assert "ddv2:failure_expand_0" not in selected_sources
    assert record["version"] == 4
    assert (
        record["build_metadata"]["teacher_contract"]
        == "gt_anchor_plus_uniform_reachable_pareto_modes"
    )
    assert (
        record["build_metadata"]["mode_selection_order"]
        == "scene_normalized_objective_fps_then_snsad_then_policy_density"
    )
    assert record["candidate_funnel"]["selected_non_gt_count"] == len(selected_sources) - 1
    selected_modes = record["mode_ids"][record["support_indices"]]
    assert sorted(selected_modes.tolist()) == list(range(len(selected_sources)))
    assert record["teacher_eligible_mask"][record["support_indices"]].all()
    raw_ddv2_index = record["sources"].index("ddv2")
    assert bool(record["source_conditioned_mask"][raw_ddv2_index])
    assert np.isfinite(record["policy_reachability_snsad"][record["support_indices"]]).all()
    assert np.isfinite(record["pareto_objective_novelty"][record["support_indices"]]).all()


def test_mode_pareto_v4_near_gt_candidate_cannot_suppress_distinct_mode() -> None:
    gt = _legacy_record("gt", 0, valid=True, reward=0.90)
    near_gt = _legacy_record("policy", 0, valid=True, reward=0.99)
    near_gt.trajectory = gt.trajectory.copy()
    near_gt.trajectory[:, 1] += 0.01
    distinct = _legacy_record("progress_endpoint", 1, valid=True, reward=0.92)
    distinct.trajectory = gt.trajectory.copy()
    distinct.trajectory[:, 1] += 0.6

    gt.components.update(ego_progress=0.80, time_to_collision_within_bound=1.0, history_comfort=1.0)
    near_gt.components.update(
        ego_progress=1.0,
        time_to_collision_within_bound=1.0,
        history_comfort=1.0,
    )
    distinct.components.update(
        ego_progress=0.90,
        time_to_collision_within_bound=0.90,
        history_comfort=1.0,
    )
    cfg = {
        "support_archive_version": 4,
        "support_selection_strategy": "mode_pareto_v4",
        "support_top_m": 4,
        "support_v4_max_gt_ade_m": 1.5,
        "support_v4_max_gt_fde_m": 4.0,
        "support_v4_mode_distance_threshold": 0.25,
        "support_v4_max_gt_reward_drop": 0.05,
        "support_v4_pareto_eps": 0.01,
        "support_v4_require_policy_reachability": True,
        "support_v4_max_policy_snsad": 0.75,
        "support_v4_min_policy_neighbors": 1,
    }

    candidates = [gt, near_gt, distinct]
    selected = select_feasible_pareto_support(candidates, gt.components, cfg)
    record = build_archive_record("scene", candidates, selected, ref=gt.components, cfg=cfg)
    selected_sources = [record["sources"][index] for index in record["support_indices"]]

    assert selected_sources == ["gt", "progress_endpoint"]
    assert not bool(record["pareto_front_mask"][1])
    assert bool(record["pareto_front_mask"][2])
    assert record["candidate_funnel"]["distinct_from_gt_count"] == 1
    assert record["candidate_funnel"]["version"] == 2


def test_mode_pareto_v4_rejects_mode_outside_policy_learning_frontier() -> None:
    gt = _legacy_record("gt", 0, valid=True, reward=0.90)
    policy = _legacy_record("policy", 1, valid=True, reward=0.91)
    unreachable = _legacy_record("progress_endpoint", 12, valid=True, reward=0.95)
    cfg = {
        "support_archive_version": 4,
        "support_selection_strategy": "mode_pareto_v4",
        "support_top_m": 4,
        "support_v4_max_gt_ade_m": 10.0,
        "support_v4_max_gt_fde_m": 20.0,
        "support_v4_mode_distance_threshold": 0.01,
        "support_v4_require_policy_reachability": True,
        "support_v4_max_policy_snsad": 0.05,
        "support_v4_min_policy_neighbors": 1,
    }

    selected = select_feasible_pareto_support([gt, policy, unreachable], gt.components, cfg)

    assert "progress_endpoint" not in [candidate.source for candidate in selected]


def test_mode_pareto_v4_does_not_fill_with_gt_dominated_candidate() -> None:
    gt = _legacy_record("gt", 0, valid=True, reward=1.0)
    dominated = _legacy_record("progress_speed", 1, valid=True, reward=0.96)
    dominated.components["ego_progress"] = gt.components["ego_progress"] - 0.1
    cfg = {
        "support_archive_version": 4,
        "support_selection_strategy": "mode_pareto_v4",
        "support_top_m": 4,
        "support_v4_mode_distance_threshold": 0.01,
        "support_v4_max_gt_reward_drop": 0.05,
        "support_v4_pareto_eps": 0.01,
    }

    selected = select_feasible_pareto_support([gt, dominated], gt.components, cfg)

    assert [candidate.source for candidate in selected] == ["gt"]


def test_mode_pareto_v4_covers_objective_front_before_scalar_reward() -> None:
    gt = _legacy_record("gt", 0, valid=True, reward=0.90)
    near_scalar_best = _legacy_record("policy", 1, valid=True, reward=0.99)
    tradeoff = _legacy_record("progress_tradeoff", 4, valid=True, reward=0.91)
    gt.components.update(ego_progress=0.50, time_to_collision_within_bound=0.50)
    near_scalar_best.components.update(ego_progress=0.51, time_to_collision_within_bound=0.51)
    tradeoff.components.update(ego_progress=0.90, time_to_collision_within_bound=0.20)
    cfg = {
        "support_archive_version": 4,
        "support_selection_strategy": "mode_pareto_v4",
        "support_top_m": 2,
        "support_v4_max_gt_ade_m": 10.0,
        "support_v4_max_gt_fde_m": 20.0,
        "support_v4_mode_distance_threshold": 0.01,
        "support_v4_max_gt_reward_drop": 0.05,
        "support_v4_pareto_eps": 0.0,
    }

    selected = select_feasible_pareto_support([gt, near_scalar_best, tradeoff], gt.components, cfg)

    assert [candidate.source for candidate in selected] == ["gt", "progress_tradeoff"]
    assert float(getattr(selected[1], "_sg_fps_objective_novelty")) > 0.0


def test_mode_pareto_v4_preserves_diversity_after_policy_density_gate() -> None:
    gt = _legacy_record("gt", 0, valid=True, reward=0.90)
    policies = [
        _legacy_record(f"policy:{index}", index, valid=True, reward=0.90)
        for index in (1, 2, 3, 4, 8, 9)
    ]
    far_mode = _legacy_record("lateral_offset", 10, valid=True, reward=0.90)
    candidates = [gt, *policies, far_mode]
    for candidate in candidates:
        candidate.components.update(ego_progress=0.80, time_to_collision_within_bound=1.0)
    cfg = {
        "support_archive_version": 4,
        "support_selection_strategy": "mode_pareto_v4",
        "support_top_m": 2,
        "support_v4_max_gt_ade_m": 10.0,
        "support_v4_max_gt_fde_m": 20.0,
        "support_v4_mode_distance_threshold": 0.05,
        "support_v4_pareto_eps": 0.0,
        "support_v4_require_policy_reachability": True,
        "support_v4_max_policy_snsad": 0.06,
        "support_v4_min_policy_neighbors": 2,
    }

    selected = select_feasible_pareto_support(candidates, gt.components, cfg)

    assert [candidate.source for candidate in selected] == ["gt", "lateral_offset"]
    assert getattr(selected[1], "_sg_fps_policy_neighbor_count") == 2


def test_mode_pareto_v4_uses_only_its_explicit_gt_relative_reward_gate() -> None:
    gt = _legacy_record("gt", 0, valid=True, reward=0.50)
    tradeoff = _legacy_record("progress_tradeoff", 2, valid=True, reward=0.48)
    gt.components.update(ego_progress=0.50, time_to_collision_within_bound=0.90)
    tradeoff.components.update(ego_progress=0.80, time_to_collision_within_bound=0.70)
    cfg = {
        "support_archive_version": 4,
        "support_selection_strategy": "mode_pareto_v4",
        "support_top_m": 2,
        "support_quality_enable": True,
        "support_min_non_gt_reward": 0.90,
        "support_reward_gate_mode": "absolute",
        "support_v4_max_gt_reward_drop": 0.05,
        "support_v4_mode_distance_threshold": 0.01,
        "support_v4_pareto_eps": 0.0,
    }

    selected = select_feasible_pareto_support([gt, tradeoff], gt.components, cfg)
    record = build_archive_record("scene", [gt, tradeoff], selected, ref=gt.components, cfg=cfg)

    assert [candidate.source for candidate in selected] == ["gt", "progress_tradeoff"]
    assert record["build_metadata"]["legacy_reward_gate_disabled"] is True
    assert record["support_quality_mask"][record["support_indices"]].all()


def test_reselect_archive_record_replaces_invalid_non_anchor_support() -> None:
    candidates = [
        _legacy_record("gt", 0, valid=True, reward=0.5),
        _legacy_record("il", 1, valid=True, reward=0.6),
        _legacy_record("progress_endpoint", 2, valid=False, reward=1.0),
        _legacy_record("progress_speed", 3, valid=True, reward=0.7),
    ]
    candidates[2].support_tag = "best_pdms"
    old_record = build_archive_record(
        "scene",
        candidates,
        [candidates[0], candidates[1], candidates[2]],
        ref=candidates[0].components,
        cfg={"support_top_m": 4},
    )

    new_record, stats = reselect_record(
        old_record,
        {
            "support_top_m": 4,
            "fpv3_ddc_min_absolute": 0.95,
            "fpv3_ddc_drop_tolerance": 0.01,
            "fpv3_feas_cost_max": 0.10,
            "fpv3_comfort_min": 0.95,
        },
    )
    tagged_sources = {
        source: tag
        for source, tag in zip(new_record["sources"], new_record["support_tags"])
        if tag
    }

    assert stats["records_changed"] == 1
    assert "progress_endpoint" not in tagged_sources
    assert "progress_speed" in tagged_sources
