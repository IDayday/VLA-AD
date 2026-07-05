from __future__ import annotations

import numpy as np

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
