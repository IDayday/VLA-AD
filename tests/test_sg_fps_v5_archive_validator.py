from __future__ import annotations

import json
import lzma
import pickle

import numpy as np

from navsim.agents.recogdrive.pareto_support import (
    CandidateRecord,
    build_archive_record,
    select_feasible_pareto_support,
)
from scripts.tools.validate_sg_fps_v5_archive import validate_archive, validate_record


def _trajectory(lateral_offset: float) -> np.ndarray:
    x = np.linspace(0.5, 8.0, 8, dtype=np.float32)
    y = np.full_like(x, lateral_offset)
    return np.stack((x, y, np.zeros_like(x)), axis=-1)


def _candidate(source: str, lateral_offset: float) -> CandidateRecord:
    return CandidateRecord(
        trajectory=_trajectory(lateral_offset),
        source=source,
        token="scene",
        components={
            "pdms": 0.95,
            "no_at_fault_collisions": 1.0,
            "drivable_area_compliance": 1.0,
            "time_to_collision_within_bound": 1.0,
            "ego_progress": 0.95,
            "history_comfort": 1.0,
            "driving_direction_compliance": 1.0,
        },
        reward=0.95,
        feas={"feas_cost": 0.0},
        selection_score=0.95,
    )


def _config() -> dict:
    return {
        "support_archive_version": 5,
        "support_selection_strategy": "mode_learning_frontier_v5",
        "support_top_m": 4,
        "support_v5_max_gt_ade_m": 1.5,
        "support_v5_max_gt_fde_m": 4.0,
        "support_v5_mode_distance_threshold": 0.05,
        "support_v5_max_gt_reward_drop": 0.05,
        "support_v5_pareto_eps": 0.01,
        "support_v5_max_policy_snsad": 1.0,
        "support_v5_min_policy_neighbors": 2,
        "support_v5_mode_evidence_radius": 0.20,
        "support_v5_mode_evidence_margin": 0.01,
        "support_v5_min_policy_witnesses": 2,
        "support_v5_min_source_families": 2,
        "support_build_metadata": {
            "raw_internal_candidates": True,
            "expand_external_candidates": False,
            "policy_samples_per_scene": 2,
            "policy_checkpoint_sha256": "checkpoint-sha",
            "fs_norm_stats_sha256": "stats-sha",
            "build_log_split": "train_val",
            "dataset_scene_count": 1,
            "scene_seed_scheme": "sha256_token_xor_build_seed_v1",
        },
    }


def _record(*, with_evidence: bool) -> dict:
    policy_offsets = (0.58, 0.62) if with_evidence else (0.01, 0.02)
    candidates = [
        _candidate("gt", 0.0),
        _candidate("policy:0", policy_offsets[0]),
        _candidate("policy:1", policy_offsets[1]),
        _candidate("lateral_offset", 0.60),
    ]
    selected = select_feasible_pareto_support(candidates, candidates[0].components, _config())
    return build_archive_record(
        "scene",
        candidates,
        selected,
        ref=candidates[0].components,
        cfg=_config(),
    )


def test_v5_validator_recomputes_independent_mode_evidence() -> None:
    record = _record(with_evidence=True)

    result = validate_record(record)

    assert result["errors"] == []
    assert result["non_gt_count"] == 1
    assert result["evidence_count"] >= 1


def test_v5_validator_rejects_forged_policy_witness_count() -> None:
    record = _record(with_evidence=True)
    selected_non_gt = next(index for index in record["support_indices"] if index != 0)
    record["policy_assignment_witness_count"][selected_non_gt] += 1

    result = validate_record(record)

    assert any("policy_assignment_witness_count" in error for error in result["errors"])


def test_v5_gt_only_scene_passes_static_contract_without_diversity_quota(tmp_path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    with lzma.open(archive / "scene.pkl.xz", "wb") as stream:
        pickle.dump(_record(with_evidence=False), stream)

    report = validate_archive(archive)

    assert report["hard_contract_pass"] is True
    assert report["static_promotion_pass"] is True
    assert report["multimode_scene_ratio"] == 0.0
    assert report["diversity_statistics_are_promotion_gates"] is False
    assert report["gt_only_reason_counts"] == {"no_independent_mode_evidence": 1}


def test_v5_validator_writes_no_quota_stage2_scene_index(tmp_path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    with lzma.open(archive / "scene.pkl.xz", "wb") as stream:
        pickle.dump(_record(with_evidence=True), stream)
    index_path = tmp_path / "scene_index.json"

    report = validate_archive(archive, output_scene_index=index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))

    assert report["static_promotion_pass"] is True
    assert index["no_per_scene_candidate_quota"] is True
    assert index["capacity_semantics"] == "observed_selected_support_not_scene_intrinsic"
    assert index["frontier_eligible_count"] == 1
    assert index["records"][0]["frontier_eligible"] is True
    assert index["records"][0]["priority"] > 0.0
