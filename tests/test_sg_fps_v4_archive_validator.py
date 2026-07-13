from __future__ import annotations

import lzma
import pickle

import numpy as np

from navsim.agents.recogdrive.pareto_support import (
    CandidateRecord,
    build_archive_record,
    compute_policy_credit_readiness,
)
from navsim.agents.recogdrive.pareto_support import select_feasible_pareto_support
from scripts.tools.validate_sg_fps_v4_archive import validate_archive, validate_record
from scripts.tools.reselect_sg_fps_support_archive import reselect_record


def _trajectory(offset: float) -> np.ndarray:
    x = np.linspace(0.5, 8.0, 8, dtype=np.float32)
    y = np.linspace(0.0, offset, 8, dtype=np.float32)
    return np.stack((x, y, np.zeros_like(x)), axis=-1)


def _candidate(source: str, offset: float, ep: float, reward: float) -> CandidateRecord:
    return CandidateRecord(
        trajectory=_trajectory(offset),
        source=source,
        token="scene",
        components={
            "pdms": reward,
            "no_at_fault_collisions": 1.0,
            "drivable_area_compliance": 1.0,
            "time_to_collision_within_bound": 1.0,
            "ego_progress": ep,
            "history_comfort": 1.0,
            "driving_direction_compliance": 1.0,
        },
        reward=reward,
        feas={"feas_cost": 0.0},
        selection_score=reward,
    )


def _record() -> dict:
    candidates = [
        _candidate("gt", 0.0, 0.95, 0.95),
        _candidate("policy", 1.0, 0.96, 0.96),
    ]
    cfg = {
        "support_archive_version": 4,
        "support_selection_strategy": "mode_pareto_v4",
        "support_top_m": 4,
        "support_v4_max_gt_ade_m": 1.5,
        "support_v4_max_gt_fde_m": 4.0,
        "support_v4_mode_distance_threshold": 0.10,
        "support_v4_max_gt_reward_drop": 0.05,
        "support_v4_pareto_eps": 0.01,
        "support_v4_require_policy_reachability": True,
        "support_v4_max_policy_snsad": 0.75,
        "support_v4_min_policy_neighbors": 1,
        "support_build_metadata": {
            "raw_internal_candidates": True,
            "expand_external_candidates": False,
            "policy_samples_per_scene": 1,
            "policy_checkpoint_sha256": "checkpoint-sha",
            "fs_norm_stats_sha256": "stats-sha",
            "build_log_split": "train_val",
            "dataset_scene_count": 1,
            "scene_seed_scheme": "sha256_token_xor_build_seed_v1",
        },
    }
    selected = select_feasible_pareto_support(candidates, candidates[0].components, cfg)
    return build_archive_record("scene", candidates, selected, ref=candidates[0].components, cfg=cfg)


def _gt_only_record() -> dict:
    candidates = [
        _candidate("gt", 0.0, 0.95, 0.95),
        _candidate("policy", 1.0, 0.96, 0.96),
    ]
    candidates[1].components["no_at_fault_collisions"] = 0.0
    cfg = {
        "support_archive_version": 4,
        "support_selection_strategy": "mode_pareto_v4",
        "support_top_m": 4,
        "support_v4_max_gt_ade_m": 1.5,
        "support_v4_max_gt_fde_m": 4.0,
        "support_v4_mode_distance_threshold": 0.10,
        "support_v4_max_gt_reward_drop": 0.05,
        "support_v4_pareto_eps": 0.01,
        "support_v4_require_policy_reachability": True,
        "support_v4_max_policy_snsad": 0.75,
        "support_v4_min_policy_neighbors": 1,
        "support_build_metadata": {
            "raw_internal_candidates": True,
            "expand_external_candidates": False,
            "policy_samples_per_scene": 1,
            "policy_checkpoint_sha256": "checkpoint-sha",
            "fs_norm_stats_sha256": "stats-sha",
            "build_log_split": "train_val",
            "dataset_scene_count": 1,
            "scene_seed_scheme": "sha256_token_xor_build_seed_v1",
        },
    }
    selected = select_feasible_pareto_support(candidates, candidates[0].components, cfg)
    return build_archive_record("hard-scene", candidates, selected, ref=candidates[0].components, cfg=cfg)


def test_v4_validator_accepts_coherent_learnable_mode_record() -> None:
    result = validate_record(_record())

    assert result["errors"] == []
    assert result["selected_count"] == 2
    assert result["multimode"] == 1
    assert result["reachable_pareto_mode_capacity"] == 1
    assert result["capacity_normalized_coverage"] == 1.0


def test_v4_validator_accepts_gt_only_scene_without_forcing_a_mode() -> None:
    record = _gt_only_record()
    result = validate_record(record)

    assert result["errors"] == []
    assert record["support_indices"] == [0]
    assert result["candidate_funnel"]["supervision_type"] == "gt_only"
    assert result["candidate_funnel"]["gt_only_reason"] == "no_evaluator_valid_candidate"
    assert result["reachable_pareto_mode_capacity"] == 0
    assert np.isnan(result["capacity_normalized_coverage"])


def test_v4_validator_rejects_inconsistent_candidate_funnel() -> None:
    record = _gt_only_record()
    record["candidate_funnel"]["selected_non_gt_count"] = 1

    result = validate_record(record)

    assert any("candidate_funnel selected_non_gt_count" in error for error in result["errors"])


def test_v4_validator_rejects_selected_non_pareto_candidate() -> None:
    record = _record()
    non_gt_index = next(index for index in record["support_indices"] if record["sources"][index] != "gt")
    record["pareto_front_mask"][non_gt_index] = False

    result = validate_record(record)

    assert any("epsilon-Pareto" in error for error in result["errors"])


def test_v4_validator_recomputes_gate_masks() -> None:
    record = _record()
    record["valid_mask"][1] = False

    result = validate_record(record)

    assert any("valid_mask disagrees" in error for error in result["errors"])


def test_v4_validator_rejects_incoherent_external_provenance() -> None:
    record = _record()
    record["build_metadata"]["external_candidate_roots"] = {"ddv2": "/data/ddv2"}

    result = validate_record(record)

    assert any("provenance source keys disagree" in error for error in result["errors"])


def test_raw_v4_reselection_preserves_policy_provenance() -> None:
    record = _record()
    source_gate_config = dict(record["build_metadata"]["selection_gate_config"])
    rebuilt, _ = reselect_record(
        record,
        {
            "support_archive_version": 4,
            "support_selection_strategy": "mode_pareto_v4",
            "support_top_m": 4,
            "support_v4_mode_distance_threshold": 0.10,
            "support_v4_require_policy_reachability": True,
            "support_v4_max_policy_snsad": 0.50,
        },
    )

    metadata = rebuilt["build_metadata"]
    assert metadata["raw_internal_candidates"] is True
    assert metadata["reselected_from_raw_v4"] is True
    assert metadata["policy_checkpoint_sha256"] == "checkpoint-sha"
    assert metadata["max_policy_snsad"] == record["build_metadata"]["max_policy_snsad"]
    assert metadata["selection_gate_config"] == source_gate_config
    assert validate_record(rebuilt)["errors"] == []


def test_v4_archive_coverage_rejects_missing_expected_token(tmp_path) -> None:
    archive = tmp_path / "archive"
    expected = tmp_path / "expected.txt"
    archive.mkdir()
    with lzma.open(archive / "scene.pkl.xz", "wb") as stream:
        pickle.dump(_record(), stream)
    expected.write_text("scene\nmissing\n", encoding="utf-8")

    report = validate_archive(archive, expected_token_source=expected)

    assert report["hard_contract_pass"] is True
    assert report["token_coverage_pass"] is False
    assert report["missing_token_count"] == 1
    assert report["missing_token_examples"] == ["missing"]
    assert report["static_promotion_pass"] is False


def test_gt_only_scene_does_not_fail_static_data_contract_on_policy_readiness(tmp_path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    with lzma.open(archive / "hard-scene.pkl.xz", "wb") as stream:
        pickle.dump(_gt_only_record(), stream)

    report = validate_archive(archive, min_multimode_scene_ratio=0.0)

    assert report["hard_contract_pass"] is True
    assert report["static_promotion_pass"] is True
    assert report["initial_policy_readiness_pass"] is False
    assert report["supervision_type_counts"] == {"gt_only": 1}


def test_policy_credit_readiness_requires_bidirectional_credit() -> None:
    candidates = [
        _candidate("gt", 0.0, 0.95, 0.95),
        _candidate("policy", 0.2, 0.98, 0.98),
        _candidate("policy", 0.4, 0.96, 0.96),
        _candidate("policy", 0.6, 0.92, 0.92),
        _candidate("policy", 0.8, 0.90, 0.90),
    ]
    diagnostics = compute_policy_credit_readiness(
        candidates,
        np.ones(len(candidates), dtype=np.bool_),
        np.ones(len(candidates), dtype=np.bool_),
        0,
        {
            "support_v4_stage3_min_feasible_rollouts": 2,
            "support_v4_stage3_min_score_span": 0.01,
        },
    )

    assert diagnostics["positive_fraction"] > 0.0
    assert diagnostics["negative_fraction"] > 0.0
    assert diagnostics["bidirectional_energy"] > 0.0
    assert diagnostics["credit_ready"] is True


def test_policy_credit_readiness_rejects_saturated_equal_rollouts() -> None:
    candidates = [_candidate("gt", 0.0, 0.95, 0.95)] + [
        _candidate("policy", float(index) * 0.2, 0.95, 0.95)
        for index in range(1, 5)
    ]
    diagnostics = compute_policy_credit_readiness(
        candidates,
        np.ones(len(candidates), dtype=np.bool_),
        np.ones(len(candidates), dtype=np.bool_),
        0,
        {},
    )

    assert diagnostics["positive_fraction"] == 0.0
    assert diagnostics["negative_fraction"] == 0.0
    assert diagnostics["bidirectional_energy"] == 0.0
    assert diagnostics["credit_ready"] is False


def test_v4_validator_rejects_fake_policy_density_witness() -> None:
    record = _record()
    non_gt_index = next(index for index in record["support_indices"] if record["sources"][index] != "gt")
    record["policy_neighbor_count"][non_gt_index] = 0

    result = validate_record(record)

    assert any("policy_neighbor_count disagrees" in error for error in result["errors"])


def test_v4_validator_recomputes_policy_density_instead_of_trusting_archive() -> None:
    record = _record()
    non_gt_index = next(index for index in record["support_indices"] if record["sources"][index] != "gt")
    record["policy_neighbor_count"][non_gt_index] = 7
    record["policy_neighbor_fraction"][non_gt_index] = 1.0

    result = validate_record(record)

    assert any("policy_neighbor_count disagrees" in error for error in result["errors"])
