from __future__ import annotations

import lzma
import pickle

import numpy as np

from navsim.agents.recogdrive.pareto_support import CandidateRecord, build_archive_record
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
        "support_build_metadata": {
            "raw_internal_candidates": True,
            "expand_external_candidates": False,
            "policy_samples_per_scene": 8,
            "policy_checkpoint_sha256": "checkpoint-sha",
            "fs_norm_stats_sha256": "stats-sha",
        },
    }
    selected = select_feasible_pareto_support(candidates, candidates[0].components, cfg)
    return build_archive_record("scene", candidates, selected, ref=candidates[0].components, cfg=cfg)


def test_v4_validator_accepts_coherent_learnable_mode_record() -> None:
    result = validate_record(_record())

    assert result["errors"] == []
    assert result["selected_count"] == 2
    assert result["multimode"] == 1
    assert result["reachable_pareto_mode_capacity"] == 1
    assert result["capacity_normalized_coverage"] == 1.0


def test_v4_validator_rejects_selected_non_pareto_candidate() -> None:
    record = _record()
    non_gt_index = next(index for index in record["support_indices"] if record["sources"][index] != "gt")
    record["pareto_front_mask"][non_gt_index] = False

    result = validate_record(record)

    assert any("epsilon-Pareto" in error for error in result["errors"])


def test_raw_v4_reselection_preserves_policy_provenance() -> None:
    record = _record()
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
    assert metadata["max_policy_snsad"] == 0.50
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
