#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import multiprocessing as mp
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.pareto_support import (  # noqa: E402
    CandidateRecord,
    is_valid_candidate,
    trajectory_snsad_distance,
)
from navsim.agents.recogdrive.pareto_support.pareto_archive import (  # noqa: E402
    support_trajectory_quality_pass,
)
from scripts.tools.validate_sg_fps_v4_archive import (  # noqa: E402
    _iter_paths,
    _load_expected_tokens,
    _stats,
)


def _load(path: Path) -> dict[str, Any]:
    with lzma.open(path, "rb") as stream:
        record = pickle.load(stream)
    if not isinstance(record, dict):
        raise TypeError(f"record must be a dict, got {type(record).__name__}: {path}")
    return record


def _is_gt(source: str) -> bool:
    source = str(source).lower()
    return source == "gt" or source.startswith("gt:")


def _is_policy(source: str) -> bool:
    source = str(source).lower()
    return source == "policy" or source.startswith("policy:") or source.startswith("current_policy")


def _is_derived_external(source: str) -> bool:
    source = str(source).lower()
    external = source.startswith(("ddv2", "driveor", "drivor", "diffusiondrivev2"))
    return external and (":failure_expand_" in source or ":trust_region_" in source)


def _direct_family(source: str) -> str:
    source = str(source).lower()
    if _is_policy(source):
        return "policy"
    if source == "ddv2" or source.startswith(("ddv2:", "diffusiondrivev2")):
        return "ddv2"
    if source == "driveor" or source.startswith(("driveor:", "drivor")):
        return "driveor"
    return ""


def _source_family(source: str) -> str:
    source = str(source).lower()
    if _is_gt(source):
        return "gt"
    direct = _direct_family(source)
    if direct:
        return direct
    if source.startswith("progress"):
        return "progress"
    if "lateral" in source:
        return "lateral"
    if source.startswith("timing"):
        return "timing"
    return "other"


def _array(record: Mapping[str, Any], key: str, count: int, dtype: Any) -> np.ndarray:
    if key not in record:
        raise KeyError(f"missing required field {key!r}")
    value = np.asarray(record[key], dtype=dtype).reshape(-1)
    if value.shape != (count,):
        raise ValueError(f"{key} has shape {value.shape}, expected {(count,)}")
    return value


def _matrix(
    record: Mapping[str, Any], key: str, shape: tuple[int, ...], dtype: Any
) -> np.ndarray:
    if key not in record:
        raise KeyError(f"missing required field {key!r}")
    value = np.asarray(record[key], dtype=dtype)
    if value.shape != shape:
        raise ValueError(f"{key} has shape {value.shape}, expected {shape}")
    return value


def _candidate_records(record: Mapping[str, Any], candidates: np.ndarray) -> list[CandidateRecord]:
    count = int(candidates.shape[0])
    rewards = _array(record, "rewards", count, np.float32)
    selection = np.asarray(record.get("selection_score", rewards), dtype=np.float32).reshape(-1)
    if selection.shape != (count,):
        raise ValueError(f"selection_score has shape {selection.shape}, expected {(count,)}")
    sources = [str(source) for source in record.get("sources", [])]
    components = record.get("components")
    feasibility = record.get("feasibility", {})
    if not isinstance(components, Mapping) or not isinstance(feasibility, Mapping):
        raise TypeError("components and feasibility must be mappings")
    component_arrays = {
        str(key): np.asarray(value, dtype=np.float32).reshape(-1)
        for key, value in components.items()
    }
    feasibility_arrays = {
        str(key): np.asarray(value, dtype=np.float32).reshape(-1)
        for key, value in feasibility.items()
    }
    for key, value in {**component_arrays, **feasibility_arrays}.items():
        if value.shape != (count,):
            raise ValueError(f"candidate field {key!r} has shape {value.shape}, expected {(count,)}")
    return [
        CandidateRecord(
            trajectory=candidates[index],
            source=sources[index],
            token=str(record.get("token", "")),
            components={key: float(value[index]) for key, value in component_arrays.items()},
            reward=float(rewards[index]),
            feas={key: float(value[index]) for key, value in feasibility_arrays.items()},
            selection_score=float(selection[index]),
        )
        for index in range(count)
    ]


def _scene_normalize(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.bool_)
    output = np.full(values.shape, 0.5, dtype=np.float64)
    finite = mask & np.isfinite(values)
    if not bool(finite.any()):
        return output
    lower = float(values[finite].min())
    upper = float(values[finite].max())
    if upper - lower > 1e-6:
        output = (np.nan_to_num(values, nan=lower, posinf=upper, neginf=lower) - lower) / (
            upper - lower
        )
    return np.clip(output, 0.0, 1.0)


def _pareto_front(values: np.ndarray, valid: np.ndarray, eps: float) -> np.ndarray:
    front = np.zeros((values.shape[0],), dtype=np.bool_)
    indices = np.flatnonzero(valid)
    for index in indices:
        others = indices[indices != index]
        dominates = np.all(values[others] >= values[index] - eps, axis=1) & np.any(
            values[others] > values[index] + eps, axis=1
        )
        front[index] = not bool(np.any(dominates))
    return front


def _pairwise_policy_distance(
    candidates: np.ndarray, policy_indices: np.ndarray
) -> np.ndarray:
    return np.asarray(
        [
            [
                trajectory_snsad_distance(candidates[index], candidates[policy_index])
                for policy_index in policy_indices
            ]
            for index in range(candidates.shape[0])
        ],
        dtype=np.float64,
    )


_FUNNEL_COUNT_KEYS = (
    "non_gt_proposal_count",
    "current_policy_proposal_count",
    "evaluator_valid_count",
    "trajectory_quality_count",
    "trust_region_count",
    "one_neighbor_reachable_count",
    "required_neighbors_reachable_count",
    "hard_teacher_eligible_count",
    "distinct_hypothesis_count",
    "independent_mode_evidence_count",
    "teacher_eligible_count",
    "distinct_from_gt_count",
    "pareto_eligible_count",
    "selected_non_gt_count",
)


def _expected_funnel(
    *,
    sources: list[str],
    selected: np.ndarray,
    gt_index: int,
    valid: np.ndarray,
    quality: np.ndarray,
    trust_region: np.ndarray,
    one_neighbor: np.ndarray,
    required_neighbors: np.ndarray,
    hard_eligible: np.ndarray,
    hypothesis: np.ndarray,
    evidence: np.ndarray,
    teacher: np.ndarray,
    distinct: np.ndarray,
    pareto: np.ndarray,
) -> dict[str, Any]:
    count = len(sources)
    non_gt = np.arange(count, dtype=np.int64) != gt_index
    counts = {
        "non_gt_proposal_count": int(non_gt.sum()),
        "current_policy_proposal_count": int(sum(_is_policy(source) for source in sources)),
        "evaluator_valid_count": int((non_gt & valid).sum()),
        "trajectory_quality_count": int((non_gt & valid & quality).sum()),
        "trust_region_count": int(trust_region.sum()),
        "one_neighbor_reachable_count": int(one_neighbor.sum()),
        "required_neighbors_reachable_count": int(required_neighbors.sum()),
        "hard_teacher_eligible_count": int((hard_eligible & non_gt).sum()),
        "distinct_hypothesis_count": int(hypothesis.sum()),
        "independent_mode_evidence_count": int((evidence & hypothesis).sum()),
        "teacher_eligible_count": int((teacher & non_gt).sum()),
        "distinct_from_gt_count": int((teacher & distinct & non_gt).sum()),
        "pareto_eligible_count": int((pareto & teacher & non_gt).sum()),
        "selected_non_gt_count": int(np.sum(selected != gt_index)),
    }
    if counts["selected_non_gt_count"] > 0:
        supervision_type, reason = "learning_frontier_modes", ""
    elif counts["non_gt_proposal_count"] == 0:
        supervision_type, reason = "gt_only", "no_non_gt_proposals"
    elif counts["evaluator_valid_count"] == 0:
        supervision_type, reason = "gt_only", "no_evaluator_valid_candidate"
    elif counts["trajectory_quality_count"] == 0:
        supervision_type, reason = "gt_only", "no_trajectory_quality_candidate"
    elif counts["trust_region_count"] == 0:
        supervision_type, reason = "gt_only", "no_trust_region_candidate"
    elif counts["required_neighbors_reachable_count"] == 0:
        supervision_type, reason = "gt_only", "no_policy_reachable_candidate"
    elif counts["distinct_hypothesis_count"] == 0:
        supervision_type, reason = "gt_only", "no_distinct_mode_hypothesis"
    elif counts["independent_mode_evidence_count"] == 0:
        supervision_type, reason = "gt_only", "no_independent_mode_evidence"
    elif counts["pareto_eligible_count"] == 0:
        supervision_type, reason = "gt_only", "no_learning_frontier_candidate"
    else:
        supervision_type, reason = "gt_only", "no_pairwise_distinct_frontier_mode"
    return {
        "version": 3,
        **counts,
        "supervision_type": supervision_type,
        "gt_only_reason": reason,
        "capacity_semantics": "observed_under_frozen_generator_not_scene_intrinsic",
    }


def validate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    token = str(record.get("token", ""))
    errors: list[str] = []
    if int(record.get("version", 0)) != 5:
        errors.append(f"version={record.get('version')!r}, expected 5")
    candidates = np.asarray(record.get("candidates"), dtype=np.float32)
    if candidates.ndim != 3 or candidates.shape[-1] != 3 or not np.isfinite(candidates).all():
        return {"token": token, "errors": [f"invalid candidates shape/content: {candidates.shape}"]}
    count = int(candidates.shape[0])
    sources = [str(source) for source in record.get("sources", [])]
    if len(sources) != count:
        return {"token": token, "errors": ["source count does not match candidates"]}
    gt_indices = [index for index, source in enumerate(sources) if _is_gt(source)]
    if len(gt_indices) != 1:
        return {"token": token, "errors": [f"GT candidate count={len(gt_indices)}, expected 1"]}
    gt_index = int(gt_indices[0])
    try:
        candidate_records = _candidate_records(record, candidates)
        selected = np.asarray(record["support_indices"], dtype=np.int64).reshape(-1)
        if selected.size == 0 or selected.min() < 0 or selected.max() >= count:
            raise ValueError("support_indices must be non-empty and in range")
        rewards = _array(record, "rewards", count, np.float64)
        valid = _array(record, "valid_mask", count, np.bool_)
        quality = _array(record, "support_quality_mask", count, np.bool_)
        teacher = _array(record, "teacher_eligible_mask", count, np.bool_)
        pareto = _array(record, "pareto_front_mask", count, np.bool_)
        mode_ids = _array(record, "mode_ids", count, np.int64)
        gt_ade = _array(record, "gt_relative_ade", count, np.float64)
        gt_fde = _array(record, "gt_relative_fde", count, np.float64)
        reachability = _array(record, "policy_reachability_snsad", count, np.float64)
        neighbor_count = _array(record, "policy_neighbor_count", count, np.int64)
        neighbor_fraction = _array(record, "policy_neighbor_fraction", count, np.float64)
        policy_witness = _array(record, "policy_assignment_witness_count", count, np.int64)
        source_witness = _array(record, "independent_source_witness_count", count, np.int64)
        evidence = _array(record, "mode_evidence_mask", count, np.bool_)
        hypothesis = _array(record, "mode_hypothesis_mask", count, np.bool_)
        evidence_score = _array(record, "mode_evidence_score", count, np.float64)
        difficulty = _array(record, "learning_frontier_difficulty", count, np.float64)
        objective_novelty = _array(record, "pareto_objective_novelty", count, np.float64)
        objectives = _matrix(record, "learning_frontier_objectives", (count, 3), np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        return {"token": token, "errors": [str(exc)]}

    metadata = dict(record.get("build_metadata", {}) or {})
    required_metadata = {
        "selection_strategy",
        "teacher_contract",
        "mode_selection_order",
        "learnability_tiebreak",
        "learnability_contract",
        "candidate_capacity_contract",
        "capacity_semantics",
        "no_per_scene_candidate_quota",
        "selection_gate_config",
        "legacy_reward_gate_disabled",
        "support_top_m",
        "max_gt_ade_m",
        "max_gt_fde_m",
        "mode_distance_threshold",
        "max_gt_reward_drop",
        "pareto_eps",
        "pareto_objectives",
        "exclude_derived_external",
        "policy_reachability_required",
        "max_policy_snsad",
        "min_policy_neighbors",
        "mode_evidence_radius",
        "mode_evidence_margin",
        "min_policy_witnesses",
        "min_source_families",
        "raw_internal_candidates",
        "expand_external_candidates",
        "policy_samples_per_scene",
        "policy_checkpoint_sha256",
        "fs_norm_stats_sha256",
        "build_log_split",
        "dataset_scene_count",
        "scene_seed_scheme",
        "external_candidate_roots",
        "external_candidate_root_fingerprints",
        "external_candidate_root_file_counts",
    }
    missing = sorted(required_metadata.difference(metadata))
    if missing:
        errors.append(f"missing build metadata: {missing}")
    expected_contract = {
        "selection_strategy": "mode_learning_frontier_v5",
        "teacher_contract": "gt_anchor_plus_independently_evidenced_learning_frontier_modes",
        "mode_selection_order": (
            "evidence_gate_then_performance_diversity_learnability_pareto_then_snsad_fps"
        ),
        "learnability_tiebreak": "independent_evidence_then_frontier_difficulty",
        "learnability_contract": "closer_than_gt_independent_witness_v1",
        "candidate_capacity_contract": "observed_independent_evidence_no_quota_v3",
        "capacity_semantics": "observed_selected_support_not_scene_intrinsic",
        "pareto_objectives": (
            "scene_relative_performance,gt_relative_snsad_diversity,independent_learnability"
        ),
    }
    for key, expected in expected_contract.items():
        if metadata.get(key) != expected:
            errors.append(f"{key}={metadata.get(key)!r}, expected {expected!r}")
    for key in (
        "no_per_scene_candidate_quota",
        "legacy_reward_gate_disabled",
        "exclude_derived_external",
        "policy_reachability_required",
        "raw_internal_candidates",
    ):
        if not bool(metadata.get(key, False)):
            errors.append(f"{key}=false")
    if bool(metadata.get("expand_external_candidates", True)):
        errors.append("expand_external_candidates=true")
    gate_cfg = metadata.get("selection_gate_config")
    if not isinstance(gate_cfg, Mapping):
        errors.append("selection_gate_config must be a mapping")
        gate_cfg = {}
    roots = dict(metadata.get("external_candidate_roots", {}) or {})
    root_hashes = dict(metadata.get("external_candidate_root_fingerprints", {}) or {})
    root_counts = dict(metadata.get("external_candidate_root_file_counts", {}) or {})
    if set(roots) != set(root_hashes) or set(roots) != set(root_counts):
        errors.append("external candidate provenance source keys disagree")

    expected_valid = np.asarray(
        [
            is_valid_candidate(candidate.components, candidate.feas, candidate_records[gt_index].components, gate_cfg)
            for candidate in candidate_records
        ],
        dtype=np.bool_,
    )
    expected_quality = np.asarray(
        [
            support_trajectory_quality_pass(candidate, candidates[gt_index], gate_cfg)
            for candidate in candidate_records
        ],
        dtype=np.bool_,
    )
    if not np.array_equal(valid, expected_valid):
        errors.append("valid_mask disagrees with evaluator gates")
    if not np.array_equal(quality, expected_quality):
        errors.append("support_quality_mask disagrees with trajectory gates")

    expected_gt_ade = np.linalg.norm(
        candidates[..., :2] - candidates[gt_index, None, ..., :2], axis=-1
    ).mean(axis=-1)
    expected_gt_fde = np.linalg.norm(
        candidates[:, -1, :2] - candidates[gt_index, -1, :2], axis=-1
    )
    expected_gt_distance = np.asarray(
        [trajectory_snsad_distance(candidate, candidates[gt_index]) for candidate in candidates],
        dtype=np.float64,
    )
    if not np.allclose(gt_ade, expected_gt_ade, atol=1e-6, rtol=1e-5):
        errors.append("gt_relative_ade disagrees with trajectories")
    if not np.allclose(gt_fde, expected_gt_fde, atol=1e-6, rtol=1e-5):
        errors.append("gt_relative_fde disagrees with trajectories")

    policy_indices = np.asarray(
        [index for index, source in enumerate(sources) if _is_policy(source)], dtype=np.int64
    )
    if policy_indices.size == 0:
        return {"token": token, "errors": [*errors, "no current-policy proposals"]}
    policy_distance = _pairwise_policy_distance(candidates, policy_indices)
    expected_reachability = policy_distance.min(axis=1)
    expected_neighbor_count = np.sum(
        policy_distance <= float(metadata.get("max_policy_snsad", 0.0)) + 1e-6, axis=1
    ).astype(np.int64)
    expected_neighbor_fraction = expected_neighbor_count / float(policy_indices.size)
    if not np.allclose(reachability, expected_reachability, atol=1e-6, rtol=1e-5):
        errors.append("policy_reachability_snsad disagrees with trajectories")
    if not np.array_equal(neighbor_count, expected_neighbor_count):
        errors.append("policy_neighbor_count disagrees with trajectories")
    if not np.allclose(neighbor_fraction, expected_neighbor_fraction, atol=1e-6, rtol=1e-5):
        errors.append("policy_neighbor_fraction disagrees with trajectories")

    non_gt = np.arange(count, dtype=np.int64) != gt_index
    trust_region = (
        non_gt
        & expected_valid
        & expected_quality
        & (expected_gt_ade <= float(metadata.get("max_gt_ade_m", math.inf)) + 1e-6)
        & (expected_gt_fde <= float(metadata.get("max_gt_fde_m", math.inf)) + 1e-6)
        & (
            rewards
            >= rewards[gt_index] - float(metadata.get("max_gt_reward_drop", math.inf)) - 1e-6
        )
    )
    if bool(metadata.get("exclude_derived_external", False)):
        trust_region &= np.asarray(
            [not _is_derived_external(source) for source in sources], dtype=np.bool_
        )
    within_policy = expected_reachability <= float(metadata.get("max_policy_snsad", 0.0)) + 1e-6
    one_neighbor = trust_region & within_policy & (expected_neighbor_count >= 1)
    required_neighbors = trust_region & within_policy & (
        expected_neighbor_count >= int(metadata.get("min_policy_neighbors", 0))
    )
    hard_eligible = required_neighbors.copy()
    hard_eligible[gt_index] = True
    distinct = expected_gt_distance + 1e-6 >= float(metadata.get("mode_distance_threshold", 0.0))
    expected_hypothesis = hard_eligible & non_gt & distinct

    expected_policy_witness = np.zeros((count,), dtype=np.int64)
    expected_source_witness = np.zeros((count,), dtype=np.int64)
    expected_evidence_score = np.zeros((count,), dtype=np.float64)
    expected_evidence = np.zeros((count,), dtype=np.bool_)
    policy_gt_distance = policy_distance[gt_index]
    eligible_policy = hard_eligible[policy_indices]
    direct_indices = {
        family: np.asarray(
            [
                index
                for index, source in enumerate(sources)
                if _direct_family(source) == family and hard_eligible[index]
            ],
            dtype=np.int64,
        )
        for family in ("policy", "ddv2", "driveor")
    }
    evidence_radius = float(metadata.get("mode_evidence_radius", 0.0))
    evidence_margin = float(metadata.get("mode_evidence_margin", 0.0))
    for index in np.flatnonzero(expected_hypothesis):
        advantage = policy_gt_distance - policy_distance[index]
        witness = (
            eligible_policy
            & (policy_distance[index] <= evidence_radius + 1e-6)
            & (advantage + 1e-6 >= evidence_margin)
        )
        expected_policy_witness[index] = int(witness.sum())
        for family, family_indices in direct_indices.items():
            del family
            if not family_indices.size:
                continue
            family_distance = np.asarray(
                [
                    trajectory_snsad_distance(candidates[index], candidates[witness_index])
                    for witness_index in family_indices
                ],
                dtype=np.float64,
            )
            family_advantage = expected_gt_distance[family_indices] - family_distance
            expected_source_witness[index] += int(
                np.any(
                    (family_distance <= evidence_radius + 1e-6)
                    & (family_advantage + 1e-6 >= evidence_margin)
                )
            )
        expected_evidence[index] = bool(
            expected_policy_witness[index] >= int(metadata.get("min_policy_witnesses", 0))
            or expected_source_witness[index] >= int(metadata.get("min_source_families", 0))
        )
        expected_evidence_score[index] = (
            0.50 * min(expected_policy_witness[index] / float(policy_indices.size), 1.0)
            + 0.25 * min(expected_source_witness[index] / 3.0, 1.0)
            + 0.25 * (1.0 - min(expected_reachability[index] / evidence_radius, 1.0))
        )
    if not np.array_equal(policy_witness, expected_policy_witness):
        errors.append("policy_assignment_witness_count disagrees with trajectories")
    if not np.array_equal(source_witness, expected_source_witness):
        errors.append("independent_source_witness_count disagrees with trajectories")
    if not np.array_equal(evidence, expected_evidence):
        errors.append("mode_evidence_mask disagrees with independent witnesses")
    if not np.array_equal(hypothesis, expected_hypothesis):
        errors.append("mode_hypothesis_mask disagrees with hard gates")
    if not np.allclose(evidence_score, expected_evidence_score, atol=1e-6, rtol=1e-5):
        errors.append("mode_evidence_score disagrees with witnesses")

    expected_teacher = hard_eligible & (expected_evidence | (~non_gt))
    domain = expected_teacher & (distinct | (~non_gt))
    expected_objectives = np.stack(
        (
            _scene_normalize(rewards, domain),
            _scene_normalize(expected_gt_distance, domain),
            expected_evidence_score.copy(),
        ),
        axis=-1,
    )
    expected_objectives[gt_index, 2] = 1.0
    expected_pareto = _pareto_front(
        expected_objectives, domain, float(metadata.get("pareto_eps", 0.0))
    )
    if not np.array_equal(teacher, expected_teacher):
        errors.append("teacher_eligible_mask disagrees with mode evidence")
    if not np.allclose(objectives, expected_objectives, atol=1e-6, rtol=1e-5):
        errors.append("learning_frontier_objectives disagree with candidates")
    if not np.array_equal(pareto, expected_pareto):
        errors.append("pareto_front_mask disagrees with learning-frontier objectives")

    expected_difficulty = np.clip(
        0.40 * np.minimum(expected_reachability / float(metadata.get("max_policy_snsad", 1.0)), 1.0)
        + 0.20 * np.minimum(expected_gt_ade / float(metadata.get("max_gt_ade_m", 1.0)), 1.0)
        + 0.20 * np.minimum(expected_gt_fde / float(metadata.get("max_gt_fde_m", 1.0)), 1.0)
        + 0.20 * (1.0 - expected_evidence_score),
        0.0,
        1.0,
    )
    expected_difficulty[gt_index] = 0.0
    if not np.allclose(difficulty, expected_difficulty, atol=1e-6, rtol=1e-5):
        errors.append("learning_frontier_difficulty disagrees with contract")

    top_m = int(metadata.get("support_top_m", 0))
    if selected.size > top_m:
        errors.append(f"selected count {selected.size} exceeds support_top_m={top_m}")
    if gt_index not in selected.tolist():
        errors.append("GT anchor is not selected")
    selected_non_gt = selected[selected != gt_index]
    if selected_non_gt.size and not bool(np.all(expected_teacher[selected_non_gt])):
        errors.append("selected non-GT candidate is not teacher eligible")
    if selected_non_gt.size and not bool(np.all(expected_evidence[selected_non_gt])):
        errors.append("selected non-GT candidate lacks independent mode evidence")
    if selected_non_gt.size and not bool(np.all(expected_pareto[selected_non_gt])):
        errors.append("selected non-GT candidate is outside the learning-frontier Pareto front")
    selected_mode_ids = mode_ids[selected]
    if set(selected_mode_ids.tolist()) != set(range(selected.size)):
        errors.append(f"selected mode IDs are not contiguous: {selected_mode_ids.tolist()}")
    if selected_non_gt.size and (
        not np.isfinite(objective_novelty[selected_non_gt]).all()
        or bool(np.any(objective_novelty[selected_non_gt] < 0.0))
    ):
        errors.append("selected objective novelty is invalid")
    min_pairwise = math.inf
    threshold = float(metadata.get("mode_distance_threshold", 0.0))
    for left_position, left in enumerate(selected.tolist()):
        for right in selected[left_position + 1 :].tolist():
            distance = trajectory_snsad_distance(candidates[left], candidates[right])
            min_pairwise = min(min_pairwise, distance)
            if distance + 1e-6 < threshold:
                errors.append(
                    f"selected modes {left}/{right} have SNSAD={distance:.6f} below {threshold:.6f}"
                )
    if not math.isfinite(min_pairwise):
        min_pairwise = 0.0

    expected_funnel = _expected_funnel(
        sources=sources,
        selected=selected,
        gt_index=gt_index,
        valid=expected_valid,
        quality=expected_quality,
        trust_region=trust_region,
        one_neighbor=one_neighbor,
        required_neighbors=required_neighbors,
        hard_eligible=hard_eligible,
        hypothesis=expected_hypothesis,
        evidence=expected_evidence,
        teacher=expected_teacher,
        distinct=distinct,
        pareto=expected_pareto,
    )
    funnel = dict(record.get("candidate_funnel", {}) or {})
    for key, expected in expected_funnel.items():
        if funnel.get(key) != expected:
            errors.append(f"candidate_funnel {key}={funnel.get(key)!r}, expected {expected!r}")

    selected_families = Counter(_source_family(sources[index]) for index in selected)
    return {
        "token": token,
        "errors": errors,
        "candidate_count": count,
        "selected_count": int(selected.size),
        "non_gt_count": int(selected_non_gt.size),
        "multimode": int(selected_non_gt.size > 0),
        "hypothesis_count": int(expected_hypothesis.sum()),
        "evidence_count": int(expected_evidence.sum()),
        "hypothesis_without_evidence": int(expected_hypothesis.any() and not expected_evidence.any()),
        "raw_internal_candidates": int(bool(metadata.get("raw_internal_candidates", False))),
        "external_expansion_disabled": int(not bool(metadata.get("expand_external_candidates", True))),
        "policy_reachability_required": int(bool(metadata.get("policy_reachability_required", False))),
        "selected_families": selected_families,
        "selected_reward_delta": (rewards[selected_non_gt] - rewards[gt_index]).tolist(),
        "selected_reachability": expected_reachability[selected_non_gt].tolist(),
        "selected_difficulty": expected_difficulty[selected_non_gt].tolist(),
        "selected_policy_witness": expected_policy_witness[selected_non_gt].tolist(),
        "selected_source_witness": expected_source_witness[selected_non_gt].tolist(),
        "selected_evidence_score": expected_evidence_score[selected_non_gt].tolist(),
        "min_pairwise_snsad": float(min_pairwise),
        "mode_distance_threshold": float(threshold),
        "candidate_funnel": expected_funnel,
    }


def _validate_path(path: Path) -> dict[str, Any]:
    try:
        return validate_record(_load(path))
    except Exception as exc:  # pragma: no cover - corrupt archive path
        return {"token": path.stem, "errors": [f"{path}: {exc}"]}


def validate_archive(
    root: Path,
    *,
    max_records: int = 0,
    workers: int = 1,
    require_raw_build: bool = True,
    expected_token_source: Optional[Path] = None,
    output_scene_index: Optional[Path] = None,
) -> dict[str, Any]:
    paths = _iter_paths(root)
    if max_records > 0:
        paths = paths[:max_records]
    if workers > 1:
        with mp.Pool(workers) as pool:
            rows = list(pool.imap_unordered(_validate_path, paths, chunksize=64))
    else:
        rows = [_validate_path(path) for path in paths]
    totals: Counter[str] = Counter()
    values: dict[str, list[float]] = defaultdict(list)
    selected_families: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    error_examples: list[dict[str, Any]] = []
    actual_token_counts: Counter[str] = Counter()
    scene_index_records: list[dict[str, Any]] = []
    for row in rows:
        totals["records"] += 1
        actual_token_counts[str(row.get("token", ""))] += 1
        if row.get("errors"):
            totals["contract_errors"] += 1
            if len(error_examples) < 20:
                error_examples.append({"token": row.get("token", ""), "errors": row["errors"]})
            continue
        for key in (
            "candidate_count",
            "selected_count",
            "non_gt_count",
            "multimode",
            "hypothesis_count",
            "evidence_count",
            "hypothesis_without_evidence",
            "raw_internal_candidates",
            "external_expansion_disabled",
            "policy_reachability_required",
        ):
            totals[key] += int(row[key])
        selected_families.update(row["selected_families"])
        values["selected_reward_delta"].extend(row["selected_reward_delta"])
        values["selected_reachability"].extend(row["selected_reachability"])
        values["selected_difficulty"].extend(row["selected_difficulty"])
        values["selected_policy_witness"].extend(row["selected_policy_witness"])
        values["selected_source_witness"].extend(row["selected_source_witness"])
        values["min_pairwise_snsad"].append(float(row["min_pairwise_snsad"]))
        selected_evidence = np.asarray(row["selected_evidence_score"], dtype=np.float64)
        selected_difficulty = np.asarray(row["selected_difficulty"], dtype=np.float64)
        selected_reward_delta = np.asarray(row["selected_reward_delta"], dtype=np.float64)
        frontier_eligible = int(row["non_gt_count"]) > 0
        scene_index_records.append(
            {
                "token": str(row["token"]),
                "frontier_eligible": bool(frontier_eligible),
                "selected_non_gt_count": int(row["non_gt_count"]),
                "priority": (
                    float(max(selected_evidence.mean(), 1e-3))
                    if frontier_eligible
                    else 0.0
                ),
                "selected_evidence_score_mean": (
                    float(selected_evidence.mean()) if selected_evidence.size else 0.0
                ),
                "selected_difficulty_mean": (
                    float(selected_difficulty.mean()) if selected_difficulty.size else 0.0
                ),
                "selected_reward_delta_mean": (
                    float(selected_reward_delta.mean()) if selected_reward_delta.size else 0.0
                ),
                "selected_min_pairwise_snsad": float(row["min_pairwise_snsad"]),
                "mode_distance_threshold": float(row["mode_distance_threshold"]),
            }
        )
        funnel = dict(row["candidate_funnel"])
        if funnel["gt_only_reason"]:
            reasons[str(funnel["gt_only_reason"])] += 1

    valid_records = max(totals["records"] - totals["contract_errors"], 1)
    expected_list = (
        _load_expected_tokens(expected_token_source, workers=workers)
        if expected_token_source
        else None
    )
    expected_counts = Counter(expected_list or [])
    actual_tokens = set(actual_token_counts)
    expected_tokens = set(expected_counts) if expected_list is not None else None
    duplicate_tokens = sorted(token for token, count in actual_token_counts.items() if count > 1)
    expected_duplicates = sorted(token for token, count in expected_counts.items() if count > 1)
    missing_tokens = sorted(expected_tokens - actual_tokens) if expected_tokens is not None else []
    extra_tokens = sorted(actual_tokens - expected_tokens) if expected_tokens is not None else []
    coverage_pass = not duplicate_tokens and not expected_duplicates and (
        expected_tokens is None or (not missing_tokens and not extra_tokens)
    )
    hard_contract_pass = totals["records"] > 0 and totals["contract_errors"] == 0
    raw_ratio = totals["raw_internal_candidates"] / valid_records
    expansion_disabled_ratio = totals["external_expansion_disabled"] / valid_records
    reachability_ratio = totals["policy_reachability_required"] / valid_records
    static_promotion_pass = (
        hard_contract_pass
        and coverage_pass
        and (not require_raw_build or raw_ratio == 1.0)
        and (not require_raw_build or expansion_disabled_ratio == 1.0)
        and reachability_ratio == 1.0
    )
    report = {
        "archive": str(root),
        "record_count": totals["records"],
        "contract_error_count": totals["contract_errors"],
        "error_examples": error_examples,
        "hard_contract_pass": hard_contract_pass,
        "token_coverage_pass": coverage_pass,
        "static_promotion_pass": static_promotion_pass,
        "diversity_statistics_are_promotion_gates": False,
        "expected_token_count": len(expected_tokens) if expected_tokens is not None else None,
        "actual_unique_token_count": len(actual_tokens),
        "missing_token_count": len(missing_tokens),
        "extra_token_count": len(extra_tokens),
        "duplicate_token_count": len(duplicate_tokens),
        "candidate_count_per_scene": totals["candidate_count"] / valid_records,
        "selected_count_per_scene": totals["selected_count"] / valid_records,
        "non_gt_count_per_scene": totals["non_gt_count"] / valid_records,
        "multimode_scene_ratio": totals["multimode"] / valid_records,
        "mode_hypothesis_count_per_scene": totals["hypothesis_count"] / valid_records,
        "independent_evidence_count_per_scene": totals["evidence_count"] / valid_records,
        "hypothesis_without_evidence_scene_ratio": totals["hypothesis_without_evidence"]
        / valid_records,
        "raw_internal_candidate_build_ratio": raw_ratio,
        "external_expansion_disabled_ratio": expansion_disabled_ratio,
        "policy_reachability_required_ratio": reachability_ratio,
        "selected_source_family_counts": dict(selected_families),
        "gt_only_reason_counts": dict(reasons),
        "selected_reward_delta": _stats(values["selected_reward_delta"]),
        "selected_policy_reachability_snsad": _stats(values["selected_reachability"]),
        "selected_learning_frontier_difficulty": _stats(values["selected_difficulty"]),
        "selected_policy_assignment_witness_count": _stats(values["selected_policy_witness"]),
        "selected_independent_source_witness_count": _stats(values["selected_source_witness"]),
        "selected_min_pairwise_snsad": _stats(values["min_pairwise_snsad"]),
    }
    if output_scene_index is not None:
        if totals["contract_errors"]:
            raise ValueError("Cannot write a Stage2 frontier scene index from invalid archive records.")
        scene_index_records.sort(key=lambda record: record["token"])
        thresholds = sorted({record["mode_distance_threshold"] for record in scene_index_records})
        if len(thresholds) != 1:
            raise ValueError(f"Stage2 frontier scene index requires one mode threshold, got {thresholds}.")
        encoded_records = json.dumps(
            scene_index_records,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        index_payload = {
            "version": 1,
            "archive_version": 5,
            "archive_path": str(root),
            "archive_fingerprint": hashlib.sha256(encoded_records).hexdigest(),
            "capacity_semantics": "observed_selected_support_not_scene_intrinsic",
            "no_per_scene_candidate_quota": True,
            "priority_semantics": "independent_evidence_confidence_not_candidate_count",
            "mode_distance_threshold": thresholds[0],
            "record_count": len(scene_index_records),
            "frontier_eligible_count": sum(
                int(record["frontier_eligible"]) for record in scene_index_records
            ),
            "records": scene_index_records,
        }
        output_scene_index.parent.mkdir(parents=True, exist_ok=True)
        output_scene_index.write_text(
            json.dumps(index_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report["scene_index_path"] = str(output_scene_index)
        report["scene_index_fingerprint"] = index_payload["archive_fingerprint"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the no-quota SG-FPS v5 contract.")
    parser.add_argument("--archive-path", required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--require-raw-build", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected-token-source", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-scene-index", default="")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    report = validate_archive(
        Path(args.archive_path),
        max_records=int(args.max_records),
        workers=max(int(args.workers), 1),
        require_raw_build=bool(args.require_raw_build),
        expected_token_source=Path(args.expected_token_source) if args.expected_token_source else None,
        output_scene_index=Path(args.output_scene_index) if args.output_scene_index else None,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    if not args.report_only and not bool(report["static_promotion_pass"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
