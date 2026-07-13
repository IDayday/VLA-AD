#!/usr/bin/env python
from __future__ import annotations

import argparse
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

from navsim.agents.recogdrive.pareto_support import (
    CandidateRecord,
    is_valid_candidate,
    trajectory_snsad_distance,
)
from navsim.agents.recogdrive.pareto_support.pareto_archive import (
    support_trajectory_quality_pass,
)


def _iter_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.glob("*.pkl.xz"))


def _load_expected_token(path: Path) -> str:
    token = str(_load(path).get("token", "")).strip()
    if not token:
        raise ValueError(f"Expected-token archive record has no token: {path}")
    return token


def _load_expected_tokens(source: Path, workers: int = 1) -> list[str]:
    if source.is_dir():
        paths = _iter_paths(source)
        if workers > 1:
            with mp.Pool(workers) as pool:
                return list(pool.imap_unordered(_load_expected_token, paths, chunksize=64))
        return [_load_expected_token(path) for path in paths]
    if not source.is_file():
        raise FileNotFoundError(source)
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, list):
        return [str(token).strip() for token in payload if str(token).strip()]
    if isinstance(payload, Mapping) and isinstance(payload.get("tokens"), list):
        return [str(token).strip() for token in payload["tokens"] if str(token).strip()]
    return [
        line.strip().split()[0]
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _load(path: Path) -> dict[str, Any]:
    with lzma.open(path, "rb") as file:
        record = pickle.load(file)
    if not isinstance(record, dict):
        raise TypeError(f"record must be a dict, got {type(record).__name__}.")
    return record


def _is_gt(source: str) -> bool:
    source = str(source).lower()
    return source == "gt" or source.startswith("gt:")


def _is_derived_external(source: str) -> bool:
    source = str(source).lower()
    external = source.startswith(("ddv2", "driveor", "drivor", "diffusiondrivev2"))
    return external and (":failure_expand_" in source or ":trust_region_" in source)


def _is_current_policy(source: str) -> bool:
    source = str(source).lower()
    return source == "policy" or source.startswith("policy:") or source.startswith("current_policy")


def _source_family(source: str) -> str:
    source = str(source).lower()
    if _is_gt(source):
        return "gt"
    if source.startswith(("ddv2", "diffusiondrivev2")):
        return "ddv2"
    if source.startswith(("driveor", "drivor")):
        return "driveor"
    if source.startswith("progress"):
        return "progress"
    if "lateral" in source:
        return "lateral"
    if source.startswith("timing"):
        return "timing"
    return "other"


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _array(record: Mapping[str, Any], key: str, count: int, dtype: Any) -> np.ndarray:
    if key not in record:
        raise KeyError(f"missing required field {key!r}")
    value = np.asarray(record[key], dtype=dtype).reshape(-1)
    if value.shape != (count,):
        raise ValueError(f"{key} has shape {value.shape}, expected {(count,)}")
    return value


def _component_array(
    record: Mapping[str, Any],
    count: int,
    *aliases: str,
) -> np.ndarray:
    components = record.get("components")
    if not isinstance(components, Mapping):
        raise KeyError("missing required field 'components'")
    for key in aliases:
        if key in components:
            value = np.asarray(components[key], dtype=np.float64).reshape(-1)
            if value.shape != (count,):
                raise ValueError(
                    f"components[{key!r}] has shape {value.shape}, expected {(count,)}"
                )
            return value
    raise KeyError(f"missing required metric component; expected one of {aliases!r}")


def _pareto_front_mask(values: np.ndarray, valid: np.ndarray, eps: float) -> np.ndarray:
    front = np.zeros((values.shape[0],), dtype=np.bool_)
    valid_indices = np.flatnonzero(valid)
    for index in valid_indices:
        others = valid_indices[valid_indices != index]
        if others.size == 0:
            front[index] = True
            continue
        dominates = np.all(values[others] >= values[index] - eps, axis=1) & np.any(
            values[others] > values[index] + eps,
            axis=1,
        )
        front[index] = not bool(np.any(dominates))
    return front


def _greedy_mode_capacity(
    candidates: np.ndarray,
    candidate_indices: np.ndarray,
    gt_index: int,
    mode_threshold: float,
) -> int:
    remaining = set(int(index) for index in candidate_indices.tolist())
    selected = [int(gt_index)]
    capacity = 0
    while remaining:
        distances = [
            (
                min(
                    trajectory_snsad_distance(candidates[index], candidates[chosen])
                    for chosen in selected
                ),
                index,
            )
            for index in remaining
        ]
        distance, index = max(distances, key=lambda item: (item[0], -item[1]))
        if distance + 1e-6 < mode_threshold:
            break
        selected.append(index)
        remaining.remove(index)
        capacity += 1
    return capacity


def _policy_reachability_stats(
    candidates: np.ndarray,
    policy_mask: np.ndarray,
    max_policy_snsad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = int(candidates.shape[0])
    policy = candidates[np.asarray(policy_mask, dtype=np.bool_)]
    if policy.shape[0] == 0:
        return (
            np.full((count,), np.inf, dtype=np.float64),
            np.zeros((count,), dtype=np.int64),
            np.zeros((count,), dtype=np.float64),
        )
    delta_xy = candidates[:, None, :, :2].astype(np.float64) - policy[None, :, :, :2]
    xy_distance = np.linalg.norm(delta_xy, axis=-1)
    heading_delta = (
        candidates[:, None, :, 2].astype(np.float64)
        - policy[None, :, :, 2]
        + np.pi
    ) % (2.0 * np.pi) - np.pi
    snsad = (
        0.30 * xy_distance[..., -1] / 3.0
        + 0.30 * xy_distance.mean(axis=-1) / 1.5
        + 0.15 * np.abs(delta_xy[..., 0]).mean(axis=-1) / 1.5
        + 0.15 * np.abs(delta_xy[..., 1]).mean(axis=-1) / 0.8
        + 0.10 * np.abs(heading_delta).mean(axis=-1) / 0.35
    )
    neighbor_count = np.sum(snsad <= float(max_policy_snsad) + 1e-6, axis=1).astype(np.int64)
    return (
        snsad.min(axis=1),
        neighbor_count,
        neighbor_count.astype(np.float64) / float(policy.shape[0]),
    )


_CANDIDATE_FUNNEL_COUNT_KEYS = (
    "non_gt_proposal_count",
    "current_policy_proposal_count",
    "evaluator_valid_count",
    "trajectory_quality_count",
    "trust_region_count",
    "one_neighbor_reachable_count",
    "required_neighbors_reachable_count",
    "teacher_eligible_count",
    "distinct_from_gt_count",
    "pareto_eligible_count",
    "selected_non_gt_count",
)


def _recompute_candidate_funnel(
    *,
    sources: list[str],
    selected: np.ndarray,
    gt_index: int,
    valid: np.ndarray,
    quality: np.ndarray,
    teacher_eligible: np.ndarray,
    pareto: np.ndarray,
    gt_ade: np.ndarray,
    gt_fde: np.ndarray,
    rewards: np.ndarray,
    policy_reachability: np.ndarray,
    policy_neighbor_count: np.ndarray,
    gt_mode_distance: np.ndarray,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    count = len(sources)
    non_gt = np.arange(count, dtype=np.int64) != int(gt_index)
    policy = np.asarray([_is_current_policy(source) for source in sources], dtype=np.bool_)
    evaluator_valid = non_gt & valid
    trajectory_quality = evaluator_valid & quality
    trust_region = (
        trajectory_quality
        & (gt_ade <= float(metadata.get("max_gt_ade_m", math.inf)) + 1e-6)
        & (gt_fde <= float(metadata.get("max_gt_fde_m", math.inf)) + 1e-6)
        & (
            rewards
            >= rewards[int(gt_index)]
            - float(metadata.get("max_gt_reward_drop", math.inf))
            - 1e-6
        )
    )
    if bool(metadata.get("exclude_derived_external", False)):
        trust_region &= np.asarray(
            [not _is_derived_external(source) for source in sources], dtype=np.bool_
        )
    within_policy_radius = np.isfinite(policy_reachability) & (
        policy_reachability <= float(metadata.get("max_policy_snsad", math.inf)) + 1e-6
    )
    one_neighbor = trust_region & within_policy_radius & (policy_neighbor_count >= 1)
    required_neighbors = trust_region & within_policy_radius & (
        policy_neighbor_count >= int(metadata.get("min_policy_neighbors", 0))
    )
    teacher = non_gt & teacher_eligible
    distinct = teacher & (
        gt_mode_distance + 1e-6 >= float(metadata.get("mode_distance_threshold", 0.0))
    )
    pareto_eligible = teacher & pareto
    selected_non_gt_count = int(np.sum(selected != int(gt_index)))
    counts = {
        "non_gt_proposal_count": int(non_gt.sum()),
        "current_policy_proposal_count": int(policy.sum()),
        "evaluator_valid_count": int(evaluator_valid.sum()),
        "trajectory_quality_count": int(trajectory_quality.sum()),
        "trust_region_count": int(trust_region.sum()),
        "one_neighbor_reachable_count": int(one_neighbor.sum()),
        "required_neighbors_reachable_count": int(required_neighbors.sum()),
        "teacher_eligible_count": int(teacher.sum()),
        "distinct_from_gt_count": int(distinct.sum()),
        "pareto_eligible_count": int(pareto_eligible.sum()),
        "selected_non_gt_count": selected_non_gt_count,
    }
    if selected_non_gt_count > 0:
        supervision_type = "frontier_modes"
        gt_only_reason = ""
    elif counts["non_gt_proposal_count"] == 0:
        supervision_type = "gt_only"
        gt_only_reason = "no_non_gt_proposals"
    elif counts["evaluator_valid_count"] == 0:
        supervision_type = "gt_only"
        gt_only_reason = "no_evaluator_valid_candidate"
    elif counts["trajectory_quality_count"] == 0:
        supervision_type = "gt_only"
        gt_only_reason = "no_trajectory_quality_candidate"
    elif counts["trust_region_count"] == 0:
        supervision_type = "gt_only"
        gt_only_reason = "no_trust_region_candidate"
    elif counts["required_neighbors_reachable_count"] == 0:
        supervision_type = "gt_only"
        gt_only_reason = "no_policy_reachable_candidate"
    elif counts["distinct_from_gt_count"] == 0:
        supervision_type = "gt_only"
        gt_only_reason = "no_distinct_mode"
    elif counts["pareto_eligible_count"] == 0:
        supervision_type = "gt_only"
        gt_only_reason = "no_pareto_candidate"
    else:
        supervision_type = "gt_only"
        gt_only_reason = "no_distinct_mode"
    return {
        "version": 2,
        **counts,
        "supervision_type": supervision_type,
        "gt_only_reason": gt_only_reason,
    }


def validate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    token = str(record.get("token", ""))
    version = int(record.get("version", 0))
    if version < 4:
        errors.append(f"version={version}, expected >=4")

    candidates = np.asarray(record.get("candidates"), dtype=np.float32)
    if candidates.ndim != 3 or candidates.shape[-1] != 3 or not np.isfinite(candidates).all():
        return {"token": token, "errors": [f"invalid candidates shape/content: {candidates.shape}"]}
    count = int(candidates.shape[0])
    sources = [str(source) for source in record.get("sources", [])]
    if len(sources) != count:
        return {"token": token, "errors": [f"source count={len(sources)}, candidate count={count}"]}
    gt_indices = [index for index, source in enumerate(sources) if _is_gt(source)]
    if len(gt_indices) != 1:
        return {"token": token, "errors": [f"GT candidate count={len(gt_indices)}, expected 1"]}
    gt_index = gt_indices[0]

    try:
        selected = np.asarray(record["support_indices"], dtype=np.int64).reshape(-1)
        if selected.size == 0 or selected.min() < 0 or selected.max() >= count:
            raise ValueError("support_indices must be non-empty and in range")
        mode_ids = _array(record, "mode_ids", count, np.int64)
        teacher_eligible = _array(record, "teacher_eligible_mask", count, np.bool_)
        source_conditioned = _array(record, "source_conditioned_mask", count, np.bool_)
        gt_ade = _array(record, "gt_relative_ade", count, np.float64)
        gt_fde = _array(record, "gt_relative_fde", count, np.float64)
        policy_reachability = _array(record, "policy_reachability_snsad", count, np.float64)
        policy_neighbor_count = _array(record, "policy_neighbor_count", count, np.int64)
        policy_neighbor_fraction = _array(record, "policy_neighbor_fraction", count, np.float64)
        frontier_difficulty = _array(record, "learning_frontier_difficulty", count, np.float64)
        objective_novelty = _array(record, "pareto_objective_novelty", count, np.float64)
        pareto = _array(record, "pareto_front_mask", count, np.bool_)
        quality = _array(record, "support_quality_mask", count, np.bool_)
        valid = _array(record, "valid_mask", count, np.bool_)
        rewards = _array(record, "rewards", count, np.float64)
        pareto_ep = _component_array(record, count, "ego_progress", "ep")
        pareto_ttc = _component_array(
            record,
            count,
            "time_to_collision_within_bound",
            "ttc",
        )
        pareto_quality = _component_array(record, count, "history_comfort", "comfort")
    except (KeyError, ValueError) as exc:
        return {"token": token, "errors": [str(exc)]}

    metadata = dict(record.get("build_metadata", {}) or {})
    required_metadata = {
        "selection_strategy",
        "teacher_contract",
        "mode_selection_order",
        "learnability_tiebreak",
        "learnability_contract",
        "candidate_capacity_contract",
        "selection_gate_config",
        "legacy_reward_gate_disabled",
        "support_top_m",
        "max_gt_ade_m",
        "max_gt_fde_m",
        "mode_distance_threshold",
        "max_gt_reward_drop",
        "pareto_eps",
        "raw_internal_candidates",
        "expand_external_candidates",
        "policy_reachability_required",
        "max_policy_snsad",
        "min_policy_neighbors",
        "policy_samples_per_scene",
        "policy_checkpoint_sha256",
        "fs_norm_stats_sha256",
        "build_log_split",
        "dataset_scene_count",
        "exclude_derived_external",
        "scene_seed_scheme",
        "external_candidate_roots",
        "external_candidate_root_fingerprints",
        "external_candidate_root_file_counts",
    }
    missing_metadata = sorted(required_metadata.difference(metadata))
    if missing_metadata:
        errors.append(f"missing build metadata: {missing_metadata}")
    if metadata.get("selection_strategy") != "mode_pareto_v4":
        errors.append(f"selection_strategy={metadata.get('selection_strategy')!r}")
    if metadata.get("teacher_contract") != "gt_anchor_plus_uniform_reachable_pareto_modes":
        errors.append(f"teacher_contract={metadata.get('teacher_contract')!r}")
    if metadata.get("mode_selection_order") != "scene_normalized_objective_fps_then_snsad_then_policy_density":
        errors.append(f"mode_selection_order={metadata.get('mode_selection_order')!r}")
    if metadata.get("learnability_tiebreak") != "policy_density_then_frontier_difficulty":
        errors.append(f"learnability_tiebreak={metadata.get('learnability_tiebreak')!r}")
    if metadata.get("learnability_contract") != "policy_density_bounded_frontier_v1":
        errors.append(f"learnability_contract={metadata.get('learnability_contract')!r}")
    if (
        metadata.get("candidate_capacity_contract")
        != "observed_funnel_distinct_before_pareto_no_quota_v2"
    ):
        errors.append(f"candidate_capacity_contract={metadata.get('candidate_capacity_contract')!r}")
    if not bool(metadata.get("legacy_reward_gate_disabled", False)):
        errors.append("legacy_reward_gate_disabled=false")
    if not bool(metadata.get("exclude_derived_external", False)):
        errors.append("exclude_derived_external=false")
    if not bool(metadata.get("policy_reachability_required", False)):
        errors.append("policy_reachability_required=false")
    selection_gate_config = metadata.get("selection_gate_config")
    if not isinstance(selection_gate_config, Mapping):
        errors.append("selection_gate_config must be a mapping")
        selection_gate_config = {}
    required_gate_fields = {
        "fpv3_ddc_min_absolute",
        "fpv3_ddc_drop_tolerance",
        "support_ddc_gate_mode",
        "fpv3_feas_cost_max",
        "support_feas_gate_mode",
        "support_feas_cost_tolerance",
        "fpv3_comfort_min",
        "support_comfort_gate_mode",
        "support_comfort_drop_tolerance",
        "support_quality_enable",
        "support_min_non_gt_reward",
        "support_reward_gate_mode",
        "support_max_first_xy_error_m",
        "support_max_first_heading_error_rad",
        "support_max_xy_turn_rad",
        "support_max_early_xy_turn_rad",
        "support_max_step_m",
        "support_semantic_enable",
        "support_allow_turn_class_mismatch",
        "support_max_semantic_final_heading_error_rad",
        "support_max_semantic_path_angle_error_rad",
        "support_max_semantic_endpoint_lateral_error_m",
    }
    missing_gate_fields = sorted(required_gate_fields.difference(selection_gate_config))
    if missing_gate_fields:
        errors.append(f"selection_gate_config missing fields: {missing_gate_fields}")
    external_roots = metadata.get("external_candidate_roots")
    external_fingerprints = metadata.get("external_candidate_root_fingerprints")
    external_file_counts = metadata.get("external_candidate_root_file_counts")
    if not all(
        isinstance(value, Mapping)
        for value in (external_roots, external_fingerprints, external_file_counts)
    ):
        errors.append("external candidate provenance fields must be mappings")
    else:
        root_keys = set(external_roots)
        if root_keys != set(external_fingerprints) or root_keys != set(external_file_counts):
            errors.append("external candidate provenance source keys disagree")
        coherent_keys = root_keys & set(external_fingerprints) & set(external_file_counts)
        for source in coherent_keys:
            fingerprint = str(external_fingerprints[source])
            if len(fingerprint) != 64 or any(char not in "0123456789abcdef" for char in fingerprint.lower()):
                errors.append(f"invalid external candidate fingerprint for source={source!r}")
            if int(external_file_counts[source]) <= 0:
                errors.append(f"invalid external candidate file count for source={source!r}")
    if int(metadata.get("policy_samples_per_scene", 0)) <= 0:
        errors.append("policy_samples_per_scene must be positive")
    if not str(metadata.get("policy_checkpoint_sha256", "")):
        errors.append("policy_checkpoint_sha256 is empty")
    if not str(metadata.get("fs_norm_stats_sha256", "")):
        errors.append("fs_norm_stats_sha256 is empty")
    if str(metadata.get("build_log_split", "")).strip().lower() not in {"train", "val", "train_val"}:
        errors.append(f"invalid build_log_split={metadata.get('build_log_split')!r}")
    if int(metadata.get("dataset_scene_count", 0)) <= 0:
        errors.append("dataset_scene_count must be positive")
    if metadata.get("scene_seed_scheme") != "sha256_token_xor_build_seed_v1":
        errors.append(f"scene_seed_scheme={metadata.get('scene_seed_scheme')!r}")

    expected_valid = valid.copy()
    expected_quality = quality.copy()
    components_payload = record.get("components")
    feasibility_payload = record.get("feasibility")
    if isinstance(components_payload, Mapping) and isinstance(feasibility_payload, Mapping):
        try:
            component_rows = [
                {
                    key: float(np.asarray(values).reshape(-1)[index])
                    for key, values in components_payload.items()
                }
                for index in range(count)
            ]
            feasibility_rows = [
                {
                    key: float(np.asarray(values).reshape(-1)[index])
                    for key, values in feasibility_payload.items()
                }
                for index in range(count)
            ]
            candidate_rows = [
                CandidateRecord(
                    trajectory=candidates[index],
                    source=sources[index],
                    token=token,
                    components=component_rows[index],
                    reward=float(rewards[index]),
                    feas=feasibility_rows[index],
                    selection_score=float(rewards[index]),
                )
                for index in range(count)
            ]
            reference_components = component_rows[gt_index]
            expected_valid = np.asarray(
                [
                    is_valid_candidate(
                        candidate.components,
                        candidate.feas,
                        reference_components,
                        selection_gate_config,
                    )
                    for candidate in candidate_rows
                ],
                dtype=np.bool_,
            )
            expected_quality = np.asarray(
                [
                    support_trajectory_quality_pass(
                        candidate,
                        candidates[gt_index],
                        selection_gate_config,
                    )
                    for candidate in candidate_rows
                ],
                dtype=np.bool_,
            )
            if not np.array_equal(valid, expected_valid):
                errors.append("valid_mask disagrees with persisted selection gates")
            if not np.array_equal(quality, expected_quality):
                errors.append("support_quality_mask disagrees with persisted selection gates")
        except (IndexError, TypeError, ValueError) as exc:
            errors.append(f"could not recompute selection gates: {exc}")
    else:
        errors.append("components and feasibility must be mappings")

    policy_mask = np.asarray([_is_current_policy(source) for source in sources], dtype=np.bool_)
    policy_sample_count = int(policy_mask.sum())
    if policy_sample_count != int(metadata.get("policy_samples_per_scene", 0)):
        errors.append(
            f"current-policy candidate count={policy_sample_count}, "
            f"metadata policy_samples_per_scene={metadata.get('policy_samples_per_scene')!r}"
        )

    top_m = int(metadata.get("support_top_m", 0))
    max_ade = float(metadata.get("max_gt_ade_m", math.inf))
    max_fde = float(metadata.get("max_gt_fde_m", math.inf))
    max_reward_drop = float(metadata.get("max_gt_reward_drop", math.inf))
    mode_threshold = float(metadata.get("mode_distance_threshold", 0.0))
    max_policy_snsad = float(metadata.get("max_policy_snsad", math.inf))
    min_policy_neighbors = int(metadata.get("min_policy_neighbors", 0))
    if min_policy_neighbors <= 0:
        errors.append("min_policy_neighbors must be positive")
    if min_policy_neighbors > policy_sample_count:
        errors.append("min_policy_neighbors exceeds current-policy candidate count")
    if not bool(np.isfinite(policy_neighbor_fraction).all()) or bool(
        np.any((policy_neighbor_fraction < 0.0) | (policy_neighbor_fraction > 1.0))
    ):
        errors.append("policy_neighbor_fraction must be finite and in [0,1]")
    if not bool(np.isfinite(frontier_difficulty).all()) or bool(
        np.any((frontier_difficulty < 0.0) | (frontier_difficulty > 1.0 + 1e-6))
    ):
        errors.append("learning_frontier_difficulty must be finite and in [0,1]")

    gt_xy_distance = np.linalg.norm(
        candidates[:, :, :2].astype(np.float64)
        - candidates[gt_index : gt_index + 1, :, :2].astype(np.float64),
        axis=-1,
    )
    expected_gt_ade = gt_xy_distance.mean(axis=1)
    expected_gt_fde = gt_xy_distance[:, -1]
    if not np.allclose(gt_ade, expected_gt_ade, rtol=1e-5, atol=1e-6):
        errors.append("gt_relative_ade disagrees with candidate trajectories")
    if not np.allclose(gt_fde, expected_gt_fde, rtol=1e-5, atol=1e-6):
        errors.append("gt_relative_fde disagrees with candidate trajectories")

    expected_policy_reachability, expected_policy_neighbor_count, expected_policy_neighbor_fraction = (
        _policy_reachability_stats(candidates, policy_mask, max_policy_snsad)
    )
    if not np.allclose(
        policy_reachability,
        expected_policy_reachability,
        rtol=1e-5,
        atol=1e-6,
        equal_nan=False,
    ):
        errors.append("policy_reachability_snsad disagrees with candidate trajectories")
    if not np.array_equal(policy_neighbor_count, expected_policy_neighbor_count):
        errors.append("policy_neighbor_count disagrees with candidate trajectories")
    if not np.allclose(
        policy_neighbor_fraction,
        expected_policy_neighbor_fraction,
        rtol=1e-5,
        atol=1e-6,
    ):
        errors.append("policy_neighbor_fraction disagrees with candidate trajectories")
    expected_frontier_difficulty = (
        0.50 * np.minimum(expected_policy_reachability / max(max_policy_snsad, 1e-6), 1.0)
        + 0.25 * np.minimum(expected_gt_ade / max(max_ade, 1e-6), 1.0)
        + 0.25 * np.minimum(expected_gt_fde / max(max_fde, 1e-6), 1.0)
    )
    if not np.allclose(
        frontier_difficulty,
        expected_frontier_difficulty,
        rtol=1e-5,
        atol=1e-6,
    ):
        errors.append("learning_frontier_difficulty disagrees with candidate trajectories")

    expected_teacher_eligible = expected_valid & expected_quality
    non_gt_mask = np.arange(count, dtype=np.int64) != gt_index
    expected_teacher_eligible &= (
        (~non_gt_mask)
        | (
            (expected_gt_ade <= max_ade + 1e-6)
            & (expected_gt_fde <= max_fde + 1e-6)
            & (rewards >= rewards[gt_index] - max_reward_drop - 1e-6)
            & (expected_policy_reachability <= max_policy_snsad + 1e-6)
            & (expected_policy_neighbor_count >= min_policy_neighbors)
        )
    )
    if bool(metadata.get("exclude_derived_external", False)):
        expected_teacher_eligible &= (~non_gt_mask) | np.asarray(
            [not _is_derived_external(source) for source in sources], dtype=np.bool_
        )
    expected_teacher_eligible[gt_index] = True
    if not np.array_equal(teacher_eligible, expected_teacher_eligible):
        errors.append("teacher_eligible_mask disagrees with candidate gates")

    expected_gt_mode_distance = np.asarray(
        [
            trajectory_snsad_distance(candidates[index], candidates[gt_index])
            for index in range(count)
        ],
        dtype=np.float64,
    )
    mode_distinct_from_gt = expected_gt_mode_distance + 1e-6 >= mode_threshold
    pareto_domain = expected_teacher_eligible & (mode_distinct_from_gt | (~non_gt_mask))
    expected_pareto = _pareto_front_mask(
        np.stack((pareto_ep, pareto_ttc, pareto_quality), axis=-1),
        pareto_domain,
        float(metadata.get("pareto_eps", 0.0)),
    )
    if not np.array_equal(pareto, expected_pareto):
        errors.append("pareto_front_mask disagrees with distinct-mode Pareto domain")
    if top_m <= 0 or selected.size > top_m:
        errors.append(f"selected count={selected.size} exceeds support_top_m={top_m}")
    if gt_index not in selected.tolist():
        errors.append("GT anchor is not selected")
    if not bool(np.all(teacher_eligible[selected])):
        errors.append("a selected candidate is not teacher eligible")
    if not bool(np.all(quality[selected])):
        errors.append("a selected candidate fails the persisted v4 trajectory-quality mask")
    selected_modes = mode_ids[selected]
    if len(set(selected_modes.tolist())) != selected.size or set(selected_modes.tolist()) != set(range(selected.size)):
        errors.append(f"selected mode IDs are not one-to-one contiguous IDs: {selected_modes.tolist()}")

    non_gt = np.asarray([index for index in selected if int(index) != gt_index], dtype=np.int64)
    if non_gt.size:
        if not bool(np.isfinite(objective_novelty[non_gt]).all()):
            errors.append("a selected non-GT candidate has non-finite objective novelty")
        elif bool(np.any(objective_novelty[non_gt] < -1e-8)):
            errors.append("a selected non-GT candidate has negative objective novelty")
        if not bool(np.all(pareto[non_gt])):
            errors.append("a selected non-GT candidate is outside the epsilon-Pareto front")
        if bool(np.any(expected_gt_ade[non_gt] > max_ade + 1e-6)):
            errors.append("a selected non-GT candidate exceeds max_gt_ade_m")
        if bool(np.any(expected_gt_fde[non_gt] > max_fde + 1e-6)):
            errors.append("a selected non-GT candidate exceeds max_gt_fde_m")
        if bool(np.any(rewards[non_gt] < rewards[gt_index] - max_reward_drop - 1e-6)):
            errors.append("a selected non-GT candidate exceeds max_gt_reward_drop")
        if not bool(np.isfinite(expected_policy_reachability[non_gt]).all()):
            errors.append("a selected non-GT candidate has no finite current-policy reachability")
        elif bool(np.any(expected_policy_reachability[non_gt] > max_policy_snsad + 1e-6)):
            errors.append("a selected non-GT candidate exceeds max_policy_snsad")
        if bool(np.any(expected_policy_neighbor_count[non_gt] < min_policy_neighbors)):
            errors.append("a selected non-GT candidate has insufficient current-policy neighbors")
        if any(_is_derived_external(sources[int(index)]) for index in non_gt):
            errors.append("a derived external expansion is selected")

    expected_funnel = _recompute_candidate_funnel(
        sources=sources,
        selected=selected,
        gt_index=gt_index,
        valid=expected_valid,
        quality=expected_quality,
        teacher_eligible=expected_teacher_eligible,
        pareto=pareto,
        gt_ade=expected_gt_ade,
        gt_fde=expected_gt_fde,
        rewards=rewards,
        policy_reachability=expected_policy_reachability,
        policy_neighbor_count=expected_policy_neighbor_count,
        gt_mode_distance=expected_gt_mode_distance,
        metadata=metadata,
    )
    candidate_funnel = dict(record.get("candidate_funnel", {}) or {})
    missing_funnel = sorted(expected_funnel.keys() - candidate_funnel.keys())
    if missing_funnel:
        errors.append(f"missing candidate_funnel fields: {missing_funnel}")
    for key, expected in expected_funnel.items():
        if candidate_funnel.get(key) != expected:
            errors.append(
                f"candidate_funnel {key}={candidate_funnel.get(key)!r}, expected {expected!r}"
            )

    min_pairwise_distance = math.inf
    for left_pos, left in enumerate(selected.tolist()):
        for right in selected[left_pos + 1 :].tolist():
            distance = trajectory_snsad_distance(candidates[left], candidates[right])
            min_pairwise_distance = min(min_pairwise_distance, distance)
            if distance + 1e-6 < mode_threshold:
                errors.append(
                    f"selected modes {left}/{right} have SNSAD={distance:.6f} below {mode_threshold:.6f}"
                )
    if not math.isfinite(min_pairwise_distance):
        min_pairwise_distance = 0.0

    eligible_pareto = np.flatnonzero(
        expected_teacher_eligible & expected_pareto & (np.arange(count, dtype=np.int64) != gt_index)
    )
    mode_capacity = _greedy_mode_capacity(
        candidates,
        eligible_pareto,
        gt_index,
        mode_threshold,
    )
    target_mode_count = min(mode_capacity, max(top_m - 1, 0))
    capacity_coverage = (
        min(float(non_gt.size) / float(target_mode_count), 1.0)
        if target_mode_count > 0
        else math.nan
    )

    selected_families = Counter(_source_family(sources[int(index)]) for index in selected)
    readiness = dict(record.get("stage3_readiness", {}) or {})
    required_readiness = {
        "benchmark",
        "policy_sample_count",
        "feasible_count",
        "feasible_ratio",
        "positive_fraction",
        "negative_fraction",
        "advantage_magnitude",
        "bidirectional_energy",
        "feasible_score_span",
        "credit_ready",
    }
    missing_readiness = sorted(required_readiness.difference(readiness))
    if missing_readiness:
        errors.append(f"missing stage3 readiness fields: {missing_readiness}")
    readiness_values = [
        float(readiness.get(key, math.nan))
        for key in (
            "feasible_ratio",
            "positive_fraction",
            "negative_fraction",
            "advantage_magnitude",
            "bidirectional_energy",
            "feasible_score_span",
        )
    ]
    if not bool(np.isfinite(readiness_values).all()):
        errors.append("stage3 readiness contains non-finite scalars")
    expected_feasible_count = int((policy_mask & expected_valid & expected_quality).sum())
    if int(readiness.get("policy_sample_count", -1)) != policy_sample_count:
        errors.append("stage3 readiness policy_sample_count disagrees with candidate sources")
    if int(readiness.get("feasible_count", -1)) != expected_feasible_count:
        errors.append("stage3 readiness feasible_count disagrees with valid/quality masks")
    positive_fraction = float(readiness.get("positive_fraction", 0.0))
    negative_fraction = float(readiness.get("negative_fraction", 0.0))
    advantage_magnitude = float(readiness.get("advantage_magnitude", 0.0))
    expected_energy = 4.0 * positive_fraction * negative_fraction * advantage_magnitude
    if not math.isclose(
        float(readiness.get("bidirectional_energy", math.nan)),
        expected_energy,
        rel_tol=1e-5,
        abs_tol=1e-6,
    ):
        errors.append("stage3 readiness bidirectional_energy is inconsistent")
    min_feasible = int(metadata.get("stage3_min_feasible_rollouts", 2))
    min_score_span = float(metadata.get("stage3_min_score_span", 0.01))
    expected_credit_ready = bool(
        expected_feasible_count >= min_feasible
        and positive_fraction > 0.0
        and negative_fraction > 0.0
        and float(readiness.get("feasible_score_span", 0.0)) >= min_score_span
    )
    if bool(readiness.get("credit_ready", False)) != expected_credit_ready:
        errors.append("stage3 readiness credit_ready is inconsistent")
    return {
        "token": token,
        "errors": errors,
        "candidate_count": count,
        "selected_count": int(selected.size),
        "non_gt_count": int(non_gt.size),
        "multimode": int(non_gt.size > 0),
        "reachable_pareto_mode_capacity": int(mode_capacity),
        "target_mode_count": int(target_mode_count),
        "capacity_normalized_coverage": float(capacity_coverage),
        "capacity_undercovered": int(target_mode_count > 0 and non_gt.size < target_mode_count),
        "raw_internal_candidates": int(bool(metadata.get("raw_internal_candidates", False))),
        "external_expansion_disabled": int(not bool(metadata.get("expand_external_candidates", True))),
        "policy_reachability_required": int(bool(metadata.get("policy_reachability_required", False))),
        "selected_source_conditioned": int(source_conditioned[selected].sum()),
        "candidate_families": Counter(_source_family(source) for source in sources),
        "selected_families": selected_families,
        "reward_delta": (rewards[non_gt] - rewards[gt_index]).tolist(),
        "gt_ade": expected_gt_ade[non_gt].tolist(),
        "gt_fde": expected_gt_fde[non_gt].tolist(),
        "policy_reachability": expected_policy_reachability[non_gt].tolist(),
        "objective_novelty": objective_novelty[non_gt].tolist(),
        "min_pairwise_snsad": float(min_pairwise_distance),
        "selected_frontier_difficulty": frontier_difficulty[non_gt].tolist(),
        "policy_feasible_ratio": float(readiness.get("feasible_ratio", 0.0)),
        "stage3_credit_ready": int(bool(readiness.get("credit_ready", False))),
        "stage3_bidirectional_energy": float(readiness.get("bidirectional_energy", 0.0)),
        "stage3_feasible_score_span": float(readiness.get("feasible_score_span", 0.0)),
        "candidate_funnel": expected_funnel,
    }


def _validate_path(path: Path) -> dict[str, Any]:
    try:
        return validate_record(_load(path))
    except Exception as exc:  # pragma: no cover - archive I/O error path
        return {"token": path.stem, "errors": [f"{path}: {exc}"]}


def validate_archive(
    root: Path,
    *,
    max_records: int = 0,
    workers: int = 1,
    require_raw_build: bool = True,
    min_multimode_scene_ratio: float = 0.25,
    min_capacity_normalized_coverage: float = 0.90,
    min_policy_feasible_ratio: float = 0.90,
    min_stage3_credit_ready_scene_ratio: float = 0.05,
    expected_token_source: Optional[Path] = None,
) -> dict[str, Any]:
    paths = _iter_paths(root)
    if max_records > 0:
        paths = paths[:max_records]
    rows: Iterable[dict[str, Any]]
    if workers > 1:
        with mp.Pool(workers) as pool:
            rows = list(pool.imap_unordered(_validate_path, paths, chunksize=64))
    else:
        rows = [_validate_path(path) for path in paths]

    totals: Counter[str] = Counter()
    source_families: Counter[str] = Counter()
    candidate_source_families: Counter[str] = Counter()
    candidate_source_scenes: Counter[str] = Counter()
    values: dict[str, list[float]] = defaultdict(list)
    gt_only_reasons: Counter[str] = Counter()
    supervision_types: Counter[str] = Counter()
    error_examples: list[dict[str, Any]] = []
    actual_token_counts: Counter[str] = Counter()
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
            "reachable_pareto_mode_capacity",
            "target_mode_count",
            "capacity_undercovered",
            "raw_internal_candidates",
            "external_expansion_disabled",
            "policy_reachability_required",
            "selected_source_conditioned",
            "stage3_credit_ready",
        ):
            totals[key] += int(row[key])
        source_families.update(row["selected_families"])
        candidate_source_families.update(row["candidate_families"])
        candidate_source_scenes.update(row["candidate_families"].keys())
        values["reward_delta"].extend(row["reward_delta"])
        values["gt_ade"].extend(row["gt_ade"])
        values["gt_fde"].extend(row["gt_fde"])
        values["policy_reachability"].extend(row["policy_reachability"])
        values["objective_novelty"].extend(row["objective_novelty"])
        values["min_pairwise_snsad"].append(float(row["min_pairwise_snsad"]))
        values["selected_frontier_difficulty"].extend(row["selected_frontier_difficulty"])
        values["policy_feasible_ratio"].append(float(row["policy_feasible_ratio"]))
        values["stage3_bidirectional_energy"].append(float(row["stage3_bidirectional_energy"]))
        values["stage3_feasible_score_span"].append(float(row["stage3_feasible_score_span"]))
        funnel = dict(row["candidate_funnel"])
        supervision_types[str(funnel["supervision_type"])] += 1
        if str(funnel["gt_only_reason"]):
            gt_only_reasons[str(funnel["gt_only_reason"])] += 1
        for key in _CANDIDATE_FUNNEL_COUNT_KEYS:
            values[f"candidate_funnel_{key}"].append(float(funnel[key]))
        if math.isfinite(float(row["capacity_normalized_coverage"])):
            values["capacity_normalized_coverage"].append(
                float(row["capacity_normalized_coverage"])
            )

    valid_records = max(totals["records"] - totals["contract_errors"], 1)
    multimode_ratio = totals["multimode"] / valid_records
    raw_ratio = totals["raw_internal_candidates"] / valid_records
    expansion_disabled_ratio = totals["external_expansion_disabled"] / valid_records
    policy_reachability_ratio = totals["policy_reachability_required"] / valid_records
    capacity_coverage_values = np.asarray(values["capacity_normalized_coverage"], dtype=np.float64)
    capacity_normalized_coverage = (
        float(capacity_coverage_values.mean()) if capacity_coverage_values.size else 1.0
    )
    policy_feasible_ratio = float(np.mean(values["policy_feasible_ratio"])) if values["policy_feasible_ratio"] else 0.0
    stage3_credit_ready_ratio = totals["stage3_credit_ready"] / valid_records
    duplicate_tokens = sorted(token for token, count in actual_token_counts.items() if count > 1)
    expected_token_list = (
        _load_expected_tokens(expected_token_source, workers=workers)
        if expected_token_source
        else None
    )
    expected_token_counts = Counter(expected_token_list or [])
    expected_tokens = set(expected_token_counts) if expected_token_list is not None else None
    expected_duplicate_tokens = sorted(
        token for token, count in expected_token_counts.items() if count > 1
    )
    actual_tokens = set(actual_token_counts)
    missing_tokens = sorted(expected_tokens - actual_tokens) if expected_tokens is not None else []
    extra_tokens = sorted(actual_tokens - expected_tokens) if expected_tokens is not None else []
    coverage_pass = not duplicate_tokens and not expected_duplicate_tokens and (
        expected_tokens is None or (not missing_tokens and not extra_tokens)
    )
    hard_contract_pass = totals["records"] > 0 and totals["contract_errors"] == 0
    initial_policy_readiness_pass = (
        policy_feasible_ratio >= float(min_policy_feasible_ratio)
        and stage3_credit_ready_ratio >= float(min_stage3_credit_ready_scene_ratio)
    )
    static_promotion_pass = (
        hard_contract_pass
        and coverage_pass
        and multimode_ratio >= float(min_multimode_scene_ratio)
        and capacity_normalized_coverage >= float(min_capacity_normalized_coverage)
        and (not require_raw_build or raw_ratio == 1.0)
        and (not require_raw_build or expansion_disabled_ratio == 1.0)
        and policy_reachability_ratio == 1.0
    )
    return {
        "archive": str(root),
        "record_count": totals["records"],
        "contract_error_count": totals["contract_errors"],
        "error_examples": error_examples,
        "hard_contract_pass": hard_contract_pass,
        "token_coverage_pass": coverage_pass,
        "expected_token_source": str(expected_token_source) if expected_token_source else "",
        "expected_token_count": len(expected_tokens) if expected_tokens is not None else None,
        "expected_duplicate_token_count": len(expected_duplicate_tokens),
        "expected_duplicate_token_examples": expected_duplicate_tokens[:20],
        "actual_unique_token_count": len(actual_tokens),
        "missing_token_count": len(missing_tokens),
        "missing_token_examples": missing_tokens[:20],
        "extra_token_count": len(extra_tokens),
        "extra_token_examples": extra_tokens[:20],
        "duplicate_token_count": len(duplicate_tokens),
        "duplicate_token_examples": duplicate_tokens[:20],
        "static_promotion_pass": static_promotion_pass,
        "initial_policy_readiness_pass": initial_policy_readiness_pass,
        "require_raw_build": bool(require_raw_build),
        "min_multimode_scene_ratio": float(min_multimode_scene_ratio),
        "min_capacity_normalized_coverage": float(min_capacity_normalized_coverage),
        "min_policy_feasible_ratio": float(min_policy_feasible_ratio),
        "min_stage3_credit_ready_scene_ratio": float(min_stage3_credit_ready_scene_ratio),
        "candidate_count_per_scene": totals["candidate_count"] / valid_records,
        "selected_count_per_scene": totals["selected_count"] / valid_records,
        "non_gt_count_per_scene": totals["non_gt_count"] / valid_records,
        "multimode_scene_ratio": multimode_ratio,
        "reachable_pareto_mode_capacity_per_scene": totals["reachable_pareto_mode_capacity"]
        / valid_records,
        "target_mode_count_per_scene": totals["target_mode_count"] / valid_records,
        "capacity_normalized_coverage": capacity_normalized_coverage,
        "capacity_undercovered_scene_ratio": totals["capacity_undercovered"] / valid_records,
        "raw_internal_candidate_build_ratio": raw_ratio,
        "external_expansion_disabled_ratio": expansion_disabled_ratio,
        "policy_reachability_required_ratio": policy_reachability_ratio,
        "selected_source_conditioned_ratio": totals["selected_source_conditioned"]
        / max(totals["selected_count"], 1),
        "selected_source_family_counts": dict(source_families),
        "candidate_source_family_counts": dict(candidate_source_families),
        "candidate_source_scene_counts": dict(candidate_source_scenes),
        "non_gt_reward_delta": _stats(values["reward_delta"]),
        "non_gt_gt_relative_ade_m": _stats(values["gt_ade"]),
        "non_gt_gt_relative_fde_m": _stats(values["gt_fde"]),
        "non_gt_policy_reachability_snsad": _stats(values["policy_reachability"]),
        "non_gt_pareto_objective_novelty": _stats(values["objective_novelty"]),
        "min_pairwise_snsad_per_scene": _stats(values["min_pairwise_snsad"]),
        "selected_frontier_difficulty": _stats(values["selected_frontier_difficulty"]),
        "policy_feasible_ratio": policy_feasible_ratio,
        "stage3_credit_ready_scene_ratio": stage3_credit_ready_ratio,
        "stage3_bidirectional_energy": _stats(values["stage3_bidirectional_energy"]),
        "stage3_feasible_score_span": _stats(values["stage3_feasible_score_span"]),
        "candidate_funnel_mean": {
            key: float(np.mean(values[f"candidate_funnel_{key}"]))
            if values[f"candidate_funnel_{key}"]
            else 0.0
            for key in _CANDIDATE_FUNNEL_COUNT_KEYS
        },
        "supervision_type_counts": dict(supervision_types),
        "gt_only_reason_counts": dict(gt_only_reasons),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            "# SG-FPS v4 archive validation",
            "",
            f"- Archive: `{report['archive']}`",
            f"- Records: {report['record_count']}",
            f"- Contract errors: {report['contract_error_count']}",
            f"- Hard contract: **{'PASS' if report['hard_contract_pass'] else 'FAIL'}**",
            f"- Token coverage: **{'PASS' if report['token_coverage_pass'] else 'FAIL'}**",
            f"- Expected / actual unique tokens: {report['expected_token_count']} / {report['actual_unique_token_count']}",
            f"- Missing / extra / duplicate tokens: {report['missing_token_count']} / {report['extra_token_count']} / {report['duplicate_token_count']}",
            f"- Static promotion gate: **{'PASS' if report['static_promotion_pass'] else 'FAIL'}**",
            f"- Initial-policy readiness diagnostic: **{'PASS' if report['initial_policy_readiness_pass'] else 'FAIL'}**",
            f"- Mean selected trajectories: {report['selected_count_per_scene']:.4f}",
            f"- Multi-mode scene ratio: {report['multimode_scene_ratio']:.4f}",
            f"- Reachable Pareto mode capacity per scene: {report['reachable_pareto_mode_capacity_per_scene']:.4f}",
            f"- Capacity-normalized coverage: {report['capacity_normalized_coverage']:.4f}",
            f"- Capacity-undercovered scene ratio: {report['capacity_undercovered_scene_ratio']:.4f}",
            f"- Current-policy feasible ratio: {report['policy_feasible_ratio']:.4f}",
            f"- Stage3 credit-ready scene ratio: {report['stage3_credit_ready_scene_ratio']:.4f}",
            f"- GT-only scenes: {report['supervision_type_counts'].get('gt_only', 0)}",
            f"- GT-only reasons: {report['gt_only_reason_counts']}",
            f"- Raw candidate build ratio: {report['raw_internal_candidate_build_ratio']:.4f}",
            f"- External expansion disabled ratio: {report['external_expansion_disabled_ratio']:.4f}",
            f"- Policy reachability required ratio: {report['policy_reachability_required_ratio']:.4f}",
            "",
            "Static promotion does not replace a Stage2 learnability smoke or Stage3 improvement probe.",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the SG-FPS v4 data and teacher contract.")
    parser.add_argument("--archive-path", required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--require-raw-build", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-multimode-scene-ratio", type=float, default=0.25)
    parser.add_argument("--min-capacity-normalized-coverage", type=float, default=0.90)
    parser.add_argument("--min-policy-feasible-ratio", type=float, default=0.90)
    parser.add_argument("--min-stage3-credit-ready-scene-ratio", type=float, default=0.05)
    parser.add_argument(
        "--expected-token-source",
        default="",
        help="Archive directory or text/JSON manifest defining the exact expected token set.",
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    report = validate_archive(
        Path(args.archive_path),
        max_records=int(args.max_records),
        workers=max(int(args.workers), 1),
        require_raw_build=bool(args.require_raw_build),
        min_multimode_scene_ratio=float(args.min_multimode_scene_ratio),
        min_capacity_normalized_coverage=float(args.min_capacity_normalized_coverage),
        min_policy_feasible_ratio=float(args.min_policy_feasible_ratio),
        min_stage3_credit_ready_scene_ratio=float(args.min_stage3_credit_ready_scene_ratio),
        expected_token_source=Path(args.expected_token_source) if args.expected_token_source else None,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    if args.output_md:
        output = Path(args.output_md)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_markdown(report) + "\n", encoding="utf-8")
    if not args.report_only and not bool(report["static_promotion_pass"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
