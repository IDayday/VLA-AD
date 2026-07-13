from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .pareto_archive import pareto_front_mask, trajectory_snsad_distance


V6_SELECTION_STRATEGY = "mode_full_training_v6"
V6_TEACHER_CONTRACT = "gt_anchor_plus_policy_independent_evaluator_valid_modes"
V6_OBJECTIVES = "safety,efficiency,gt_relative_diversity,source_confidence"


@dataclass(frozen=True)
class FullTrainingV6Config:
    support_top_m: int = 4
    max_gt_ade_m: float = 1.5
    max_gt_fde_m: float = 4.0
    max_gt_reward_drop: float = 0.05
    mode_distance_threshold: float = 0.40
    pareto_eps: float = 0.01
    exclude_policy_candidates: bool = True
    exclude_derived_external: bool = True


def source_family(source: str) -> str:
    value = str(source).lower()
    if value == "gt" or value.startswith("gt:"):
        return "gt"
    if value == "policy" or value.startswith("policy:"):
        return "policy"
    if value == "ddv2" or value.startswith(("ddv2:", "diffusiondrivev2")):
        return "ddv2"
    if value == "driveor" or value.startswith(("driveor:", "drivor")):
        return "driveor"
    if value.startswith("progress"):
        return "progress"
    if "lateral" in value:
        return "lateral"
    if value.startswith("timing"):
        return "timing"
    return "other"


def is_derived_external(source: str) -> bool:
    value = str(source).lower()
    return (
        "failure_expand" in value
        or "external_expand" in value
        or value.startswith("expanded_")
    )


def _array(record: Mapping[str, Any], key: str, count: int, dtype: Any) -> np.ndarray:
    value = np.asarray(record[key], dtype=dtype).reshape(-1)
    if value.shape != (count,):
        raise ValueError(f"{key} has shape {value.shape}, expected {(count,)}")
    return value


def _component(record: Mapping[str, Any], key: str, count: int) -> np.ndarray:
    components = dict(record.get("components", {}) or {})
    if key not in components:
        raise KeyError(f"components is missing {key!r}")
    return _array(components, key, count, np.float32)


def _scene_normalize(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.bool_)
    output = np.full(values.shape, 0.5, dtype=np.float32)
    finite = mask & np.isfinite(values)
    if not bool(finite.any()):
        return output
    lower = float(values[finite].min())
    upper = float(values[finite].max())
    if upper - lower > 1e-6:
        output = (
            np.nan_to_num(values, nan=lower, posinf=upper, neginf=lower) - lower
        ) / (upper - lower)
    return np.clip(output, 0.0, 1.0)


def _source_confidence(sources: Sequence[str]) -> np.ndarray:
    # Direct external planners have scene-conditioned predictions. Structured
    # perturbations are useful coverage teachers after exact evaluator gating,
    # but receive lower confidence and are exposed later by the curriculum.
    values = {
        "gt": 1.00,
        "ddv2": 0.90,
        "driveor": 0.90,
        "lateral": 0.68,
        "timing": 0.66,
        "progress": 0.62,
        "other": 0.55,
        "policy": 0.00,
    }
    return np.asarray([values[source_family(source)] for source in sources], dtype=np.float32)


def _source_conditioned(sources: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [source_family(source) in {"ddv2", "driveor"} for source in sources],
        dtype=np.bool_,
    )


def _gt_distances(candidates: np.ndarray, gt_index: int) -> np.ndarray:
    return np.asarray(
        [trajectory_snsad_distance(candidate, candidates[gt_index]) for candidate in candidates],
        dtype=np.float32,
    )


def full_training_v6_selection(
    record: Mapping[str, Any],
    cfg: FullTrainingV6Config = FullTrainingV6Config(),
) -> dict[str, Any]:
    """Select a policy-independent full Stage2 teacher set from a raw v5 pool.

    Current-policy rollouts and their reachability/evidence fields are ignored.
    This prevents a previous Stage2 checkpoint from becoming a hidden admission
    dependency when the new Stage2 model is initialized from random weights.
    """

    candidates = np.asarray(record["candidates"], dtype=np.float32)
    if candidates.ndim != 3 or candidates.shape[-1] != 3 or candidates.shape[0] == 0:
        raise ValueError(f"candidates must have shape [M,H,3], got {candidates.shape}")
    count = int(candidates.shape[0])
    sources = [str(source) for source in record["sources"]]
    if len(sources) != count:
        raise ValueError("sources length does not match candidates")
    gt_indices = [index for index, source in enumerate(sources) if source_family(source) == "gt"]
    if len(gt_indices) != 1:
        raise ValueError(f"full_training_v6 requires exactly one GT candidate, found {len(gt_indices)}")
    gt_index = int(gt_indices[0])
    rewards = _array(record, "rewards", count, np.float32)
    valid = _array(record, "valid_mask", count, np.bool_)
    quality = _array(record, "support_quality_mask", count, np.bool_)
    gt_ade = _array(record, "gt_relative_ade", count, np.float32)
    gt_fde = _array(record, "gt_relative_fde", count, np.float32)
    gt_distance = _gt_distances(candidates, gt_index)
    non_gt = np.arange(count, dtype=np.int64) != gt_index
    allowed_source = np.asarray(
        [
            (not cfg.exclude_policy_candidates or source_family(source) != "policy")
            and (not cfg.exclude_derived_external or not is_derived_external(source))
            for source in sources
        ],
        dtype=np.bool_,
    )
    trust_region = (
        non_gt
        & valid
        & quality
        & allowed_source
        & (gt_ade <= float(cfg.max_gt_ade_m) + 1e-6)
        & (gt_fde <= float(cfg.max_gt_fde_m) + 1e-6)
        & (rewards >= float(rewards[gt_index]) - float(cfg.max_gt_reward_drop) - 1e-6)
    )
    teacher_eligible = trust_region & (
        gt_distance + 1e-6 >= float(cfg.mode_distance_threshold)
    )
    teacher_eligible[gt_index] = True

    nc = _component(record, "no_at_fault_collisions", count)
    dac = _component(record, "drivable_area_compliance", count)
    ttc = _component(record, "time_to_collision_within_bound", count)
    ddc = _component(record, "driving_direction_compliance", count)
    efficiency = _component(record, "ego_progress", count)
    safety = np.minimum.reduce((nc, dac, ttc, ddc)).astype(np.float32)
    confidence = _source_confidence(sources)
    objective_domain = teacher_eligible & non_gt
    objectives = np.stack(
        (
            _scene_normalize(safety, objective_domain),
            _scene_normalize(efficiency, objective_domain),
            _scene_normalize(gt_distance, objective_domain),
            confidence,
        ),
        axis=-1,
    ).astype(np.float32)
    pareto_front = pareto_front_mask(
        objectives,
        objective_domain,
        eps=float(cfg.pareto_eps),
    )

    selected = [gt_index]
    objective_novelty = np.zeros((count,), dtype=np.float32)
    remaining = set(int(index) for index in np.flatnonzero(pareto_front & objective_domain))
    while remaining and len(selected) < int(cfg.support_top_m):
        choices: list[tuple[tuple[float, ...], int, float]] = []
        selected_non_gt = selected[1:]
        selected_families = {source_family(sources[index]) for index in selected_non_gt}
        for index in remaining:
            trajectory_novelty = min(
                trajectory_snsad_distance(candidates[index], candidates[chosen])
                for chosen in selected
            )
            if trajectory_novelty + 1e-6 < float(cfg.mode_distance_threshold):
                continue
            if selected_non_gt:
                novelty = min(
                    float(np.abs(objectives[index] - objectives[chosen]).mean())
                    for chosen in selected_non_gt
                )
            else:
                novelty = float(objectives[index].mean())
            family = source_family(sources[index])
            key = (
                float(family not in selected_families),
                novelty,
                float(trajectory_novelty),
                float(confidence[index]),
                float(safety[index]),
                float(efficiency[index]),
                float(rewards[index]),
                float(-index),
            )
            choices.append((key, index, novelty))
        if not choices:
            break
        _, index, novelty = max(choices, key=lambda item: item[0])
        selected.append(index)
        remaining.remove(index)
        objective_novelty[index] = float(novelty)

    mode_ids = np.full((count,), -1, dtype=np.int64)
    mode_ids[gt_index] = 0
    support_tags = [""] * count
    support_tags[gt_index] = "gt_anchor"
    teacher_tier = np.full((count,), -1, dtype=np.int8)
    teacher_tier[gt_index] = 0
    for mode_id, index in enumerate(selected[1:], start=1):
        mode_ids[index] = mode_id
        family = source_family(sources[index])
        if family in {"ddv2", "driveor"}:
            teacher_tier[index] = 1
            support_tags[index] = "full_training_direct"
        else:
            teacher_tier[index] = 2
            support_tags[index] = "full_training_structured"

    difficulty = np.clip(
        0.35 * np.minimum(gt_ade / max(float(cfg.max_gt_ade_m), 1e-6), 1.0)
        + 0.20 * np.minimum(gt_fde / max(float(cfg.max_gt_fde_m), 1e-6), 1.0)
        + 0.25 * (1.0 - confidence)
        + 0.20 * (1.0 - np.clip(safety, 0.0, 1.0)),
        0.0,
        1.0,
    ).astype(np.float32)
    difficulty[gt_index] = 0.0

    selected_non_gt = selected[1:]
    if selected_non_gt:
        supervision_type = "full_training_modes"
        gt_only_reason = ""
    elif not bool((non_gt & valid).any()):
        supervision_type = "gt_only"
        gt_only_reason = "no_evaluator_valid_candidate"
    elif not bool(trust_region.any()):
        supervision_type = "gt_only"
        gt_only_reason = "no_policy_independent_trust_region_candidate"
    elif not bool(objective_domain.any()):
        supervision_type = "gt_only"
        gt_only_reason = "no_distinct_policy_independent_mode"
    else:
        supervision_type = "gt_only"
        gt_only_reason = "no_pairwise_distinct_pareto_mode"

    funnel = {
        "version": 4,
        "non_gt_proposal_count": int(non_gt.sum()),
        "policy_candidate_count": int(
            sum(source_family(source) == "policy" for source in sources)
        ),
        "evaluator_valid_count": int((non_gt & valid).sum()),
        "trajectory_quality_count": int((non_gt & valid & quality).sum()),
        "policy_independent_trust_region_count": int(trust_region.sum()),
        "distinct_from_gt_count": int(objective_domain.sum()),
        "pareto_eligible_count": int((pareto_front & objective_domain).sum()),
        "selected_non_gt_count": int(len(selected_non_gt)),
        "supervision_type": supervision_type,
        "gt_only_reason": gt_only_reason,
    }
    return {
        "support_indices": selected,
        "support_tags": support_tags,
        "mode_ids": mode_ids,
        "teacher_eligible_mask": teacher_eligible,
        "source_conditioned_mask": _source_conditioned(sources),
        "teacher_tier": teacher_tier,
        "teacher_confidence": confidence,
        "learning_frontier_difficulty": difficulty,
        "pareto_objective_novelty": objective_novelty,
        "pareto_front_mask": pareto_front,
        "full_training_objectives": objectives,
        "gt_mode_distance_snsad": gt_distance,
        "candidate_funnel": funnel,
    }


def promote_v5_record_to_v6(
    record: Mapping[str, Any],
    cfg: FullTrainingV6Config = FullTrainingV6Config(),
) -> dict[str, Any]:
    if int(record.get("version", 0)) != 5:
        raise ValueError(f"full_training_v6 promotion requires a v5 record, got {record.get('version')!r}")
    selection = full_training_v6_selection(record, cfg)
    rebuilt = dict(record)
    rebuilt.update(selection)
    rebuilt["version"] = 6
    support_indices = list(selection["support_indices"])
    rewards = np.asarray(record["rewards"], dtype=np.float32)
    sources = [str(source) for source in record["sources"]]
    best_index = max(support_indices, key=lambda index: float(rewards[index]))
    rebuilt["best_selected_reward"] = float(rewards[best_index])
    rebuilt["best_selected_source"] = sources[best_index]
    metadata = dict(record.get("build_metadata", {}) or {})
    metadata.update(
        {
            "selection_strategy": V6_SELECTION_STRATEGY,
            "teacher_contract": V6_TEACHER_CONTRACT,
            "mode_selection_order": (
                "policy_independent_hard_gates_then_safety_efficiency_diversity_confidence_"
                "pareto_then_family_aware_snsad_fps"
            ),
            "candidate_capacity_contract": "full_dataset_policy_independent_evaluator_modes_v1",
            "capacity_semantics": "selected_full_training_support_cap_not_scene_quota",
            "support_top_m": int(cfg.support_top_m),
            "support_cap_not_quota": True,
            "policy_candidates_excluded": bool(cfg.exclude_policy_candidates),
            "policy_fields_used_for_admission": False,
            "policy_fields_used_for_ranking": False,
            "exclude_derived_external": bool(cfg.exclude_derived_external),
            "max_gt_ade_m": float(cfg.max_gt_ade_m),
            "max_gt_fde_m": float(cfg.max_gt_fde_m),
            "max_gt_reward_drop": float(cfg.max_gt_reward_drop),
            "mode_distance_threshold": float(cfg.mode_distance_threshold),
            "pareto_eps": float(cfg.pareto_eps),
            "pareto_objectives": V6_OBJECTIVES,
            "teacher_confidence_contract": "source_prior_after_exact_evaluator_gate_v1",
            "frontier_difficulty_formula": (
                "0.35*gt_ade+0.20*gt_fde+0.25*(1-confidence)+0.20*(1-safety)"
            ),
        }
    )
    rebuilt["build_metadata"] = metadata
    return rebuilt
