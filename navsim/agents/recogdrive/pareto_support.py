from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


PARETO_COMPONENT_KEYS = (
    "ego_progress",
    "time_to_collision_within_bound",
    "driving_direction_compliance",
)


@dataclass
class CandidateRecord:
    trajectory: np.ndarray
    source: str
    token: str
    components: dict[str, float]
    reward: float
    feas: dict[str, float]
    selection_score: float
    support_tag: str = ""
    parent_id: str = ""


def _cfg_value(cfg: Any, name: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _component_value(components: Mapping[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        if name in components:
            return float(components[name])
    return float(default)


def _feas_cost(feas: Mapping[str, Any]) -> float:
    return _component_value(feas, "feas_cost", "cost", default=0.0)


def compute_utility(
    components: dict | torch.Tensor,
    feas_cost: float | torch.Tensor,
    weights: dict,
) -> float | torch.Tensor:
    ep_w = float(weights.get("ep", weights.get("fp_ep_weight", 0.60)))
    ttc_w = float(weights.get("ttc", weights.get("fp_ttc_weight", 0.25)))
    ddc_w = float(weights.get("ddc", weights.get("fp_ddc_weight", 0.10)))
    feas_w = float(weights.get("feas", weights.get("fp_feas_weight", 0.05)))
    if isinstance(components, torch.Tensor):
        if components.shape[-1] < 4:
            raise ValueError("Tensor components must contain [EP, TTC, DDC, -FeasCost].")
        return ep_w * components[..., 0] + ttc_w * components[..., 1] + ddc_w * components[..., 2] - feas_w * feas_cost
    ep = _component_value(components, "ego_progress", "ep")
    ttc = _component_value(components, "time_to_collision_within_bound", "ttc")
    ddc = _component_value(components, "driving_direction_compliance", "ddc")
    return ep_w * ep + ttc_w * ttc + ddc_w * ddc - feas_w * float(feas_cost)


def is_valid_candidate(components: Mapping[str, Any], feas: Mapping[str, Any], ref_components: Mapping[str, Any], cfg: Any) -> bool:
    nc = _component_value(components, "no_at_fault_collisions", "nc")
    dac = _component_value(components, "drivable_area_compliance", "dac")
    ddc = _component_value(components, "driving_direction_compliance", "ddc")
    comfort = _component_value(components, "history_comfort", "comfort", default=1.0)
    ref_ddc = _component_value(ref_components, "driving_direction_compliance", "ddc", default=ddc)
    ddc_min = max(
        float(_cfg_value(cfg, "fp_ddc_min_absolute", 0.95)),
        ref_ddc - float(_cfg_value(cfg, "fp_ddc_ref_tolerance", 0.01)),
    )
    feas_threshold = float(_cfg_value(cfg, "fp_feas_max", 0.0))
    if feas_threshold <= 0.0:
        feas_threshold = float(_cfg_value(cfg, "support_feas_max", 1.0))
    return (
        nc >= float(_cfg_value(cfg, "fp_nc_min", 1.0))
        and dac >= float(_cfg_value(cfg, "fp_dac_min", 1.0))
        and ddc >= ddc_min
        and comfort >= float(_cfg_value(cfg, "fp_comfort_min", 0.95))
        and _feas_cost(feas) <= feas_threshold
    )


def pareto_front_mask(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=np.bool_)
    if values.ndim != 2:
        raise ValueError(f"values must have shape [K, D], got {values.shape}.")
    if valid_mask.shape != (values.shape[0],):
        raise ValueError(f"valid_mask must have shape [K], got {valid_mask.shape}.")
    front = np.zeros(values.shape[0], dtype=np.bool_)
    valid_idx = np.flatnonzero(valid_mask)
    for idx in valid_idx:
        others = valid_idx[valid_idx != idx]
        if others.size == 0:
            front[idx] = True
            continue
        dominates = np.all(values[others] >= values[idx], axis=1) & np.any(values[others] > values[idx], axis=1)
        front[idx] = not bool(np.any(dominates))
    return front


def _fingerprint_traj(traj: np.ndarray) -> str:
    arr = np.asarray(traj, dtype=np.float32)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def _pareto_values(candidates: Sequence[CandidateRecord]) -> np.ndarray:
    rows = []
    for cand in candidates:
        feas = _feas_cost(cand.feas)
        rows.append(
            [
                _component_value(cand.components, "ego_progress", "ep"),
                _component_value(cand.components, "time_to_collision_within_bound", "ttc"),
                _component_value(cand.components, "driving_direction_compliance", "ddc"),
                -feas,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _append_unique(
    selected: list[CandidateRecord],
    used: set[str],
    candidates: Iterable[CandidateRecord],
    tag: str,
    limit: int,
) -> None:
    for cand in candidates:
        if len(selected) >= limit:
            return
        key = _fingerprint_traj(cand.trajectory)
        if key in used:
            continue
        used.add(key)
        cand.support_tag = cand.support_tag or tag
        selected.append(cand)


def _sort_by_reward(candidates: Iterable[CandidateRecord]) -> list[CandidateRecord]:
    return sorted(candidates, key=lambda c: (float(c.reward), float(c.selection_score)), reverse=True)


def _sort_by_utility(candidates: Iterable[CandidateRecord], weights: Mapping[str, float]) -> list[CandidateRecord]:
    return sorted(
        candidates,
        key=lambda c: (
            float(compute_utility(c.components, _feas_cost(c.feas), dict(weights))),
            float(c.reward),
            float(c.selection_score),
        ),
        reverse=True,
    )


def _diversity_order(candidates: Sequence[CandidateRecord]) -> list[CandidateRecord]:
    if not candidates:
        return []
    chosen = [max(candidates, key=lambda c: float(c.reward))]
    remaining = [c for c in candidates if c is not chosen[0]]
    while remaining:
        endpoints = np.stack([np.asarray(c.trajectory, dtype=np.float32)[-1, :2] for c in chosen], axis=0)
        best_idx = 0
        best_dist = -1.0
        for idx, cand in enumerate(remaining):
            end = np.asarray(cand.trajectory, dtype=np.float32)[-1, :2]
            dist = float(np.linalg.norm(endpoints - end[None, :], axis=1).min())
            if dist > best_dist:
                best_dist = dist
                best_idx = idx
        chosen.append(remaining.pop(best_idx))
    return chosen


def select_feasible_pareto_support(
    candidates: list[CandidateRecord],
    ref: dict,
    cfg: Any,
) -> list[CandidateRecord]:
    if not candidates:
        return []
    support_top_m = int(_cfg_value(cfg, "support_top_m", _cfg_value(cfg, "dpsi_top_m", 12)))
    weights = {
        "ep": float(_cfg_value(cfg, "fp_ep_weight", 0.60)),
        "ttc": float(_cfg_value(cfg, "fp_ttc_weight", 0.25)),
        "ddc": float(_cfg_value(cfg, "fp_ddc_weight", 0.10)),
        "feas": float(_cfg_value(cfg, "fp_feas_weight", 0.05)),
    }
    valid = np.asarray([is_valid_candidate(c.components, c.feas, ref, cfg) for c in candidates], dtype=np.bool_)
    values = _pareto_values(candidates)
    front = pareto_front_mask(values, valid)
    ref_ep = _component_value(ref, "ego_progress", "ep", default=0.0)
    ep_margin = float(_cfg_value(cfg, "safe_ep_improve_margin", 0.01))

    selected: list[CandidateRecord] = []
    used: set[str] = set()
    by_source = lambda name: [c for c in candidates if name in c.source.lower()]
    valid_candidates = [c for c, ok in zip(candidates, valid) if ok]
    front_candidates = [c for c, ok in zip(candidates, front) if ok]

    quotas: list[tuple[str, list[CandidateRecord], int]] = [
        ("gt_anchor", _sort_by_reward(by_source("gt")), 1),
        ("il_anchor", _sort_by_reward(by_source("il")), 1),
        ("best_pdms_valid", _sort_by_reward(valid_candidates), 1),
        (
            "safe_ep_improver",
            _sort_by_reward(
                [
                    c
                    for c in valid_candidates
                    if _component_value(c.components, "ego_progress", "ep") >= ref_ep + ep_margin
                ]
            ),
            2,
        ),
        (
            "safety_repair",
            _sort_by_reward(
                [
                    c
                    for c in candidates
                    if _component_value(c.components, "no_at_fault_collisions", "nc")
                    + _component_value(c.components, "drivable_area_compliance", "dac")
                    + _component_value(c.components, "time_to_collision_within_bound", "ttc")
                    >= 2.0
                ]
            ),
            1,
        ),
        (
            "ddc_repair",
            sorted(candidates, key=lambda c: _component_value(c.components, "driving_direction_compliance", "ddc"), reverse=True),
            1,
        ),
        ("smooth_feasible", sorted(valid_candidates, key=lambda c: _feas_cost(c.feas)), 1),
        ("vector_pareto", _sort_by_utility(front_candidates, weights), 2),
        ("diversity_max_valid", _diversity_order(valid_candidates), 1),
    ]

    for tag, pool, count in quotas:
        _append_unique(selected, used, pool, tag, min(support_top_m, len(selected) + count))
        if len(selected) >= support_top_m:
            return selected[:support_top_m]

    fallback = _sort_by_utility(front_candidates, weights) + _sort_by_utility(valid_candidates, weights) + _sort_by_reward(candidates)
    _append_unique(selected, used, fallback, "fallback_best", support_top_m)
    return selected[:support_top_m]


def build_archive_record(
    token: str,
    candidates: Sequence[CandidateRecord],
    selected: Sequence[CandidateRecord],
    ref: Mapping[str, Any] | None = None,
    cfg: Any = None,
) -> dict[str, Any]:
    ref = dict(ref or {})
    cfg = cfg or {}
    all_values = _pareto_values(candidates) if candidates else np.zeros((0, 4), dtype=np.float32)
    valid_mask = np.asarray([is_valid_candidate(c.components, c.feas, ref, cfg) for c in candidates], dtype=np.bool_)
    front_mask = pareto_front_mask(all_values, valid_mask) if candidates else np.zeros((0,), dtype=np.bool_)
    utilities = np.asarray(
        [compute_utility(c.components, _feas_cost(c.feas), dict(cfg or {})) for c in candidates],
        dtype=np.float32,
    )
    selected_fps = {_fingerprint_traj(c.trajectory) for c in selected}
    selected_mask = np.asarray([_fingerprint_traj(c.trajectory) in selected_fps for c in candidates], dtype=np.bool_)
    rewards = np.asarray([float(c.reward) for c in candidates], dtype=np.float32)
    selection_score = np.asarray([float(c.selection_score) for c in candidates], dtype=np.float32)
    raw_best_idx = int(np.argmax(rewards)) if rewards.size else 0
    valid_indices = np.flatnonzero(valid_mask)
    valid_best_idx = int(valid_indices[np.argmax(rewards[valid_indices])]) if valid_indices.size else raw_best_idx
    selected_indices = np.flatnonzero(selected_mask)
    selected_best_idx = int(selected_indices[np.argmax(rewards[selected_indices])]) if selected_indices.size else valid_best_idx
    gt_rewards = [float(c.reward) for c in candidates if "gt" in c.source.lower()]
    il_rewards = [float(c.reward) for c in candidates if "il" in c.source.lower()]
    gt_reward = gt_rewards[0] if gt_rewards else 0.0
    il_reward = il_rewards[0] if il_rewards else gt_reward
    best_valid_reward = float(rewards[valid_best_idx]) if rewards.size else 0.0
    best_selected_reward = float(rewards[selected_mask].max()) if np.any(selected_mask) else best_valid_reward
    if candidates:
        first = np.asarray(candidates[0].trajectory, dtype=np.float32)
        anchor_distance = np.asarray(
            [
                float(np.linalg.norm(np.asarray(c.trajectory, dtype=np.float32)[..., :2] - first[..., :2], axis=-1).mean())
                for c in candidates
            ],
            dtype=np.float32,
        )
    else:
        anchor_distance = np.zeros((0,), dtype=np.float32)
    return {
        "version": 3,
        "token": str(token),
        "candidates": np.stack([np.asarray(c.trajectory, dtype=np.float32) for c in candidates], axis=0)
        if candidates
        else np.zeros((0, 0, 3), dtype=np.float32),
        "rewards": rewards,
        "anchor_distance": anchor_distance,
        "components": {
            key: np.asarray([_component_value(c.components, key) for c in candidates], dtype=np.float32)
            for key in (
                "pdms",
                "no_at_fault_collisions",
                "drivable_area_compliance",
                "time_to_collision_within_bound",
                "ego_progress",
                "history_comfort",
                "lane_keeping",
                "driving_direction_compliance",
                "traffic_light_compliance",
            )
        },
        "sources": [str(c.source) for c in candidates],
        "valid_mask": valid_mask,
        "selection_score": selection_score,
        "feasibility": {key: np.asarray([float(c.feas.get(key, 0.0)) for c in candidates], dtype=np.float32) for key in ("feas_cost", "early_kink_rate", "tail_reverse_rate", "curvature_violation_rate")},
        "support_tags": [str(c.support_tag) for c in candidates],
        "pareto_front_mask": front_mask,
        "utility": utilities,
        "gt_reward": float(gt_reward),
        "il_reward": float(il_reward),
        "best_reward": float(rewards[raw_best_idx]) if rewards.size else 0.0,
        "best_source": str(candidates[raw_best_idx].source) if candidates else "",
        "best_raw_reward": float(rewards[raw_best_idx]) if rewards.size else 0.0,
        "best_valid_reward": best_valid_reward,
        "best_selected_reward": best_selected_reward,
        "best_raw_source": str(candidates[raw_best_idx].source) if candidates else "",
        "best_valid_source": str(candidates[valid_best_idx].source) if candidates else "",
        "best_selected_source": str(candidates[selected_best_idx].source) if candidates else "",
        "has_valid_candidate": bool(np.any(valid_mask)),
    }
