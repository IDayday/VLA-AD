from __future__ import annotations

import math
import time
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .dataclasses import CandidateRecord, ParetoSupportArchive, TrajectoryCandidate, trajectory_fingerprint
from .metrics import cfg_value, compute_metric_utility, is_valid_metrics, metric_value


PARETO_COMPONENT_KEYS = (
    "ego_progress",
    "time_to_collision_within_bound",
    "driving_direction_compliance",
)


def _component_value(components: Mapping[str, Any], *names: str, default: float = 0.0) -> float:
    return metric_value(components, *names, default=default)


def _feas_cost(feas: Mapping[str, Any]) -> float:
    return metric_value(feas, "feas_cost", "cost", default=0.0)


def _wrap_angle_np(x: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(x), np.cos(x))


def _source_matches(source: str, names: Sequence[str]) -> bool:
    """Match candidate sources without substring false positives.

    The old selector used ``name in source``. That incorrectly matched
    ``failure_expand`` as an ``il`` source and produced misleading
    ``il_anchor`` tags for DDV2/DriveOR candidates.
    """

    lower = str(source).lower()
    for raw_name in names:
        name = str(raw_name).lower()
        if name == "gt":
            if lower == "gt" or lower.startswith("gt:"):
                return True
            continue
        if name == "il":
            if lower == "il" or lower.startswith("il:"):
                return True
            continue
        if name == "recogdrive_stage3":
            if lower == "recogdrive_stage3" or lower.startswith("recogdrive_stage3:"):
                return True
            continue
        if lower == name or lower.startswith(f"{name}:"):
            return True
    return False


def _xy_turn_angles(traj: np.ndarray) -> np.ndarray:
    arr = np.asarray(traj, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 3:
        return np.zeros((0,), dtype=np.float32)
    step = arr[1:, :2] - arr[:-1, :2]
    norm = np.linalg.norm(step, axis=-1)
    valid = norm > 1e-4
    if valid.sum() < 2:
        return np.zeros((0,), dtype=np.float32)
    step = step[valid]
    angles = np.arctan2(step[:, 1], step[:, 0])
    return np.abs(_wrap_angle_np(np.diff(angles))).astype(np.float32)


def _trajectory_intent_class(
    traj: np.ndarray,
    *,
    turn_threshold_rad: float = 0.55,
    min_progress_m: float = 1.0,
) -> str:
    arr = np.asarray(traj, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[-1] != 3 or arr.shape[0] == 0:
        return "invalid"
    disp = arr[-1, :2] - arr[0, :2]
    if float(np.linalg.norm(disp)) < min_progress_m:
        return "stop"
    heading_delta = float(_wrap_angle_np(float(arr[-1, 2] - arr[0, 2])))
    path_angle = float(math.atan2(float(disp[1]), float(disp[0])) if float(np.linalg.norm(disp)) > 1e-4 else 0.0)
    turn_signal = heading_delta if abs(heading_delta) >= abs(path_angle) else path_angle
    if abs(turn_signal) < turn_threshold_rad:
        return "straight"
    return "left" if turn_signal > 0.0 else "right"


def support_quality_metrics(
    traj: np.ndarray,
    ref_traj: np.ndarray | None = None,
) -> dict[str, float]:
    arr = np.asarray(traj, dtype=np.float32)
    out = {
        "first_xy_error_m": 0.0,
        "first_heading_error_rad": 0.0,
        "mean_xy_error_m": 0.0,
        "endpoint_xy_error_m": 0.0,
        "max_xy_turn_rad": 0.0,
        "early_xy_turn_rad": 0.0,
        "max_step_m": 0.0,
        "semantic_final_heading_error_rad": 0.0,
        "semantic_path_angle_error_rad": 0.0,
        "semantic_endpoint_lateral_error_m": 0.0,
        "semantic_turn_class_match": 1.0,
    }
    if arr.ndim != 2 or arr.shape[-1] != 3 or arr.shape[0] == 0:
        return {key: float("inf") for key in out}
    step = arr[1:, :2] - arr[:-1, :2]
    if step.size:
        out["max_step_m"] = float(np.linalg.norm(step, axis=-1).max(initial=0.0))
    turns = _xy_turn_angles(arr)
    if turns.size:
        out["max_xy_turn_rad"] = float(turns.max(initial=0.0))
        out["early_xy_turn_rad"] = float(turns[:2].max(initial=0.0))
    if ref_traj is not None:
        ref = np.asarray(ref_traj, dtype=np.float32)
        n = min(arr.shape[0], ref.shape[0])
        if ref.ndim == 2 and ref.shape[-1] == 3 and n > 0:
            out["first_xy_error_m"] = float(np.linalg.norm(arr[0, :2] - ref[0, :2]))
            out["first_heading_error_rad"] = float(abs(_wrap_angle_np(float(arr[0, 2] - ref[0, 2]))))
            out["mean_xy_error_m"] = float(np.linalg.norm(arr[:n, :2] - ref[:n, :2], axis=-1).mean())
            out["endpoint_xy_error_m"] = float(np.linalg.norm(arr[n - 1, :2] - ref[n - 1, :2]))
            out["semantic_final_heading_error_rad"] = float(
                abs(_wrap_angle_np(float(arr[n - 1, 2] - ref[n - 1, 2])))
            )
            cand_disp = arr[n - 1, :2] - arr[0, :2]
            ref_disp = ref[n - 1, :2] - ref[0, :2]
            cand_norm = float(np.linalg.norm(cand_disp))
            ref_norm = float(np.linalg.norm(ref_disp))
            if cand_norm > 1e-4 and ref_norm > 1e-4:
                cand_angle = math.atan2(float(cand_disp[1]), float(cand_disp[0]))
                ref_angle = math.atan2(float(ref_disp[1]), float(ref_disp[0]))
                out["semantic_path_angle_error_rad"] = float(abs(_wrap_angle_np(cand_angle - ref_angle)))
            out["semantic_endpoint_lateral_error_m"] = float(abs(arr[n - 1, 1] - ref[n - 1, 1]))
            out["semantic_turn_class_match"] = float(
                _trajectory_intent_class(arr[:n]) == _trajectory_intent_class(ref[:n])
            )
    return out


def support_semantic_pass(
    traj: np.ndarray,
    ref_traj: np.ndarray | None,
    cfg: Any,
    *,
    metrics: Mapping[str, float] | None = None,
) -> bool:
    if not bool(cfg_value(cfg, "support_semantic_enable", False)) or ref_traj is None:
        return True
    quality_metrics = dict(metrics) if metrics is not None else support_quality_metrics(traj, ref_traj)
    if quality_metrics["semantic_turn_class_match"] < 0.5 and not bool(
        cfg_value(cfg, "support_allow_turn_class_mismatch", False)
    ):
        return False
    semantic_checks = (
        ("semantic_final_heading_error_rad", "support_max_semantic_final_heading_error_rad"),
        ("semantic_path_angle_error_rad", "support_max_semantic_path_angle_error_rad"),
        ("semantic_endpoint_lateral_error_m", "support_max_semantic_endpoint_lateral_error_m"),
    )
    for metric_name, cfg_name in semantic_checks:
        limit = float(cfg_value(cfg, cfg_name, np.inf))
        if quality_metrics[metric_name] > limit:
            return False
    return True


def support_quality_pass(
    candidate: CandidateRecord,
    ref_traj: np.ndarray | None,
    cfg: Any,
    *,
    ref_reward: float | None = None,
) -> bool:
    if not bool(cfg_value(cfg, "support_quality_enable", False)):
        return True
    if _source_matches(candidate.source, ("gt",)) and bool(cfg_value(cfg, "support_quality_exempt_gt", True)):
        return True
    if not support_reward_pass(candidate, ref_reward, cfg):
        return False
    metrics = support_quality_metrics(candidate.trajectory, ref_traj)
    checks = (
        ("first_xy_error_m", "support_max_first_xy_error_m"),
        ("first_heading_error_rad", "support_max_first_heading_error_rad"),
        ("max_xy_turn_rad", "support_max_xy_turn_rad"),
        ("early_xy_turn_rad", "support_max_early_xy_turn_rad"),
        ("max_step_m", "support_max_step_m"),
    )
    for metric_name, cfg_name in checks:
        limit = float(cfg_value(cfg, cfg_name, np.inf))
        if metrics[metric_name] > limit:
            return False
    if not support_semantic_pass(candidate.trajectory, ref_traj, cfg, metrics=metrics):
        return False
    return True


def support_reward_pass(
    candidate: CandidateRecord,
    ref_reward: float | None,
    cfg: Any,
) -> bool:
    """Check reward eligibility for non-GT training support candidates.

    The default is the old absolute reward floor. SG-FPS can opt into a
    GT-relative improver mode for scenes where GT is itself low scoring.
    """

    if _source_matches(candidate.source, ("gt",)) and bool(cfg_value(cfg, "support_quality_exempt_gt", True)):
        return True
    reward = float(candidate.reward)
    min_reward = float(cfg_value(cfg, "support_min_non_gt_reward", 0.0))
    if reward >= min_reward:
        return True

    mode = str(cfg_value(cfg, "support_reward_gate_mode", "absolute")).lower()
    if mode == "absolute":
        return False
    if ref_reward is None or not math.isfinite(float(ref_reward)):
        return False

    ref_reward_f = float(ref_reward)
    min_improver_reward = float(cfg_value(cfg, "support_min_non_gt_improver_reward", 0.70))
    improver_margin = float(cfg_value(cfg, "support_gt_improver_margin", 0.05))
    ref_max = float(cfg_value(cfg, "support_gt_improver_ref_max_reward", min_reward))

    if mode == "absolute_or_gt_improver":
        if ref_reward_f >= ref_max:
            return False
        return reward >= min_improver_reward and reward >= ref_reward_f + improver_margin
    if mode == "gt_relative":
        return reward >= min_improver_reward and reward >= ref_reward_f + improver_margin
    return False


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
        if components.shape[-1] < 3:
            raise ValueError("Tensor components must contain at least [EP, TTC, DDC].")
        return ep_w * components[..., 0] + ttc_w * components[..., 1] + ddc_w * components[..., 2] - feas_w * feas_cost
    ep = _component_value(components, "ego_progress", "ep")
    ttc = _component_value(components, "time_to_collision_within_bound", "ttc")
    ddc = _component_value(components, "driving_direction_compliance", "ddc")
    return ep_w * ep + ttc_w * ttc + ddc_w * ddc - feas_w * float(feas_cost)


def is_valid_candidate(components: Mapping[str, Any], feas: Mapping[str, Any], ref_components: Mapping[str, Any], cfg: Any) -> bool:
    return is_valid_metrics({**dict(components), **dict(feas)}, ref_components, cfg)


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


def _pareto_values_legacy(candidates: Sequence[CandidateRecord]) -> np.ndarray:
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
        key = trajectory_fingerprint(cand.trajectory)
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


def _quality_pareto_score(candidate: CandidateRecord, weights: Mapping[str, float], cfg: Any) -> float:
    reward_w = float(cfg_value(cfg, "support_score_reward_weight", 1.0))
    utility_w = float(cfg_value(cfg, "support_score_utility_weight", 0.5))
    pareto_w = float(cfg_value(cfg, "support_score_pareto_weight", 0.15))
    feas_w = float(cfg_value(cfg, "support_score_feas_penalty_weight", 0.1))
    utility = float(compute_utility(candidate.components, _feas_cost(candidate.feas), dict(weights)))
    return (
        reward_w * float(candidate.reward)
        + utility_w * utility
        + pareto_w * float(getattr(candidate, "_sg_fps_pareto_front", False))
        - feas_w * _feas_cost(candidate.feas)
        + 1e-4 * float(candidate.selection_score)
    )


def _trajectory_diversity_metrics(traj: np.ndarray, others: Sequence[np.ndarray]) -> dict[str, float]:
    arr = np.asarray(traj, dtype=np.float32)
    if not others:
        return {
            "min_endpoint_dist_m": float("inf"),
            "min_mean_xy_dist_m": float("inf"),
            "min_progress_curve_dist_m": float("inf"),
            "min_lateral_curve_dist_m": float("inf"),
            "min_heading_curve_dist_rad": float("inf"),
        }
    endpoint_dists = []
    mean_xy_dists = []
    progress_dists = []
    lateral_dists = []
    heading_dists = []
    for other_raw in others:
        other = np.asarray(other_raw, dtype=np.float32)
        n = min(arr.shape[0], other.shape[0])
        if n <= 0:
            continue
        endpoint_dists.append(float(np.linalg.norm(arr[n - 1, :2] - other[n - 1, :2])))
        mean_xy_dists.append(float(np.linalg.norm(arr[:n, :2] - other[:n, :2], axis=-1).mean()))
        progress_dists.append(float(np.mean(np.abs(arr[:n, 0] - other[:n, 0]))))
        lateral_dists.append(float(np.mean(np.abs(arr[:n, 1] - other[:n, 1]))))
        heading_dists.append(float(np.mean(np.abs(_wrap_angle_np(arr[:n, 2] - other[:n, 2])))))
    return {
        "min_endpoint_dist_m": min(endpoint_dists) if endpoint_dists else 0.0,
        "min_mean_xy_dist_m": min(mean_xy_dists) if mean_xy_dists else 0.0,
        "min_progress_curve_dist_m": min(progress_dists) if progress_dists else 0.0,
        "min_lateral_curve_dist_m": min(lateral_dists) if lateral_dists else 0.0,
        "min_heading_curve_dist_rad": min(heading_dists) if heading_dists else 0.0,
    }


def trajectory_diversity_score(traj: np.ndarray, others: Sequence[np.ndarray], cfg: Any) -> float:
    metrics = _trajectory_diversity_metrics(traj, others)
    if math.isinf(metrics["min_endpoint_dist_m"]):
        return 1.0
    endpoint_scale = float(cfg_value(cfg, "support_diversity_endpoint_scale_m", 3.0))
    mean_scale = float(cfg_value(cfg, "support_diversity_mean_scale_m", 1.5))
    progress_scale = float(cfg_value(cfg, "support_diversity_progress_scale_m", 1.5))
    lateral_scale = float(cfg_value(cfg, "support_diversity_lateral_scale_m", 0.8))
    heading_scale = float(cfg_value(cfg, "support_diversity_heading_scale_rad", 0.35))
    components = [
        min(metrics["min_endpoint_dist_m"] / max(endpoint_scale, 1e-6), 1.0),
        min(metrics["min_mean_xy_dist_m"] / max(mean_scale, 1e-6), 1.0),
        min(metrics["min_progress_curve_dist_m"] / max(progress_scale, 1e-6), 1.0),
        min(metrics["min_lateral_curve_dist_m"] / max(lateral_scale, 1e-6), 1.0),
        min(metrics["min_heading_curve_dist_rad"] / max(heading_scale, 1e-6), 1.0),
    ]
    weights = np.asarray(
        [
            float(cfg_value(cfg, "support_diversity_endpoint_weight", 0.30)),
            float(cfg_value(cfg, "support_diversity_mean_weight", 0.30)),
            float(cfg_value(cfg, "support_diversity_progress_weight", 0.15)),
            float(cfg_value(cfg, "support_diversity_lateral_weight", 0.15)),
            float(cfg_value(cfg, "support_diversity_heading_weight", 0.10)),
        ],
        dtype=np.float32,
    )
    weights = weights / max(float(weights.sum()), 1e-6)
    return float(np.dot(np.asarray(components, dtype=np.float32), weights))


def _append_diverse_ranked(
    selected: list[CandidateRecord],
    used: set[str],
    pool: Sequence[CandidateRecord],
    tag: str,
    limit: int,
    weights: Mapping[str, float],
    cfg: Any,
) -> None:
    diversity_w = float(cfg_value(cfg, "support_score_diversity_weight", 0.5))
    min_diversity = float(cfg_value(cfg, "support_min_trajectory_diversity_score", 0.0))
    remaining = sorted(pool, key=lambda c: _quality_pareto_score(c, weights, cfg), reverse=True)
    while remaining and len(selected) < limit:
        if not selected:
            candidate = remaining.pop(0)
        else:
            selected_traj = [np.asarray(c.trajectory, dtype=np.float32) for c in selected]
            best_idx = -1
            best_score = -float("inf")
            for idx, cand in enumerate(remaining):
                traj = np.asarray(cand.trajectory, dtype=np.float32)
                diversity = trajectory_diversity_score(traj, selected_traj, cfg)
                if diversity < min_diversity:
                    score = -float("inf")
                else:
                    score = _quality_pareto_score(cand, weights, cfg) + diversity_w * diversity
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx < 0 or not math.isfinite(best_score):
                break
            candidate = remaining.pop(best_idx)
        _append_unique(selected, used, [candidate], tag, limit)


def _select_quality_pareto_support(
    candidates: list[CandidateRecord],
    ref: dict,
    cfg: Any,
    weights: Mapping[str, float],
) -> list[CandidateRecord]:
    support_top_m = int(cfg_value(cfg, "support_top_m", cfg_value(cfg, "dpsi_top_m", 12)))
    if support_top_m <= 0:
        return []
    ref_traj = next((np.asarray(c.trajectory, dtype=np.float32) for c in candidates if _source_matches(c.source, ("gt",))), None)
    ref_reward = next((float(c.reward) for c in candidates if _source_matches(c.source, ("gt",))), None)
    evaluator_valid = np.asarray([is_valid_candidate(c.components, c.feas, ref, cfg) for c in candidates], dtype=np.bool_)
    quality = np.asarray([support_quality_pass(c, ref_traj, cfg, ref_reward=ref_reward) for c in candidates], dtype=np.bool_)
    selectable = evaluator_valid & quality
    selectable_candidates = [c for c, ok in zip(candidates, selectable) if ok]
    gt_candidates = _sort_by_reward([c for c, ok in zip(candidates, quality) if ok and _source_matches(c.source, ("gt",))])
    gt_candidate = gt_candidates[0] if gt_candidates else None
    gt_reward = float(gt_candidate.reward) if gt_candidate is not None else float("-inf")
    best_selectable_reward = max((float(c.reward) for c in selectable_candidates), default=float("-inf"))
    gt_floor = float(cfg_value(cfg, "support_gt_keep_min_reward", 0.85))
    gt_replace_margin = float(cfg_value(cfg, "support_gt_replace_margin", 0.05))
    keep_gt = bool(cfg_value(cfg, "support_keep_gt_by_default", True))
    if not selectable_candidates:
        if gt_candidate is not None and keep_gt:
            gt_candidate.support_tag = gt_candidate.support_tag or "gt_anchor"
            return [gt_candidate]
        return []

    values = _pareto_values_legacy(candidates)
    front_mask = pareto_front_mask(values, selectable)
    for candidate, is_front in zip(candidates, front_mask):
        setattr(candidate, "_sg_fps_pareto_front", bool(is_front))
    front_candidates = [c for c, ok in zip(candidates, front_mask) if ok]

    selected: list[CandidateRecord] = []
    used: set[str] = set()
    if gt_candidate is not None and keep_gt:
        gt_too_poor = gt_reward < gt_floor and best_selectable_reward >= gt_reward + gt_replace_margin
        if not gt_too_poor:
            _append_unique(
                selected,
                used,
                [gt_candidate],
                "gt_anchor",
                min(support_top_m, len(selected) + 1),
            )
    if bool(cfg_value(cfg, "support_include_il_anchor", False)):
        _append_unique(
            selected,
            used,
            _sort_by_reward(
                [
                    c
                    for c, ok in zip(candidates, selectable)
                    if ok and _source_matches(c.source, ("il", "recogdrive_stage3"))
                ]
            ),
            "il_anchor",
            min(support_top_m, len(selected) + 1),
        )

    high_pdms = float(cfg_value(cfg, "support_high_pdms_threshold", 0.95))
    top_pdms_count = int(cfg_value(cfg, "support_top_pdms_count", 3))
    pareto_count = int(cfg_value(cfg, "support_pareto_count", 5))
    diverse_count = max(0, support_top_m - len(selected))
    _append_unique(
        selected,
        used,
        _sort_by_reward([c for c in selectable_candidates if float(c.reward) >= high_pdms]),
        "best_pdms",
        min(support_top_m, len(selected) + top_pdms_count),
    )
    _append_unique(
        selected,
        used,
        sorted(front_candidates, key=lambda c: _quality_pareto_score(c, weights, cfg), reverse=True),
        "vector_pareto",
        min(support_top_m, len(selected) + pareto_count),
    )
    _append_diverse_ranked(
        selected,
        used,
        selectable_candidates,
        "diversity_max",
        min(support_top_m, len(selected) + diverse_count),
        weights,
        cfg,
    )
    if len(selected) < support_top_m:
        _append_unique(
            selected,
            used,
            sorted(selectable_candidates, key=lambda c: _quality_pareto_score(c, weights, cfg), reverse=True),
            "fallback_best",
            support_top_m,
        )
    return selected[:support_top_m]


def select_feasible_pareto_support(candidates: list[CandidateRecord], ref: dict, cfg: Any) -> list[CandidateRecord]:
    if not candidates:
        return []
    support_top_m = int(cfg_value(cfg, "support_top_m", cfg_value(cfg, "dpsi_top_m", 12)))
    weights = {
        "ep": float(cfg_value(cfg, "fp_ep_weight", 0.60)),
        "ttc": float(cfg_value(cfg, "fp_ttc_weight", 0.25)),
        "ddc": float(cfg_value(cfg, "fp_ddc_weight", 0.10)),
        "feas": float(cfg_value(cfg, "fp_feas_weight", 0.05)),
    }
    strategy = str(cfg_value(cfg, "support_selection_strategy", "quota")).lower()
    if strategy in {"quality_pareto", "pareto_diverse", "quality"}:
        return _select_quality_pareto_support(candidates, ref, cfg, weights)

    ref_traj = next((np.asarray(c.trajectory, dtype=np.float32) for c in candidates if _source_matches(c.source, ("gt",))), None)
    ref_reward = next((float(c.reward) for c in candidates if _source_matches(c.source, ("gt",))), None)
    valid = np.asarray([is_valid_candidate(c.components, c.feas, ref, cfg) for c in candidates], dtype=np.bool_)
    quality = np.asarray([support_quality_pass(c, ref_traj, cfg, ref_reward=ref_reward) for c in candidates], dtype=np.bool_)
    selectable = valid & quality
    values = _pareto_values_legacy(candidates)
    front = pareto_front_mask(values, selectable)
    ref_ep = _component_value(ref, "ego_progress", "ep", default=0.0)
    ep_margin = float(cfg_value(cfg, "safe_ep_improve_margin", 0.01))

    selected: list[CandidateRecord] = []
    used: set[str] = set()
    valid_candidates = [c for c, ok in zip(candidates, selectable) if ok]
    front_candidates = [c for c, ok in zip(candidates, front) if ok]

    def by_source(*names: str) -> list[CandidateRecord]:
        return [c for c in candidates if _source_matches(c.source, names)]

    def selectable_by_source(*names: str) -> list[CandidateRecord]:
        return [c for c, ok in zip(candidates, selectable) if ok and _source_matches(c.source, names)]

    quotas: list[tuple[str, list[CandidateRecord], int]] = [
        ("gt_anchor", _sort_by_reward(by_source("gt")), 1),
        ("il_anchor", _sort_by_reward(selectable_by_source("il", "recogdrive_stage3")), 1),
        ("best_pdms", _sort_by_reward(valid_candidates), 1),
        (
            "safe_ep_improver",
            _sort_by_reward([c for c in valid_candidates if _component_value(c.components, "ego_progress", "ep") >= ref_ep + ep_margin]),
            2,
        ),
        (
            "safety_repair",
            _sort_by_reward(
                [
                    c
                    for c in valid_candidates
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
            sorted(valid_candidates, key=lambda c: _component_value(c.components, "driving_direction_compliance", "ddc"), reverse=True),
            1,
        ),
        ("smooth_feasible", sorted(valid_candidates, key=lambda c: _feas_cost(c.feas)), 1),
        ("vector_pareto", _sort_by_utility(front_candidates, weights), 2),
        ("diversity_max", _diversity_order(valid_candidates), 1),
    ]
    for tag, pool, count in quotas:
        _append_unique(selected, used, pool, tag, min(support_top_m, len(selected) + count))
        if len(selected) >= support_top_m:
            return selected[:support_top_m]
    fallback = _sort_by_utility(front_candidates, weights) + _sort_by_utility(valid_candidates, weights)
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
    ref_traj = next((np.asarray(c.trajectory, dtype=np.float32) for c in candidates if _source_matches(c.source, ("gt",))), None)
    ref_reward = next((float(c.reward) for c in candidates if _source_matches(c.source, ("gt",))), None)
    all_values = _pareto_values_legacy(candidates) if candidates else np.zeros((0, 4), dtype=np.float32)
    valid_mask = np.asarray([is_valid_candidate(c.components, c.feas, ref, cfg) for c in candidates], dtype=np.bool_)
    quality_mask = np.asarray([support_quality_pass(c, ref_traj, cfg, ref_reward=ref_reward) for c in candidates], dtype=np.bool_)
    front_mask = pareto_front_mask(all_values, valid_mask) if candidates else np.zeros((0,), dtype=np.bool_)
    utilities = np.asarray([compute_utility(c.components, _feas_cost(c.feas), dict(cfg or {})) for c in candidates], dtype=np.float32)
    selected_tag_by_fp = {
        trajectory_fingerprint(c.trajectory): str(c.support_tag or "support")
        for c in selected
    }
    selected_fps = set(selected_tag_by_fp)
    selected_mask = np.asarray([trajectory_fingerprint(c.trajectory) in selected_fps for c in candidates], dtype=np.bool_)
    rewards = np.asarray([float(c.reward) for c in candidates], dtype=np.float32)
    selection_score = np.asarray([float(c.selection_score) for c in candidates], dtype=np.float32)
    raw_best_idx = int(np.argmax(rewards)) if rewards.size else 0
    valid_indices = np.flatnonzero(valid_mask)
    valid_best_idx = int(valid_indices[np.argmax(rewards[valid_indices])]) if valid_indices.size else raw_best_idx
    selected_indices = np.flatnonzero(selected_mask)
    selected_best_idx = int(selected_indices[np.argmax(rewards[selected_indices])]) if selected_indices.size else valid_best_idx
    gt_rewards = [float(c.reward) for c in candidates if _source_matches(c.source, ("gt",))]
    il_rewards = [float(c.reward) for c in candidates if _source_matches(c.source, ("il", "recogdrive_stage3"))]
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
    component_keys = (
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
            for key in component_keys
        },
        "sources": [str(c.source) for c in candidates],
        "valid_mask": valid_mask,
        "support_quality_mask": quality_mask,
        "selection_score": selection_score,
        "feasibility": {
            key: np.asarray([float(c.feas.get(key, 0.0)) for c in candidates], dtype=np.float32)
            for key in ("feas_cost", "early_kink_rate", "tail_reverse_rate", "curvature_violation_rate")
        },
        "support_tags": [
            selected_tag_by_fp.get(trajectory_fingerprint(c.trajectory), "")
            for c in candidates
        ],
        "support_indices": selected_indices.astype(np.int64).tolist(),
        "support_quality": {
            key: np.asarray(
                [support_quality_metrics(c.trajectory, ref_traj).get(key, 0.0) for c in candidates],
                dtype=np.float32,
            )
            for key in (
                "first_xy_error_m",
                "first_heading_error_rad",
                "mean_xy_error_m",
                "endpoint_xy_error_m",
                "max_xy_turn_rad",
                "early_xy_turn_rad",
                "max_step_m",
            )
        },
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


def _candidate_metrics(candidate: TrajectoryCandidate) -> dict[str, float]:
    if candidate.true_metrics is None:
        return {}
    return dict(candidate.true_metrics)


def _candidate_objectives(candidate: TrajectoryCandidate) -> np.ndarray:
    metrics = _candidate_metrics(candidate)
    return np.asarray(
        [
            metric_value(metrics, "ego_progress", "ep"),
            metric_value(metrics, "time_to_collision_within_bound", "ttc"),
            metric_value(metrics, "driving_direction_compliance", "ddc"),
            -metric_value(metrics, "feas_cost"),
        ],
        dtype=np.float32,
    )


def _assign_category(candidate: TrajectoryCandidate, category: str) -> None:
    candidate.support_category = category
    weight_map = {
        "safe_ep_improver": 1.0,
        "vector_pareto": 0.9,
        "safety_repair": 0.8,
        "ddc_repair": 0.8,
        "best_pdms": 0.7,
        "smooth_feasible": 0.6,
        "il_anchor": 0.5,
        "gt_anchor": 0.4,
        "diversity_max": 0.6,
    }
    candidate.support_weight = float(weight_map.get(category, candidate.support_weight))


def _append_candidate(
    selected: list[TrajectoryCandidate],
    used: set[str],
    pool: Iterable[TrajectoryCandidate],
    category: str,
    limit: int,
) -> None:
    for cand in pool:
        if len(selected) >= limit:
            return
        fp = trajectory_fingerprint(cand.trajectory)
        if fp in used:
            continue
        used.add(fp)
        _assign_category(cand, category)
        selected.append(cand)


def _trajectory_diversity_order(candidates: Sequence[TrajectoryCandidate]) -> list[TrajectoryCandidate]:
    if not candidates:
        return []
    selected = [max(candidates, key=lambda c: metric_value(_candidate_metrics(c), "pdms"))]
    remaining = [c for c in candidates if c is not selected[0]]
    while remaining:
        best_idx = 0
        best_dist = -1.0
        selected_trajs = np.stack([c.trajectory[:, :2] for c in selected], axis=0)
        for idx, cand in enumerate(remaining):
            dist = float(np.linalg.norm(selected_trajs - cand.trajectory[None, :, :2], axis=-1).mean(axis=-1).min())
            if dist > best_dist:
                best_dist = dist
                best_idx = idx
        selected.append(remaining.pop(best_idx))
    return selected


def build_support_set(
    evaluated_candidates: list[TrajectoryCandidate],
    ref_candidate: TrajectoryCandidate | None,
    cfg: Any,
) -> ParetoSupportArchive:
    if not evaluated_candidates:
        raise ValueError("Cannot build a support archive without evaluated candidates.")
    scene_token = evaluated_candidates[0].scene_token
    max_support = int(cfg_value(cfg, "support_top_m", 12))
    ref_metrics = _candidate_metrics(ref_candidate or evaluated_candidates[0])
    valid = np.asarray([is_valid_metrics(_candidate_metrics(c), ref_metrics, cfg) for c in evaluated_candidates], dtype=np.bool_)
    values = np.stack([_candidate_objectives(c) for c in evaluated_candidates], axis=0)
    front = pareto_front_mask(values, valid)
    for cand, ok, pf in zip(evaluated_candidates, valid, front):
        cand.valid_mask = bool(ok)
        cand.pareto_front = bool(pf)

    valid_candidates = [c for c, ok in zip(evaluated_candidates, valid) if ok]
    front_candidates = [c for c, ok in zip(evaluated_candidates, front) if ok]
    by_pdms = sorted(valid_candidates, key=lambda c: metric_value(_candidate_metrics(c), "pdms"), reverse=True)
    by_utility = sorted(valid_candidates, key=lambda c: compute_metric_utility(_candidate_metrics(c), cfg), reverse=True)
    ref_ep = metric_value(ref_metrics, "ego_progress", "ep")
    ref_ttc = metric_value(ref_metrics, "time_to_collision_within_bound", "ttc")
    ref_ddc = metric_value(ref_metrics, "driving_direction_compliance", "ddc")
    ref_feas = metric_value(ref_metrics, "feas_cost")
    ref_pdms = metric_value(ref_metrics, "pdms")
    selected: list[TrajectoryCandidate] = []
    used: set[str] = set()

    def source_pool(name: str) -> list[TrajectoryCandidate]:
        return sorted([c for c in evaluated_candidates if name in c.source], key=lambda c: metric_value(_candidate_metrics(c), "pdms"), reverse=True)

    _append_candidate(selected, used, source_pool("gt"), "gt_anchor", min(max_support, len(selected) + 1))
    _append_candidate(selected, used, source_pool("recogdrive_stage3") + source_pool("il"), "il_anchor", min(max_support, len(selected) + 1))
    _append_candidate(selected, used, by_pdms, "best_pdms", min(max_support, len(selected) + 1))

    safe_ep = [
        c
        for c in valid_candidates
        if metric_value(_candidate_metrics(c), "ego_progress", "ep") > ref_ep + float(cfg_value(cfg, "safe_ep_improve_margin", 0.01))
        and metric_value(_candidate_metrics(c), "time_to_collision_within_bound", "ttc") >= ref_ttc - float(cfg_value(cfg, "ttc_drop_tolerance", 0.03))
        and metric_value(_candidate_metrics(c), "driving_direction_compliance", "ddc") >= ref_ddc - float(cfg_value(cfg, "ddc_drop_tolerance", 0.01))
        and metric_value(_candidate_metrics(c), "feas_cost") <= ref_feas + float(cfg_value(cfg, "feas_drop_tolerance", 0.05))
    ]
    _append_candidate(selected, used, sorted(safe_ep, key=lambda c: metric_value(_candidate_metrics(c), "ego_progress", "ep"), reverse=True), "safe_ep_improver", min(max_support, len(selected) + 2))

    safety_repair = [
        c
        for c in valid_candidates
        if metric_value(_candidate_metrics(c), "pdms") >= ref_pdms - float(cfg_value(cfg, "repair_pdms_tolerance", 0.05))
        and (
            metric_value(_candidate_metrics(c), "no_at_fault_collisions") > metric_value(ref_metrics, "no_at_fault_collisions")
            or metric_value(_candidate_metrics(c), "drivable_area_compliance") > metric_value(ref_metrics, "drivable_area_compliance")
            or metric_value(_candidate_metrics(c), "time_to_collision_within_bound") > ref_ttc
            or metric_value(_candidate_metrics(c), "driving_direction_compliance") > ref_ddc
        )
    ]
    _append_candidate(selected, used, sorted(safety_repair, key=lambda c: metric_value(_candidate_metrics(c), "pdms"), reverse=True), "safety_repair", min(max_support, len(selected) + 1))

    ddc_repair = [
        c
        for c in valid_candidates
        if metric_value(_candidate_metrics(c), "driving_direction_compliance", "ddc") > ref_ddc + float(cfg_value(cfg, "ddc_repair_margin", 0.01))
        and metric_value(_candidate_metrics(c), "pdms") >= ref_pdms - float(cfg_value(cfg, "repair_pdms_tolerance", 0.05))
    ]
    _append_candidate(selected, used, sorted(ddc_repair, key=lambda c: metric_value(_candidate_metrics(c), "driving_direction_compliance", "ddc"), reverse=True), "ddc_repair", min(max_support, len(selected) + 1))

    smooth = [c for c in valid_candidates if metric_value(_candidate_metrics(c), "pdms") >= ref_pdms - float(cfg_value(cfg, "repair_pdms_tolerance", 0.05))]
    _append_candidate(selected, used, sorted(smooth, key=lambda c: metric_value(_candidate_metrics(c), "feas_cost")), "smooth_feasible", min(max_support, len(selected) + 1))
    _append_candidate(selected, used, sorted(front_candidates, key=lambda c: compute_metric_utility(_candidate_metrics(c), cfg), reverse=True), "vector_pareto", min(max_support, len(selected) + 2))
    _append_candidate(selected, used, _trajectory_diversity_order(valid_candidates), "diversity_max", min(max_support, len(selected) + 1))
    _append_candidate(selected, used, sorted(front_candidates, key=lambda c: compute_metric_utility(_candidate_metrics(c), cfg), reverse=True) + by_utility + by_pdms, "fallback_best", max_support)

    hard_negatives = [
        c
        for c in evaluated_candidates
        if (not c.valid_mask and metric_value(_candidate_metrics(c), "pdms") > ref_pdms - 0.05)
        or metric_value(_candidate_metrics(c), "feas_cost") > float(cfg_value(cfg, "hard_negative_feas_cost", 0.30))
    ]
    oracle_best = {
        "best_pdms": max((metric_value(_candidate_metrics(c), "pdms") for c in evaluated_candidates), default=0.0),
        "best_utility": max((compute_metric_utility(_candidate_metrics(c), cfg) for c in evaluated_candidates), default=0.0),
        "best_ep": max((metric_value(_candidate_metrics(c), "ego_progress", "ep") for c in evaluated_candidates), default=0.0),
        "best_ddc": max((metric_value(_candidate_metrics(c), "driving_direction_compliance", "ddc") for c in evaluated_candidates), default=0.0),
        "best_feas": min((metric_value(_candidate_metrics(c), "feas_cost") for c in evaluated_candidates), default=0.0),
    }
    operator_stats = {
        "candidate_count": len(evaluated_candidates),
        "valid_count": int(valid.sum()),
        "pareto_front_count": int(front.sum()),
        "support_count": len(selected),
        "support_category_counts": dict(Counter(str(c.support_category) for c in selected)),
    }
    return ParetoSupportArchive(
        scene_token=scene_token,
        reference={
            "gt_metrics": next((_candidate_metrics(c) for c in evaluated_candidates if c.source == "gt"), {}),
            "recogdrive_stage3_metrics": next((_candidate_metrics(c) for c in evaluated_candidates if c.source == "recogdrive_stage3"), {}),
            "ref_source": (ref_candidate.source if ref_candidate is not None else evaluated_candidates[0].source),
            "ref_trajectory": (ref_candidate.trajectory if ref_candidate is not None else evaluated_candidates[0].trajectory),
            "ref_metrics": ref_metrics,
        },
        seed_candidates=[c for c in evaluated_candidates if c.source in {"gt", "recogdrive_stage3", "ddv2", "driveor"}],
        generated_candidates=[c for c in evaluated_candidates if c.source not in {"gt", "recogdrive_stage3", "ddv2", "driveor"}],
        evaluated_candidates=evaluated_candidates,
        support_set=selected,
        hard_negatives=hard_negatives,
        operator_stats=operator_stats,
        archive_hypervolume=float(front.sum()),
        oracle_best=oracle_best,
        build_metadata={"timestamp": time.time(), "version": 3},
    )
