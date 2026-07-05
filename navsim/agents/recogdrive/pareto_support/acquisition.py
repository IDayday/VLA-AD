from __future__ import annotations

from typing import Any

import numpy as np

from .dataclasses import TrajectoryCandidate
from .metrics import cfg_value, compute_metric_utility, metric_value


def _trajectory_distance(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return float("inf")
    return float(np.linalg.norm(a[:n, :2] - b[:n, :2], axis=-1).mean())


def _fallback_score(candidate: TrajectoryCandidate, archive: list[TrajectoryCandidate], cfg: Any = None) -> float:
    metrics = candidate.true_metrics or candidate.cheap_metrics or {}
    score = compute_metric_utility(metrics, cfg) if metrics else 0.0
    if archive:
        score += 0.05 * min(_trajectory_distance(candidate.trajectory, prev.trajectory) for prev in archive)
    score -= 0.2 * metric_value(metrics, "feas_cost", default=0.0)
    return float(score)


def select_candidates_for_evaluation(
    candidates: list[TrajectoryCandidate],
    archive: list[TrajectoryCandidate],
    scorer: Any = None,
    cfg: Any = None,
) -> list[TrajectoryCandidate]:
    if not candidates:
        return []
    budget = int(cfg_value(cfg, "evaluator_budget", cfg_value(cfg, "true_eval_budget", 16)))
    duplicate_threshold = float(cfg_value(cfg, "acquisition_duplicate_distance_m", 0.15))
    selected: list[TrajectoryCandidate] = []
    selected_trajs = [c.trajectory for c in archive]
    scored: list[tuple[float, TrajectoryCandidate]] = []
    for candidate in candidates:
        if selected_trajs and min(_trajectory_distance(candidate.trajectory, prev) for prev in selected_trajs) < duplicate_threshold:
            continue
        if scorer is not None:
            pred = scorer(candidate)
            if isinstance(pred, dict):
                candidate.scorer_pred.update({str(k): float(v) for k, v in pred.items()})
                valid_prob = float(pred.get("valid_prob", pred.get("p_valid", 1.0)))
                score = valid_prob * float(pred.get("utility", pred.get("pdms_value", 0.0)))
            else:
                score = float(pred)
        else:
            score = _fallback_score(candidate, archive, cfg)
        scored.append((score, candidate))
    category_need = {
        "progress_tune": 2,
        "ddc_repair": 1,
        "yield_delay": 1,
        "feas_repair": 1,
    }
    used = set()
    for operator, need in category_need.items():
        pool = [(score, cand) for score, cand in scored if cand.operator == operator]
        for _, cand in sorted(pool, key=lambda item: item[0], reverse=True)[:need]:
            if cand.candidate_id not in used and len(selected) < budget:
                selected.append(cand)
                used.add(cand.candidate_id)
    for _, cand in sorted(scored, key=lambda item: item[0], reverse=True):
        if len(selected) >= budget:
            break
        if cand.candidate_id in used:
            continue
        selected.append(cand)
        used.add(cand.candidate_id)
    return selected
