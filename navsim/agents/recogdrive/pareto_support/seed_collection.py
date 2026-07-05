from __future__ import annotations

from typing import Any, Mapping

from .dataclasses import TrajectoryCandidate
from .external_generators import BaseTrajectoryGenerator, build_generators
from .metrics import compute_true_metrics_batch, merge_true_and_feasibility_metrics


def collect_seed_candidates(
    scene_token: str,
    features: Mapping[str, Any],
    targets: Mapping[str, Any],
    cfg: Any,
    generators: list[BaseTrajectoryGenerator] | None = None,
) -> list[TrajectoryCandidate]:
    generators = generators if generators is not None else build_generators(cfg)
    candidates: list[TrajectoryCandidate] = []
    for generator in generators:
        records = generator.generate(scene_token, features, targets, int(getattr(generator, "top_k", 8)))
        candidates.extend(records)
    return [candidate for candidate in candidates if candidate.trajectory.size and candidate.scene_token == str(scene_token)]


def ensure_evaluator_verified(
    candidates: list[TrajectoryCandidate],
    metric_context: Any,
    cfg: Any,
) -> list[TrajectoryCandidate]:
    missing = [idx for idx, cand in enumerate(candidates) if cand.true_metrics is None or "pdms" not in cand.true_metrics]
    if missing:
        rows = compute_true_metrics_batch(
            [candidates[idx].trajectory for idx in missing],
            [candidates[idx].scene_token for idx in missing],
            metric_context,
            cfg,
        )
        for idx, row in zip(missing, rows):
            candidates[idx].true_metrics = row
    for candidate in candidates:
        candidate.true_metrics = merge_true_and_feasibility_metrics(candidate.true_metrics, candidate.trajectory, cfg)
    return candidates
