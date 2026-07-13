from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


TRAJECTORY_SOURCES = {
    "gt",
    "recogdrive_stage3",
    "ddv2",
    "driveor",
    "path_speed_recombine",
    "progress_tune",
    "yield_delay",
    "ddc_repair",
    "feas_repair",
    "control_expand",
    "il",
    "current_policy",
    "unknown",
}

SUPPORT_CATEGORIES = {
    "gt_anchor",
    "il_anchor",
    "best_pdms",
    "best_pdms_valid",
    "safe_ep_improver",
    "safety_repair",
    "ddc_repair",
    "smooth_feasible",
    "vector_pareto",
    "mode_pareto",
    "mode_support",
    "diversity_max",
    "diversity_max_valid",
    "hard_negative",
    "fallback_best",
    None,
}


def ensure_trajectory_array(value: Any, *, name: str = "trajectory") -> np.ndarray:
    if hasattr(value, "poses"):
        value = getattr(value, "poses")
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [H, 3], got {arr.shape}.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def trajectory_fingerprint(traj: np.ndarray) -> str:
    arr = np.asarray(traj, dtype=np.float32)
    return hashlib.sha1(arr.tobytes()).hexdigest()


@dataclass
class CandidateRecord:
    """Legacy v3 elite-buffer candidate used by existing ReCogDrive scripts."""

    trajectory: np.ndarray
    source: str
    token: str
    components: dict[str, float]
    reward: float
    feas: dict[str, float]
    selection_score: float
    support_tag: str = ""
    parent_id: str = ""

    def __post_init__(self) -> None:
        self.trajectory = ensure_trajectory_array(self.trajectory)
        self.source = str(self.source)
        self.token = str(self.token)
        self.components = {str(k): float(v) for k, v in dict(self.components or {}).items()}
        self.reward = float(self.reward)
        self.feas = {str(k): float(v) for k, v in dict(self.feas or {}).items()}
        self.selection_score = float(self.selection_score)
        self.support_tag = str(self.support_tag or "")
        self.parent_id = str(self.parent_id or "")


@dataclass
class TrajectoryCandidate:
    scene_token: str
    candidate_id: str
    source: str
    parent_ids: list[str] = field(default_factory=list)
    operator: str = ""
    trajectory: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    path_atom_id: Optional[str] = None
    speed_atom_id: Optional[str] = None
    cheap_metrics: dict[str, float] = field(default_factory=dict)
    scorer_pred: dict[str, float] = field(default_factory=dict)
    true_metrics: Optional[dict[str, float]] = None
    valid_mask: bool = False
    pareto_front: bool = False
    support_category: Optional[str] = None
    support_weight: float = 1.0
    repair_distance: float = 0.0
    duplicate_group_id: Optional[str] = None
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.scene_token = str(self.scene_token)
        self.candidate_id = str(self.candidate_id)
        self.source = str(self.source or "unknown")
        if self.source not in TRAJECTORY_SOURCES:
            self.source_metadata.setdefault("original_source", self.source)
            self.source = "unknown"
        if self.support_category not in SUPPORT_CATEGORIES:
            raise ValueError(f"Unsupported support_category={self.support_category!r}.")
        self.trajectory = ensure_trajectory_array(self.trajectory)
        self.parent_ids = [str(item) for item in self.parent_ids]
        self.cheap_metrics = {str(k): float(v) for k, v in dict(self.cheap_metrics or {}).items()}
        self.scorer_pred = {str(k): float(v) for k, v in dict(self.scorer_pred or {}).items()}
        if self.true_metrics is not None:
            self.true_metrics = {str(k): float(v) for k, v in dict(self.true_metrics).items()}
        self.valid_mask = bool(self.valid_mask)
        self.pareto_front = bool(self.pareto_front)
        self.support_weight = float(self.support_weight)
        self.repair_distance = float(self.repair_distance)

    def to_legacy_record(self) -> CandidateRecord:
        metrics = dict(self.true_metrics or {})
        feas = {
            k: float(v)
            for k, v in metrics.items()
            if k in {"feas_cost", "early_kink_rate", "tail_reverse_rate", "curvature_violation_rate"}
        }
        reward = float(metrics.get("pdms", metrics.get("score", self.scorer_pred.get("pdms_value", 0.0))))
        return CandidateRecord(
            trajectory=self.trajectory,
            source=self.source,
            token=self.scene_token,
            components=metrics,
            reward=reward,
            feas=feas,
            selection_score=float(metrics.get("utility", self.scorer_pred.get("utility", reward))),
            support_tag=str(self.support_category or ""),
            parent_id=",".join(self.parent_ids),
        )

    @classmethod
    def from_legacy_record(cls, record: CandidateRecord, *, candidate_id: str | None = None) -> "TrajectoryCandidate":
        return cls(
            scene_token=record.token,
            candidate_id=candidate_id or f"{record.token}:{record.source}:{trajectory_fingerprint(record.trajectory)[:12]}",
            source=record.source if record.source in TRAJECTORY_SOURCES else "unknown",
            parent_ids=[record.parent_id] if record.parent_id else [],
            operator=record.source,
            trajectory=record.trajectory,
            cheap_metrics={},
            true_metrics={**record.components, **record.feas},
            valid_mask=False,
            pareto_front=False,
            support_category=record.support_tag or None,
            support_weight=1.0,
        )


@dataclass
class ParetoSupportArchive:
    scene_token: str
    reference: dict[str, Any]
    seed_candidates: list[TrajectoryCandidate] = field(default_factory=list)
    generated_candidates: list[TrajectoryCandidate] = field(default_factory=list)
    evaluated_candidates: list[TrajectoryCandidate] = field(default_factory=list)
    support_set: list[TrajectoryCandidate] = field(default_factory=list)
    hard_negatives: list[TrajectoryCandidate] = field(default_factory=list)
    operator_stats: dict[str, Any] = field(default_factory=dict)
    archive_hypervolume: float = 0.0
    oracle_best: dict[str, Any] = field(default_factory=dict)
    build_metadata: dict[str, Any] = field(default_factory=dict)

    def all_candidates(self) -> list[TrajectoryCandidate]:
        by_id: dict[str, TrajectoryCandidate] = {}
        for candidate in self.seed_candidates + self.generated_candidates + self.evaluated_candidates:
            by_id[candidate.candidate_id] = candidate
        return list(by_id.values())
