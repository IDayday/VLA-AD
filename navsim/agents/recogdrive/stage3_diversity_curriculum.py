from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional

import torch


@dataclass(frozen=True)
class DiversityCapacityBatch:
    support_pairwise_ade_m: torch.Tensor
    reference_mode_count: torch.Tensor


@dataclass(frozen=True)
class DiversityFrontierOutput:
    energy: torch.Tensor
    mode_capacity: torch.Tensor
    coverage_ratio: torch.Tensor
    coverage_gap: torch.Tensor
    bonus: torch.Tensor
    active_mask: torch.Tensor


class Stage3DiversityCapacityCache:
    VERSION = 2

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if payload is None:
            if path is None or not str(path):
                raise ValueError("Diversity-capacity curriculum requires a non-empty cache path.")
            self.path = Path(path).expanduser().resolve()
            if not self.path.is_file():
                raise FileNotFoundError(f"Diversity-capacity cache not found: {self.path}")
            try:
                payload = torch.load(self.path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(self.path, map_location="cpu")
        else:
            self.path = Path(path).expanduser().resolve() if path else None
        if not isinstance(payload, Mapping):
            raise TypeError("Diversity-capacity cache payload must be a mapping.")
        self.metadata = dict(payload.get("metadata", {}))
        version = int(self.metadata.get("version", payload.get("version", 0)))
        if version != self.VERSION:
            raise ValueError(
                f"Unsupported diversity-capacity cache version {version}; expected {self.VERSION}."
            )
        records = payload.get("records")
        if not isinstance(records, Mapping) or not records:
            raise ValueError("Diversity-capacity cache must contain a non-empty records mapping.")
        self.records = {
            str(token): self._validate_record(str(token), record)
            for token, record in records.items()
        }
        canonical = json.dumps(self.metadata, sort_keys=True, separators=(",", ":"), default=str)
        self.metadata_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_record(token: str, record: Any) -> Dict[str, float]:
        if not isinstance(record, Mapping):
            raise TypeError(f"Diversity-capacity record {token!r} must be a mapping.")
        required = ("support_pairwise_ade_m", "reference_mode_count")
        missing = [name for name in required if name not in record]
        if missing:
            raise KeyError(f"Diversity-capacity record {token!r} is missing {missing}.")
        dispersion = float(record["support_pairwise_ade_m"])
        mode_count = float(record["reference_mode_count"])
        if not math.isfinite(dispersion) or dispersion < 0.0:
            raise ValueError(
                f"Diversity-capacity record {token!r} has invalid support_pairwise_ade_m={dispersion}."
            )
        if not math.isfinite(mode_count) or mode_count < 1.0:
            raise ValueError(
                f"Diversity-capacity record {token!r} has invalid reference_mode_count={mode_count}."
            )
        return {
            "support_pairwise_ade_m": dispersion,
            "reference_mode_count": mode_count,
        }

    def get(
        self,
        tokens: list[str],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> DiversityCapacityBatch:
        normalized = [str(token) for token in tokens]
        missing = [token for token in normalized if token not in self.records]
        if missing:
            preview = ", ".join(repr(token) for token in missing[:8])
            raise KeyError(
                f"Diversity-capacity cache is missing {len(missing)} requested token(s): {preview}."
            )
        rows = [self.records[token] for token in normalized]
        return DiversityCapacityBatch(
            support_pairwise_ade_m=torch.tensor(
                [row["support_pairwise_ade_m"] for row in rows],
                device=device,
                dtype=dtype,
            ),
            reference_mode_count=torch.tensor(
                [row["reference_mode_count"] for row in rows],
                device=device,
                dtype=dtype,
            ),
        )


def compute_capacity_normalized_frontier_energy(
    base_energy: torch.Tensor,
    feasible_mask: torch.Tensor,
    group_pairwise_ade_m: torch.Tensor,
    capacity: DiversityCapacityBatch,
    *,
    gap_weight: float,
    dispersion_floor: float,
    priority_mode: Literal[
        "credit_multiplicative", "additive_preservation"
    ] = "credit_multiplicative",
) -> DiversityFrontierOutput:
    """Combines a policy frontier with its normalized diversity gap.

    A coverage gap is not itself a policy-learning signal.  In particular, a
    scene with zero bidirectional advantage has zero REINFORCE gradient, so
    the default mode only increases the priority of an existing frontier.  The
    explicit additive ablation can instead schedule such a scene for frozen
    Stage2 KL preservation; callers must not interpret that uplift as new
    policy credit.
    """
    if base_energy.ndim != 1:
        raise ValueError("base_energy must have shape [B].")
    if feasible_mask.ndim != 2 or feasible_mask.shape[0] != base_energy.shape[0]:
        raise ValueError("feasible_mask must have shape [B, G].")
    expected = base_energy.shape
    for name, value in (
        ("group_pairwise_ade_m", group_pairwise_ade_m),
        ("support_pairwise_ade_m", capacity.support_pairwise_ade_m),
        ("reference_mode_count", capacity.reference_mode_count),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {tuple(expected)}, got {tuple(value.shape)}.")
    if gap_weight < 0.0:
        raise ValueError("gap_weight must be non-negative.")
    if dispersion_floor <= 0.0:
        raise ValueError("dispersion_floor must be positive.")
    if priority_mode not in {"credit_multiplicative", "additive_preservation"}:
        raise ValueError(f"Unsupported diversity priority mode: {priority_mode!r}.")

    support_dispersion = capacity.support_pairwise_ade_m.to(base_energy).clamp_min(0.0)
    group_dispersion = group_pairwise_ade_m.to(base_energy).clamp_min(0.0)
    mode_count = capacity.reference_mode_count.to(base_energy).clamp_min(1.0)
    mode_capacity = (1.0 - mode_count.reciprocal()).clamp(0.0, 1.0)
    coverage_ratio = (
        (group_dispersion + float(dispersion_floor))
        / (support_dispersion + float(dispersion_floor))
    ).clamp(0.0, 1.0)
    coverage_gap = mode_capacity * (1.0 - coverage_ratio)
    all_feasible = feasible_mask.all(dim=1)
    has_capacity = (mode_count > 1.0) & (support_dispersion > 0.0)
    active_mask = all_feasible & has_capacity
    safe_base_energy = torch.nan_to_num(base_energy)
    weighted_gap = float(gap_weight) * coverage_gap * active_mask.to(base_energy)
    if priority_mode == "additive_preservation":
        # Explicit ablation: this can schedule a zero-advantage scene so the
        # frozen Stage2 KL preserves its learned support, but it cannot create
        # a new policy-gradient direction.
        energy = torch.nan_to_num(safe_base_energy + weighted_gap)
    else:
        energy = torch.nan_to_num(safe_base_energy * (1.0 + weighted_gap))
    # Keep the existing diagnostic/checkpoint field.  It records the realized
    # priority uplift under either ablation mode.
    bonus = energy - safe_base_energy
    return DiversityFrontierOutput(
        energy=energy,
        mode_capacity=mode_capacity,
        coverage_ratio=coverage_ratio,
        coverage_gap=coverage_gap,
        bonus=bonus,
        active_mask=active_mask,
    )
