from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import torch


@dataclass
class CanonicalMetricBatch:
    scalar: torch.Tensor
    ep: torch.Tensor
    ttc: torch.Tensor
    quality: torch.Tensor
    nc: torch.Tensor
    dac: torch.Tensor
    ddc_guard_value: torch.Tensor
    tlc: Optional[torch.Tensor]
    diagnostics: Dict[str, torch.Tensor]


class Stage3MetricAdapter:
    """Strictly maps NAVSIM evaluator output to the LFP metric contract."""

    _COMMON_ALIASES = {
        "ep": ("ego_progress", "ep", "progress"),
        "ttc": ("time_to_collision_within_bound", "ttc"),
        "nc": ("no_at_fault_collisions", "nc"),
        "dac": ("drivable_area_compliance", "dac"),
    }
    _RAW_DDC_ALIASES = (
        "raw_driving_direction_compliance",
        "driving_direction_compliance_raw",
        "raw_ddc",
    )
    _FILTERED_DDC_ALIASES = (
        "driving_direction_compliance",
        "filtered_driving_direction_compliance",
        "ddc",
    )

    def __init__(self, benchmark: str = "navsim_v1") -> None:
        benchmark = str(benchmark)
        if benchmark not in {"navsim_v1", "navsim_v2"}:
            raise ValueError("benchmark must be 'navsim_v1' or 'navsim_v2'.")
        self.benchmark = benchmark

    @staticmethod
    def _find(
        metrics: Mapping[str, Any],
        aliases: tuple[str, ...],
        canonical_name: str,
    ) -> tuple[Any, str]:
        lowered = {str(key).lower(): key for key in metrics}
        for alias in aliases:
            original = lowered.get(alias.lower())
            if original is not None:
                return metrics[original], str(original)
        raise KeyError(
            f"Missing required NAVSIM metric {canonical_name!r}; expected one of "
            f"{list(aliases)}, available={sorted(str(key) for key in metrics)}."
        )

    @staticmethod
    def _matrix(
        value: Any,
        *,
        name: str,
        batch_size: Optional[int],
        group_size: Optional[int],
    ) -> torch.Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        tensor = tensor.float()
        if batch_size is not None or group_size is not None:
            if batch_size is None or group_size is None:
                raise ValueError("batch_size and group_size must be provided together.")
            expected = int(batch_size) * int(group_size)
            if tensor.numel() != expected:
                raise ValueError(
                    f"Metric {name!r} has {tensor.numel()} values, expected {expected} "
                    f"for shape [{batch_size}, {group_size}]."
                )
            tensor = tensor.reshape(int(batch_size), int(group_size))
        elif tensor.ndim != 2:
            raise ValueError(f"Metric {name!r} must have shape [B, G], got {tuple(tensor.shape)}.")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Metric {name!r} contains NaN or Inf.")
        return tensor

    def canonicalize(
        self,
        metrics: Mapping[str, Any],
        *,
        batch_size: Optional[int] = None,
        group_size: Optional[int] = None,
    ) -> CanonicalMetricBatch:
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping of evaluator field names to tensors.")

        field_mapping: Dict[str, str] = {}

        def required(name: str, aliases: tuple[str, ...]) -> torch.Tensor:
            value, source = self._find(metrics, aliases, name)
            field_mapping[name] = source
            return self._matrix(
                value,
                name=source,
                batch_size=batch_size,
                group_size=group_size,
            )

        ep = required("ep", self._COMMON_ALIASES["ep"])
        ttc = required("ttc", self._COMMON_ALIASES["ttc"])
        nc = required("nc", self._COMMON_ALIASES["nc"])
        dac = required("dac", self._COMMON_ALIASES["dac"])

        try:
            ddc_value, ddc_source = self._find(metrics, self._RAW_DDC_ALIASES, "raw_ddc")
            ddc_fallback = False
        except KeyError:
            ddc_value, ddc_source = self._find(
                metrics,
                self._FILTERED_DDC_ALIASES,
                "driving_direction_compliance",
            )
            ddc_fallback = True
        field_mapping["ddc_guard_value"] = ddc_source
        ddc = self._matrix(
            ddc_value,
            name=ddc_source,
            batch_size=batch_size,
            group_size=group_size,
        )

        if self.benchmark == "navsim_v1":
            scalar = required("scalar", ("pdms", "score", "pdm_score", "official_pdms"))
            quality = required("quality", ("history_comfort", "comfort", "comfortable"))
            tlc = None
        else:
            scalar = required(
                "scalar",
                ("epdms", "score", "official_epdms", "one_stage_epdms"),
            )
            lane_keeping = required("lane_keeping", ("lane_keeping", "lane_keeping_compliance", "lk"))
            history_comfort = required("history_comfort", ("history_comfort", "comfort", "hc"))
            extended_comfort = required(
                "extended_comfort",
                ("two_frame_extended_comfort", "extended_comfort", "ec"),
            )
            quality = (lane_keeping + history_comfort + extended_comfort) / 3.0
            tlc = required(
                "tlc",
                ("filtered_traffic_light_compliance", "traffic_light_compliance", "tlc"),
            )

        canonical_tensors = {
            "scalar": scalar,
            "ep": ep,
            "ttc": ttc,
            "quality": quality,
            "nc": nc,
            "dac": dac,
            "ddc_guard_value": ddc,
        }
        if tlc is not None:
            canonical_tensors["tlc"] = tlc
        mismatched = {
            name: tuple(tensor.shape)
            for name, tensor in canonical_tensors.items()
            if tensor.shape != scalar.shape
        }
        if mismatched:
            raise ValueError(
                f"Canonical metric tensors must share shape {tuple(scalar.shape)}; "
                f"mismatched={mismatched}."
            )

        ref = scalar
        diagnostics: Dict[str, torch.Tensor] = {
            "ddc_fallback_ratio": ref.new_tensor(float(ddc_fallback)),
            "raw_ddc_available_ratio": ref.new_tensor(float(not ddc_fallback)),
        }
        # Retained for cache metadata/reporting without putting strings in the logger.
        diagnostics["benchmark_code"] = ref.new_tensor(1.0 if self.benchmark == "navsim_v1" else 2.0)
        self.last_field_mapping = field_mapping
        return CanonicalMetricBatch(
            scalar=scalar,
            ep=ep,
            ttc=ttc,
            quality=quality,
            nc=nc,
            dac=dac,
            ddc_guard_value=ddc,
            tlc=tlc,
            diagnostics=diagnostics,
        )

    def feasible_mask(
        self,
        metrics: CanonicalMetricBatch,
        gt_ddc: torch.Tensor,
        cfg: Any,
    ) -> torch.Tensor:
        gt_ddc = gt_ddc.to(device=metrics.scalar.device, dtype=metrics.scalar.dtype)
        if gt_ddc.ndim != 1 or gt_ddc.shape[0] != metrics.scalar.shape[0]:
            raise ValueError(
                f"gt_ddc must have shape [B], got {tuple(gt_ddc.shape)} for B={metrics.scalar.shape[0]}."
            )
        feasible = torch.ones_like(metrics.scalar, dtype=torch.bool)
        if bool(getattr(cfg, "require_nc", True)):
            feasible &= metrics.nc >= 1.0
        if bool(getattr(cfg, "require_dac", True)):
            feasible &= metrics.dac >= 1.0
        tolerance = float(getattr(cfg, "ddc_gt_tolerance", 0.01))
        feasible &= metrics.ddc_guard_value >= gt_ddc[:, None] - tolerance
        if self.benchmark == "navsim_v2" and bool(getattr(cfg, "v2_require_tlc", True)):
            if metrics.tlc is None:
                raise KeyError("NAVSIM v2 feasibility requires traffic_light_compliance.")
            feasible &= metrics.tlc >= 1.0
        return feasible

    @staticmethod
    def pareto_objectives(metrics: CanonicalMetricBatch) -> torch.Tensor:
        return torch.stack((metrics.ep, metrics.ttc, metrics.quality), dim=-1)
