from types import SimpleNamespace

import torch

from navsim.agents.recogdrive.stage3_metric_adapter import CanonicalMetricBatch, Stage3MetricAdapter


def _metrics(ddc: torch.Tensor) -> CanonicalMetricBatch:
    ones = torch.ones_like(ddc)
    return CanonicalMetricBatch(
        scalar=ones,
        ep=ones,
        ttc=ones,
        quality=ones,
        nc=ones,
        dac=ones,
        ddc_guard_value=ddc,
        tlc=None,
        diagnostics={},
    )


def test_ddc_guard_is_gt_relative_not_stage2_relative() -> None:
    adapter = Stage3MetricAdapter("navsim_v1")
    cfg = SimpleNamespace(require_nc=True, require_dac=True, ddc_gt_tolerance=0.01)
    feasible = adapter.feasible_mask(_metrics(torch.tensor([[0.50, 0.49, 0.48]])), torch.tensor([0.50]), cfg)
    assert feasible.tolist() == [[True, True, False]]


def test_gt_ddc_half_candidate_half_passes_even_if_reference_ddc_is_higher() -> None:
    adapter = Stage3MetricAdapter("navsim_v1")
    cfg = SimpleNamespace(require_nc=True, require_dac=True, ddc_gt_tolerance=0.0)
    candidate = _metrics(torch.tensor([[0.5]]))
    # A selected Stage2 reference DDC of 1.0 is intentionally irrelevant here.
    assert adapter.feasible_mask(candidate, torch.tensor([0.5]), cfg).item()
