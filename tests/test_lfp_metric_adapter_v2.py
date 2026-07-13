import pytest
import torch

from navsim.agents.recogdrive.stage3_metric_adapter import Stage3MetricAdapter


def _v2_metrics():
    return {
        "score": torch.tensor([[0.8, 0.9]]),
        "ego_progress": torch.tensor([[0.7, 0.8]]),
        "time_to_collision_within_bound": torch.ones(1, 2),
        "lane_keeping": torch.tensor([[0.6, 0.9]]),
        "history_comfort": torch.tensor([[0.9, 0.6]]),
        "two_frame_extended_comfort": torch.tensor([[0.3, 0.9]]),
        "no_at_fault_collisions": torch.ones(1, 2),
        "drivable_area_compliance": torch.ones(1, 2),
        "raw_ddc": torch.ones(1, 2),
        "traffic_light_compliance": torch.ones(1, 2),
    }


def test_v2_quality_is_official_three_component_mean() -> None:
    metrics = _v2_metrics()
    output = Stage3MetricAdapter("navsim_v2").canonicalize(metrics)
    expected = (
        metrics["lane_keeping"]
        + metrics["history_comfort"]
        + metrics["two_frame_extended_comfort"]
    ) / 3.0
    torch.testing.assert_close(output.quality, expected)
    assert output.tlc is not None
    assert output.diagnostics["ddc_fallback_ratio"].item() == 0.0


@pytest.mark.parametrize(
    "missing",
    [
        "score",
        "ego_progress",
        "time_to_collision_within_bound",
        "lane_keeping",
        "history_comfort",
        "two_frame_extended_comfort",
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "raw_ddc",
        "traffic_light_compliance",
    ],
)
def test_v2_missing_required_fields_fails_fast(missing: str) -> None:
    metrics = _v2_metrics()
    metrics.pop(missing)
    if missing == "raw_ddc":
        # DDC is the one allowed raw->filtered fallback. Removing both must fail.
        metrics.pop("driving_direction_compliance", None)
    with pytest.raises(KeyError):
        Stage3MetricAdapter("navsim_v2").canonicalize(metrics)
