import torch

from navsim.agents.recogdrive.stage3_metric_adapter import Stage3MetricAdapter


def test_v1_canonical_vector_and_ddc_fallback() -> None:
    metrics = {
        "score": torch.tensor([[0.8, 0.9]]),
        "ego_progress": torch.tensor([[0.7, 0.8]]),
        "time_to_collision_within_bound": torch.tensor([[1.0, 0.9]]),
        "comfort": torch.tensor([[0.95, 0.85]]),
        "no_at_fault_collisions": torch.ones(1, 2),
        "drivable_area_compliance": torch.ones(1, 2),
        "driving_direction_compliance": torch.tensor([[0.5, 1.0]]),
    }
    output = Stage3MetricAdapter("navsim_v1").canonicalize(metrics)
    torch.testing.assert_close(output.scalar, metrics["score"])
    torch.testing.assert_close(output.ep, metrics["ego_progress"])
    torch.testing.assert_close(output.quality, metrics["comfort"])
    torch.testing.assert_close(output.ddc_guard_value, metrics["driving_direction_compliance"])
    assert output.tlc is None
    assert output.diagnostics["ddc_fallback_ratio"].item() == 1.0


def test_v1_missing_required_metric_fails_fast() -> None:
    metrics = {
        "score": torch.ones(1, 1),
        "ego_progress": torch.ones(1, 1),
        "time_to_collision_within_bound": torch.ones(1, 1),
        "comfort": torch.ones(1, 1),
        "no_at_fault_collisions": torch.ones(1, 1),
        "driving_direction_compliance": torch.ones(1, 1),
    }
    try:
        Stage3MetricAdapter("navsim_v1").canonicalize(metrics)
    except KeyError as error:
        assert "dac" in str(error)
    else:
        raise AssertionError("missing DAC must fail")
