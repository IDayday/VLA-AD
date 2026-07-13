from pathlib import Path

import torch

from navsim.agents.recogdrive.stage3_lfp_grpo import (
    LFPGRPOConfig,
    compute_bidirectional_pareto_advantage_energy,
    compute_lfp_advantages,
)
from navsim.agents.recogdrive.stage3_metric_adapter import CanonicalMetricBatch
from navsim.agents.recogdrive.stage3_reference_cache import ReferenceBatch


def _case(
    advantage_clip: float = 3.0,
    *,
    config_overrides=None,
    group_pairwise_ade_m=None,
):
    scalar = torch.tensor(
        [
            [0.7, 0.9, 0.6, 0.8],
            [0.9, 0.8, 0.7, 0.6],
            [0.9, 0.8, 0.7, 0.6],
            [0.9, 0.8, 0.7, 0.6],
        ]
    )
    ep = torch.tensor(
        [
            [0.7, 0.9, 0.6, 0.8],
            [0.9, 0.8, 0.7, 0.6],
            [0.7, 0.8, 0.9, 0.6],
            [0.9, 0.8, 0.7, 0.6],
        ]
    )
    ones = torch.ones_like(scalar)
    nc = ones.clone()
    nc[2, 2:] = 0.0
    nc[3] = 0.0
    metrics = CanonicalMetricBatch(
        scalar=scalar,
        ep=ep,
        ttc=ones,
        quality=ep,
        nc=nc,
        dac=ones,
        ddc_guard_value=ones,
        tlc=None,
        diagnostics={},
    )
    reference = ReferenceBatch(
        scalar=torch.full((4,), 0.7),
        ep=torch.full((4,), 0.7),
        ttc=torch.ones(4),
        quality=torch.ones(4),
        nc=torch.ones(4),
        dac=torch.ones(4),
        ddc=torch.ones(4),
        tlc=None,
        gt_ddc=torch.ones(4),
        selected_source_code=torch.ones(4, dtype=torch.long),
        fallback_mask=torch.zeros(4, dtype=torch.bool),
    )
    config_values = {"advantage_clip": advantage_clip}
    config_values.update(config_overrides or {})
    return compute_lfp_advantages(
        metrics,
        reference,
        LFPGRPOConfig(**config_values),
        group_pairwise_ade_m=group_pairwise_ade_m,
    )


def test_pareto_eligible_can_receive_positive_advantage() -> None:
    output = _case()
    assert output.pareto_front[0, 1]
    assert output.advantages[0, 1] > 0.0


def test_dominated_and_progress_fail_positive_credit_is_capped() -> None:
    output = _case()
    # Row 0 candidate 3 is feasible/progress-ok but dominated by candidate 1.
    assert not output.pareto_front[0, 3]
    assert output.advantages[0, 3] <= 0.0
    # Row 1 candidate 3 fails progress and cannot receive positive credit.
    assert not output.progress_ok[1, 3]
    assert output.advantages[1, 3] <= 0.0


def test_mixed_unsafe_is_minus_one_and_all_infeasible_is_zero() -> None:
    output = _case()
    torch.testing.assert_close(output.advantages[2, 2:], torch.tensor([-1.0, -1.0]))
    torch.testing.assert_close(output.advantages[3], torch.zeros(4))


def test_advantage_clamp_and_energy_are_finite() -> None:
    output = _case(advantage_clip=0.2)
    assert output.advantages.abs().max() <= 0.2
    assert output.diagnostics["lfp_advantage_clip_ratio"] > 0.0
    assert torch.isfinite(output.energy).all()


def test_default_tradeoff_weight_preserves_original_bpae() -> None:
    output = _case()
    expected, _ = compute_bidirectional_pareto_advantage_energy(output.advantages)
    torch.testing.assert_close(output.energy, expected)
    torch.testing.assert_close(
        output.diagnostics["lfp_frontier_tradeoff_multiplier_mean"],
        torch.tensor(1.0),
    )


def test_low_spread_group_is_removed_before_frontier_energy() -> None:
    output = _case(
        config_overrides={"min_group_pairwise_ade_m": 0.05},
        group_pairwise_ade_m=torch.tensor([0.01, 0.2, 0.2, 0.2]),
    )
    torch.testing.assert_close(output.advantages[0], torch.zeros(4))
    torch.testing.assert_close(output.energy[0], torch.tensor(0.0))
    assert output.diagnostics["lfp_low_spread_group_ratio"].item() == 0.25
    assert output.diagnostics["lfp_credit_active_group_ratio"].item() == 0.75
    assert output.diagnostics["lfp_global_normalization_count"].item() == 6.0


def test_ttc_regression_cannot_receive_positive_credit() -> None:
    output = _case()
    metrics_ttc = torch.ones_like(output.advantages)
    metrics_ttc[0, 1] = 0.8
    scalar = torch.tensor(
        [[0.7, 0.9, 0.6, 0.8], [0.9, 0.8, 0.7, 0.6], [0.9, 0.8, 0.7, 0.6], [0.9, 0.8, 0.7, 0.6]]
    )
    ep = scalar.clone()
    ones = torch.ones_like(scalar)
    nc = ones.clone()
    nc[2, 2:] = 0.0
    nc[3] = 0.0
    metrics = CanonicalMetricBatch(
        scalar=scalar,
        ep=ep,
        ttc=metrics_ttc,
        quality=ep,
        nc=nc,
        dac=ones,
        ddc_guard_value=ones,
        tlc=None,
        diagnostics={},
    )
    reference = ReferenceBatch(
        scalar=torch.full((4,), 0.7),
        ep=torch.full((4,), 0.7),
        ttc=torch.ones(4),
        quality=torch.ones(4),
        nc=torch.ones(4),
        dac=torch.ones(4),
        ddc=torch.ones(4),
        tlc=None,
        gt_ddc=torch.ones(4),
        selected_source_code=torch.ones(4, dtype=torch.long),
        fallback_mask=torch.zeros(4, dtype=torch.bool),
    )
    guarded = compute_lfp_advantages(
        metrics,
        reference,
        LFPGRPOConfig(ttc_positive_credit_guard=True, ttc_reference_tolerance=0.01),
    )
    assert not guarded.ttc_ok[0, 1]
    assert guarded.advantages[0, 1] <= 0.0
    assert guarded.diagnostics["lfp_ttc_fail_ratio"] > 0.0
    assert guarded.diagnostics["lfp_ttc_fail_advantage_mean"] <= 0.0
    assert 0.0 <= guarded.diagnostics["lfp_ttc_fail_zero_credit_ratio"] <= 1.0


def _reference_gate_case(**config_overrides):
    scalar = torch.tensor([[0.95, 0.90, 0.70]])
    metrics = CanonicalMetricBatch(
        scalar=scalar,
        ep=torch.tensor([[0.89, 0.95, 0.70]]),
        ttc=torch.tensor([[0.99, 0.90, 0.80]]),
        quality=torch.tensor([[0.85, 0.80, 0.70]]),
        nc=torch.ones_like(scalar),
        dac=torch.ones_like(scalar),
        ddc_guard_value=torch.ones_like(scalar),
        tlc=None,
        diagnostics={},
    )
    reference = ReferenceBatch(
        scalar=torch.tensor([0.80]),
        ep=torch.tensor([0.90]),
        ttc=torch.tensor([1.0]),
        quality=torch.tensor([0.90]),
        nc=torch.ones(1),
        dac=torch.ones(1),
        ddc=torch.ones(1),
        tlc=None,
        gt_ddc=torch.ones(1),
        selected_source_code=torch.ones(1, dtype=torch.long),
        fallback_mask=torch.zeros(1, dtype=torch.bool),
    )
    return compute_lfp_advantages(metrics, reference, LFPGRPOConfig(**config_overrides))


def test_coherent_reference_pareto_gate_removes_reference_dominated_positive_credit() -> None:
    baseline = _reference_gate_case()
    guarded = _reference_gate_case(reference_pareto_gate_enabled=True)

    assert baseline.advantages[0, 0] > 0.0
    assert not guarded.reference_pareto_ok[0, 0]
    assert guarded.reference_pareto_ok[0, 1]
    assert guarded.advantages[0, 0] <= 0.0
    assert guarded.diagnostics["lfp_positive_advantage_reference_dominated_ratio"] == 0.0


def test_quality_guard_bounds_pareto_tradeoff_regression() -> None:
    baseline = _reference_gate_case()
    guarded = _reference_gate_case(
        quality_positive_credit_guard=True,
        quality_reference_tolerance=0.02,
    )

    assert baseline.advantages[0, 0] > 0.0
    assert not guarded.quality_ok[0, 0]
    assert guarded.advantages[0, 0] <= 0.0
    assert guarded.diagnostics["lfp_quality_fail_ratio"] > 0.0


def test_reference_regression_diagnostics_expose_group_relative_positive_credit() -> None:
    output = _reference_gate_case()
    diagnostics = output.diagnostics

    assert diagnostics["lfp_reference_dominated_ratio"] > 0.0
    assert diagnostics["lfp_positive_advantage_reference_dominated_ratio"] > 0.0
    for key in (
        "lfp_positive_advantage_below_ref_scalar_ratio",
        "lfp_positive_advantage_below_ref_quality_ratio",
        "lfp_positive_advantage_scalar_delta_mean",
        "lfp_positive_advantage_quality_delta_mean",
    ):
        assert diagnostics[key].numel() == 1
        assert torch.isfinite(diagnostics[key])


def test_credit_diagnostics_are_scalar_and_match_final_advantages() -> None:
    output = _case()
    diagnostics = output.diagnostics
    torch.testing.assert_close(diagnostics["lfp_advantage_mean"], output.advantages.mean())
    torch.testing.assert_close(
        diagnostics["lfp_advantage_std"],
        output.advantages.std(unbiased=False),
    )
    for key in (
        "lfp_positive_eligible_advantage_mean",
        "lfp_ttc_fail_advantage_mean",
        "lfp_ttc_fail_zero_credit_ratio",
        "lfp_quality_fail_advantage_mean",
        "lfp_reference_dominated_advantage_mean",
        "lfp_positive_advantage_reference_dominated_ratio",
        "lfp_progress_fail_advantage_mean",
        "lfp_progress_fail_zero_credit_ratio",
        "lfp_unsafe_advantage_mean",
    ):
        assert diagnostics[key].numel() == 1
        assert torch.isfinite(diagnostics[key])


def test_credit_diagnostics_are_in_lightning_logging_contract() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "navsim/planning/training/agent_lightning_module.py"
    )
    source = module_path.read_text()
    for key in (
        "lfp_advantage_mean",
        "lfp_ttc_fail_advantage_mean",
        "lfp_ttc_fail_zero_credit_ratio",
        "lfp_quality_fail_advantage_mean",
        "lfp_reference_dominated_advantage_mean",
        "lfp_positive_advantage_reference_dominated_ratio",
        "lfp_progress_fail_advantage_mean",
        "lfp_unsafe_advantage_mean",
    ):
        assert f'"{key}"' in source


def test_all_infeasible_rescue_only_assigns_negative_credit() -> None:
    scalar = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
    ones = torch.ones_like(scalar)
    metrics = CanonicalMetricBatch(
        scalar=scalar,
        ep=scalar,
        ttc=ones,
        quality=scalar,
        nc=torch.zeros_like(scalar),
        dac=ones,
        ddc_guard_value=ones,
        tlc=None,
        diagnostics={},
    )
    reference = ReferenceBatch(
        scalar=torch.tensor([0.7]),
        ep=torch.tensor([0.7]),
        ttc=torch.ones(1),
        quality=torch.ones(1),
        nc=torch.ones(1),
        dac=torch.ones(1),
        ddc=torch.ones(1),
        tlc=None,
        gt_ddc=torch.ones(1),
        selected_source_code=torch.ones(1, dtype=torch.long),
        fallback_mask=torch.zeros(1, dtype=torch.bool),
    )
    rescued = compute_lfp_advantages(metrics, reference, LFPGRPOConfig(all_infeasible_rescue=True))
    assert (rescued.advantages <= 0.0).all()
    assert (rescued.advantages < 0.0).any()
