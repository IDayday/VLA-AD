import torch

from navsim.agents.recogdrive.stage3_lfp_grpo import LFPGRPOConfig, compute_lfp_advantages
from navsim.agents.recogdrive.stage3_metric_adapter import CanonicalMetricBatch
from navsim.agents.recogdrive.stage3_reference_cache import ReferenceBatch


def _case(advantage_clip: float = 3.0):
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
    return compute_lfp_advantages(
        metrics,
        reference,
        LFPGRPOConfig(advantage_clip=advantage_clip),
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
