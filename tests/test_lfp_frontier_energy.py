import torch

from navsim.agents.recogdrive.stage3_lfp_grpo import compute_bidirectional_pareto_advantage_energy


def test_degenerate_advantage_signs_have_zero_energy() -> None:
    advantages = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0, 2.0],
            [-1.0, -2.0, -1.0, -2.0],
        ]
    )
    energy, _ = compute_bidirectional_pareto_advantage_energy(advantages)
    torch.testing.assert_close(energy, torch.zeros(3))


def test_balanced_bidirectional_advantage_has_energy() -> None:
    energy, diag = compute_bidirectional_pareto_advantage_energy(torch.tensor([[2.0, 2.0, -2.0, -2.0]]))
    assert energy.item() == 2.0
    assert diag["positive_fraction"].item() == 0.5
    assert diag["negative_fraction"].item() == 0.5
