import torch

from navsim.agents.recogdrive.stage3_lfp_grpo import (
    compute_bidirectional_pareto_advantage_energy,
    compute_pareto_tradeoff_intensity,
    compute_safety_first_frontier_energy,
)


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


def test_pareto_tradeoff_intensity_distinguishes_conflict_from_shared_ordering() -> None:
    objectives = torch.tensor(
        [
            [[0.1, 0.1, 0.1], [0.5, 0.5, 0.5], [0.9, 0.9, 0.9]],
            [[0.9, 0.1, 0.5], [0.1, 0.9, 0.5], [0.5, 0.5, 0.9]],
            [[0.9, 0.1, 0.5], [0.1, 0.9, 0.5], [0.5, 0.5, 0.9]],
        ]
    )
    valid = torch.tensor(
        [
            [True, True, True],
            [True, True, True],
            [True, False, False],
        ]
    )

    intensity = compute_pareto_tradeoff_intensity(objectives, valid, eps=0.01)

    torch.testing.assert_close(intensity, torch.tensor([0.0, 1.0, 0.0]))


def test_pareto_tradeoff_intensity_rejects_negative_epsilon() -> None:
    objectives = torch.zeros(1, 2, 3)
    valid = torch.ones(1, 2, dtype=torch.bool)
    try:
        compute_pareto_tradeoff_intensity(objectives, valid, eps=-0.01)
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("negative tradeoff epsilon must fail fast")


def test_safety_first_frontier_energy_is_lexicographic() -> None:
    pareto_energy = torch.tensor([0.25, 0.25, 0.25])
    advantages = torch.tensor(
        [
            [1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 0.0],
        ]
    )
    feasible = torch.tensor(
        [
            [True, True],
            [True, False],
            [False, False],
        ]
    )

    energy, diagnostics = compute_safety_first_frontier_energy(
        pareto_energy,
        advantages,
        feasible,
    )

    torch.testing.assert_close(energy, torch.tensor([0.25, 1.0, 0.5]))
    torch.testing.assert_close(diagnostics["all_feasible"], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(diagnostics["mixed_feasibility"], torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(diagnostics["all_infeasible"], torch.tensor([0.0, 0.0, 1.0]))


def test_safety_first_frontier_energy_rejects_shape_mismatch() -> None:
    with torch.no_grad():
        try:
            compute_safety_first_frontier_energy(
                torch.zeros(2),
                torch.zeros(1, 2),
                torch.ones(1, 2, dtype=torch.bool),
            )
        except ValueError as error:
            assert "pareto_energy" in str(error)
        else:
            raise AssertionError("mismatched Pareto energy shape must fail fast")
