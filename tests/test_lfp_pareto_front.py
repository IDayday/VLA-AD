import torch

from navsim.agents.recogdrive.stage3_lfp_grpo import compute_pareto_front_mask


def test_pareto_dominance_and_invalid_rows() -> None:
    objectives = torch.tensor(
        [[
            [1.0, 1.0, 1.0],
            [0.9, 0.9, 0.9],
            [1.1, 0.8, 1.0],
            [2.0, 2.0, 2.0],
        ]]
    )
    valid = torch.tensor([[True, True, True, False]])
    front = compute_pareto_front_mask(objectives, valid)
    assert front.tolist() == [[True, False, True, False]]


def test_equal_points_are_both_non_dominated() -> None:
    objectives = torch.ones(1, 2, 3)
    front = compute_pareto_front_mask(objectives, torch.ones(1, 2, dtype=torch.bool))
    assert front.all()
