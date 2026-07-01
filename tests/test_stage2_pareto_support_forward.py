from __future__ import annotations

import pytest
import torch

from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner


def _planner(*, training: bool) -> ReCogDriveDiffusionPlanner:
    planner = ReCogDriveDiffusionPlanner.__new__(ReCogDriveDiffusionPlanner)
    planner.training = training
    return planner


def _physical_actions(batch: int) -> torch.Tensor:
    actions = torch.zeros(batch, 8, 3)
    actions[..., 0] = torch.linspace(0.0, 6.0, 8)
    return actions


def test_stage2_pareto_support_training_selects_weighted_support_and_falls_back_for_missing() -> None:
    planner = _planner(training=True)
    gt_physical = _physical_actions(2)
    gt_norm = planner.norm_odo(gt_physical)

    support = torch.zeros(2, 3, 8, 3)
    support[0, 0] = gt_physical[0]
    support[0, 1] = gt_physical[0] + torch.tensor([2.0, 1.0, 0.1])
    support[0, 2] = gt_physical[0] + torch.tensor([4.0, -1.0, -0.1])
    support[1, 0] = gt_physical[1] + torch.tensor([10.0, 0.0, 0.0])

    action_input = {
        "support_trajectories": support,
        "support_mask": torch.tensor([[True, True, True], [True, False, False]]),
        "support_weights": torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        "support_scores": torch.tensor([[0.7, 0.9, 0.8], [0.1, 0.0, 0.0]]),
        "support_missing_mask": torch.tensor([False, True]),
    }

    selected, diagnostics = planner._select_stage2_pareto_support_target(action_input, gt_norm)

    assert torch.allclose(selected[0], planner.norm_odo(support[0, 1]))
    assert torch.allclose(selected[1], gt_norm[1])
    assert diagnostics["stage2_pareto_enabled"].item() == 1.0
    assert diagnostics["stage2_pareto_used_ratio"].item() == pytest.approx(0.5)
    assert diagnostics["stage2_pareto_missing_ratio"].item() == pytest.approx(0.5)
    assert diagnostics["stage2_pareto_selected_index_mean"].item() == pytest.approx(1.0)


def test_stage2_pareto_support_eval_mode_ignores_support_fields() -> None:
    planner = _planner(training=False)
    gt_physical = _physical_actions(1)
    gt_norm = planner.norm_odo(gt_physical)
    support = gt_physical[:, None].repeat(1, 3, 1, 1) + 5.0

    selected, diagnostics = planner._select_stage2_pareto_support_target(
        {
            "support_trajectories": support,
            "support_mask": torch.ones(1, 3, dtype=torch.bool),
            "support_weights": torch.tensor([[0.0, 1.0, 0.0]]),
        },
        gt_norm,
    )

    assert torch.allclose(selected, gt_norm)
    assert diagnostics["stage2_pareto_enabled"].item() == 0.0


def test_stage2_pareto_support_rejects_bad_shapes() -> None:
    planner = _planner(training=True)
    gt_norm = planner.norm_odo(_physical_actions(1))

    with pytest.raises(ValueError, match="support_trajectories"):
        planner._select_stage2_pareto_support_target(
            {
                "support_trajectories": torch.zeros(1, 2, 8, 3),
                "support_mask": torch.ones(1, 3, dtype=torch.bool),
                "support_weights": torch.ones(1, 3),
            },
            gt_norm,
        )
