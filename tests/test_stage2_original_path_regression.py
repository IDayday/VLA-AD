from __future__ import annotations

import torch

from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner


def _planner() -> ReCogDriveDiffusionPlanner:
    planner = ReCogDriveDiffusionPlanner.__new__(ReCogDriveDiffusionPlanner)
    planner.training = True
    return planner


def test_stage2_original_target_path_is_unchanged_without_support_fields() -> None:
    planner = _planner()
    gt = torch.randn(4, 8, 3)

    selected, diagnostics = planner._select_stage2_pareto_support_target({}, gt)

    assert selected.data_ptr() == gt.data_ptr()
    assert torch.equal(selected, gt)
    assert diagnostics["stage2_pareto_enabled"].item() == 0.0
    assert diagnostics["stage2_pareto_used_ratio"].item() == 0.0


def test_stage2_original_target_path_is_unchanged_with_incomplete_support_fields() -> None:
    planner = _planner()
    gt = torch.randn(2, 8, 3)

    selected, diagnostics = planner._select_stage2_pareto_support_target(
        {
            "support_trajectories": torch.randn(2, 3, 8, 3),
            "support_mask": torch.ones(2, 3, dtype=torch.bool),
        },
        gt,
    )

    assert selected.data_ptr() == gt.data_ptr()
    assert torch.equal(selected, gt)
    assert diagnostics["stage2_pareto_enabled"].item() == 0.0
