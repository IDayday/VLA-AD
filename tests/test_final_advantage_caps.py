from __future__ import annotations

import torch

from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner


def _planner() -> ReCogDriveDiffusionPlanner:
    planner = ReCogDriveDiffusionPlanner.__new__(ReCogDriveDiffusionPlanner)
    planner.core_pareto_positive_slow_fail_cap = 0.0
    planner.core_pareto_dominated_positive_adv_cap = 0.0
    return planner


def test_sign_preserving_rms_scale_keeps_zero_and_signs() -> None:
    planner = _planner()
    advantage = torch.tensor([0.0, 3.0, -4.0, 12.0, -6.0])

    scaled = planner._sign_preserving_rms_scale(advantage)

    assert scaled[0].item() == 0.0
    assert torch.equal(torch.sign(scaled), torch.sign(advantage))
    assert scaled.abs().max() < advantage.abs().max()


def test_reapply_final_advantage_caps_removes_forbidden_positive_advantages() -> None:
    planner = _planner()
    advantage = torch.tensor([2.0, -1.0, 1.5, -0.2, 0.7, 0.9])
    aux = {
        "core_pareto_valid_mask": torch.tensor([[False, False, True, True, True, True]]),
        "core_pareto_ep_floor_ok_mask": torch.tensor([[False, False, False, False, True, True]]),
        "core_pareto_pareto_front_mask": torch.tensor([[False, False, False, False, False, True]]),
    }

    capped, logs = planner._reapply_final_advantage_caps(advantage, aux, B=1, G=6)

    assert torch.allclose(capped, torch.tensor([0.0, -1.0, 0.0, -0.2, 0.0, 0.9]))
    assert logs["final_positive_invalid_ratio"].item() == 0.0
    assert logs["final_positive_slow_fail_ratio"].item() == 0.0
    assert logs["final_positive_dominated_ratio"].item() == 0.0
