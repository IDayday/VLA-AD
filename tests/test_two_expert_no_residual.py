from __future__ import annotations

import torch

from tests.test_two_expert_stage2_context import make_two_expert_batch, make_two_expert_planner


def test_two_expert_diffusion_target_is_gt_not_residual():
    planner = make_two_expert_planner()
    assert planner.config.last_vla_use_residual_diffusion is False
    gt = torch.randn(2, 8, 3)
    target, alpha = planner._last_vla_diffusion_target(gt, training=True)

    assert alpha == 0.0
    assert torch.allclose(target, gt)


def test_two_expert_get_action_outputs_direct_pred_traj_only():
    torch.manual_seed(41)
    planner = make_two_expert_planner()
    planner.eval()
    vl, action_input = make_two_expert_batch()
    context_only = type(action_input)(data={key: value for key, value in action_input.items() if key != "action"})

    with torch.no_grad():
        out = planner.get_action(
            vl,
            context_only,
            init_actions=torch.zeros(vl.shape[0], 8, 3),
            deterministic=True,
        )

    assert out["pred_traj"].shape == (vl.shape[0], 8, 3)
    assert "pred_residual_norm" not in out
    assert "pred_coarse_traj" not in out
