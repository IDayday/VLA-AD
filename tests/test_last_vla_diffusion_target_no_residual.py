from __future__ import annotations

import torch

from tests.test_last_vla_helpers import make_last_vla_batch, make_last_vla_planner


def test_last_vla_diffusion_target_is_gt_not_coarse_residual():
    planner = make_last_vla_planner()
    assert planner.config.last_vla_use_residual_diffusion is False

    gt = torch.ones(2, 8, 3)
    coarse = torch.full_like(gt, 0.25)
    target, alpha = planner._last_vla_diffusion_target(gt, training=True)

    assert alpha == 0.0
    assert torch.allclose(target, gt)
    assert not torch.allclose(target, gt - coarse)


def test_last_vla_get_action_outputs_direct_pred_traj_and_diagnostic_coarse():
    torch.manual_seed(901)
    planner = make_last_vla_planner()
    planner.eval()
    vl_features, action_input = make_last_vla_batch(include_targets=False)
    context_only = type(action_input)(
        data={key: value for key, value in action_input.items() if "target" not in key and key != "action"}
    )

    with torch.no_grad():
        pred = planner.get_action(
            vl_features,
            context_only,
            init_actions=torch.zeros(vl_features.shape[0], 8, 3),
            deterministic=True,
        )

    assert pred["pred_traj"].shape == (vl_features.shape[0], 8, 3)
    assert pred["pred_coarse_traj"].shape == (vl_features.shape[0], 8, 3)
    assert "pred_residual_norm" not in pred
    assert not torch.allclose(pred["pred_traj"], pred["pred_coarse_traj"])
