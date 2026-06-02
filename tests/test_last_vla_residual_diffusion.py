from __future__ import annotations

import torch

from tests.test_last_vla_helpers import make_last_vla_batch, make_last_vla_planner


def test_last_vla_residual_target_and_sampling_shapes():
    torch.manual_seed(103)
    planner = make_last_vla_planner(residual=True)
    vl_features, action_input = make_last_vla_batch(include_targets=True)
    gt_norm = planner.norm_odo(action_input["action"])
    with torch.no_grad():
        context = planner._prepare_dit_context(
            vl_features,
            action_input,
            training=True,
            target_action_norm=gt_norm,
            allow_target_tokens=True,
        )
    last_vla = context["last_vla_output"]
    assert last_vla.residual_target_norm.shape == gt_norm.shape
    reconstructed = last_vla.coarse_traj_norm.detach() + last_vla.residual_target_norm
    assert torch.allclose(reconstructed, gt_norm, atol=1e-5)
    assert torch.isfinite(last_vla.residual_target_norm).all()

    planner.eval()
    context_only = type(action_input)(data={key: value for key, value in action_input.items() if "target" not in key and key != "action"})
    with torch.no_grad():
        pred = planner.get_action(
            vl_features,
            context_only,
            init_actions=torch.zeros(vl_features.shape[0], 8, 3),
            deterministic=True,
        )
    assert pred["pred_traj"].shape == (vl_features.shape[0], 8, 3)
    assert torch.isfinite(pred["pred_traj"]).all()
