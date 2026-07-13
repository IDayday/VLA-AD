from __future__ import annotations

import torch

from tests.test_last_vla_helpers import make_last_vla_batch, make_last_vla_planner


def test_last_vla_get_action_ignores_future_teacher_targets():
    torch.manual_seed(106)
    planner = make_last_vla_planner(residual=True)
    planner.eval()
    vl_features, action_input = make_last_vla_batch(include_targets=False)
    context = type(action_input)(data={key: value for key, value in action_input.items() if key != "action"})
    with_targets = type(action_input)(data=dict(context))
    with_targets["jepa_target_tokens"] = torch.randn(2, 12, 1024)
    with_targets["vggt_target_tokens"] = torch.randn(2, 12, 2048)
    with_targets["vggt_geometry_target_tokens"] = torch.randn(2, 12, 2048)
    with_targets["teacher_trajectory_norm"] = torch.randn(2, 8, 3)
    with_targets["teacher_score"] = torch.randn(2)
    init_actions = torch.zeros(2, 8, 3)

    with torch.no_grad():
        out_context = planner.get_action(vl_features, context, init_actions=init_actions.clone(), deterministic=True)
        out_targets = planner.get_action(vl_features, with_targets, init_actions=init_actions.clone(), deterministic=True)

    assert torch.allclose(out_context["pred_traj"], out_targets["pred_traj"], atol=1e-6, rtol=0.0)
