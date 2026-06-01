from __future__ import annotations

import torch

from tests.test_last_rd_shapes import make_last_rd_batch, make_last_rd_planner
from transformers.feature_extraction_utils import BatchFeature


def test_last_rd_get_action_ignores_future_targets():
    torch.manual_seed(11)
    planner = make_last_rd_planner()
    planner.eval()
    vl_features, action_input = make_last_rd_batch(include_targets=False)
    context = BatchFeature(data={key: value for key, value in action_input.items() if key != "action"})
    with_targets = BatchFeature(data=dict(context))
    with_targets["jepa_target_tokens"] = torch.randn(2, 12, 1024)
    with_targets["vggt_target_tokens"] = torch.randn(2, 12, 2048)
    with_targets["vggt_geometry_target_tokens"] = torch.randn(2, 12, 2048)
    init_actions = torch.zeros(2, 8, 3)

    with torch.no_grad():
        out_context = planner.get_action(vl_features, context, init_actions=init_actions.clone(), deterministic=True)
        out_targets = planner.get_action(vl_features, with_targets, init_actions=init_actions.clone(), deterministic=True)

    assert torch.allclose(out_context["pred_traj"], out_targets["pred_traj"], atol=1e-6, rtol=0.0)
