from __future__ import annotations

import torch

from tests.test_last_vla_helpers import make_last_vla_batch, make_last_vla_planner


def test_last_vla_forward_cot_alignment_and_progressive_get_action():
    torch.manual_seed(101)
    vl_features, action_input = make_last_vla_batch(include_targets=True)

    align = make_last_vla_planner(stage="cot_alignment", diffusion_loss_weight=0.0, dynamic_loss_weight=0.1)
    align.train()
    out = align(vl_features, action_input)
    for key in (
        "loss",
        "diffusion_loss",
        "last_vla_geometry_loss",
        "last_vla_dynamic_loss",
        "last_vla_coarse_loss",
        "last_vla_heading_loss",
        "last_vla_progress_loss",
    ):
        assert key in out
        assert torch.isfinite(out[key]).all()
    assert out["diffusion_loss"].item() == 0.0

    progressive = make_last_vla_planner(stage="progressive_sft_bottleneck", residual=True)
    progressive.train()
    out = progressive(vl_features, action_input)
    assert torch.isfinite(out["loss"]).all()
    assert torch.isfinite(out["diffusion_loss"]).all()

    progressive.eval()
    context_only = {key: value for key, value in action_input.items() if "target" not in key and key != "action"}
    with torch.no_grad():
        pred = progressive.get_action(
            vl_features,
            type(action_input)(data=context_only),
            init_actions=torch.zeros(vl_features.shape[0], 8, 3),
            deterministic=True,
        )
    assert pred["pred_traj"].shape == (vl_features.shape[0], 8, 3)
    assert pred["pred_coarse_traj"].shape == (vl_features.shape[0], 8, 3)
    assert pred["pred_residual_norm"].shape == (vl_features.shape[0], 8, 3)
    assert torch.isfinite(pred["pred_traj"]).all()
