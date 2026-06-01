from __future__ import annotations

import torch

from tests.test_last_rd_shapes import make_last_rd_batch, make_last_rd_planner


def test_last_rd_stage1_5_aux_loss_backward_reaches_adapter():
    torch.manual_seed(13)
    planner = make_last_rd_planner(stage="stage1_5", diffusion_loss_weight=0.0)
    for name, parameter in planner.named_parameters():
        if not name.startswith("last_rd."):
            parameter.requires_grad = False
    planner.train()
    vl_features, action_input = make_last_rd_batch(include_targets=True)

    out = planner(vl_features, action_input)
    assert torch.isfinite(out["loss"])
    assert out["diffusion_loss"].item() == 0.0
    assert out["future_jepa_loss"].item() > 0.0
    assert out["coarse_traj_loss"].item() > 0.0

    out["loss"].backward()
    last_rd_grads = [
        parameter.grad
        for name, parameter in planner.named_parameters()
        if name.startswith("last_rd.") and parameter.requires_grad
    ]
    assert any(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in last_rd_grads)
