from __future__ import annotations

import torch

from tests.test_last_rd_shapes import make_last_rd_batch, make_last_rd_planner


def test_last_rd_progressive_sft_diffusion_aux_and_diagnostics():
    torch.manual_seed(17)
    planner = make_last_rd_planner(stage="progressive_sft", diffusion_loss_weight=1.0)
    planner.train()
    vl_features, action_input = make_last_rd_batch(include_targets=True)

    out = planner(vl_features, action_input)
    assert torch.isfinite(out["loss"])
    assert out["diffusion_loss"].item() >= 0.0
    assert out["future_jepa_loss"].item() > 0.0
    assert out["vggt_geometry_loss"].item() > 0.0
    for key in (
        "last_rd_group_weight_vlm",
        "last_rd_group_weight_dynamic",
        "last_rd_group_weight_geometry",
        "last_rd_group_weight_ego",
        "last_rd_group_weight_risk",
        "last_rd_token_norms",
    ):
        assert key in out
        assert torch.isfinite(out[key])

    out["loss"].backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in planner.parameters()
        if parameter.requires_grad
    )
