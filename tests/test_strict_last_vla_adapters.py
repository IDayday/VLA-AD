from __future__ import annotations

import torch

from navsim.agents.recogdrive.last_vla_teacher_adapters import DynamicsAdapter, GeometryAdapter


def test_strict_last_vla_adapter_output_shapes():
    torch.manual_seed(11)
    image_hidden = torch.randn(2, 5, 1536)
    h_dyn = torch.randn(2, 64, 1536)
    h_geo = torch.randn(2, 64, 1536)
    jepa_target = torch.randn(2, 128, 1024)
    geometry_target = torch.randn(2, 192, 512)

    dyn = DynamicsAdapter(mask_ratio=0.0, strict_teacher=True)
    geo = GeometryAdapter(mask_ratio=0.0, strict_teacher=True)

    dyn_out = dyn(image_hidden, h_dyn, jepa_target_tokens=jepa_target)
    geo_out = geo(image_hidden, h_geo, vggt_geometry_tokens=geometry_target)

    assert dyn_out["pred"].shape == (2, 128, 1024)
    assert geo_out["pred"].shape == (2, 192, 512)
    assert torch.isfinite(dyn_out["loss"])
    assert torch.isfinite(geo_out["loss"])
