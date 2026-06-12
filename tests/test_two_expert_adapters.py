from __future__ import annotations

import torch

from navsim.agents.recogdrive.two_expert_adapters import JEPADynamicAdapter, RandomTokenMask, VGGTFeature23Adapter


def test_random_mask_only_training_and_seed_deterministic():
    tokens = torch.randn(2, 10, 8)
    mask = RandomTokenMask(mask_ratio=0.4, token_dim=8)
    mask.eval()
    out_eval, info_eval = mask(tokens)
    assert torch.allclose(out_eval, tokens)
    assert float(info_eval["actual_mask_ratio"]) == 0.0

    mask.train()
    torch.manual_seed(10)
    out_a, info_a = mask(tokens)
    torch.manual_seed(10)
    out_b, info_b = mask(tokens)
    assert torch.allclose(out_a, out_b)
    assert torch.equal(info_a["mask"], info_b["mask"])
    assert float(info_a["actual_mask_ratio"]) > 0.0


def test_jepa_dynamic_adapter_output_and_loss_shape():
    torch.manual_seed(11)
    adapter = JEPADynamicAdapter(vlm_hidden_dim=16, adapter_dim=8, teacher_dim=1024, mask_ratio=0.0)
    image_hidden = torch.randn(2, 5, 16)
    h_dyn = torch.randn(2, 3, 12, 16)
    target = torch.randn(2, 3, 12, 1024)
    out = adapter(image_hidden, h_dyn, target)

    assert out["pred"].shape == (2, 3, 12, 1024)
    assert torch.isfinite(out["loss"])
    assert "dynamic_loss_short" in out["diagnostics"]


def test_vggt_feature23_adapter_output_and_image_dependency():
    torch.manual_seed(12)
    adapter = VGGTFeature23Adapter(vlm_hidden_dim=16, adapter_dim=8, vggt_feature_dim=32, mask_ratio=0.0)
    adapter.eval()
    image_hidden = torch.randn(2, 5, 16)
    h_geo = torch.randn(2, 12, 16)
    target = torch.randn(2, 12, 32)
    out = adapter(image_hidden, h_geo, target)
    out_zero_image = adapter(torch.zeros_like(image_hidden), h_geo, target)

    assert out["pred"].shape == (2, 12, 32)
    assert torch.isfinite(out["loss"])
    assert not torch.allclose(out["pred"], out_zero_image["pred"])
