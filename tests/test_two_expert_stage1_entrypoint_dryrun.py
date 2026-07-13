from __future__ import annotations

from scripts.last_vla_v2.two_expert_slot.smoke_stage1_two_expert_sft import run_smoke


def test_stage1_synthetic_smoke_forward_backward():
    report = run_smoke(train_mode="frozen", vggt_dim=768)

    assert report["ok"] is True
    assert report["vggt_feature_dim"] == 768
    assert report["slot_grad_l1"] > 0.0
    assert report["adapter_grad_l1"] > 0.0
    assert report["trainable_parameter_report"]["vlm_trainable_param_count"] == 0


def test_stage1_top_layers_smoke_has_vlm_trainable_params():
    report = run_smoke(train_mode="top_layers", vggt_dim=64)

    assert report["ok"] is True
    assert report["trainable_parameter_report"]["vlm_trainable_param_count"] > 0
