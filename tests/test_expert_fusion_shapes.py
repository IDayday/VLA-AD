from __future__ import annotations

import torch

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

_install_dependency_stubs()

from transformers.feature_extraction_utils import BatchFeature
from navsim.agents.recogdrive.expert_fusion import AlignmentHead, ExpertAdapter768, TeacherTokenProjector, normalized_mse_loss
from navsim.agents.recogdrive.recogdrive_diffusion_planner import DDIMConfig, ReCogDriveDiffusionPlanner, ReCogDriveDiffusionPlannerConfig


def test_expert_modules_shapes():
    b = 2
    jepa = TeacherTokenProjector(1024, 768, 384)
    vggt = TeacherTokenProjector(2048, 768, 384)
    assert jepa(torch.randn(b, 12, 1024)).shape == (b, 12, 384)
    assert vggt(torch.randn(b, 12, 2048)).shape == (b, 12, 384)
    adapter = ExpertAdapter768(384, 768, 12)
    z768, z384 = adapter(torch.randn(b, 5, 384))
    assert z768.shape == (b, 12, 768)
    assert z384.shape == (b, 12, 384)
    assert torch.allclose(z384, torch.zeros_like(z384), atol=1e-6)
    assert AlignmentHead(768, 1024)(z768).shape == (b, 12, 1024)
    assert torch.isfinite(normalized_mse_loss(torch.randn(b, 12, 8), torch.randn(b, 12, 8)))


def _planner(use_expert_features: bool) -> ReCogDriveDiffusionPlanner:
    cfg = ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg={
            "num_heads": 8,
            "head_dim": 48,
            "num_layers": 2,
            "output_dim": 512,
            "dropout": 0.0,
            "attention_bias": True,
            "norm_eps": 1e-5,
            "interleave_attention": True,
        },
        input_embedding_dim=384,
        planner_dim=384,
        hidden_size=384,
        action_dim=3,
        action_horizon=8,
        max_seq_len=8,
        sampling_method="ddim",
        num_inference_steps=2,
        vlm_size="small",
        ddim_cfg=DDIMConfig(num_train_timesteps=10),
        use_expert_features=use_expert_features,
        jepa_dim=1024,
        vggt_dim=2048,
        num_jepa_tokens=12,
        num_vggt_tokens=12,
        expert_adapter_dim=768,
        jepa_alignment_weight=0.03,
        vggt_alignment_weight=0.05,
        alignment_loss_type="normalized_mse",
    )
    return ReCogDriveDiffusionPlanner(cfg)


def _batch(include_experts: bool, include_targets: bool = True) -> tuple[torch.Tensor, BatchFeature]:
    b = 2
    vl = torch.randn(b, 7, 1536)
    data = {
        "his_traj": torch.randn(b, 12),
        "status_feature": torch.randn(b, 8),
        "action": torch.randn(b, 8, 3),
    }
    if include_experts:
        data.update({
            "jepa_context_tokens": torch.randn(b, 12, 1024),
            "vggt_context_tokens": torch.randn(b, 12, 2048),
        })
        if include_targets:
            data.update({
                "jepa_target_tokens": torch.randn(b, 12, 1024),
                "vggt_target_tokens": torch.randn(b, 12, 2048),
            })
    return vl, BatchFeature(data=data)


def test_planner_forward_backward_expert_and_baseline():
    for enabled in (False, True):
        planner = _planner(enabled)
        planner.train()
        vl, action_input = _batch(include_experts=enabled)
        out = planner(vl, action_input)
        assert torch.isfinite(out["loss"])
        out["loss"].backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in planner.parameters())


def test_align_only_loss_excludes_diffusion_path():
    planner = _planner(True)
    planner.config.diffusion_loss_weight = 0.0
    planner.train()
    vl, action_input = _batch(include_experts=True)
    out = planner(vl, action_input)
    expected = (
        planner.config.jepa_alignment_weight * out["jepa_alignment_loss"]
        + planner.config.vggt_alignment_weight * out["vggt_alignment_loss"]
    )
    assert torch.allclose(out["loss"], expected)
    out["loss"].backward()
    assert planner.action_decoder.fc1.weight.grad is not None
    assert planner.action_decoder.fc1.weight.grad.abs().sum().item() == 0.0
    assert planner.jepa_adapter.up.weight.grad is not None
    assert planner.vggt_adapter.up.weight.grad is not None


def test_freeze_expert_trainability_keeps_action_head_trainable():
    from argparse import Namespace

    from scripts.train_recogdrive_expert_chunked import set_trainable

    planner = _planner(True)
    args = Namespace(train_expert_only=False, freeze_base_action_head=False, freeze_expert=True)
    set_trainable(planner, args)
    expert_params = [param for name, param in planner.named_parameters() if "jepa_" in name or "vggt_" in name or name == "branch_logits"]
    assert expert_params
    assert not any(param.requires_grad for param in expert_params)
    assert planner.action_decoder.fc1.weight.requires_grad
