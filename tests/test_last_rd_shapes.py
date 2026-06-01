from __future__ import annotations

import torch

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

_install_dependency_stubs()

from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.recogdrive_diffusion_planner import (
    DDIMConfig,
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)


def make_last_rd_planner(stage: str = "progressive_sft", diffusion_loss_weight: float = 1.0) -> ReCogDriveDiffusionPlanner:
    cfg = ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg={
            "num_heads": 4,
            "head_dim": 96,
            "num_layers": 2,
            "output_dim": 384,
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
        use_expert_features=True,
        use_jepa=True,
        use_vggt=True,
        jepa_dim=1024,
        vggt_dim=2048,
        num_jepa_tokens=12,
        num_vggt_tokens=12,
        expert_adapter_dim=768,
        expert_dropout=0.0,
        expert_stream_dropout=0.0,
        jepa_alignment_weight=0.0,
        vggt_alignment_weight=0.0,
        use_last_rd=True,
        last_rd_stage=stage,
        diffusion_loss_weight=diffusion_loss_weight,
        last_rd_latent_dim=384,
        num_dynamic_tokens=12,
        num_geometry_tokens=12,
        num_ego_tokens=8,
        num_risk_tokens=8,
        future_jepa_loss_weight=0.3,
        vggt_geometry_loss_weight=0.1,
        coarse_traj_loss_weight=0.5,
        coarse_heading_loss_weight=0.1,
        risk_loss_weight=0.05,
        policy_kd_loss_weight=0.0,
    )
    return ReCogDriveDiffusionPlanner(cfg)


def make_last_rd_batch(include_targets: bool = True) -> tuple[torch.Tensor, BatchFeature]:
    batch = 2
    data = {
        "his_traj": torch.randn(batch, 12),
        "history_trajectory": torch.randn(batch, 4, 3),
        "status_feature": torch.randn(batch, 8),
        "action": torch.randn(batch, 8, 3),
        "high_command_one_hot": torch.eye(3)[torch.tensor([0, 2])].float(),
        "jepa_context_tokens": torch.randn(batch, 12, 1024),
        "vggt_context_tokens": torch.randn(batch, 12, 2048),
    }
    if include_targets:
        data.update({
            "jepa_target_tokens": torch.randn(batch, 12, 1024),
            "vggt_target_tokens": torch.randn(batch, 12, 2048),
        })
    return torch.randn(batch, 16, 1536), BatchFeature(data=data)


def test_last_rd_forward_and_get_action_shapes():
    torch.manual_seed(7)
    planner = make_last_rd_planner()
    planner.train()
    vl_features, action_input = make_last_rd_batch(include_targets=True)
    out = planner(vl_features, action_input)
    for key in (
        "loss",
        "diffusion_loss",
        "future_jepa_loss",
        "vggt_geometry_loss",
        "coarse_traj_loss",
        "coarse_heading_loss",
        "risk_loss",
        "policy_kd_loss",
    ):
        assert key in out
        assert torch.isfinite(out[key])

    planner.eval()
    context_only = BatchFeature(data={key: value for key, value in action_input.items() if "target" not in key and key != "action"})
    with torch.no_grad():
        pred = planner.get_action(
            vl_features,
            context_only,
            init_actions=torch.zeros(2, 8, 3),
            deterministic=True,
        )
    assert pred["pred_traj"].shape == (2, 8, 3)
    assert torch.isfinite(pred["pred_traj"]).all()
