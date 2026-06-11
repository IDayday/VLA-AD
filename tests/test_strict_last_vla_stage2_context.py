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


def make_strict_planner(stage: str = "stage2_diffusion_sft", diffusion_loss_weight: float = 1.0) -> ReCogDriveDiffusionPlanner:
    cfg = ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg={
            "num_heads": 4,
            "head_dim": 96,
            "num_layers": 1,
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
        num_inference_steps=1,
        vlm_size="small",
        ddim_cfg=DDIMConfig(num_train_timesteps=10),
        use_last_vla=True,
        use_strict_last_vla=True,
        last_vla_reasoner_type="strict_latent_slots",
        last_vla_stage=stage,
        last_vla_latent_cache_mode=True,
        use_last_rd=False,
        last_rd_stage="disabled",
        use_expert_features=False,
        use_jepa=False,
        use_vggt=False,
        last_vla_num_dyn_latent_tokens=2,
        last_vla_num_geo_latent_tokens=2,
        last_vla_num_plan_latent_tokens=1,
        last_vla_wm_teacher_source="jepa_dev",
        last_vla_random_mask_ratio=0.0,
        last_vla_use_residual_diffusion=False,
        last_vla_residual_anchor_source="none",
        last_vla_teacher_traj_mode="none",
        diffusion_loss_weight=diffusion_loss_weight,
        last_vla_dynamic_loss_weight=0.3,
        last_vla_geometry_loss_weight=0.3,
        last_vla_coarse_loss_weight=1.0,
        last_vla_heading_loss_weight=0.2,
        last_vla_progress_loss_weight=0.2,
    )
    return ReCogDriveDiffusionPlanner(cfg)


def make_strict_batch(batch: int = 2) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    data = {
        "his_traj": torch.randn(batch, 12),
        "history_trajectory": torch.randn(batch, 4, 3),
        "status_feature": torch.randn(batch, 8),
        "high_command_one_hot": torch.eye(3)[torch.arange(batch) % 3].float(),
        "action": torch.randn(batch, 8, 3),
        "last_vla_h_dyn": torch.randn(batch, 2, 1536),
        "last_vla_h_geo": torch.randn(batch, 2, 1536),
        "last_vla_h_plan": torch.randn(batch, 1, 1536),
        "image_hidden_states": torch.randn(batch, 4, 1536),
        "jepa_target_tokens": torch.randn(batch, 128, 1024),
        "vggt_geometry_tokens": torch.randn(batch, 192, 512),
    }
    return torch.randn(batch, 6, 1536), data


def test_strict_last_vla_stage2_context_uses_raw_vlm_and_latent_condition():
    planner = make_strict_planner()
    planner.eval()
    vl_features, data = make_strict_batch()
    action_input = BatchFeature(data={key: value for key, value in data.items() if key != "action"})
    context = planner._prepare_dit_context(vl_features, action_input, training=False, allow_target_tokens=False)

    assert context["context_tokens"].shape == (2, 6, 384)
    assert context["cot_condition_tokens"].shape == (2, 5, 384)
    assert context["expert_step_condition"] is None


def test_strict_last_vla_diffusion_target_is_gt():
    planner = make_strict_planner()
    gt = torch.randn(2, 8, 3)
    target, alpha = planner._last_vla_diffusion_target(gt, training=True)
    assert alpha == 0.0
    assert torch.allclose(target, gt)
