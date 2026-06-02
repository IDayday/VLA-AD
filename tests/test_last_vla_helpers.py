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


def make_last_vla_planner(
    *,
    stage: str = "progressive_sft_bottleneck",
    residual: bool = True,
    diffusion_loss_weight: float = 1.0,
    dynamic_loss_weight: float = 0.0,
    cot_tokens: int = 8,
    summary_tokens: int = 2,
    risk_head: bool = False,
    action_conditioned: bool = True,
    teacher_mode: str = "none",
) -> ReCogDriveDiffusionPlanner:
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
        use_expert_features=False,
        use_last_vla=True,
        last_vla_stage=stage,
        use_last_rd=False,
        last_rd_stage="disabled",
        jepa_dim=1024,
        vggt_dim=2048,
        num_dynamic_tokens=6,
        num_geometry_tokens=6,
        num_ego_tokens=4,
        num_risk_tokens=4,
        diffusion_loss_weight=diffusion_loss_weight,
        last_vla_cot_num_tokens=cot_tokens,
        last_vla_vlm_summary_tokens=summary_tokens,
        last_vla_use_risk_head=risk_head,
        last_vla_allow_patch_geometry_fallback=True,
        last_vla_use_residual_diffusion=residual,
        last_vla_use_action_conditioned_dynamics=action_conditioned,
        last_vla_geometry_loss_weight=0.1,
        last_vla_dynamic_loss_weight=dynamic_loss_weight,
        last_vla_coarse_loss_weight=0.2,
        last_vla_heading_loss_weight=0.1,
        last_vla_progress_loss_weight=0.1,
        last_vla_teacher_traj_mode=teacher_mode,
        policy_kd_loss_weight=0.0,
        policy_kd_mode="none",
    )
    return ReCogDriveDiffusionPlanner(cfg)


def make_last_vla_batch(batch: int = 2, include_targets: bool = True) -> tuple[torch.Tensor, BatchFeature]:
    data = {
        "his_traj": torch.randn(batch, 12),
        "history_trajectory": torch.randn(batch, 4, 3),
        "status_feature": torch.randn(batch, 8),
        "action": torch.randn(batch, 8, 3),
        "high_command_one_hot": torch.eye(3)[torch.arange(batch) % 3].float(),
        "jepa_context_tokens": torch.randn(batch, 12, 1024),
        "vggt_context_tokens": torch.randn(batch, 12, 2048),
    }
    if include_targets:
        data.update(
            {
                "jepa_target_tokens": torch.randn(batch, 12, 1024),
                "vggt_geometry_target_tokens": torch.randn(batch, 12, 2048),
            }
        )
    return torch.randn(batch, 10, 1536), BatchFeature(data=data)
