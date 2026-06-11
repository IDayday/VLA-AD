from __future__ import annotations

import torch

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

_install_dependency_stubs()

from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.last_vla_cot_planning import GEOMETRY_MODE_TO_CODE
from navsim.agents.recogdrive.recogdrive_diffusion_planner import (
    DDIMConfig,
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)


def _highcap_planner() -> ReCogDriveDiffusionPlanner:
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
        use_expert_features=False,
        use_last_vla=True,
        last_vla_stage="progressive_sft_decoupled",
        use_last_rd=False,
        last_rd_stage="disabled",
        use_jepa=True,
        use_vggt=True,
        jepa_dim=1024,
        vggt_dim=2048,
        num_jepa_tokens=128,
        num_dynamic_tokens=128,
        num_vggt_tokens=128,
        num_geometry_tokens=192,
        num_ego_tokens=8,
        num_risk_tokens=0,
        last_vla_cot_num_tokens=192,
        last_vla_cot_num_steps=5,
        last_vla_use_risk_head=False,
        last_vla_risk_loss_weight=0.0,
        last_vla_geometry_teacher_dim=512,
        last_vla_geometry_grid_rows=12,
        last_vla_geometry_grid_cols=16,
        last_vla_require_full_geometry=True,
        last_vla_allow_patch_geometry_fallback=False,
        last_vla_condition_mode="decoupled_cot_residual",
        last_vla_raw_vlm_context_to_dit=True,
        last_vla_cot_bottleneck_mode=False,
        last_vla_use_residual_diffusion=False,
        diffusion_loss_weight=1.0,
        last_vla_geometry_loss_weight=0.1,
        last_vla_dynamic_loss_weight=0.1,
        last_vla_coarse_loss_weight=0.1,
        last_vla_heading_loss_weight=0.1,
        last_vla_progress_loss_weight=0.1,
        policy_kd_loss_weight=0.0,
        policy_kd_mode="none",
    )
    return ReCogDriveDiffusionPlanner(cfg)


def _highcap_action_input(batch: int = 2) -> tuple[torch.Tensor, BatchFeature]:
    data = {
        "his_traj": torch.randn(batch, 12),
        "history_trajectory": torch.randn(batch, 4, 3),
        "status_feature": torch.randn(batch, 8),
        "high_command_one_hot": torch.eye(3)[torch.arange(batch) % 3].float(),
        "jepa_context_tokens": torch.randn(batch, 128, 1024),
        "vggt_geometry_tokens": torch.randn(batch, 192, 512),
        "vggt_geometry_mode_code": torch.full((batch,), GEOMETRY_MODE_TO_CODE["full_geometry"]),
    }
    return torch.randn(batch, 6, 1536), BatchFeature(data=data)


def test_last_vla_decoupled_sample_chain_passes_cot_condition_tokens(monkeypatch):
    torch.manual_seed(902)
    planner = _highcap_planner()
    planner.eval()
    vl_features, action_input = _highcap_action_input()
    seen_cot_conditions = []
    original_forward = planner.model.forward

    def wrapped_forward(*args, **kwargs):
        seen_cot_conditions.append(kwargs.get("cot_condition_tokens"))
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(planner.model, "forward", wrapped_forward)

    with torch.no_grad():
        chain, final_actions = planner.sample_chain(
            vl_features,
            action_input["his_traj"],
            action_input["status_feature"],
            init_actions=torch.zeros(vl_features.shape[0], 8, 3),
            deterministic=True,
            action_input=action_input,
        )

    assert chain.shape[0] == vl_features.shape[0]
    assert chain.shape[-2:] == (8, 3)
    assert chain.shape[1] >= 2
    assert final_actions.shape == (vl_features.shape[0], 8, 3)
    assert any(tokens is not None for tokens in seen_cot_conditions)
    for tokens in seen_cot_conditions:
        assert tokens is not None
        assert tokens.shape[0] == vl_features.shape[0]
        assert tokens.shape[-1] == 384
