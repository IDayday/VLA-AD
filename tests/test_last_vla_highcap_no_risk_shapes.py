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


def _planner() -> ReCogDriveDiffusionPlanner:
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
        last_vla_stage="progressive_sft_bottleneck",
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
        last_vla_vlm_summary_tokens=64,
        last_vla_use_risk_head=False,
        last_vla_risk_loss_weight=0.0,
        last_vla_geometry_teacher_dim=512,
        last_vla_geometry_grid_rows=12,
        last_vla_geometry_grid_cols=16,
        last_vla_require_full_geometry=True,
        last_vla_allow_patch_geometry_fallback=False,
        last_vla_raw_vlm_context_to_dit=False,
        last_vla_cot_bottleneck_mode=True,
        last_vla_use_residual_diffusion=True,
        last_vla_vlm_summary_keep_start=1.0,
        last_vla_vlm_summary_keep_end=1.0,
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


def _batch(batch: int = 1, include_targets: bool = True) -> tuple[torch.Tensor, BatchFeature]:
    data = {
        "his_traj": torch.randn(batch, 12),
        "history_trajectory": torch.randn(batch, 4, 3),
        "status_feature": torch.randn(batch, 8),
        "action": torch.randn(batch, 8, 3),
        "high_command_one_hot": torch.eye(3)[torch.arange(batch) % 3].float(),
        "jepa_context_tokens": torch.randn(batch, 128, 1024),
        "vggt_geometry_tokens": torch.randn(batch, 192, 512),
        "vggt_geometry_mode_code": torch.full((batch,), GEOMETRY_MODE_TO_CODE["full_geometry"]),
    }
    if include_targets:
        data["jepa_target_tokens"] = torch.randn(batch, 128, 1024)
    return torch.randn(batch, 6, 1536), BatchFeature(data=data)


def test_highcap_no_risk_planner_shapes_and_get_action():
    torch.manual_seed(128)
    planner = _planner()
    vl_features, action_input = _batch(include_targets=True)

    planner.train()
    train_out = planner(vl_features, action_input)
    train_context = planner._prepare_dit_context(vl_features, action_input, training=True, allow_target_tokens=True)
    last_vla = train_context["last_vla_output"]
    assert train_out["last_vla_risk_loss"].item() == 0.0
    assert last_vla.risk_logits is None
    assert last_vla.predicted_geometry.shape == (1, 192, 512)
    assert last_vla.predicted_future_jepa.shape == (1, 128, 1024)
    assert last_vla.coarse_traj_norm.shape == (1, 8, 3)
    assert torch.isfinite(train_out["last_vla_geometry_loss"])
    assert torch.isfinite(train_out["last_vla_dynamic_loss"])

    planner.eval()
    context_only = BatchFeature(data={key: value for key, value in action_input.items() if "target" not in key and key != "action"})
    with torch.no_grad():
        dit_context = planner._prepare_dit_context(vl_features, context_only, training=False, allow_target_tokens=False)
        last_vla_eval = dit_context["last_vla_output"]
        pred = planner.get_action(
            vl_features,
            context_only,
            init_actions=torch.zeros(1, 8, 3),
            deterministic=True,
        )
    assert dit_context["context_tokens"].shape == (1, 256, 384)
    assert last_vla_eval.diagnostics["last_vla_use_risk_head"].item() == 0.0
    assert last_vla_eval.diagnostics["last_vla_num_risk_tokens"].item() == 0.0
    assert last_vla_eval.diagnostics["last_vla_context_token_count"].item() == 256.0
    assert last_vla_eval.diagnostics["last_vla_cot_token_count"].item() == 192.0
    assert last_vla_eval.diagnostics["last_vla_vlm_summary_token_count"].item() == 64.0
    assert pred["pred_traj"].shape == (1, 8, 3)
    assert torch.isfinite(pred["pred_traj"]).all()
