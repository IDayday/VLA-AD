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


def make_two_expert_planner() -> ReCogDriveDiffusionPlanner:
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
        use_two_expert_slots=True,
        two_expert_cache_mode=True,
        two_expert_condition_mode="horizon_hmef_lite",
        two_expert_dit_condition_mode="horizon_hmef_lite",
        two_expert_zero_init_deltas=True,
        use_expert_features=False,
        use_last_vla=False,
        last_vla_stage="disabled",
        use_last_rd=False,
        last_rd_stage="disabled",
        last_vla_use_residual_diffusion=False,
        last_vla_teacher_traj_mode="none",
    )
    return ReCogDriveDiffusionPlanner(cfg)


def make_two_expert_batch(batch: int = 2) -> tuple[torch.Tensor, BatchFeature]:
    return torch.randn(batch, 9, 1536), BatchFeature(
        data={
            "his_traj": torch.randn(batch, 12),
            "history_trajectory": torch.randn(batch, 4, 3),
            "status_feature": torch.randn(batch, 8),
            "action": torch.randn(batch, 8, 3),
            "two_expert_h_dyn": torch.randn(batch, 3, 12, 1536),
            "two_expert_h_geo": torch.randn(batch, 12, 1536),
        }
    )


def test_two_expert_stage2_context_preserves_raw_vlm_and_horizon_aligns():
    planner = make_two_expert_planner()
    vl, action_input = make_two_expert_batch()
    context = planner._prepare_dit_context(vl, action_input, training=True)

    assert context["context_tokens"].shape == (2, 9, 384)
    assert context["two_expert_f_dyn"].shape == (2, 8, 384)
    assert context["two_expert_f_geo"].shape == (2, 8, 384)
    assert context["expert_step_condition"].shape == (2, 8, 384)
    assert torch.allclose(context["expert_step_condition"], torch.zeros_like(context["expert_step_condition"]))
    assert context["cot_condition_tokens"] is None
    assert not hasattr(planner, "last_vla_cot")
    assert context["diagnostics"]["two_expert_raw_vlm_context_used"].item() == 1.0
    assert context["diagnostics"]["two_expert_zero_init_dyn"].item() == 1.0
    assert context["diagnostics"]["two_expert_zero_init_geo"].item() == 1.0


def test_two_expert_training_forward_outputs_diagnostics():
    torch.manual_seed(31)
    planner = make_two_expert_planner()
    planner.train()
    vl, action_input = make_two_expert_batch()
    out = planner(vl, action_input)

    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["diffusion_loss"])
    assert out["two_expert_condition_enabled"].item() == 1.0
    assert out["two_expert_raw_vlm_context_used"].item() == 1.0
