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


def _make_original_residual_anchor_planner() -> ReCogDriveDiffusionPlanner:
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
        num_inference_steps=2,
        vlm_size="small",
        ddim_cfg=DDIMConfig(num_train_timesteps=10),
        use_expert_features=False,
        use_last_rd=False,
        last_rd_stage="disabled",
        use_last_vla=False,
        last_vla_stage="disabled",
        last_vla_use_residual_diffusion=True,
        last_vla_residual_anchor_source="vlm_text_traj",
        last_vla_require_residual_anchor=True,
        last_vla_residual_alpha_start=1.0,
        last_vla_residual_alpha_end=1.0,
        last_vla_residual_alpha_warmup_epochs=0,
        policy_kd_loss_weight=0.0,
        policy_kd_mode="none",
    )
    return ReCogDriveDiffusionPlanner(cfg)


def _make_original_batch(batch: int = 2) -> tuple[torch.Tensor, BatchFeature]:
    data = {
        "his_traj": torch.randn(batch, 12),
        "history_trajectory": torch.randn(batch, 4, 3),
        "status_feature": torch.randn(batch, 8),
        "action": torch.randn(batch, 8, 3),
        "high_command_one_hot": torch.eye(3)[torch.arange(batch) % 3].float(),
        "vlm_text_trajectory_norm": torch.randn(batch, 8, 3).clamp(-0.3, 0.3),
        "vlm_text_parse_ok": torch.ones(batch),
    }
    return torch.randn(batch, 10, 1536), BatchFeature(data=data)


def test_original_recogdrive_residual_anchor_target_without_last_vla():
    planner = _make_original_residual_anchor_planner()
    assert planner.config.use_last_vla is False

    _, action_input = _make_original_batch()
    gt_norm = planner.norm_odo(action_input["action"])
    target, alpha = planner._last_vla_diffusion_target(
        gt_norm,
        training=True,
        action_input=action_input,
    )

    assert alpha == 1.0
    assert torch.allclose(target, gt_norm - action_input["vlm_text_trajectory_norm"])


def test_original_recogdrive_forward_uses_residual_anchor_without_last_vla():
    planner = _make_original_residual_anchor_planner()
    planner.train()
    vl_features, action_input = _make_original_batch()
    calls = {"target_info": 0}

    original_target_info = planner._last_vla_diffusion_target_info

    def spy_target_info(*args, **kwargs):
        calls["target_info"] += 1
        return original_target_info(*args, **kwargs)

    planner._last_vla_diffusion_target_info = spy_target_info
    planner._denoise_model_output = lambda noisy_actions, *args, **kwargs: torch.zeros_like(noisy_actions)

    out = planner(vl_features, action_input)

    assert calls["target_info"] == 1
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["diffusion_loss"])
    assert torch.allclose(out["last_vla_residual_alpha"], torch.tensor(1.0, device=out["loss"].device))
    assert planner.config.use_last_vla is False


def test_original_recogdrive_get_action_adds_fixed_anchor_without_last_vla():
    planner = _make_original_residual_anchor_planner()
    planner.eval()
    vl_features, action_input = _make_original_batch()
    anchor_norm = torch.full((vl_features.shape[0], 8, 3), 0.2)
    action_input["vlm_text_trajectory_norm"] = anchor_norm

    def fake_p_mean_variance(x, *args, **kwargs):
        return torch.zeros_like(x), torch.full_like(x, -100.0), torch.zeros_like(x)

    planner.p_mean_variance = fake_p_mean_variance
    init_actions = torch.zeros(vl_features.shape[0], 8, 3)

    with torch.no_grad():
        pred = planner.get_action(vl_features, action_input, init_actions=init_actions, deterministic=True)

    assert planner.config.use_last_vla is False
    assert "pred_coarse_traj" not in pred
    assert torch.allclose(pred["pred_residual_norm"], torch.zeros_like(anchor_norm))
    assert torch.allclose(pred["pred_traj"], planner.denorm_odo(anchor_norm), atol=1e-5)
