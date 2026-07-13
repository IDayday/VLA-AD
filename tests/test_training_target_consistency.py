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


def _planner() -> ReCogDriveDiffusionPlanner:
    cfg = ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg={
            "num_heads": 2,
            "head_dim": 16,
            "num_layers": 2,
            "output_dim": 32,
            "dropout": 0.0,
            "attention_bias": True,
            "norm_eps": 1e-5,
            "interleave_attention": True,
        },
        input_embedding_dim=32,
        planner_dim=32,
        hidden_size=64,
        action_horizon=8,
        sampling_method="ddim",
        num_inference_steps=2,
        vlm_size="small",
        ddim_cfg=DDIMConfig(num_train_timesteps=10),
    )
    return ReCogDriveDiffusionPlanner(cfg)


def test_diffusion_x0_and_geometry_share_selected_target(monkeypatch) -> None:
    torch.manual_seed(31)
    planner = _planner()
    selected = torch.randn(1, 2, 8, 3)
    action_input = BatchFeature(
        data={
            "action": torch.full((1, 8, 3), 100.0),
            "his_traj": torch.randn(1, 12),
            "history_trajectory": torch.randn(1, 4, 3),
            "status_feature": torch.randn(1, 8),
            "high_command_one_hot": torch.tensor([[1.0, 0.0, 0.0]]),
        }
    )
    seen = {}
    original_build = planner._build_training_target

    def build_spy(**kwargs):
        target = original_build(**kwargs)
        seen["training_target"] = target
        return target

    def denoise_spy(noisy_actions, timesteps, dit_context, repeated_action_input):
        seen["noisy_actions"] = noisy_actions.detach().clone()
        return torch.zeros_like(noisy_actions)

    def legacy_aux_spy(x0_repr, target_raw, timesteps, *, method):
        seen["geometry_target"] = target_raw.detach().clone()
        count = target_raw.shape[0]
        zeros = target_raw.new_zeros((count,))
        return zeros, zeros, torch.ones(count, device=target_raw.device, dtype=torch.bool), {
            "delta_aux_per_sample": zeros,
            "early_kink_rate": zeros.new_zeros(()),
            "tail_reverse_rate": zeros.new_zeros(()),
            "curvature_violation_rate": zeros.new_zeros(()),
        }

    def pta_aux_spy(x0_repr, training_target, timesteps, *, method):
        seen["pta_target"] = training_target
        zeros = x0_repr.new_zeros((x0_repr.shape[0],))
        return zeros, zeros, {}

    monkeypatch.setattr(planner, "_build_training_target", build_spy)
    monkeypatch.setattr(planner, "_prepare_dit_context", lambda *args, **kwargs: {})
    monkeypatch.setattr(planner, "_denoise_model_output", denoise_spy)
    monkeypatch.setattr(planner, "_compute_x0_geo_aux_per_sample_losses", legacy_aux_spy)
    monkeypatch.setattr(planner, "_compute_pta_aux_per_sample_losses", pta_aux_spy)

    flat_shape = (2, 8, 3)
    noise = torch.full(flat_shape, 0.25)
    timesteps = torch.tensor([2, 7])
    planner._diffusion_per_target_loss_on_targets(
        torch.randn(1, 4, 1536),
        action_input,
        selected,
        noise=noise,
        t_discrete=timesteps,
        target_source_code=torch.tensor([[3, 1]]),
    )

    target = seen["training_target"]
    flat_selected = selected.reshape(flat_shape)
    assert torch.allclose(target.raw_trajectory, flat_selected)
    assert torch.allclose(seen["geometry_target"], flat_selected)
    assert seen["pta_target"] is target
    assert not torch.allclose(target.raw_trajectory, action_input.action.expand_as(target.raw_trajectory))
    expected_noisy = (
        planner.extract(planner.ddpm_sqrt_alphas_cumprod, timesteps, flat_shape) * target.diffusion_target_repr
        + planner.extract(planner.ddpm_sqrt_one_minus_alphas_cumprod, timesteps, flat_shape) * noise
    )
    assert torch.allclose(seen["noisy_actions"], expected_noisy)
