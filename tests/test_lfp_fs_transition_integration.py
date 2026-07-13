from types import MethodType

import torch

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

_install_dependency_stubs()

from navsim.agents.recogdrive.fs_norm import FSNormStats, save_fs_norm_stats
from navsim.agents.recogdrive.recogdrive_diffusion_planner import (
    DDIMConfig,
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)
from navsim.agents.recogdrive.stage3_lfp_grpo import LFPGRPOConfig
from navsim.agents.recogdrive.stage3_policy_geometry import (
    endpoint_std_from_normalized_transition_floor,
)


def test_sampling_logprob_and_kl_distribution_share_fs_floor(tmp_path) -> None:
    scale = torch.tensor(
        [[1.0, 0.10, 0.05]] * 8,
        dtype=torch.float32,
    )
    stats_path = tmp_path / "fs_stats.pt"
    save_fs_norm_stats(
        str(stats_path),
        FSNormStats(
            mean=torch.zeros_like(scale),
            std=scale,
            clip_lower=torch.full_like(scale, -10.0),
            clip_upper=torch.full_like(scale, 10.0),
            version=2,
            scene_balanced=True,
            heading_center_zero=True,
        ),
    )
    planner = ReCogDriveDiffusionPlanner(
        ReCogDriveDiffusionPlannerConfig(
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
            use_fs_norm=True,
            fs_norm_stats_path=str(stats_path),
            fs_norm_min_version=2,
            fs_norm_output_clip_mode="stats_bounds",
        )
    )
    lfp_cfg = LFPGRPOConfig(
        fs_transition_std_enabled=True,
        fs_endpoint_std_x_m=0.40,
        fs_endpoint_std_y_m=0.16,
        fs_endpoint_std_heading_rad=0.014,
    )
    planner.stage3_algorithm = "lfp_grpo"
    planner.lfp_grpo_cfg = lfp_cfg
    planner.config.lfp_grpo_cfg = lfp_cfg
    planner.min_sampling_denoising_std = 0.04
    planner.min_logprob_denoising_std = 0.04

    calls = []
    original_floor = planner._apply_lfp_transition_std_floor

    def recording_floor(self, std, scalar_floor):
        output = original_floor(std, scalar_floor)
        calls.append((float(scalar_floor), output.detach().clone()))
        return output

    def zero_transition(self, x, *args, **kwargs):
        logvar = torch.full((x.shape[0], 1, 1), -100.0, device=x.device, dtype=x.dtype)
        return torch.zeros_like(x), logvar, torch.zeros_like(x)

    planner._apply_lfp_transition_std_floor = MethodType(recording_floor, planner)
    planner.p_mean_variance = MethodType(zero_transition, planner)
    context_tokens = torch.randn(1, 4, 32)
    prepared_context = {
        "context_tokens": context_tokens,
        "context_mean": context_tokens.mean(dim=1),
        "expert_step_condition": None,
        "planning_condition_tokens": None,
        "cot_condition_tokens": None,
        "diagnostics": {},
    }
    vl_features = torch.randn(1, 4, 1536)
    history = torch.randn(1, 12)
    status = torch.randn(1, 8)
    chains, _ = planner.sample_chain(
        vl_features,
        history,
        status,
        init_actions=torch.zeros(1, 8, 3),
        deterministic=False,
        prepared_dit_context=prepared_context,
    )
    assert len(calls) == 2
    sampling_floor = calls[0][1][0]

    calls.clear()
    distribution = planner._chain_transition_distribution(
        vl_features,
        history,
        status,
        chains,
        deterministic=False,
        prepared_dit_context=prepared_context,
    )
    assert len(calls) == 1
    assert calls[0][0] == planner.min_logprob_denoising_std
    torch.testing.assert_close(distribution.scale[0], sampling_floor)
    endpoint_std = endpoint_std_from_normalized_transition_floor(sampling_floor, scale)
    torch.testing.assert_close(
        endpoint_std,
        torch.tensor([0.40, 0.16, 0.014]),
        rtol=1e-5,
        atol=1e-6,
    )
