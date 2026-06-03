from __future__ import annotations

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

_install_dependency_stubs()

from navsim.agents.recogdrive.recogdrive_diffusion_planner import (  # noqa: E402
    DDIMConfig,
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)


def _config(**overrides):
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
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_default_config_disables_risk_vla():
    cfg = _config()
    assert cfg.use_risk_vla is False
    planner = ReCogDriveDiffusionPlanner(cfg)
    assert not hasattr(planner, "risk_state_encoder")


def test_config_enabled_constructs_risk_modules_on_cpu():
    planner = ReCogDriveDiffusionPlanner(_config(use_risk_vla=True))
    assert hasattr(planner, "risk_state_encoder")
    assert hasattr(planner, "risk_strategy_router")
    assert hasattr(planner, "risk_strategy_bank")
