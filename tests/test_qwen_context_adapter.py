import torch

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

_install_dependency_stubs()

from navsim.agents.recogdrive.recogdrive_diffusion_planner import (  # noqa: E402
    DDIMConfig,
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)


def tiny_config(adapter_type: str = "linear") -> ReCogDriveDiffusionPlannerConfig:
    return ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg={
            "num_heads": 1,
            "head_dim": 8,
            "num_layers": 1,
            "output_dim": 8,
            "dropout": 0.0,
            "attention_bias": True,
            "norm_eps": 1e-5,
            "interleave_attention": True,
        },
        input_embedding_dim=8,
        hidden_size=16,
        action_dim=3,
        action_horizon=8,
        max_seq_len=8,
        sampling_method="ddim",
        num_inference_steps=1,
        vlm_feature_dim=6,
        use_expert_features=False,
        vlm_adapter_type=adapter_type,
        vlm_adapter_dropout=0.0,
        ddim_cfg=DDIMConfig(num_train_timesteps=10),
    )


def test_linear_adapter_matches_feature_encoder_mean_contract():
    torch.manual_seed(11)
    planner = ReCogDriveDiffusionPlanner(tiny_config("linear"))
    hidden = torch.randn(2, 5, 6)
    action_input = {"his_traj": torch.randn(2, 12), "status_feature": torch.randn(2, 8)}

    context = planner._prepare_dit_context(hidden, action_input, training=False)
    expected = planner.feature_encoder(hidden).mean(1)

    assert torch.allclose(context["context_mean"], expected)
    assert torch.isfinite(context["context_tokens"]).all()


def test_pre_ln_linear_post_ln_adapter_shape_and_finite():
    torch.manual_seed(12)
    planner = ReCogDriveDiffusionPlanner(tiny_config("pre_ln_linear_post_ln"))
    hidden = torch.randn(2, 7, 6)

    encoded = planner._encode_vlm(hidden)

    assert encoded.shape == (2, 7, 8)
    assert torch.isfinite(encoded).all()


def test_mlp_adapter_shape_and_finite():
    torch.manual_seed(13)
    planner = ReCogDriveDiffusionPlanner(tiny_config("mlp_adapter"))
    hidden = torch.randn(2, 4, 6)

    encoded = planner._encode_vlm(hidden)

    assert encoded.shape == (2, 4, 8)
    assert torch.isfinite(encoded).all()


def test_config_default_adapter_is_linear():
    cfg = tiny_config()

    assert cfg.vlm_adapter_type == "linear"
