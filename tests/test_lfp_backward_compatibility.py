import torch

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

_install_dependency_stubs()

from navsim.agents.recogdrive.recogdrive_diffusion_planner import (
    DDIMConfig,
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)


def _legacy_planner() -> ReCogDriveDiffusionPlanner:
    return ReCogDriveDiffusionPlanner(
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
        )
    )


def test_lfp_off_does_not_create_stage3_runtime_or_change_state_dict() -> None:
    planner = _legacy_planner()
    assert planner.stage3_algorithm == "legacy"
    assert planner.lfp_metric_adapter is None
    assert planner.lfp_reference_cache is None
    assert not hasattr(planner, "old_policy")
    assert not any(key.startswith("lfp_") for key in planner.state_dict())


def test_legacy_checkpoint_loads_strict_false_with_lfp_code_present() -> None:
    source = _legacy_planner()
    target = _legacy_planner()
    incompatible = target.load_state_dict(source.state_dict(), strict=False)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    for source_value, target_value in zip(source.state_dict().values(), target.state_dict().values()):
        torch.testing.assert_close(source_value, target_value)
