import torch

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

_install_dependency_stubs()

from transformers.feature_extraction_utils import BatchFeature  # noqa: E402

from navsim.agents.recogdrive.recogdrive_diffusion_planner import (  # noqa: E402
    DDIMConfig,
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)


def tiny_config(use_aux: bool = False) -> ReCogDriveDiffusionPlannerConfig:
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
        use_action_aware_aux=use_aux,
        action_aware_aux_weight=0.5 if use_aux else 0.0,
        action_aware_aux_space="norm_odo",
        ddim_cfg=DDIMConfig(num_train_timesteps=10),
    )


def synthetic_batch(batch_size: int = 2) -> tuple[torch.Tensor, BatchFeature]:
    hidden = torch.randn(batch_size, 5, 6)
    action_input = BatchFeature(
        data={
            "his_traj": torch.randn(batch_size, 12),
            "status_feature": torch.randn(batch_size, 8),
            "action": torch.randn(batch_size, 8, 3),
        }
    )
    return hidden, action_input


def test_action_aware_aux_head_adds_finite_loss_when_enabled():
    torch.manual_seed(21)
    planner = ReCogDriveDiffusionPlanner(tiny_config(use_aux=True)).train()
    hidden, action_input = synthetic_batch()

    output = planner(hidden, action_input)

    assert "action_aware_aux_loss" in output
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["action_aware_aux_loss"])
    assert output["action_aware_aux_loss"].item() >= 0.0


def test_action_aware_aux_head_default_off_is_zero():
    torch.manual_seed(22)
    planner = ReCogDriveDiffusionPlanner(tiny_config(use_aux=False)).train()
    hidden, action_input = synthetic_batch()

    output = planner(hidden, action_input)

    assert torch.isfinite(output["loss"])
    assert output["action_aware_aux_loss"].item() == 0.0
