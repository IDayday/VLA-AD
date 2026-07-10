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
        action_dim=3,
        action_horizon=8,
        max_seq_len=8,
        sampling_method="ddim",
        num_inference_steps=5,
        vlm_size="small",
        ddim_cfg=DDIMConfig(num_train_timesteps=10),
        use_planning_token_adapter=True,
        planning_num_tokens=16,
        planning_num_heads=4,
        planning_condition_dropout=0.0,
    )
    return ReCogDriveDiffusionPlanner(cfg).eval()


def test_static_planning_adapter_runs_once_in_five_step_ddim(monkeypatch) -> None:
    torch.manual_seed(29)
    planner = _planner()
    assert planner.planning_adapter is not None
    calls = 0
    original = planner.planning_adapter.forward

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(planner.planning_adapter, "forward", counted)
    batch = 2
    vl_features = torch.randn(batch, 5, 1536)
    action_input = BatchFeature(
        data={
            "his_traj": torch.randn(batch, 12),
            "history_trajectory": torch.randn(batch, 4, 3),
            "status_feature": torch.randn(batch, 8),
            "high_command_one_hot": torch.eye(3)[:batch],
        }
    )

    with torch.no_grad():
        chain, trajectory = planner.sample_chain(
            vl_features,
            action_input.his_traj,
            action_input.status_feature,
            init_actions=torch.zeros(batch, 8, 3),
            deterministic=True,
            action_input=action_input,
        )

    assert chain.shape == (batch, 6, 8, 3)
    assert trajectory.shape == (batch, 8, 3)
    assert calls == 1
    assert planner.planning_adapter.forward_count == 1
