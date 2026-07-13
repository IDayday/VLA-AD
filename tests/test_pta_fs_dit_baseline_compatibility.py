from __future__ import annotations

import torch
import pytest

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

_install_dependency_stubs()

from navsim.agents.recogdrive.fs_norm import FSNormStats, save_fs_norm_stats
from navsim.agents.recogdrive.last_vla_cot_planning import CoTCoarseTrajectoryHead, LastVLACoTConfig
from navsim.agents.recogdrive.recogdrive_dit import LightningDiT
from navsim.agents.recogdrive.recogdrive_diffusion_planner import (
    DDIMConfig,
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)


def _dit() -> LightningDiT:
    return LightningDiT(
        num_heads=2,
        head_dim=16,
        output_dim=32,
        num_layers=2,
        dropout=0.0,
        attention_bias=True,
        interleave_attention=True,
    )


def test_flag_off_output_is_unchanged_at_fixed_seed() -> None:
    torch.manual_seed(43)
    model = _dit().eval()
    inputs = (
        torch.randn(2, 8, 32),
        torch.randn(2, 5, 32),
        torch.randn(2, 32),
        torch.tensor([1, 2]),
    )
    with torch.no_grad():
        baseline = model(*inputs)
        model.set_planning_gate_init(0.05)
        no_condition = model(*inputs, planning_condition_tokens=None)
    assert torch.equal(baseline, no_condition)


def test_old_checkpoint_loads_with_strict_false() -> None:
    old_model = _dit()
    old_state = {
        key: value
        for key, value in old_model.state_dict().items()
        if ".planning_" not in key
    }
    new_model = _dit()
    incompatible = new_model.load_state_dict(old_state, strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(".planning_" in key for key in incompatible.missing_keys)


def test_last_vla_fs_rejects_legacy_heading_and_progress_losses(tmp_path) -> None:
    stats_path = tmp_path / "stats_v2.pt"
    save_fs_norm_stats(
        str(stats_path),
        FSNormStats(
            mean=torch.zeros(8, 3),
            std=torch.ones(8, 3),
            version=2,
            scene_balanced=True,
            heading_center_zero=True,
        ),
    )
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
        use_fs_norm=True,
        fs_norm_stats_path=str(stats_path),
        fs_norm_min_version=2,
        use_last_vla=True,
        last_vla_stage="progressive_sft_decoupled",
        last_vla_heading_loss_weight=0.1,
        last_vla_progress_loss_weight=0.0,
    )
    with pytest.raises(ValueError, match="heading_loss_weight=0"):
        ReCogDriveDiffusionPlanner(cfg)


def test_fs_v2_rejects_robust_mode_mismatch(tmp_path) -> None:
    stats_path = tmp_path / "robust_stats_v2.pt"
    save_fs_norm_stats(
        str(stats_path),
        FSNormStats(
            mean=torch.zeros(8, 3),
            std=torch.ones(8, 3),
            median=torch.zeros(8, 3),
            mad=torch.ones(8, 3),
            use_robust=True,
            version=2,
            scene_balanced=True,
            heading_center_zero=True,
        ),
    )
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
        use_fs_norm=True,
        fs_norm_stats_path=str(stats_path),
        fs_norm_use_robust=False,
        fs_norm_min_version=2,
    )
    with pytest.raises(ValueError, match="use_robust must match"):
        ReCogDriveDiffusionPlanner(cfg)


def test_last_vla_fs_coarse_head_uses_linear_representation_bound() -> None:
    head = CoTCoarseTrajectoryHead(
        LastVLACoTConfig(
            planner_dim=32,
            hidden_dim=64,
            action_horizon=8,
            ego_tokens=2,
            use_fs_norm=True,
        )
    )
    with torch.no_grad():
        for parameter in head.traj_head.parameters():
            parameter.zero_()
        head.traj_head[-1].bias.fill_(3.0)
    target = torch.zeros(2, 8, 3)
    coarse, _, losses = head(
        torch.randn(2, 4, 32),
        torch.randn(2, 8),
        torch.eye(3)[:2],
        torch.randn(2, 12),
        target,
        output_bound_fn=lambda value: value.clamp(-4.0, 4.0),
    )

    assert torch.allclose(coarse, torch.full_like(coarse, 3.0))
    assert coarse.abs().max() > 1.0
    assert losses["heading_loss"] == 0
    assert losses["progress_loss"] == 0
