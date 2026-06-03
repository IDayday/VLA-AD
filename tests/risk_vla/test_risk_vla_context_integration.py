from __future__ import annotations

import torch

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

_install_dependency_stubs()

from transformers.feature_extraction_utils import BatchFeature  # noqa: E402

from tests.risk_vla.test_planner_config_risk_vla import _config  # noqa: E402
from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner  # noqa: E402


def _batch(batch: int = 2):
    return torch.randn(batch, 5, 1536), BatchFeature(
        data={
            "his_traj": torch.randn(batch, 12),
            "history_trajectory": torch.randn(batch, 4, 3),
            "status_feature": torch.randn(batch, 8),
            "high_command_one_hot": torch.eye(3)[torch.tensor([0, 2])[:batch]],
            "risk_labels": torch.randint(0, 2, (batch, 8, 6)).float(),
            "bit_terminal": torch.randn(batch, 3),
            "bit_path": torch.randn(batch, 8, 3),
        }
    )


def test_context_tokens_increase_when_strategy_scale_positive():
    planner = ReCogDriveDiffusionPlanner(
        _config(use_risk_vla=True, risk_vla_strategy_token_scale=0.25, risk_vla_horizon_residual_scale=0.0)
    )
    vl, action_input = _batch()
    context = planner._prepare_dit_context(vl, action_input, training=True)
    assert context["context_tokens"].shape[1] == vl.shape[1] + 8
    assert context["expert_step_condition"] is None


def test_horizon_residual_added_only_when_scale_positive():
    vl, action_input = _batch()
    planner_zero = ReCogDriveDiffusionPlanner(
        _config(use_risk_vla=True, risk_vla_strategy_token_scale=0.0, risk_vla_horizon_residual_scale=0.0)
    )
    context_zero = planner_zero._prepare_dit_context(vl, action_input, training=True)
    assert context_zero["context_tokens"].shape[1] == vl.shape[1]
    assert context_zero["expert_step_condition"] is None

    planner_residual = ReCogDriveDiffusionPlanner(
        _config(use_risk_vla=True, risk_vla_strategy_token_scale=0.0, risk_vla_horizon_residual_scale=0.25)
    )
    context_residual = planner_residual._prepare_dit_context(vl, action_input, training=True)
    assert context_residual["context_tokens"].shape[1] == vl.shape[1]
    assert context_residual["expert_step_condition"].shape == (2, 8, 384)
