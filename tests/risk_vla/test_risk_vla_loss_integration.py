from __future__ import annotations

import torch

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

_install_dependency_stubs()

from transformers.feature_extraction_utils import BatchFeature  # noqa: E402

from tests.risk_vla.test_planner_config_risk_vla import _config  # noqa: E402
from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner  # noqa: E402


def _batch(include_labels: bool):
    batch = 2
    data = {
        "his_traj": torch.randn(batch, 12),
        "history_trajectory": torch.randn(batch, 4, 3),
        "status_feature": torch.randn(batch, 8),
        "high_command_one_hot": torch.eye(3)[torch.tensor([0, 2])],
        "action": torch.randn(batch, 8, 3),
    }
    if include_labels:
        data["risk_labels"] = torch.randint(0, 2, (batch, 8, 6)).float()
    return torch.randn(batch, 5, 1536), BatchFeature(data=data)


def test_risk_losses_zero_without_labels_and_nonzero_with_labels():
    planner = ReCogDriveDiffusionPlanner(
        _config(
            use_risk_vla=True,
            risk_vla_strategy_token_scale=0.0,
            risk_vla_horizon_residual_scale=0.0,
            risk_vla_risk_loss_weight=0.05,
        )
    )
    planner.train()
    vl, action_input = _batch(include_labels=False)
    no_label_out = planner(vl, action_input)
    assert no_label_out["risk_vla_risk_loss"].item() == 0.0
    assert no_label_out["risk_vla_total_aux_loss"].item() == 0.0

    vl, action_input = _batch(include_labels=True)
    label_out = planner(vl, action_input)
    assert label_out["risk_vla_risk_loss"].item() > 0.0
    assert label_out["risk_vla_total_aux_loss"].item() > 0.0


def test_risk_loss_zero_when_weight_zero_even_with_labels():
    planner = ReCogDriveDiffusionPlanner(
        _config(use_risk_vla=True, risk_vla_risk_loss_weight=0.0, risk_vla_strategy_token_scale=0.0, risk_vla_horizon_residual_scale=0.0)
    )
    planner.train()
    vl, action_input = _batch(include_labels=True)
    out = planner(vl, action_input)
    assert out["risk_vla_risk_loss"].item() == 0.0
    assert out["risk_vla_total_aux_loss"].item() == 0.0
