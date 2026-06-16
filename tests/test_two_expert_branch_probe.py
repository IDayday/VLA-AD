from __future__ import annotations

import torch

from navsim.agents.recogdrive.two_expert_adapters import TwoExpertTrajectoryProbe


def test_branch_probe_outputs_fused_dyn_geo_losses():
    probe = TwoExpertTrajectoryProbe(vlm_hidden_dim=16, adapter_dim=8, action_horizon=8, action_dim=3)
    out = probe(
        torch.randn(2, 3, 12, 16),
        torch.randn(2, 12, 16),
        status_feature=torch.randn(2, 8),
        high_command_one_hot=torch.eye(3)[:2],
        history_trajectory=torch.randn(2, 4, 3),
        target_action_norm=torch.randn(2, 8, 3),
    )
    assert out["probe_traj_norm"].shape == (2, 8, 3)
    assert out["probe_dyn_traj_norm"].shape == (2, 8, 3)
    assert out["probe_geo_traj_norm"].shape == (2, 8, 3)
    for key in ("probe_loss", "probe_dyn_loss", "probe_geo_loss", "geo_lateral_profile_loss", "geo_heading_profile_loss"):
        assert torch.isfinite(out["losses"][key])
