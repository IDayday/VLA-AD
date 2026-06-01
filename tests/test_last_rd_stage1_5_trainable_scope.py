from __future__ import annotations

from dataclasses import dataclass
import sys

import torch

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

_install_dependency_stubs()


def _patch_agent_stubs() -> None:
    dataclasses_mod = sys.modules["navsim.common.dataclasses"]

    @dataclass
    class AgentInput:
        pass

    class SensorConfig:
        @staticmethod
        def build_all_sensors(include=None):
            return SensorConfig()

    if not hasattr(dataclasses_mod, "AgentInput"):
        dataclasses_mod.AgentInput = AgentInput
    if not hasattr(dataclasses_mod, "SensorConfig"):
        dataclasses_mod.SensorConfig = SensorConfig


def test_stage1_5_trainable_scope_only_last_rd():
    _patch_agent_stubs()
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent

    torch.manual_seed(23)
    agent = ReCogDriveAgent(
        trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
        checkpoint_path="",
        allow_random_init=True,
        cache_hidden_state=True,
        use_last_rd=True,
        last_rd_stage="stage1_5",
        use_expert_features=False,
        use_jepa=True,
        use_vggt=True,
        allow_expert_target_features=True,
        diffusion_loss_weight=0.0,
        future_jepa_loss_weight=0.3,
        vggt_geometry_loss_weight=0.1,
        coarse_traj_loss_weight=0.5,
        risk_loss_weight=0.0,
        train_expert_only=True,
        freeze_base_action_head=True,
        freeze_expert=False,
    )
    agent.initialize()
    trainable = [name for name, parameter in agent.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all("action_head.last_rd." in name for name in trainable)
    assert all(
        not parameter.requires_grad
        for name, parameter in agent.named_parameters()
        if name.startswith("action_head.") and "action_head.last_rd." not in name
    )
    assert all(not parameter.requires_grad for name, parameter in agent.named_parameters() if name.startswith("backbone."))
