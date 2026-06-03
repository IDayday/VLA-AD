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

    dataclasses_mod.AgentInput = getattr(dataclasses_mod, "AgentInput", AgentInput)
    dataclasses_mod.SensorConfig = getattr(dataclasses_mod, "SensorConfig", SensorConfig)


_patch_agent_stubs()

from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class DummyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(1, 1)
        self.lora_A = torch.nn.Linear(1, 1)


def test_vlm_lora_scope_trains_only_lora_and_last_vla_cot():
    agent = ReCogDriveAgent(
        trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
        checkpoint_path="",
        allow_random_init=True,
        cache_hidden_state=True,
        use_last_vla=True,
        last_vla_stage="cot_alignment",
        use_last_rd=False,
        train_expert_only=True,
        freeze_base_action_head=True,
    )
    agent.initialize()
    agent.backbone = DummyBackbone()
    agent.last_vla_train_vlm_lora = True
    agent._set_trainable_parameters()
    counts = agent.count_trainable_parameters_by_group()

    assert counts["last_vla_cot"]["trainable"] > 0
    assert counts["vlm_lora"]["trainable"] > 0
    assert counts["backbone_non_lora"]["trainable"] == 0
    assert counts["action_base"]["trainable"] == 0
