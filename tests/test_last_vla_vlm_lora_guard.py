from __future__ import annotations

from dataclasses import dataclass
import sys

import pytest

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


def test_vlm_lora_rejects_cached_hidden_state_training():
    with pytest.raises(ValueError, match="VLM LoRA training requires no-cache"):
        ReCogDriveAgent(
            trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
            checkpoint_path="",
            allow_random_init=True,
            cache_hidden_state=True,
            use_last_vla=True,
            last_vla_stage="cot_alignment",
            use_last_rd=False,
            last_vla_train_vlm_lora=True,
        )
