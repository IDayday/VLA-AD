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


def _agent(adapter_path: str | None = None):
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent

    return ReCogDriveAgent(
        trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
        checkpoint_path="",
        allow_random_init=True,
        cache_hidden_state=True,
        use_last_rd=True,
        last_rd_stage="stage1_5",
        use_expert_features=False,
        allow_expert_target_features=True,
        diffusion_loss_weight=0.0,
        future_jepa_loss_weight=0.3,
        vggt_geometry_loss_weight=0.1,
        coarse_traj_loss_weight=0.5,
        risk_loss_weight=0.0,
        policy_kd_loss_weight=0.0,
        last_rd_adapter_checkpoint=adapter_path,
    )


def test_last_rd_adapter_checkpoint_loads_only_last_rd(tmp_path):
    _patch_agent_stubs()
    torch.manual_seed(29)
    reference = _agent()
    reference_state = reference.state_dict()
    adapter_state = {
        key: value.detach().clone() + 0.125
        for key, value in reference_state.items()
        if key.startswith("action_head.last_rd.") and value.is_floating_point()
    }
    assert adapter_state
    adapter_path = tmp_path / "last_rd_adapter.pt"
    torch.save({"state_dict": adapter_state}, adapter_path)

    torch.manual_seed(29)
    loaded = _agent(str(adapter_path))
    loaded_state = loaded.state_dict()

    changed_key = next(iter(adapter_state))
    assert not torch.allclose(loaded_state[changed_key], reference_state[changed_key])
    base_key = next(key for key in reference_state if key.startswith("action_head.") and "last_rd" not in key and reference_state[key].is_floating_point())
    assert torch.allclose(loaded_state[base_key], reference_state[base_key])
