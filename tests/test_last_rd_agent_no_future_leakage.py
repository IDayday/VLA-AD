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


def _agent():
    from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    return ReCogDriveAgent(
        trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
        checkpoint_path="",
        allow_random_init=True,
        cache_hidden_state=True,
        use_last_rd=True,
        use_expert_features=False,
        last_rd_stage="progressive_sft",
        policy_kd_loss_weight=0.0,
        policy_kd_mode="none",
        sampling_method="ddim",
    )


def _features(include_targets: bool) -> dict[str, torch.Tensor]:
    batch = 2
    features = {
        "history_trajectory": torch.randn(batch, 4, 3),
        "high_command_one_hot": torch.eye(3)[torch.tensor([0, 2])].float(),
        "last_hidden_state": torch.randn(batch, 6, 1536),
        "status_feature": torch.randn(batch, 8),
        "jepa_context_tokens": torch.randn(batch, 12, 1024),
        "vggt_context_tokens": torch.randn(batch, 12, 2048),
        "vggt_geometry_tokens": torch.randn(batch, 12, 2048),
    }
    if include_targets:
        features.update({
            "jepa_target_tokens": torch.randn(batch, 12, 1024),
            "vggt_target_tokens": torch.randn(batch, 12, 2048),
            "vggt_geometry_target_tokens": torch.randn(batch, 12, 2048),
        })
    return features


def test_agent_eval_forward_drops_future_teacher_targets():
    _patch_agent_stubs()
    torch.manual_seed(41)
    agent = _agent()
    agent.eval()
    captured_keys: list[set[str]] = []

    def fake_get_action(vl_features, action_input, *args, **kwargs):
        captured_keys.append(set(action_input.keys()))
        return {"pred_traj": torch.zeros(vl_features.shape[0], 8, 3, device=vl_features.device)}

    agent.action_head.get_action = fake_get_action

    with torch.no_grad():
        out_without_targets = agent.forward(_features(include_targets=False), targets=None)
        out_with_targets = agent.forward(_features(include_targets=True), targets=None)

    assert torch.allclose(out_without_targets["pred_traj"], out_with_targets["pred_traj"], atol=0.0, rtol=0.0)
    assert len(captured_keys) == 2
    assert not any("target" in key for key in captured_keys[0])
    assert not any("target" in key for key in captured_keys[1])
