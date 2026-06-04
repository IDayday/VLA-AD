from __future__ import annotations

from dataclasses import dataclass
import sys
from types import ModuleType, SimpleNamespace

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

    abstract_agent_mod = sys.modules.get("navsim.agents.abstract_agent")
    if abstract_agent_mod is None:
        abstract_agent_mod = ModuleType("navsim.agents.abstract_agent")
        sys.modules["navsim.agents.abstract_agent"] = abstract_agent_mod

    class AbstractAgent(torch.nn.Module):
        pass

    abstract_agent_mod.AbstractAgent = AbstractAgent


_patch_agent_stubs()

from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class DummyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(1, 1)
        self.lora_A = torch.nn.Linear(1, 1)


class DummyInternVL(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(2, 2, bias=False)

    def forward(
        self,
        *,
        pixel_values,
        input_ids,
        attention_mask=None,
        position_ids=None,
        image_flags=None,
        output_hidden_states=False,
        return_dict=False,
    ):
        return {"projected": self.q_proj(pixel_values.float()), "input_ids": input_ids}


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


def test_vlm_lora_uses_generic_peft_forward(monkeypatch):
    class FakeLoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakePeftModel(torch.nn.Module):
        def __init__(self, base_model, config):
            super().__init__()
            self.base_model = base_model
            self.config = config

        def forward(self, **kwargs):
            assert "inputs_embeds" not in kwargs
            return self.base_model(**kwargs)

    def fake_get_peft_model(base_model, config):
        assert "task_type" not in config.kwargs
        base_model.q_proj.lora_A = torch.nn.Parameter(torch.ones(1, 2))
        base_model.q_proj.lora_B = torch.nn.Parameter(torch.ones(2, 1))
        return FakePeftModel(base_model, config)

    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(LoraConfig=FakeLoraConfig, get_peft_model=fake_get_peft_model),
    )
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
    agent.backbone = SimpleNamespace(model=DummyInternVL())
    agent.last_vla_vlm_lora_target_modules = "q_proj"

    agent._enable_last_vla_vlm_lora()
    output = agent.backbone.model(
        pixel_values=torch.zeros(1, 2),
        input_ids=torch.ones(1, 1, dtype=torch.long),
        attention_mask=torch.ones(1, 1, dtype=torch.long),
        position_ids=torch.zeros(1, 1, dtype=torch.long),
        image_flags=torch.ones(1, dtype=torch.long),
        output_hidden_states=True,
        return_dict=True,
    )

    assert output["projected"].shape == (1, 2)
