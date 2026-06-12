from __future__ import annotations

from argparse import Namespace

import pytest
import torch
from torch import nn

from navsim.agents.recogdrive.two_expert_slots import TwoExpertSlotConfig, TwoExpertSoftSlots
from scripts.last_vla_v2.two_expert_slot.build_two_expert_hidden_cache import load_stage1_state_for_hidden_cache


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Linear(4, 4)


def _args(train_vlm_mode: str = "frozen") -> Namespace:
    return Namespace(stage1_train_mode=None, train_vlm_mode=train_vlm_mode, vlm_lora_adapter_dir=None)


def _write_checkpoint(path, *, train_mode: str, config: TwoExpertSlotConfig, fill: float = 3.0) -> torch.Tensor:
    slots = TwoExpertSoftSlots(config)
    with torch.no_grad():
        slots.dyn_slots.fill_(fill)
        slots.geo_slots.fill_(fill + 1.0)
    state = {f"two_expert_slots.{key}": value.clone() for key, value in slots.state_dict().items()}
    torch.save({"state_dict": state, "stage1_metadata": {"train_mode": train_mode}}, path)
    return slots.dyn_slots.detach().clone()


def test_hidden_cache_loader_loads_changed_slot_state(tmp_path):
    config = TwoExpertSlotConfig(vlm_hidden_dim=8, planner_dim=4)
    ckpt = tmp_path / "stage1.ckpt"
    expected_dyn = _write_checkpoint(ckpt, train_mode="frozen", config=config)

    slots, report = load_stage1_state_for_hidden_cache(
        checkpoint=ckpt,
        config=config,
        backbone=None,
        args=_args("frozen"),
    )

    assert torch.allclose(slots.dyn_slots, expected_dyn)
    assert report["stage1_train_mode"] == "frozen"
    assert "dyn_slots" in report["loaded_slot_keys"]
    assert report["loaded_lora_adapter"] is False


def test_hidden_cache_loader_lora_without_adapter_fails(tmp_path):
    config = TwoExpertSlotConfig(vlm_hidden_dim=8, planner_dim=4)
    ckpt = tmp_path / "stage1_lora.ckpt"
    _write_checkpoint(ckpt, train_mode="lora", config=config)

    with pytest.raises(RuntimeError, match="no VLM LoRA adapter"):
        load_stage1_state_for_hidden_cache(
            checkpoint=ckpt,
            config=config,
            backbone=_Backbone(),
            args=_args("lora"),
        )
