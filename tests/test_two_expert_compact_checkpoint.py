from __future__ import annotations

from argparse import Namespace

import torch
from torch import nn

from navsim.agents.recogdrive.two_expert_slots import TwoExpertSlotConfig, TwoExpertSoftSlots
from scripts.last_vla_v2.two_expert_slot.build_two_expert_hidden_cache import load_stage1_state_for_hidden_cache


class _LoadAdapterLinear(nn.Linear):
    def load_adapter(self, path: str):
        self.loaded_adapter_path = path


class _Backbone(nn.Module):
    def __init__(self, *, load_adapter: bool = False) -> None:
        super().__init__()
        self.model = _LoadAdapterLinear(4, 4) if load_adapter else nn.Linear(4, 4)


def _args(train_vlm_mode: str = "frozen", adapter_dir=None) -> Namespace:
    return Namespace(stage1_train_mode=None, train_vlm_mode=train_vlm_mode, vlm_lora_adapter_dir=adapter_dir)


def _compact_checkpoint(path, *, config: TwoExpertSlotConfig, train_mode: str, metadata_extra=None) -> TwoExpertSoftSlots:
    slots = TwoExpertSoftSlots(config)
    with torch.no_grad():
        slots.dyn_slots.fill_(2.0)
        slots.geo_slots.fill_(3.0)
    metadata = {"train_mode": train_mode}
    if metadata_extra:
        metadata.update(metadata_extra)
    torch.save(
        {
            "checkpoint_schema": "two_expert_stage1_compact_v1",
            "two_expert_slots": slots.state_dict(),
            "dynamic_adapter": {},
            "geometry_adapter": {},
            "trajectory_probe": {},
            "stage1_metadata": metadata,
        },
        path,
    )
    return slots


def test_compact_checkpoint_loads_slots(tmp_path):
    config = TwoExpertSlotConfig(vlm_hidden_dim=8, planner_dim=4)
    ckpt = tmp_path / "stage1.ckpt"
    expected = _compact_checkpoint(ckpt, config=config, train_mode="frozen")

    slots, report = load_stage1_state_for_hidden_cache(
        checkpoint=ckpt,
        config=config,
        backbone=None,
        args=_args("frozen"),
    )

    assert torch.allclose(slots.dyn_slots, expected.dyn_slots)
    assert report["checkpoint_schema"] == "two_expert_stage1_compact_v1"
    assert "dyn_slots" in report["loaded_slot_keys"]


def test_lora_compact_checkpoint_resolves_relative_adapter_dir(tmp_path):
    config = TwoExpertSlotConfig(vlm_hidden_dim=8, planner_dim=4)
    ckpt = tmp_path / "stage1.ckpt"
    adapter_dir = tmp_path / "adapters" / "vlm_lora"
    adapter_dir.mkdir(parents=True)
    _compact_checkpoint(
        ckpt,
        config=config,
        train_mode="lora",
        metadata_extra={"vlm_lora_adapter_dir": "adapters/vlm_lora"},
    )
    backbone = _Backbone(load_adapter=True)

    _, report = load_stage1_state_for_hidden_cache(
        checkpoint=ckpt,
        config=config,
        backbone=backbone,
        args=_args("lora"),
    )

    assert report["loaded_lora_adapter"] is True
    assert report["loaded_lora_adapter_dir"] == str(adapter_dir)
    assert backbone.model.loaded_adapter_path == str(adapter_dir)


def test_top_layer_compact_checkpoint_resolves_relative_trainable_state(tmp_path):
    config = TwoExpertSlotConfig(vlm_hidden_dim=8, planner_dim=4)
    ckpt = tmp_path / "stage1.ckpt"
    state_path = tmp_path / "adapters" / "vlm_trainable_state.pt"
    state_path.parent.mkdir(parents=True)
    target_weight = torch.full((4, 4), 0.25)
    target_bias = torch.full((4,), -0.5)
    torch.save({"backbone.model.weight": target_weight, "backbone.model.bias": target_bias}, state_path)
    _compact_checkpoint(
        ckpt,
        config=config,
        train_mode="top_layers",
        metadata_extra={"vlm_trainable_state_path": "adapters/vlm_trainable_state.pt"},
    )
    backbone = _Backbone()

    _, report = load_stage1_state_for_hidden_cache(
        checkpoint=ckpt,
        config=config,
        backbone=backbone,
        args=_args("top_layers"),
    )

    assert report["loaded_top_layer_keys"] == ["model.bias", "model.weight"]
    assert torch.allclose(backbone.model.weight, target_weight)
    assert torch.allclose(backbone.model.bias, target_bias)


def test_legacy_full_state_fallback_still_loads_slots(tmp_path):
    config = TwoExpertSlotConfig(vlm_hidden_dim=8, planner_dim=4)
    ckpt = tmp_path / "legacy.ckpt"
    slots = TwoExpertSoftSlots(config)
    with torch.no_grad():
        slots.dyn_slots.fill_(5.0)
    torch.save(
        {
            "state_dict": {f"two_expert_slots.{key}": value for key, value in slots.state_dict().items()},
            "stage1_metadata": {"train_mode": "frozen"},
        },
        ckpt,
    )

    loaded, _ = load_stage1_state_for_hidden_cache(
        checkpoint=ckpt,
        config=config,
        backbone=None,
        args=_args("frozen"),
    )

    assert torch.allclose(loaded.dyn_slots, slots.dyn_slots)
