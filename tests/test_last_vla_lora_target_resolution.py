from __future__ import annotations

import pytest
import torch

from navsim.agents.recogdrive.vlm_lora_utils import resolve_lora_target_modules


class Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4)
        self.k_proj = torch.nn.Linear(4, 4)
        self.v_proj = torch.nn.Linear(4, 4)
        self.o_proj = torch.nn.Linear(4, 4)
        self.gate_proj = torch.nn.Linear(4, 4)
        self.up_proj = torch.nn.Linear(4, 4)
        self.down_proj = torch.nn.Linear(4, 4)
        self.fc1 = torch.nn.Linear(4, 4)
        self.fc2 = torch.nn.Linear(4, 4)


class SyntheticVLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.llm = torch.nn.Module()
        self.llm.layers = torch.nn.ModuleList([Block(), Block()])
        self.vision_tower = torch.nn.Module()
        self.vision_tower.blocks = torch.nn.ModuleList([Block(), Block()])
        self.mm_projector = torch.nn.Linear(4, 4)
        self.lm_head = torch.nn.Linear(4, 4)
        self.action_head = torch.nn.Linear(4, 4)


def test_attention_only_resolves_attention_suffixes_only():
    targets = resolve_lora_target_modules(
        SyntheticVLM(),
        preset="attention_only",
        scope="llm",
        custom_target_modules="",
        vision_last_n=0,
    )

    assert all(name.startswith("llm.layers.") for name in targets)
    assert all(name.rsplit(".", 1)[-1] in {"k_proj", "o_proj", "q_proj", "v_proj"} for name in targets)
    assert "q_proj" not in targets
    assert "vision_tower.blocks.0.q_proj" not in targets


def test_attention_mlp_scope_llm_returns_full_names_not_suffixes():
    targets = resolve_lora_target_modules(
        SyntheticVLM(),
        preset="attention_mlp",
        scope="llm",
        custom_target_modules="",
        vision_last_n=0,
    )

    for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "fc1", "fc2"):
        assert f"llm.layers.0.{name}" in targets
        assert name not in targets
    assert all(target.startswith("llm.layers.") for target in targets)


def test_all_linear_scope_llm_does_not_return_global_all_linear():
    targets = resolve_lora_target_modules(
        SyntheticVLM(),
        preset="all_linear",
        scope="llm",
        custom_target_modules="",
        vision_last_n=0,
    )

    assert targets != "all-linear"
    assert all(name.startswith("llm.layers.") for name in targets)
    assert "vision_tower.blocks.0.q_proj" not in targets


def test_all_linear_resolves_all_allowed_linear_full_names():
    targets = resolve_lora_target_modules(
        SyntheticVLM(),
        preset="all_linear",
        scope="all",
        custom_target_modules="",
        vision_last_n=0,
    )

    assert "lm_head" not in targets
    assert "action_head" not in targets
    assert "llm.layers.0.q_proj" in targets
    assert "mm_projector" in targets


def test_all_linear_scope_all_can_use_global_all_linear_only_when_explicitly_allowed():
    targets = resolve_lora_target_modules(
        SyntheticVLM(),
        preset="all_linear",
        scope="all",
        custom_target_modules="",
        vision_last_n=0,
    )
    assert targets != "all-linear"

    global_targets = resolve_lora_target_modules(
        SyntheticVLM(),
        preset="all_linear",
        scope="all",
        custom_target_modules="",
        vision_last_n=0,
        allow_all_linear_global=True,
    )
    assert global_targets == "all-linear"


def test_custom_resolves_exact_list_and_zero_match_raises():
    model = SyntheticVLM()
    targets = resolve_lora_target_modules(
        model,
        preset="custom",
        scope="llm",
        custom_target_modules="q_proj,fc1",
        vision_last_n=0,
    )
    assert targets == ["q_proj", "fc1"]

    with pytest.raises(ValueError, match="matched zero"):
        resolve_lora_target_modules(
            model,
            preset="custom",
            scope="llm",
            custom_target_modules="does_not_exist",
            vision_last_n=0,
        )
