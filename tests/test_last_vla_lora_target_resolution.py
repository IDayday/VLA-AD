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

    assert targets == ["k_proj", "o_proj", "q_proj", "v_proj"]


def test_attention_mlp_resolves_attention_and_mlp_suffixes():
    targets = resolve_lora_target_modules(
        SyntheticVLM(),
        preset="attention_mlp",
        scope="llm",
        custom_target_modules="",
        vision_last_n=0,
    )

    for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "fc1", "fc2"):
        assert name in targets


def test_all_linear_resolves_all_allowed_linear_suffixes():
    targets = resolve_lora_target_modules(
        SyntheticVLM(),
        preset="all_linear",
        scope="all",
        custom_target_modules="",
        vision_last_n=0,
    )

    assert "lm_head" not in targets
    assert "action_head" not in targets
    assert "q_proj" in targets
    assert "mm_projector" in targets


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
