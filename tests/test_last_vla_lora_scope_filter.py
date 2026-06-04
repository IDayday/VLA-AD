from __future__ import annotations

import pytest
import torch

from tests.test_last_vla_lora_target_resolution import SyntheticVLM
from navsim.agents.recogdrive.vlm_lora_utils import (
    audit_actual_trainable_lora_modules,
    audit_lora_target_modules,
    resolve_lora_target_modules,
    validate_lora_scope_audit,
)


class FakeLoraLinear(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(4, 4)
        self.lora_A = torch.nn.Parameter(torch.randn(2, 4))
        self.lora_B = torch.nn.Parameter(torch.randn(4, 2))


class ActualLoraModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.llm = torch.nn.Module()
        self.llm.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.llm.layers[0].q_proj = FakeLoraLinear()
        self.vision_tower = torch.nn.Module()
        self.vision_tower.blocks = torch.nn.ModuleList([torch.nn.Module()])
        self.vision_tower.blocks[0].q_proj = FakeLoraLinear()
        self.mm_projector = FakeLoraLinear()


def test_scope_llm_excludes_vision_and_projector():
    model = SyntheticVLM()
    targets = resolve_lora_target_modules(model, preset="attention_mlp", scope="llm", custom_target_modules="", vision_last_n=0)
    report = audit_lora_target_modules(model, target_modules=targets, scope="llm", preset="attention_mlp")

    assert report["matched_by_category"]["llm_attention"] > 0
    assert report["matched_by_category"]["vision_attention"] == 0
    assert report["matched_by_category"]["projector"] == 0


def test_scope_vision_excludes_llm():
    model = SyntheticVLM()
    targets = resolve_lora_target_modules(model, preset="attention_mlp", scope="vision", custom_target_modules="", vision_last_n=0)
    report = audit_lora_target_modules(model, target_modules=targets, scope="vision", preset="attention_mlp")

    assert report["matched_by_category"]["vision_attention"] > 0
    assert report["matched_by_category"]["llm_attention"] == 0


def test_scope_all_includes_allowed_modules():
    model = SyntheticVLM()
    targets = resolve_lora_target_modules(model, preset="all_linear", scope="all", custom_target_modules="", vision_last_n=0)
    report = audit_lora_target_modules(model, target_modules=targets, scope="all", preset="all_linear")

    assert report["matched_by_category"]["llm_attention"] > 0
    assert report["matched_by_category"]["vision_attention"] > 0
    assert report["matched_by_category"]["projector"] > 0


def test_scope_llm_rejects_actual_vision_lora():
    report = audit_actual_trainable_lora_modules(ActualLoraModel(), scope="llm")

    assert report["actual_trainable_by_category"]["vision_attention"] > 0
    with pytest.raises(ValueError, match="scope=llm"):
        validate_lora_scope_audit(report)


def test_scope_vision_rejects_actual_llm_lora():
    report = audit_actual_trainable_lora_modules(ActualLoraModel(), scope="vision")

    assert report["actual_trainable_by_category"]["llm_attention"] > 0
    with pytest.raises(ValueError, match="scope=vision"):
        validate_lora_scope_audit(report)
