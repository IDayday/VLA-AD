from __future__ import annotations

from tests.test_last_vla_lora_target_resolution import SyntheticVLM
from navsim.agents.recogdrive.vlm_lora_utils import audit_lora_target_modules, resolve_lora_target_modules


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
