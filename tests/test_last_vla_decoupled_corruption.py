from __future__ import annotations

from transformers.feature_extraction_utils import BatchFeature

from tests.test_last_vla_helpers import make_last_vla_batch, make_last_vla_planner


def test_raw_vlm_only_zeros_cot_condition_but_keeps_raw_context():
    planner = make_last_vla_planner(cot_tokens=8)
    vl_features, action_input = make_last_vla_batch(include_targets=False)
    data = dict(action_input)
    data["last_vla_raw_vlm_only"] = True
    out = planner._prepare_dit_context(vl_features, BatchFeature(data=data), training=False, allow_target_tokens=False)

    assert out["context_tokens"].shape[1] == vl_features.shape[1]
    assert out["last_vla_output"].cot_condition_tokens.abs().sum().item() == 0.0
    assert out["last_vla_output"].diagnostics["raw_vlm_context_used"].item() == 1.0


def test_cot_only_debug_disables_raw_context_and_is_diagnostic_only():
    planner = make_last_vla_planner(cot_tokens=8)
    vl_features, action_input = make_last_vla_batch(include_targets=False)
    data = dict(action_input)
    data["last_vla_cot_only_for_debug_only"] = True
    out = planner._prepare_dit_context(vl_features, BatchFeature(data=data), training=False, allow_target_tokens=False)

    assert out["context_tokens"].abs().sum().item() == 0.0
    assert out["last_vla_output"].diagnostics["raw_vlm_context_used"].item() == 0.0
