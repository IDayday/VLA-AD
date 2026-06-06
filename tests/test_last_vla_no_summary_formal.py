from __future__ import annotations

from tests.test_last_vla_helpers import make_last_vla_batch, make_last_vla_planner


def test_formal_last_vla_has_no_summary_modules_or_diagnostics():
    planner = make_last_vla_planner()
    vl_features, action_input = make_last_vla_batch(include_targets=False)
    out = planner._prepare_dit_context(vl_features, action_input, training=False, allow_target_tokens=False)

    assert not hasattr(planner.last_vla_cot, "vlm_compressor")
    assert "vlm_summary_norm" not in out["diagnostics"]
    assert "vlm_summary_keep_prob" not in out["diagnostics"]
    assert out["context_tokens"].shape[1] == vl_features.shape[1]
    assert out["cot_condition_tokens"].shape[1] == planner.config.last_vla_cot_num_tokens
