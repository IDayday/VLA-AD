from __future__ import annotations

from tests.test_last_vla_helpers import make_last_vla_batch, make_last_vla_planner


def test_vlm_summary_path_is_removed_from_formal_last_vla():
    planner = make_last_vla_planner()

    assert not hasattr(planner.config, "last_vla_vlm_summary_tokens")
    assert not hasattr(planner.last_vla_cot.config, "vlm_summary_tokens")
    assert not hasattr(planner.last_vla_cot, "vlm_compressor")


def test_context_stays_raw_vlm_with_cot_condition_branch():
    planner = make_last_vla_planner(cot_tokens=8)
    vl_features, action_input = make_last_vla_batch(include_targets=False)

    out = planner._prepare_dit_context(vl_features, action_input, training=False, allow_target_tokens=False)
    assert out["context_tokens"].shape[1] == vl_features.shape[1]
    assert out["cot_condition_tokens"].shape[1] == 8
    assert "vlm_summary_keep_prob" not in out["diagnostics"]
