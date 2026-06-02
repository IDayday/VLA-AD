from __future__ import annotations

import torch

from tests.test_last_vla_helpers import make_last_vla_batch, make_last_vla_planner


def test_last_vla_bottleneck_context_excludes_full_vlm_tokens():
    torch.manual_seed(102)
    planner = make_last_vla_planner(cot_tokens=8, summary_tokens=2, risk_head=False)
    planner.eval()
    vl_features, action_input = make_last_vla_batch(include_targets=False)
    with torch.no_grad():
        context = planner._prepare_dit_context(
            vl_features,
            action_input,
            training=False,
            allow_target_tokens=False,
        )
    output = context["last_vla_output"]
    assert context["context_tokens"].shape[1] == output.cot_tokens.shape[1] + output.vlm_summary_tokens.shape[1]
    assert context["context_tokens"].shape[1] != vl_features.shape[1] + output.cot_tokens.shape[1]
    assert output.diagnostics["cot_bottleneck_active"].item() == 1.0
    assert output.diagnostics["raw_vlm_context_used"].item() == 0.0

    corrupted = context["context_tokens"].clone()
    corrupted[:, : output.cot_tokens.shape[1]] = 0.0
    assert (context["context_tokens"] - corrupted).abs().mean().item() > 1e-3
