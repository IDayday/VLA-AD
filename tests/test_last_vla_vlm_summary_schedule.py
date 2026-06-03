from __future__ import annotations

import torch
from transformers.feature_extraction_utils import BatchFeature

from tests.test_last_vla_helpers import make_last_vla_batch, make_last_vla_planner


def test_vlm_summary_keep_schedule_reports_probability():
    torch.manual_seed(22)
    planner = make_last_vla_planner()
    planner.config.current_train_epoch = 40
    planner.config.last_vla_vlm_summary_keep_start = 1.0
    planner.config.last_vla_vlm_summary_keep_end = 0.3
    planner.config.last_vla_vlm_summary_decay_epochs = 80
    planner.last_vla_cot.config.vlm_summary_keep_start = 1.0
    planner.last_vla_cot.config.vlm_summary_keep_end = 0.3
    planner.last_vla_cot.config.vlm_summary_decay_epochs = 80
    vl_features, action_input = make_last_vla_batch(include_targets=False)

    out = planner._prepare_dit_context(vl_features, action_input, training=True, allow_target_tokens=False)
    keep_prob = out["diagnostics"]["vlm_summary_keep_prob"]

    assert torch.isclose(keep_prob, torch.tensor(0.65, dtype=keep_prob.dtype), atol=1e-5)


def test_drop_vlm_summary_flag_removes_summary_tokens_from_context():
    planner = make_last_vla_planner(cot_tokens=8, summary_tokens=2)
    planner.eval()
    vl_features, action_input = make_last_vla_batch(include_targets=False)
    data = dict(action_input)
    data["last_vla_corrupt_drop_vlm_summary"] = True

    out = planner._prepare_dit_context(vl_features, BatchFeature(data=data), training=False, allow_target_tokens=False)

    assert out["diagnostics"]["vlm_summary_kept"].item() == 0.0
    assert out["context_tokens"].shape[1] == out["last_vla_output"].cot_tokens.shape[1]
