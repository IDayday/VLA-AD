from __future__ import annotations

import torch

from tests.test_two_expert_stage2_context import make_two_expert_batch, make_two_expert_planner


def _context_for_mode(mode: str):
    planner = make_two_expert_planner()
    vl, action_input = make_two_expert_batch()
    action_input["two_expert_corruption_mode"] = mode
    return planner._prepare_dit_context(vl, action_input, training=False)


def test_zero_h_dyn_only_zeros_dynamic_expert():
    context = _context_for_mode("zero_h_dyn")
    diag = context["diagnostics"]
    assert diag["two_expert_corruption_mode_code"].item() == 1.0
    assert diag["two_expert_h_dyn_norm"].item() == 0.0
    assert diag["two_expert_h_geo_norm"].item() > 0.0


def test_zero_h_geo_only_zeros_geometry_expert():
    context = _context_for_mode("zero_h_geo")
    diag = context["diagnostics"]
    assert diag["two_expert_corruption_mode_code"].item() == 2.0
    assert diag["two_expert_h_geo_norm"].item() == 0.0
    assert diag["two_expert_h_dyn_norm"].item() > 0.0


def test_raw_vlm_only_keeps_raw_context_and_removes_experts():
    context = _context_for_mode("raw_vlm_only")
    diag = context["diagnostics"]
    assert diag["two_expert_corruption_mode_code"].item() == 4.0
    assert diag["two_expert_condition_enabled"].item() == 0.0
    assert diag["two_expert_raw_vlm_context_used"].item() == 1.0
    assert torch.allclose(context["expert_step_condition"], torch.zeros_like(context["expert_step_condition"]))


def test_random_slots_corruption_keeps_expert_path_with_mismatched_slots():
    context = _context_for_mode("random_slots")
    diag = context["diagnostics"]
    assert diag["two_expert_corruption_mode_code"].item() == 7.0
    assert diag["two_expert_condition_enabled"].item() == 1.0
    assert diag["two_expert_h_dyn_norm"].item() > 0.0
    assert diag["two_expert_h_geo_norm"].item() > 0.0
