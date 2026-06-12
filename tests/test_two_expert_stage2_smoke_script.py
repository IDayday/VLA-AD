from __future__ import annotations

from scripts.last_vla_v2.two_expert_slot.smoke_stage2_two_expert_batch import run_smoke


def test_stage2_two_expert_synthetic_smoke():
    report = run_smoke()

    assert report["ok"] is True
    assert report["expert_step_condition_shape"] == [2, 8, 384]
    assert report["context_tokens_shape"] == [2, 9, 384]
    assert report["diffusion_target_is_gt_norm"] is True
    assert report["no_residual"] is True
