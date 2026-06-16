from __future__ import annotations

import json

from scripts.last_vla_v2.two_expert_slot.check_stage1_v2_representation_gate import DEFAULTS, check


def test_stage1_v2_gate_ready_on_passing_metrics(tmp_path):
    payload = {
        "metrics": {
            "no_signal_dyn_loss": 1.2,
            "slot_only_dyn_loss": 1.0,
            "no_signal_geo_loss": 1.3,
            "slot_only_geo_loss": 1.0,
        },
        "comparisons": {
            "image_only_dyn_loss_over_trained_dyn_loss": 1.2,
            "image_only_geo_loss_over_trained_geo_loss": 1.4,
        },
        "retrieval": {
            "trained_dyn": {"top1": 0.1, "top5": 0.3, "positive_margin": 0.01},
            "trained_geo": {"top1": 0.1, "top5": 0.3, "positive_margin": 0.01},
        },
    }
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = check(path, None, dict(DEFAULTS))
    assert result["status"] == "READY"


def test_stage1_v2_gate_fails_shortcut_metric(tmp_path):
    payload = {
        "metrics": {
            "no_signal_dyn_loss": 1.2,
            "slot_only_dyn_loss": 1.0,
            "no_signal_geo_loss": 1.3,
            "slot_only_geo_loss": 1.0,
        },
        "comparisons": {
            "image_only_dyn_loss_over_trained_dyn_loss": 1.0,
            "image_only_geo_loss_over_trained_geo_loss": 1.4,
        },
        "retrieval": {
            "trained_dyn": {"top1": 0.1, "top5": 0.3, "positive_margin": 0.01},
            "trained_geo": {"top1": 0.1, "top5": 0.3, "positive_margin": 0.01},
        },
    }
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = check(path, None, dict(DEFAULTS))
    assert result["status"] == "FAIL"
