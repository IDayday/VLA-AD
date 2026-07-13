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
            "trained_dyn_loss": 0.4,
            "random_dyn_loss": 1.0,
            "trained_geo_loss": 0.2,
            "random_geo_loss": 1.0,
            "trained_probe_loss": 0.1,
            "zero_dyn_probe_loss": 0.6,
            "zero_geo_probe_loss": 0.6,
            "dyn_only_probe_loss": 0.2,
            "geo_only_probe_loss": 0.2,
            "no_signal_probe_loss": 0.5,
        },
        "comparisons": {
            "image_only_dyn_loss_over_trained_dyn_loss": 1.2,
            "image_only_geo_loss_over_trained_geo_loss": 1.4,
            "slot_only_dyn_gain": 1.2,
            "slot_only_geo_gain": 1.3,
            "zero_dyn_probe_ratio": 6.0,
            "zero_geo_probe_ratio": 6.0,
        },
        "retrieval": {
            "trained_dyn": {"top1": 0.1, "top5": 0.3, "positive_margin": 0.01},
            "trained_geo": {"top1": 0.1, "top5": 0.3, "positive_margin": 0.01},
        },
        "direct": {"direct_eval_count": 4, "direct_traj_parse_ok_ratio": 1.0, "direct_traj_l1": 1.0},
        "hidden": {"hidden_drift_cosine": 0.95, "hidden_anchor_loss": 0.01},
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
            "trained_dyn_loss": 0.4,
            "random_dyn_loss": 1.0,
            "trained_geo_loss": 0.2,
            "random_geo_loss": 1.0,
            "trained_probe_loss": 0.1,
            "zero_dyn_probe_loss": 0.6,
            "zero_geo_probe_loss": 0.6,
            "dyn_only_probe_loss": 0.2,
            "geo_only_probe_loss": 0.2,
            "no_signal_probe_loss": 0.5,
        },
        "comparisons": {
            "image_only_dyn_loss_over_trained_dyn_loss": 1.0,
            "image_only_geo_loss_over_trained_geo_loss": 1.4,
            "slot_only_dyn_gain": 1.2,
            "slot_only_geo_gain": 1.3,
            "zero_dyn_probe_ratio": 6.0,
            "zero_geo_probe_ratio": 6.0,
        },
        "retrieval": {
            "trained_dyn": {"top1": 0.1, "top5": 0.3, "positive_margin": 0.01},
            "trained_geo": {"top1": 0.1, "top5": 0.3, "positive_margin": 0.01},
        },
        "direct": {"direct_eval_count": 4, "direct_traj_parse_ok_ratio": 1.0, "direct_traj_l1": 1.0},
        "hidden": {"hidden_drift_cosine": 0.95, "hidden_anchor_loss": 0.01},
    }
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = check(path, None, dict(DEFAULTS))
    assert result["status"] == "CONDITIONAL"
