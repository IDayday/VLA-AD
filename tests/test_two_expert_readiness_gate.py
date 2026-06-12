from __future__ import annotations

from scripts.last_vla_v2.two_expert_slot.final_two_expert_readiness_gate import (
    validate_internvl,
    validate_teacher,
    validate_vggt,
)


def test_final_readiness_gate_accepts_strict_smokes():
    teacher = {
        "ok": True,
        "teacher_status": "strict_production_teacher",
        "num_jepa_legacy_fallback": 0,
        "num_vggt_legacy_fallback": 0,
    }
    internvl = {
        "ok": True,
        "image_hidden_nonempty": True,
        "image_sensitivity_delta": 0.01,
        "slot_sensitivity_delta": 0.02,
    }
    vggt = {"ok": True, "packed_shape": [12, 1024], "feature_dim": 1024}

    assert validate_teacher(teacher) == []
    assert validate_internvl(internvl, 1e-6) == []
    assert validate_vggt(vggt) == []


def test_final_readiness_gate_rejects_fallback_teacher():
    errors = validate_teacher(
        {
            "ok": True,
            "teacher_status": "dev_fallback_not_for_final",
            "num_jepa_legacy_fallback": 1,
            "num_vggt_legacy_fallback": 0,
        }
    )

    assert any("bad_status" in item for item in errors)
    assert any("jepa_fallback" in item for item in errors)
