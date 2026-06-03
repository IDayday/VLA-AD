from __future__ import annotations

from scripts.risk_vla.check_label_leakage import check_leakage


def test_leakage_guard_rejects_navtest_for_training():
    counts, errors = check_leakage(
        [{"sample_token": "a", "split": "navtrain"}, {"sample_token": "b", "split": "navtest"}],
        allowed_splits=["navtrain", "navval"],
        for_training=True,
    )
    assert counts["navtest"] == 1
    assert errors
    assert "navtest" in errors[0]


def test_leakage_guard_requires_missing_split_ack_for_training():
    _, errors = check_leakage([{"sample_token": "a"}], allowed_splits=["navtrain"], for_training=True)
    assert errors
    _, allowed_errors = check_leakage(
        [{"sample_token": "a"}],
        allowed_splits=["navtrain"],
        for_training=True,
        allow_missing_split=True,
    )
    assert allowed_errors == []
