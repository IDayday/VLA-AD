from __future__ import annotations

import json

import pandas as pd
import pytest

from ampt_stage3.verify_results import verify_v1, verify_v2


def test_v1_and_v2_fail_closed(tmp_path) -> None:
    v1_protocol = {
        "benchmark": "NAVSIM-v1",
        "expected_scored_scenes": 2,
        "absolute_tolerance": 1e-9,
        "expected": {"score": 0.75},
    }
    v1_path = tmp_path / "v1.csv"
    pd.DataFrame(
        {"token": ["a", "b"], "valid": [True, True], "score": [0.5, 1.0]}
    ).to_csv(v1_path, index=False)
    assert verify_v1(v1_path, v1_protocol)["passed"]

    v2_protocol = {
        "benchmark": "NAVSIM-v2",
        "expected_predictions": 2,
        "expected_scored_scenes": 2,
        "extended_comfort_available": 1,
        "official_scorer_revision": "revision",
        "absolute_tolerance": 1e-9,
        "expected": {"EPDMS": 0.8},
    }
    v2_path = tmp_path / "v2.json"
    v2_path.write_text(
        json.dumps({
            "num_predictions": 2, "successful": 2, "failed": 0,
            "extended_comfort_available": 1,
            "official_navsim_revision": "revision", "EPDMS": 0.8,
        }),
        encoding="utf-8",
    )
    assert verify_v2(v2_path, v2_protocol)["passed"]
    payload = json.loads(v2_path.read_text(encoding="utf-8"))
    payload["successful"] = 1
    v2_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="successful"):
        verify_v2(v2_path, v2_protocol)
