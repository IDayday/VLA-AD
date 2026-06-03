from __future__ import annotations

import yaml

from scripts.risk_vla.print_round1_registry import format_registry, load_registry, validate_registry


def test_round1_registry_validates_oracle_analysis_only(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "round": "synthetic",
                "experiments": {
                    "R1": {"role": "oracle_upper_bound", "use_oracle_router": True, "analysis_only": True},
                    "R2": {"role": "pilot", "use_oracle_router": False},
                },
            }
        ),
        encoding="utf-8",
    )
    registry = load_registry(path)
    assert validate_registry(registry) == []
    text = format_registry(registry)
    assert "R1" in text
    assert "analysis_only: true" in text


def test_round1_registry_rejects_oracle_without_analysis_only():
    errors = validate_registry({"experiments": {"bad": {"role": "oracle_upper_bound", "use_oracle_router": True}}})
    assert errors
    assert "analysis_only" in errors[0]
