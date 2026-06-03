#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_registry(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError("Registry must be a YAML mapping.")
    return data


def validate_registry(registry: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    experiments = registry.get("experiments")
    if not isinstance(experiments, dict) or not experiments:
        errors.append("Registry must define a non-empty experiments mapping.")
        return errors
    for name, spec in experiments.items():
        if not isinstance(spec, dict):
            errors.append(f"{name}: experiment spec must be a mapping.")
            continue
        role = str(spec.get("role", ""))
        if not role:
            errors.append(f"{name}: missing role.")
        is_oracle = bool(spec.get("use_oracle_router")) or role == "oracle_upper_bound"
        if is_oracle and spec.get("analysis_only") is not True:
            errors.append(f"{name}: oracle-router experiments must set analysis_only: true.")
    return errors


def format_registry(registry: Dict[str, Any]) -> str:
    lines = [f"Round: {registry.get('round', '<missing>')}", ""]
    experiments = registry.get("experiments") or {}
    for name, spec in experiments.items():
        lines.append(f"- {name}")
        lines.append(f"  role: {spec.get('role', '<missing>')}")
        lines.append(f"  description: {spec.get('description', '')}")
        fragments = spec.get("config_fragments") or []
        lines.append(f"  config_fragments: {', '.join(fragments) if fragments else '<none>'}")
        outputs = spec.get("expected_outputs") or []
        lines.append(f"  expected_outputs: {', '.join(outputs) if outputs else '<none>'}")
        if spec.get("analysis_only"):
            lines.append("  analysis_only: true")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Print and validate the RISK-VLA round1 experiment registry.")
    parser.add_argument("--registry", type=Path, default=Path("configs/risk_vla/round1_experiment_registry.yaml"))
    args = parser.parse_args()
    registry = load_registry(args.registry)
    errors = validate_registry(registry)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(format_registry(registry))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
