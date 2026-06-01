#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIGS = {
    "hybrid_eval": REPO_ROOT / "configs" / "last_rd" / "last_rd_progressive_sft_hybrid_eval.yaml",
    "lastrd_only_eval": REPO_ROOT / "configs" / "last_rd" / "last_rd_progressive_sft_lastrd_only_eval.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check eval-safe LaST-RD configs without constructing a model.")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "reports" / "last_rd_eval_config_check.json")
    return parser.parse_args()


def check_config(name: str, path: Path) -> Dict[str, Any]:
    errors: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    expected_expert_features = name == "hybrid_eval"
    checks = {
        "use_last_rd": data.get("use_last_rd") is True,
        "policy_kd_loss_weight_zero": float(data.get("policy_kd_loss_weight", -1.0)) == 0.0,
        "policy_kd_mode_none": str(data.get("policy_kd_mode", "")).lower() == "none",
        "allow_future_targets_in_inference_false": data.get("allow_future_targets_in_inference") is False,
        "allow_expert_target_features_false": data.get("allow_expert_target_features") is False,
        "use_expert_features_expected": data.get("use_expert_features") is expected_expert_features,
    }
    for key, ok in checks.items():
        if not ok:
            errors.append(key)
    return {
        "path": str(path),
        "checks": checks,
        "resolved": {
            "use_expert_features": data.get("use_expert_features"),
            "use_last_rd": data.get("use_last_rd"),
            "policy_kd_loss_weight": data.get("policy_kd_loss_weight"),
            "policy_kd_mode": data.get("policy_kd_mode"),
            "allow_expert_target_features": data.get("allow_expert_target_features"),
            "allow_future_targets_in_inference": data.get("allow_future_targets_in_inference"),
        },
        "pass": not errors,
        "errors": errors,
    }


def main() -> int:
    args = parse_args()
    result = {"configs": {name: check_config(name, path) for name, path in CONFIGS.items()}}
    result["pass"] = all(item["pass"] for item in result["configs"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
