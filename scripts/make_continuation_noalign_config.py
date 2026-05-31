#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an A4 continuation config with alignment losses disabled.")
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    args = parse_args()
    config = load_yaml(args.base_config)
    config.update({
        "use_alignment_loss": False,
        "expert_alignment_weight": 0.0,
        "jepa_alignment_weight": 0.0,
        "vggt_alignment_weight": 0.0,
        "allow_future_targets_in_inference": False,
    })
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    with args.output_config.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(args.output_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
