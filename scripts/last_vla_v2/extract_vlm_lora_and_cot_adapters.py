#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Last-VLA CoT and VLM LoRA adapters from a cot-alignment checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _state_dict(payload: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must contain a state_dict or be a state dict.")
    return state


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = _state_dict(checkpoint)
    cot = {
        key: value
        for key, value in state.items()
        if key.startswith("action_head.last_vla_cot.") or key.startswith("last_vla_cot.")
    }
    lora = {key: value for key, value in state.items() if "lora_" in key and key.startswith("backbone.")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cot_path = args.output_dir / "last_vla_cot_adapter.pt"
    lora_path = args.output_dir / "vlm_lora_adapter_state.pt"
    torch.save({"state_dict": cot}, cot_path)
    torch.save({"state_dict": lora}, lora_path)
    report = {
        "checkpoint": str(args.checkpoint),
        "last_vla_cot_adapter": str(cot_path),
        "vlm_lora_adapter_state": str(lora_path),
        "num_cot_keys": len(cot),
        "num_lora_keys": len(lora),
    }
    (args.output_dir / "adapter_extract_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not cot:
        raise RuntimeError("No Last-VLA CoT keys found in checkpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
