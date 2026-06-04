#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import peft
except Exception:  # pragma: no cover - peft is optional in unit tests.
    peft = None

from navsim.agents.recogdrive.vlm_lora_utils import build_lora_config_payload, stable_config_hash, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Last-VLA CoT and VLM LoRA adapters from a cot-alignment checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-vlm-path", default="")
    parser.add_argument("--vlm-type", default="internvl")
    parser.add_argument("--preset", default="attention_mlp")
    parser.add_argument("--scope", default="llm")
    parser.add_argument("--target-modules", default="")
    parser.add_argument("--resolved-target-modules", default="")
    parser.add_argument("--r", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--bias", default="none")
    parser.add_argument("--use-rslora", action="store_true", default=True)
    parser.add_argument("--no-rslora", dest="use_rslora", action="store_false")
    parser.add_argument("--use-dora", action="store_true", default=False)
    parser.add_argument("--lora-target-report", type=Path, default=None)
    return parser.parse_args()


def _state_dict(payload: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must contain a state_dict or be a state dict.")
    return state


def _strip_agent_prefix(key: str) -> str:
    return key[len("agent."):] if key.startswith("agent.") else key


def _split_targets(value: str) -> list[str] | str:
    value = str(value or "").strip()
    if not value:
        return []
    if value == "all-linear":
        return value
    return [item.strip() for item in value.split(",") if item.strip()]


def _read_target_report(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = _state_dict(checkpoint)
    cot: Dict[str, torch.Tensor] = {}
    lora: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        mapped_key = _strip_agent_prefix(key)
        if mapped_key.startswith("action_head.last_vla_cot.") or mapped_key.startswith("last_vla_cot."):
            cot[mapped_key] = value
        if "lora_" not in mapped_key:
            continue
        if mapped_key.startswith("backbone.model."):
            lora[mapped_key[len("backbone.model."):]] = value
        elif mapped_key.startswith("backbone."):
            lora[mapped_key[len("backbone."):]] = value
        else:
            lora[mapped_key] = value
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vlm_lora_dir = args.output_dir / "vlm_lora"
    vlm_lora_dir.mkdir(parents=True, exist_ok=True)
    cot_path = args.output_dir / "last_vla_cot_adapter.pt"
    lora_path = args.output_dir / "vlm_lora_adapter_state.pt"
    lora_model_path = vlm_lora_dir / "adapter_model.bin"
    torch.save({"state_dict": cot}, cot_path)
    torch.save({"state_dict": lora}, lora_path)
    torch.save(lora, lora_model_path)
    target_report = _read_target_report(args.lora_target_report)
    resolved = target_report.get("resolved_target_modules") or _split_targets(args.resolved_target_modules or args.target_modules)
    config_payload = build_lora_config_payload(
        preset=args.preset,
        scope=args.scope,
        target_modules=resolved,
        r=args.r,
        alpha=args.alpha,
        dropout=args.dropout,
        bias=args.bias,
        use_rslora=args.use_rslora,
        use_dora=args.use_dora,
        init="default",
        vision_last_n=0,
        peft_version=getattr(peft, "__version__", None),
    )
    write_json(vlm_lora_dir / "adapter_config.json", config_payload)
    metadata = {
        **config_payload,
        "base_vlm_path": str(args.base_vlm_path),
        "vlm_type": str(args.vlm_type),
        "target_modules": _split_targets(args.target_modules) or resolved,
        "resolved_target_modules": resolved,
        "matched_modules": target_report.get("matched_module_names", []),
        "matched_total": target_report.get("matched_total", 0),
        "matched_by_category": target_report.get("matched_by_category", {}),
        "trainable_lora_param_count": target_report.get("trainable_lora_param_count", None),
        "peft_version": getattr(peft, "__version__", None),
        "source_checkpoint": str(args.checkpoint),
    }
    metadata["adapter_config_hash"] = stable_config_hash(config_payload)
    write_json(vlm_lora_dir / "lora_metadata.json", metadata)
    if target_report:
        write_json(vlm_lora_dir / "lora_target_report.json", target_report)
    report = {
        "checkpoint": str(args.checkpoint),
        "last_vla_cot_adapter": str(cot_path),
        "vlm_lora_adapter_state": str(lora_path),
        "vlm_lora_adapter_dir": str(vlm_lora_dir),
        "vlm_lora_adapter_model": str(lora_model_path),
        "lora_metadata": str(vlm_lora_dir / "lora_metadata.json"),
        "lora_target_report": str(vlm_lora_dir / "lora_target_report.json"),
        "num_cot_keys": len(cot),
        "num_lora_keys": len(lora),
        "adapter_config_hash": metadata["adapter_config_hash"],
    }
    (args.output_dir / "adapter_extract_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not cot:
        raise RuntimeError("No Last-VLA CoT keys found in checkpoint.")
    if not lora:
        raise RuntimeError("No VLM LoRA keys found in checkpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
