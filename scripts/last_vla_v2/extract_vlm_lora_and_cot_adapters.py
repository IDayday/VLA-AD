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
    parser.add_argument("--lora-training-config", type=Path, default=None)
    parser.add_argument("--allow-config-mismatch", action="store_true")
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


def _read_training_config(args: argparse.Namespace) -> Dict[str, Any]:
    candidates = []
    if args.lora_training_config is not None:
        candidates.append(args.lora_training_config)
    candidates.append(args.checkpoint.parent / "lora_training_config.json")
    for path in candidates:
        if path is not None and path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _config_value(config: Dict[str, Any], key: str, default: Any) -> Any:
    return config.get(key, default)


def _canonical_targets(value: Any) -> Any:
    if isinstance(value, str):
        return _split_targets(value)
    if isinstance(value, list):
        return [str(item) for item in value]
    return value


def _check_cli_matches_training_config(args: argparse.Namespace, training_config: Dict[str, Any]) -> list[str]:
    checks = {
        "preset": (args.preset, str(_config_value(training_config, "preset", args.preset))),
        "scope": (args.scope, str(_config_value(training_config, "scope", args.scope))),
        "r": (int(args.r), int(_config_value(training_config, "r", args.r))),
        "alpha": (int(args.alpha), int(_config_value(training_config, "alpha", args.alpha))),
        "dropout": (float(args.dropout), float(_config_value(training_config, "dropout", args.dropout))),
        "bias": (args.bias, str(_config_value(training_config, "bias", args.bias))),
        "use_rslora": (bool(args.use_rslora), bool(_config_value(training_config, "use_rslora", args.use_rslora))),
        "use_dora": (bool(args.use_dora), bool(_config_value(training_config, "use_dora", args.use_dora))),
    }
    warnings_list = []
    for key, (cli_value, train_value) in checks.items():
        if cli_value != train_value:
            warnings_list.append(f"{key}: cli={cli_value!r} training_config={train_value!r}")
    if args.target_modules:
        cli_targets = _canonical_targets(args.target_modules)
        train_targets = _canonical_targets(training_config.get("target_modules", training_config.get("resolved_target_modules", cli_targets)))
        if cli_targets != train_targets:
            warnings_list.append(f"target_modules: cli={cli_targets!r} training_config={train_targets!r}")
    return warnings_list


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
    training_config = _read_training_config(args)
    mismatch_warnings = _check_cli_matches_training_config(args, training_config) if training_config else []
    if mismatch_warnings and not args.allow_config_mismatch:
        raise ValueError(
            "LoRA extraction CLI arguments do not match lora_training_config.json: "
            + "; ".join(mismatch_warnings)
            + ". Pass --allow-config-mismatch only for explicit recovery/debug use."
        )
    resolved = (
        training_config.get("resolved_target_modules")
        or training_config.get("target_modules")
        or target_report.get("resolved_target_modules")
        or _split_targets(args.resolved_target_modules or args.target_modules)
    )
    config_payload = build_lora_config_payload(
        preset=str(training_config.get("preset", args.preset)),
        scope=str(training_config.get("scope", args.scope)),
        target_modules=resolved,
        r=int(training_config.get("r", args.r)),
        alpha=int(training_config.get("alpha", args.alpha)),
        dropout=float(training_config.get("dropout", args.dropout)),
        bias=str(training_config.get("bias", args.bias)),
        use_rslora=bool(training_config.get("use_rslora", args.use_rslora)),
        use_dora=bool(training_config.get("use_dora", args.use_dora)),
        init=str(training_config.get("init", "default")),
        vision_last_n=int(training_config.get("vision_last_n", 0) or 0),
        allow_all_linear_global=bool(training_config.get("allow_all_linear_global", False)),
        hidden_anchor_every_n_steps=training_config.get("hidden_anchor_every_n_steps", None),
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
        "training_config_path": str(args.lora_training_config or (args.checkpoint.parent / "lora_training_config.json")),
        "used_training_config": bool(training_config),
        "config_mismatch_warnings": mismatch_warnings,
        "high_risk_warnings": ["allowed_config_mismatch"] if mismatch_warnings and args.allow_config_mismatch else [],
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
        "used_training_config": bool(training_config),
        "config_mismatch_warnings": mismatch_warnings,
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
