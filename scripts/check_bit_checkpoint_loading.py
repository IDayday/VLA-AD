#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import torch
import yaml

if hasattr(torch, "compile"):
    torch.compile = lambda fn=None, *args, **kwargs: fn if fn is not None else (lambda f: f)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import diffusers  # noqa: F401
    import timm  # noqa: F401
    import transformers  # noqa: F401
except ModuleNotFoundError:
    from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs  # noqa: E402

    _install_dependency_stubs()

from navsim.agents.recogdrive.recogdrive_diffusion_planner import (  # noqa: E402
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)


BIT_MARKERS = (
    "bit_terminal_head",
    "bit_condition_encoder",
    "bit_reverse_decoder",
    "bit_context_gate",
    "bit_action_gate",
)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_planner(cfg_dict: Dict[str, Any]) -> ReCogDriveDiffusionPlanner:
    cfg = ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg={
            "num_heads": 8,
            "head_dim": 48,
            "num_layers": int(cfg_dict.get("num_dit_layers", 16)),
            "output_dim": 512,
            "dropout": 0.0,
            "attention_bias": True,
            "norm_eps": 1e-5,
            "interleave_attention": True,
        },
        input_embedding_dim=int(cfg_dict.get("planner_dim", 384)),
        planner_dim=int(cfg_dict.get("planner_dim", 384)),
        hidden_size=1024,
        action_dim=int(cfg_dict.get("action_dim", 3)),
        action_horizon=int(cfg_dict.get("action_horizon", 8)),
        sampling_method=str(cfg_dict.get("sampling_method", "ddim")),
        num_inference_steps=int(cfg_dict.get("num_inference_steps", 5)),
        vlm_size="small",
    )
    for key, value in cfg_dict.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.vlm_size = "small"
    cfg.use_bit_drive = True
    return ReCogDriveDiffusionPlanner(cfg)


def checkpoint_candidates(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    suffixes = (".ckpt", ".pth", ".pt", ".safetensors", ".bin")
    files: List[Path] = []
    for suffix in suffixes:
        files.extend(p for p in path.rglob(f"*{suffix}") if p.is_file())
    def score(item: Path) -> tuple[int, int, str]:
        name = item.name.lower()
        value = (1000 if "il" in name else 0) + (500 if "model" in name else 0) + (100 if item.suffix == ".safetensors" else 0)
        return value, item.stat().st_size, str(item)
    return sorted(set(files), key=score, reverse=True)


def load_state_file(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return dict(load_file(str(path), device="cpu"))
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    if not isinstance(state, dict):
        raise TypeError(f"{path} did not contain a state dict.")
    return {key: value for key, value in state.items() if isinstance(value, torch.Tensor)}


def normalize_key(key: str) -> str:
    for prefix in ("agent.action_head.", "action_head."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def is_bit_key(key: str) -> bool:
    return any(marker in key for marker in BIT_MARKERS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Base-IL checkpoint compatibility with BiT-Drive modules.")
    parser.add_argument("--base-il-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    planner = build_planner(cfg)
    model_state = planner.state_dict()
    candidates = checkpoint_candidates(args.base_il_checkpoint)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found under {args.base_il_checkpoint}")

    loaded: Dict[str, torch.Tensor] = {}
    unexpected: List[str] = []
    shape_mismatches: List[str] = []
    for file in candidates:
        for raw_key, value in load_state_file(file).items():
            key = normalize_key(raw_key)
            if key not in model_state:
                unexpected.append(key)
                continue
            if tuple(value.shape) != tuple(model_state[key].shape):
                if is_bit_key(key):
                    continue
                shape_mismatches.append(f"{key}: checkpoint {tuple(value.shape)} vs model {tuple(model_state[key].shape)}")
                continue
            loaded.setdefault(key, value)

    if shape_mismatches:
        raise RuntimeError("Shape mismatch in original checkpoint keys:\n" + "\n".join(shape_mismatches[:50]))

    incompatible = planner.load_state_dict(loaded, strict=False)
    missing_bit = [key for key in incompatible.missing_keys if is_bit_key(key)]
    missing_other = [key for key in incompatible.missing_keys if not is_bit_key(key)]
    loaded_action_head = [key for key in loaded if key.startswith(("model.", "action_decoder.", "fusion_projector.", "action_encoder."))]
    report = {
        "base_il_checkpoint": str(args.base_il_checkpoint),
        "config": str(args.config),
        "checkpoint_files": [str(path) for path in candidates],
        "loaded_key_count": len(loaded),
        "loaded_original_action_head_key_count": len(loaded_action_head),
        "missing_bit_key_count": len(missing_bit),
        "missing_non_bit_key_count": len(missing_other),
        "unexpected_key_count": len(unexpected),
        "shape_mismatch_count": len(shape_mismatches),
        "shape_mismatch_original_keys_fail": True,
        "bit_keys_randomly_initialized": bool(missing_bit),
    }

    lines = [
        "# BiT-Drive Checkpoint Loading Report",
        "",
        f"Base-IL checkpoint: `{args.base_il_checkpoint}`",
        f"Config: `{args.config}`",
        f"Loaded keys: {report['loaded_key_count']}",
        f"Loaded original action-head keys: {report['loaded_original_action_head_key_count']}",
        f"Expected missing BiT keys: {report['missing_bit_key_count']}",
        f"Missing non-BiT keys: {report['missing_non_bit_key_count']}",
        f"Unexpected skipped keys: {report['unexpected_key_count']}",
        f"Shape mismatches: {report['shape_mismatch_count']}",
        "",
        "Shape mismatches in original keys are configured to fail. BiT keys are expected to be missing from Base-IL and remain randomly initialized.",
        "",
        "```json",
        json.dumps(report, indent=2, sort_keys=True),
        "```",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
