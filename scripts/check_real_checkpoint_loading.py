#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner, ReCogDriveDiffusionPlannerConfig  # noqa: E402

EXPERT_MARKERS = (
    "jepa_projector", "vggt_projector", "jepa_adapter", "vggt_adapter", "jepa_alignment_head", "vggt_alignment_head",
    "jepa_type_embedding", "vggt_type_embedding", "z_jepa_type_embedding", "z_vggt_type_embedding", "jepa_gate", "vggt_gate", "branch_logits",
)
ORIGINAL_MARKERS = ("feature_encoder", "his_traj_encoder", "ego_status_encoder", "action_encoder", "fusion_projector", "model", "action_decoder", "position_embedding")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check real ReCogDrive-2B-IL checkpoint loading into expert model.")
    parser.add_argument("--base-il-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


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
        action_dim=3,
        action_horizon=8,
        input_embedding_dim=384,
        planner_dim=384,
        hidden_size=1024,
        sampling_method=str(cfg_dict.get("sampling_method", "ddim")),
        num_inference_steps=int(cfg_dict.get("num_inference_steps", 5)),
    )
    for key, value in cfg_dict.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.vlm_size = "small"
    cfg.planner_dim = 384
    cfg.expert_adapter_dim = 768
    cfg.use_expert_features = True
    return ReCogDriveDiffusionPlanner(cfg)


def is_expert_key(key: str) -> bool:
    return any(marker in key for marker in EXPERT_MARKERS)


def is_original_key(key: str) -> bool:
    return any(marker in key for marker in ORIGINAL_MARKERS) and not is_expert_key(key)


def normalize_key(key: str) -> str:
    for prefix in ("agent.action_head.", "action_head."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def candidates(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    suffixes = (".ckpt", ".pth", ".pt", ".safetensors", ".bin")
    files: List[Path] = []
    for suffix in suffixes:
        files.extend(p for p in path.rglob(f"*{suffix}") if p.is_file())
    def score(p: Path) -> tuple[int, int, str]:
        name = p.name.lower()
        return (1000 if "il" in name else 0) + (500 if "model" in name else 0), p.stat().st_size, str(p)
    return sorted(set(files), key=score, reverse=True)


def load_state(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return dict(load_file(str(path), device="cpu"))
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint {path} did not contain a state dict")
    return {key: value for key, value in state.items() if isinstance(value, torch.Tensor)}


def write_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Real Checkpoint Loading Report",
        "",
        f"Base checkpoint: `{report['base_il_checkpoint']}`",
        f"Loaded key count: {report['loaded_key_count']}",
        f"Loaded original key count: {report['loaded_original_key_count']}",
        f"Missing expert key count: {report['missing_expert_key_count']}",
        f"Unexpected key count: {report['unexpected_key_count']}",
        f"Shape mismatch count: {report['shape_mismatch_count']}",
        f"Original action_head loaded: {report['original_action_head_loaded']}",
        f"Expert keys randomly initialized: {report['expert_keys_randomly_initialized']}",
        "",
        "## Candidate Files",
        "",
    ]
    for item in report["candidate_files"]:
        marker = " <= selected first" if item == report["candidate_files"][0] else ""
        lines.append(f"- `{item}`{marker}")
    if report.get("shape_mismatches"):
        lines.extend(["", "## Shape Mismatches", ""])
        for item in report["shape_mismatches"]:
            lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    planner = build_planner(load_yaml(args.config))
    model_state = planner.state_dict()
    files = candidates(args.base_il_checkpoint)
    if not files:
        raise FileNotFoundError(f"No checkpoint files found under {args.base_il_checkpoint}")
    loaded: Dict[str, torch.Tensor] = {}
    unexpected: List[str] = []
    mismatches: List[str] = []
    for file in files:
        for raw_key, value in load_state(file).items():
            key = normalize_key(raw_key)
            if key not in model_state:
                unexpected.append(key)
                continue
            if tuple(value.shape) != tuple(model_state[key].shape):
                if not is_expert_key(key):
                    mismatches.append(f"{key}: checkpoint {tuple(value.shape)} vs model {tuple(model_state[key].shape)}")
                continue
            loaded.setdefault(key, value)
    if mismatches:
        report = {"base_il_checkpoint": str(args.base_il_checkpoint), "candidate_files": [str(f) for f in files], "shape_mismatches": mismatches, "shape_mismatch_count": len(mismatches)}
        write_report(args.output, report)
        raise RuntimeError("Original ReCogDrive key shape mismatches found; see report")
    incompatible = planner.load_state_dict(loaded, strict=False)
    missing_expert = [key for key in incompatible.missing_keys if is_expert_key(key)]
    loaded_original = [key for key in loaded if is_original_key(key)]
    report = {
        "base_il_checkpoint": str(args.base_il_checkpoint),
        "config": str(args.config),
        "candidate_files": [str(f) for f in files],
        "loaded_key_count": len(loaded),
        "loaded_original_key_count": len(loaded_original),
        "missing_expert_key_count": len(missing_expert),
        "unexpected_key_count": len(unexpected),
        "shape_mismatch_count": 0,
        "shape_mismatches": [],
        "original_action_head_loaded": len(loaded_original) > 0,
        "expert_keys_randomly_initialized": len(missing_expert) > 0,
        "missing_expert_keys": missing_expert,
        "unexpected_keys_sample": unexpected[:100],
    }
    write_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["original_action_head_loaded"]:
        raise RuntimeError("No original action_head keys were loaded from the base checkpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
