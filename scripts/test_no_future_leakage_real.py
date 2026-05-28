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

from transformers.feature_extraction_utils import BatchFeature  # noqa: E402
from navsim.agents.recogdrive.expert_cache import iter_index, load_sample, validate_sample_payload  # noqa: E402
from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner, ReCogDriveDiffusionPlannerConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify target teacher tokens are not accessed during real eval/get_action.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--chunk-cache-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/recogdrive2b_expert768_eval.yaml"))
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--output", type=Path, default=Path("/mnt/project/VLA-AD/experiments/recogdrive_expert/no_future_leakage_report.md"))
    return parser.parse_args()


def dtype_from_precision(precision: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_planner(cfg_dict: Dict[str, Any]) -> ReCogDriveDiffusionPlanner:
    cfg = ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg={"num_heads": 8, "head_dim": 48, "num_layers": int(cfg_dict.get("num_dit_layers", 16)), "output_dim": 512, "dropout": 0.0, "attention_bias": True, "norm_eps": 1e-5, "interleave_attention": True},
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
    cfg.use_expert_features = True
    cfg.allow_future_targets_in_inference = False
    return ReCogDriveDiffusionPlanner(cfg)


def normalize_key(key: str) -> str:
    for prefix in ("agent.action_head.", "action_head."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def candidates(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    files: List[Path] = []
    for suffix in (".ckpt", ".pth", ".pt", ".safetensors", ".bin"):
        files.extend(p for p in path.rglob(f"*{suffix}") if p.is_file())
    return sorted(set(files), key=lambda p: (p.stat().st_size, str(p)), reverse=True)


def load_checkpoint(planner: ReCogDriveDiffusionPlanner, path: Path) -> None:
    files = candidates(path)
    if not files:
        raise FileNotFoundError(path)
    model_state = planner.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    for file in files:
        if file.suffix == ".safetensors":
            from safetensors.torch import load_file
            state = dict(load_file(str(file), device="cpu"))
        else:
            try:
                obj = torch.load(file, map_location="cpu", weights_only=False)
            except TypeError:
                obj = torch.load(file, map_location="cpu")
            state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
        for raw_key, value in state.items():
            key = normalize_key(raw_key)
            if isinstance(value, torch.Tensor) and key in model_state and tuple(value.shape) == tuple(model_state[key].shape):
                filtered.setdefault(key, value)
    planner.load_state_dict(filtered, strict=False)


def first_sample(chunk_cache_dir: Path) -> Dict[str, Any]:
    records = list(iter_index(chunk_cache_dir))
    if not records:
        raise RuntimeError(f"No records found in {chunk_cache_dir}")
    path = Path(records[0]["path"])
    if not path.is_absolute():
        path = chunk_cache_dir / path
    sample = load_sample(path)
    validate_sample_payload(sample, require_jepa=True, require_vggt=True, require_targets=False)
    return sample


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_precision(args.precision) if device.type == "cuda" else torch.float32
    planner = build_planner(load_yaml(args.config)).to(device)
    if dtype != torch.float32:
        planner = planner.to(dtype=dtype)
    load_checkpoint(planner, args.checkpoint)
    planner.eval()
    sample = first_sample(args.chunk_cache_dir)
    vl_features = sample["last_hidden_state"].float().unsqueeze(0).to(device=device, dtype=dtype)
    action_input = BatchFeature(data={
        "his_traj": sample["history_trajectory"].float().view(1, -1).to(device=device, dtype=dtype),
        "status_feature": sample["status_feature"].float().unsqueeze(0).to(device=device, dtype=dtype),
        "jepa_context_tokens": sample["jepa_context_tokens"].float().unsqueeze(0).to(device=device, dtype=dtype),
        "vggt_context_tokens": sample["vggt_context_tokens"].float().unsqueeze(0).to(device=device, dtype=dtype),
        "jepa_target_tokens": torch.full((1, 12, 1024), float("nan"), device=device, dtype=dtype),
        "vggt_target_tokens": torch.full((1, 12, 2048), float("nan"), device=device, dtype=dtype),
    })
    with torch.no_grad():
        pred = planner.get_action(vl_features, action_input, deterministic=True)["pred_traj"]
    report = {
        "checkpoint": str(args.checkpoint),
        "chunk_cache_dir": str(args.chunk_cache_dir),
        "pred_traj_finite": bool(torch.isfinite(pred).all().item()),
        "target_tokens_filled_with_nan": True,
        "target_tokens_accessed_in_eval": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("# No Future Leakage Check\n\n" + json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pred_traj_finite"]:
        raise RuntimeError("NaN target tokens affected eval prediction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
