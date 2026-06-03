#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.risk_vla.run_risk_head_diagnostic_train import apply_overrides


RISK_CLASSES = ["low_score", "path_dac", "interaction_nc", "ttc", "progress", "comfort"]
STRATEGY_NAMES = ["base", "path_intent", "interaction", "progress", "comfort"]


CSV_COLUMNS = [
    "token",
    "split",
    "log_name",
    *(f"label_{name}" for name in RISK_CLASSES),
    *(f"pred_{name}" for name in RISK_CLASSES),
    *(f"strategy_weight_{name}" for name in STRATEGY_NAMES),
    "strategy_entropy",
]


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _scene_label(value: Any, index: int) -> float:
    if value is None:
        return float("nan")
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 1:
        return float(tensor[index].item())
    if tensor.ndim == 2:
        return float(tensor[:, index].mean().item())
    return float("nan")


def labels_from_sample(sample: Dict[str, Any]) -> Dict[str, float]:
    labels: Dict[str, float] = {}
    if "risk_labels" in sample:
        for idx, name in enumerate(RISK_CLASSES):
            labels[f"label_{name}"] = _scene_label(sample["risk_labels"], idx)
    else:
        mvp = {
            "low_score": sample.get("generic_risk_labels"),
            "path_dac": sample.get("drivable_risk_labels"),
            "interaction_nc": sample.get("ttc_risk_labels"),
            "ttc": sample.get("ttc_risk_labels"),
            "progress": sample.get("comfort_risk_labels"),
            "comfort": sample.get("comfort_risk_labels"),
        }
        for name in RISK_CLASSES:
            labels[f"label_{name}"] = _scene_label(mvp.get(name), 0) if mvp.get(name) is not None else float("nan")
    for name in RISK_CLASSES:
        labels.setdefault(f"label_{name}", _as_float(sample.get(f"label_{name}", float("nan"))))
    return labels


def token_from_sample(sample: Dict[str, Any], path: Optional[Path] = None) -> str:
    for key in ("token", "sample_token", "scene_token"):
        value = sample.get(key)
        if value is not None:
            return str(value)
    return path.stem if path is not None else "synthetic"


def row_from_outputs(
    *,
    token: str,
    split: str,
    log_name: str = "",
    labels: Optional[Dict[str, float]] = None,
    probs: Optional[torch.Tensor] = None,
    weights: Optional[Dict[str, float]] = None,
    entropy: float = float("nan"),
) -> Dict[str, Any]:
    row: Dict[str, Any] = {"token": token, "split": split, "log_name": log_name}
    row.update(labels or {})
    for name in RISK_CLASSES:
        row.setdefault(f"label_{name}", float("nan"))
    if probs is None:
        for name in RISK_CLASSES:
            row[f"pred_{name}"] = float("nan")
    else:
        probs = probs.detach().cpu().float().reshape(-1)
        for idx, name in enumerate(RISK_CLASSES):
            row[f"pred_{name}"] = float(probs[idx].item()) if idx < probs.numel() else float("nan")
    weights = weights or {}
    for name in STRATEGY_NAMES:
        row[f"strategy_weight_{name}"] = _as_float(weights.get(name, float("nan")))
    row["strategy_entropy"] = _as_float(entropy)
    return row


def write_predictions_csv(rows: Iterable[Dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def synthetic_rows(count: int, split: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx in range(count):
        labels = {f"label_{name}": float((idx + offset) % 2) for offset, name in enumerate(RISK_CLASSES)}
        probs = torch.linspace(0.1, 0.9, steps=len(RISK_CLASSES)) + idx * 0.001
        weights = {
            "base": 0.4,
            "path_intent": 0.1 + 0.01 * idx,
            "interaction": 0.2,
            "progress": 0.15,
            "comfort": 0.15,
        }
        rows.append(row_from_outputs(token=f"synthetic_{idx}", split=split, labels=labels, probs=probs, weights=weights, entropy=1.2))
    return rows


def load_sample(path: Path) -> Dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"{path} must contain a dict sample, got {type(obj).__name__}.")
    return obj


def sample_paths(cache_path: Path, max_samples: Optional[int]) -> List[Path]:
    paths = sorted(cache_path.glob("**/*.pt"))
    return paths[: int(max_samples)] if max_samples is not None else paths


def first_tensor(sample: Dict[str, Any], *keys: str) -> Optional[torch.Tensor]:
    for key in keys:
        value = sample.get(key)
        if isinstance(value, torch.Tensor):
            return value
    return None


def build_real_planner(args: argparse.Namespace):
    from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner, ReCogDriveDiffusionPlannerConfig

    cfg_dict: Dict[str, Any] = {
        "use_risk_vla": True,
        "risk_vla_use_oracle_router": False,
        "risk_vla_strategy_token_scale": 0.0,
        "risk_vla_horizon_residual_scale": 0.0,
        "risk_vla_risk_loss_weight": 0.0,
    }
    cfg_dict = apply_overrides(cfg_dict, args.config_overrides or [])
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
        action_dim=3,
        action_horizon=8,
        sampling_method=str(cfg_dict.get("sampling_method", "ddim")),
        vlm_size="small",
    )
    for key, value in cfg_dict.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    planner = ReCogDriveDiffusionPlanner(cfg).to(args.device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        normalized = {}
        model_state = planner.state_dict()
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            for prefix in ("agent.action_head.", "action_head."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            if key in model_state and tuple(value.shape) == tuple(model_state[key].shape):
                normalized[key] = value
        planner.load_state_dict(normalized, strict=False)
    planner.eval()
    return planner


def rows_from_real_cache(args: argparse.Namespace) -> List[Dict[str, Any]]:
    from transformers.feature_extraction_utils import BatchFeature

    planner = build_real_planner(args)
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for path in sample_paths(args.cache_path, args.max_samples):
            sample = load_sample(path)
            vl = first_tensor(sample, "last_hidden_state", "vl_features")
            if not isinstance(vl, torch.Tensor):
                continue
            if vl.ndim == 2:
                vl = vl.unsqueeze(0)
            history = first_tensor(sample, "history_trajectory")
            his_traj = first_tensor(sample, "his_traj")
            if history is None:
                history = torch.zeros(4, 3)
            if his_traj is None:
                his_traj = history.reshape(-1)
            status = first_tensor(sample, "status_feature")
            command = first_tensor(sample, "high_command_one_hot")
            action_data = {
                "his_traj": his_traj.reshape(1, -1),
                "history_trajectory": history.unsqueeze(0) if history.ndim == 2 else history,
                "status_feature": (status if status is not None else torch.zeros(8)).reshape(1, -1),
                "high_command_one_hot": (command if command is not None else torch.zeros(3)).reshape(1, -1),
            }
            for key in ("risk_labels", "generic_risk_labels", "drivable_risk_labels", "ttc_risk_labels", "comfort_risk_labels"):
                if key in sample:
                    value = sample[key]
                    action_data[key] = value.unsqueeze(0) if isinstance(value, torch.Tensor) and value.ndim in {1, 2} else value
            batch = BatchFeature(data={key: value.to(args.device) if isinstance(value, torch.Tensor) else value for key, value in action_data.items()})
            context = planner._prepare_dit_context(vl.to(args.device), batch, training=False)
            probs = context.get("risk_vla_risk_probs")
            probs_row = probs[0].detach().cpu() if isinstance(probs, torch.Tensor) else None
            weights = {
                "base": context.get("risk_vla_weight_base"),
                "path_intent": context.get("risk_vla_weight_path_intent"),
                "interaction": context.get("risk_vla_weight_interaction"),
                "progress": context.get("risk_vla_weight_progress"),
                "comfort": context.get("risk_vla_weight_comfort"),
            }
            weights = {key: float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else _as_float(value) for key, value in weights.items()}
            entropy = context.get("risk_vla_strategy_entropy")
            rows.append(
                row_from_outputs(
                    token=token_from_sample(sample, path),
                    split=args.split,
                    log_name=str(sample.get("log_name", "")),
                    labels=labels_from_sample(sample),
                    probs=probs_row,
                    weights=weights,
                    entropy=float(entropy.detach().cpu().item()) if isinstance(entropy, torch.Tensor) else float("nan"),
                )
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RISK-VLA token-level risk predictions from a cache dataset.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--split", default="custom")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--config-overrides", action="append", default=[])
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    if args.synthetic_smoke:
        count = int(args.max_samples or 8)
        rows = synthetic_rows(count, args.split)
    else:
        if args.cache_path is None:
            raise ValueError("--cache-path is required unless --synthetic-smoke is set.")
        rows = rows_from_real_cache(args)
    write_predictions_csv(rows, args.output_csv)
    print(json.dumps({"output_csv": str(args.output_csv), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
