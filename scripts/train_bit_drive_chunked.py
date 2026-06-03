#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers.feature_extraction_utils import BatchFeature  # noqa: E402
from navsim.agents.recogdrive.bit_failure_sampler import (  # noqa: E402
    batch_failure_fraction_report,
    build_failure_limited_batches,
    build_weighted_sampler,
    load_failure_index,
    summarize_sampling_records,
    validate_failure_index_for_training,
)
from navsim.agents.recogdrive.expert_cache import iter_index, load_sample  # noqa: E402
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
    "bit_risk_head",
    "bit_risk_token_encoder",
)
ACTION_HEAD_MARKERS = (
    "feature_encoder",
    "his_traj_encoder",
    "ego_status_encoder",
    "action_encoder",
    "fusion_projector",
    "model",
    "action_decoder",
    "position_embedding",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BiT-Drive Step 1-3 from ReCogDrive VLM-hidden chunks.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-il-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--chunk-cache-dir", type=Path, default=None)
    parser.add_argument("--chunk-cache-root", type=Path, default=None)
    parser.add_argument("--chunk-name-pattern", default="chunk_*")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--epochs-per-chunk", type=int, default=1)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--lr-bit", type=float, default=1e-4)
    parser.add_argument("--lr-action-head", type=float, default=2e-5)
    parser.add_argument("--freeze-vlm", action="store_true", default=True)
    parser.add_argument("--freeze-base-action-head", action="store_true")
    parser.add_argument("--failure-index-path", type=Path, default=None)
    parser.add_argument("--failure-sampling-enabled", action="store_true")
    parser.add_argument("--risk-label-jsonl", type=Path, default=None)
    parser.add_argument("--require-risk-labels", action="store_true")
    parser.add_argument("--counterfactual-label-jsonl", type=Path, default=None)
    parser.add_argument("--require-counterfactual-labels", action="store_true")
    parser.add_argument("--use-base-traj-kd", action="store_true")
    parser.add_argument(
        "--base-traj-kd-mode",
        choices=("early_longitudinal", "early_step", "early_longitudinal_step_heading"),
        default="early_longitudinal",
    )
    parser.add_argument("--safety-tag-filter", default="bit_nc_regression,bit_ttc_regression")
    parser.add_argument("--dac-tag-filter", default="bit_dac_fix")
    parser.add_argument("--max-counterfactual-label-age", type=float, default=None, help="Maximum metadata age in seconds.")
    parser.add_argument("--include-tags", default=None)
    parser.add_argument("--exclude-tags", default=None)
    parser.add_argument("--max-failure-fraction-per-batch", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--debug-overfit", action="store_true")
    parser.add_argument("--seed", type=int, default=20260601)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def tag_list(value: Any) -> Optional[List[str]]:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def resolve_failure_options(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    enabled = bool(args.failure_sampling_enabled or cfg.get("failure_sampling_enabled", False))
    path = args.failure_index_path or cfg.get("failure_index_path")
    path = Path(path) if path else None
    include_tags = tag_list(args.include_tags if args.include_tags is not None else cfg.get("include_tags"))
    exclude_tags = tag_list(args.exclude_tags if args.exclude_tags is not None else cfg.get("exclude_tags"))
    max_fraction = (
        args.max_failure_fraction_per_batch
        if args.max_failure_fraction_per_batch is not None
        else cfg.get("max_failure_fraction_per_batch")
    )
    if max_fraction is not None:
        max_fraction = float(max_fraction)
    return {
        "enabled": enabled,
        "path": path,
        "include_tags": include_tags,
        "exclude_tags": exclude_tags,
        "max_failure_fraction_per_batch": max_fraction,
        "failure_weights": cfg.get("failure_weights"),
    }


def dtype_from_precision(precision: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]


def build_planner(cfg_dict: Dict[str, Any], precision: str) -> ReCogDriveDiffusionPlanner:
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
        model_dtype={"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[precision],
        vlm_size="small",
    )
    for key, value in cfg_dict.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.vlm_size = "small"
    cfg.use_expert_features = bool(cfg_dict.get("use_expert_features", False))
    cfg.use_bit_drive = bool(cfg_dict.get("use_bit_drive", True))
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


def shape_safe_load(planner: ReCogDriveDiffusionPlanner, checkpoint_path: Path, *, strict_original: bool) -> Dict[str, Any]:
    if checkpoint_path is None:
        return {"loaded_key_count": 0, "checkpoint_files": []}
    candidates = checkpoint_candidates(checkpoint_path)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found under {checkpoint_path}")
    model_state = planner.state_dict()
    loaded: Dict[str, torch.Tensor] = {}
    mismatches: List[str] = []
    unexpected = 0
    for file in candidates:
        for raw_key, value in load_state_file(file).items():
            key = normalize_key(raw_key)
            if key not in model_state:
                unexpected += 1
                continue
            if tuple(value.shape) != tuple(model_state[key].shape):
                if is_bit_key(key) and not strict_original:
                    continue
                mismatches.append(f"{key}: checkpoint {tuple(value.shape)} vs model {tuple(model_state[key].shape)}")
                continue
            loaded.setdefault(key, value)
    if mismatches and strict_original:
        raise RuntimeError("Checkpoint shape mismatch for original keys:\n" + "\n".join(mismatches[:50]))
    incompatible = planner.load_state_dict(loaded, strict=False)
    return {
        "loaded_key_count": len(loaded),
        "missing_bit_key_count": len([key for key in incompatible.missing_keys if is_bit_key(key)]),
        "missing_non_bit_key_count": len([key for key in incompatible.missing_keys if not is_bit_key(key)]),
        "unexpected_key_count": unexpected,
        "shape_mismatch_count": len(mismatches),
        "checkpoint_files": [str(path) for path in candidates],
    }


def resolve_index_path(chunk_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_file():
        return path
    if not path.is_absolute():
        candidate = chunk_dir / path
        if candidate.is_file():
            return candidate
    return path


class BitChunkDataset(Dataset):
    def __init__(
        self,
        chunk_dir: Path,
        max_samples: Optional[int] = None,
        risk_labels: Optional[Dict[str, torch.Tensor]] = None,
        require_risk_labels: bool = False,
        counterfactual_labels: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        require_counterfactual_labels: bool = False,
    ) -> None:
        self.chunk_dir = chunk_dir
        records = list(iter_index(chunk_dir))
        self.risk_labels = risk_labels or {}
        self.counterfactual_labels = counterfactual_labels or {}
        if require_risk_labels:
            records = [record for record in records if str(record.get("sample_token")) in self.risk_labels]
        if self.counterfactual_labels:
            expanded_records = []
            for record in records:
                sample_token = str(record.get("sample_token"))
                labels = self.counterfactual_labels.get(sample_token)
                if labels:
                    for label_index in range(len(labels)):
                        expanded = dict(record)
                        expanded["_counterfactual_label_index"] = label_index
                        expanded_records.append(expanded)
                elif not require_counterfactual_labels:
                    expanded_records.append(record)
            records = expanded_records
        elif require_counterfactual_labels:
            records = []
        self.records = records
        if max_samples is not None:
            self.records = self.records[:max_samples]
        if not self.records:
            raise RuntimeError(f"No records found in {chunk_dir}")
        self.require_risk_labels = bool(require_risk_labels)
        self.require_counterfactual_labels = bool(require_counterfactual_labels)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        sample = load_sample(resolve_index_path(self.chunk_dir, record["path"]))
        for key in ("last_hidden_state", "history_trajectory", "status_feature", "trajectory"):
            if key not in sample:
                raise KeyError(f"Sample missing required BiT training key: {key}")
        sample.setdefault("scene_token", record.get("scene_token"))
        sample.setdefault("sample_token", record.get("sample_token"))
        sample_token = str(sample.get("sample_token") or record.get("sample_token"))
        if sample_token in self.risk_labels:
            sample["bit_risk_labels"] = self.risk_labels[sample_token]
        elif self.require_risk_labels:
            raise KeyError(f"Missing required bit_risk_labels for sample_token={sample_token}")
        if sample_token in self.counterfactual_labels:
            labels = self.counterfactual_labels[sample_token]
            label_index = int(record.get("_counterfactual_label_index", 0))
            label = labels[min(max(label_index, 0), len(labels) - 1)]
            for key in (
                "base_pred_traj",
                "counterfactual_candidate_mode",
                "bit_counterfactual_tags",
                "bit_safety_regression_mask",
                "hard_safety_mask",
                "soft_safety_mask",
                "safety_weight",
                "bit_nc_regression_mask",
                "bit_ttc_regression_mask",
                "bit_dac_fix_mask",
                "dac_fix_mask",
                "dac_weight",
                "bit_risk_labels",
                "base_metrics",
                "bit_metrics",
            ):
                if key in label and key not in sample:
                    sample[key] = label[key]
        elif self.require_counterfactual_labels:
            raise KeyError(f"Missing required counterfactual label for sample_token={sample_token}")
        return {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in sample.items()}


def collate(samples: List[Dict[str, Any]]) -> Tuple[torch.Tensor, BatchFeature]:
    vl_features = pad_sequence([sample["last_hidden_state"].float() for sample in samples], batch_first=True, padding_value=0.0)
    his = torch.stack([sample["history_trajectory"].float().view(-1) for sample in samples], dim=0)
    status = torch.stack([sample["status_feature"].float() for sample in samples], dim=0)
    action = torch.stack([sample["trajectory"].float() for sample in samples], dim=0)
    data = {"his_traj": his, "status_feature": status, "action": action}
    if any("bit_risk_labels" in sample for sample in samples):
        data["bit_risk_labels"] = torch.stack(
            [
                sample.get("bit_risk_labels", torch.full((4,), float("nan"))).float()
                for sample in samples
            ],
            dim=0,
        )
    if any("base_pred_traj" in sample for sample in samples):
        data["base_pred_traj"] = torch.stack(
            [
                sample.get("base_pred_traj", torch.full((8, 3), float("nan"))).float()
                for sample in samples
            ],
            dim=0,
        )
    if any("bit_safety_regression_mask" in sample for sample in samples):
        data["bit_safety_regression_mask"] = torch.tensor(
            [bool(sample.get("bit_safety_regression_mask", False)) for sample in samples],
            dtype=torch.bool,
        )
    if any("hard_safety_mask" in sample for sample in samples):
        data["hard_safety_mask"] = torch.tensor(
            [bool(sample.get("hard_safety_mask", False)) for sample in samples],
            dtype=torch.bool,
        )
    if any("soft_safety_mask" in sample for sample in samples):
        data["soft_safety_mask"] = torch.tensor(
            [bool(sample.get("soft_safety_mask", False)) for sample in samples],
            dtype=torch.bool,
        )
    if any("safety_weight" in sample for sample in samples):
        data["safety_weight"] = torch.tensor(
            [float(sample.get("safety_weight", 0.0)) for sample in samples],
            dtype=torch.float32,
        )
    if any("bit_nc_regression_mask" in sample for sample in samples):
        data["bit_nc_regression_mask"] = torch.tensor(
            [bool(sample.get("bit_nc_regression_mask", False)) for sample in samples],
            dtype=torch.bool,
        )
    if any("bit_ttc_regression_mask" in sample for sample in samples):
        data["bit_ttc_regression_mask"] = torch.tensor(
            [bool(sample.get("bit_ttc_regression_mask", False)) for sample in samples],
            dtype=torch.bool,
        )
    if any("bit_dac_fix_mask" in sample for sample in samples):
        data["bit_dac_fix_mask"] = torch.tensor(
            [bool(sample.get("bit_dac_fix_mask", False)) for sample in samples],
            dtype=torch.bool,
        )
    if any("dac_fix_mask" in sample for sample in samples):
        data["dac_fix_mask"] = torch.tensor(
            [bool(sample.get("dac_fix_mask", False)) for sample in samples],
            dtype=torch.bool,
        )
    if any("dac_weight" in sample for sample in samples):
        data["dac_weight"] = torch.tensor(
            [float(sample.get("dac_weight", 0.0)) for sample in samples],
            dtype=torch.float32,
        )
    if any("bit_counterfactual_tags" in sample for sample in samples):
        data["bit_counterfactual_tags"] = [sample.get("bit_counterfactual_tags", []) for sample in samples]
    if any("counterfactual_candidate_mode" in sample for sample in samples):
        data["counterfactual_candidate_mode"] = [sample.get("counterfactual_candidate_mode") for sample in samples]
    if any("base_metrics" in sample for sample in samples):
        data["base_metrics"] = [sample.get("base_metrics") for sample in samples]
    if any("bit_metrics" in sample for sample in samples):
        data["bit_metrics"] = [sample.get("bit_metrics") for sample in samples]
    return vl_features, BatchFeature(data=data)


def _validate_training_label_metadata(path: Path, *, max_age_seconds: Optional[float] = None) -> Dict[str, Any]:
    metadata_path = path.parent / "metadata.json"
    metadata: Dict[str, Any] = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        split = str(metadata.get("split", "")).lower()
        training_allowed = metadata.get("training_allowed")
        if "test" in split or bool(metadata.get("analysis_only", False)) or training_allowed is False:
            raise RuntimeError(f"Refusing to use test/analysis counterfactual labels for training: {path}")
        if max_age_seconds is not None and metadata.get("created_at"):
            created_at = str(metadata["created_at"]).replace("Z", "+00:00")
            try:
                created = datetime.fromisoformat(created_at)
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age = (datetime.now(timezone.utc) - created).total_seconds()
                if age > float(max_age_seconds):
                    raise RuntimeError(
                        f"Counterfactual labels are older than --max-counterfactual-label-age: "
                        f"{age:.1f}s > {max_age_seconds:.1f}s"
                    )
            except ValueError:
                raise RuntimeError(f"Could not parse metadata.created_at={metadata['created_at']!r} for {path}") from None
    return metadata


def _row_uses_test_split(row: Dict[str, Any]) -> bool:
    split = str(row.get("split") or row.get("split_alias") or "").lower()
    return "test" in split


def risk_tensor_from_metrics(metrics: Dict[str, Any]) -> torch.Tensor:
    return torch.tensor(
        [
            1.0 if float(metrics.get("pdm") or 0.0) <= 1e-9 else 0.0,
            1.0 if float(metrics.get("dac") or 0.0) <= 1e-9 else 0.0,
            1.0 if float(metrics.get("nc") or 0.0) <= 1e-9 else 0.0,
            1.0 if float(metrics.get("ttc") or 0.0) <= 1e-9 else 0.0,
        ],
        dtype=torch.float32,
    )


def load_risk_labels(path: Optional[Path]) -> Dict[str, torch.Tensor]:
    if path is None:
        return {}
    _validate_training_label_metadata(path)
    labels: Dict[str, torch.Tensor] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if _row_uses_test_split(row):
                raise RuntimeError(f"Refusing to use navtest/test risk labels for training: {path}")
            sample_token = row.get("sample_token")
            metrics = row.get("bit_metrics") or {}
            if not sample_token or not metrics:
                continue
            labels[str(sample_token)] = risk_tensor_from_metrics(metrics)
    return labels


def infer_counterfactual_tags(row: Dict[str, Any]) -> List[str]:
    tags = list(row.get("tags") or row.get("bit_counterfactual_tags") or [])
    if tags:
        return [str(tag) for tag in tags]
    base = row.get("base_metrics") or {}
    bit = row.get("bit_metrics") or {}
    if not base or not bit:
        return ["neutral"]
    eps = 1e-9
    if float(base.get("dac") or 0.0) <= eps and float(bit.get("dac") or 0.0) > eps:
        tags.append("bit_dac_fix")
    if float(base.get("nc") or 0.0) > eps and float(bit.get("nc") or 0.0) <= eps:
        tags.append("bit_nc_regression")
    if float(base.get("ttc") or 0.0) > eps and float(bit.get("ttc") or 0.0) <= eps:
        tags.append("bit_ttc_regression")
    if float(base.get("pdm") or 0.0) <= eps and float(bit.get("pdm") or 0.0) > eps:
        tags.append("bit_zero_fix")
    if float(bit.get("pdm") or 0.0) + eps < float(base.get("pdm") or 0.0):
        tags.append("bit_bad")
    return tags or ["neutral"]


def load_counterfactual_labels(
    path: Optional[Path],
    *,
    safety_tags: Optional[List[str]],
    dac_tags: Optional[List[str]],
    max_age_seconds: Optional[float],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    if path is None:
        return {}, {}
    metadata = _validate_training_label_metadata(path, max_age_seconds=max_age_seconds)
    safety_set = set(
        safety_tags
        or ["bit_nc_regression", "bit_ttc_regression", "hard_nc_regression", "hard_ttc_regression", "soft_safety_regression"]
    )
    hard_safety_tags = {"bit_nc_regression", "bit_ttc_regression", "hard_nc_regression", "hard_ttc_regression"}
    soft_safety_tags = {"soft_safety_regression", "soft_ttc_regression"}
    dac_set = set(dac_tags or ["bit_dac_fix", "dac_fix"])
    labels: Dict[str, List[Dict[str, Any]]] = {}
    summary = {
        "path": str(path),
        "metadata": metadata,
        "rows": 0,
        "usable": 0,
        "safety_regression": 0,
        "hard_safety_regression": 0,
        "soft_safety_regression": 0,
        "dac_fix": 0,
        "tag_counts": {},
    }
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            summary["rows"] += 1
            row = json.loads(line)
            if _row_uses_test_split(row):
                raise RuntimeError(f"Refusing to use navtest/test counterfactual labels for training: {path}")
            sample_token = row.get("sample_token")
            base_traj = row.get("base_pred_traj")
            base_metrics = row.get("base_metrics") or {}
            bit_metrics = row.get("bit_metrics") or {}
            if not sample_token or base_traj is None:
                continue
            tags = infer_counterfactual_tags(row)
            tag_set = set(tags)
            for tag in tags:
                summary["tag_counts"][tag] = summary["tag_counts"].get(tag, 0) + 1
            hard_mask = bool(row.get("hard_safety_mask", False) or tag_set & hard_safety_tags)
            soft_mask = bool(row.get("soft_safety_mask", False) or tag_set & soft_safety_tags)
            safety_mask = bool(row.get("bit_safety_regression_mask", False) or hard_mask or soft_mask or tag_set & safety_set)
            nc_mask = "bit_nc_regression" in tag_set or "hard_nc_regression" in tag_set
            ttc_mask = "bit_ttc_regression" in tag_set or "hard_ttc_regression" in tag_set
            dac_mask = bool(row.get("dac_fix_mask", False) or row.get("bit_dac_fix_mask", False) or tag_set & dac_set)
            if hard_mask:
                safety_weight = 1.0
            elif soft_mask:
                safety_weight = float(row.get("safety_weight", 0.25))
            else:
                safety_weight = float(row.get("safety_weight", 0.0))
            label = {
                "base_pred_traj": torch.tensor(base_traj, dtype=torch.float32),
                "counterfactual_candidate_mode": row.get("counterfactual_candidate_mode") or row.get("candidate_mode"),
                "bit_counterfactual_tags": tags,
                "bit_safety_regression_mask": safety_mask,
                "hard_safety_mask": hard_mask,
                "soft_safety_mask": soft_mask,
                "safety_weight": safety_weight,
                "bit_nc_regression_mask": nc_mask,
                "bit_ttc_regression_mask": ttc_mask,
                "bit_dac_fix_mask": dac_mask,
                "dac_fix_mask": dac_mask,
                "dac_weight": float(row.get("dac_weight", 1.0 if dac_mask else 0.0)),
                "base_metrics": base_metrics,
                "bit_metrics": bit_metrics,
            }
            if bit_metrics:
                label["bit_risk_labels"] = risk_tensor_from_metrics(bit_metrics)
            labels.setdefault(str(sample_token), []).append(label)
            summary["usable"] += 1
            summary["safety_regression"] += int(safety_mask)
            summary["hard_safety_regression"] += int(hard_mask)
            summary["soft_safety_regression"] += int(soft_mask)
            summary["dac_fix"] += int(dac_mask)
    return labels, summary


def chunk_dirs(args: argparse.Namespace) -> List[Path]:
    if args.chunk_cache_dir is not None:
        return [args.chunk_cache_dir]
    if args.chunk_cache_root is None:
        raise ValueError("Set --chunk-cache-dir or --chunk-cache-root")
    chunks = sorted(path for path in args.chunk_cache_root.glob(args.chunk_name_pattern) if path.is_dir())
    if not chunks:
        raise FileNotFoundError(f"No chunks matching {args.chunk_name_pattern!r} under {args.chunk_cache_root}")
    return chunks


def set_trainable(planner: ReCogDriveDiffusionPlanner, freeze_base_action_head: bool) -> None:
    for name, param in planner.named_parameters():
        if is_bit_key(name):
            param.requires_grad = True
        elif freeze_base_action_head:
            param.requires_grad = False
        else:
            param.requires_grad = any(marker in name for marker in ACTION_HEAD_MARKERS)


def optimizer_for(planner: ReCogDriveDiffusionPlanner, args: argparse.Namespace) -> torch.optim.Optimizer:
    bit_params = []
    action_params = []
    for name, param in planner.named_parameters():
        if not param.requires_grad:
            continue
        if is_bit_key(name):
            bit_params.append(param)
        else:
            action_params.append(param)
    groups = []
    if bit_params:
        groups.append({"params": bit_params, "lr": args.lr_bit, "weight_decay": 1e-4, "name": "bit"})
    if action_params and args.lr_action_head > 0:
        groups.append({"params": action_params, "lr": args.lr_action_head, "weight_decay": 1e-4, "name": "action_head"})
    if not groups:
        raise RuntimeError("No trainable BiT/action_head parameters.")
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)


def save_checkpoint(path: Path, planner: ReCogDriveDiffusionPlanner, optimizer: torch.optim.Optimizer, step: int, cfg: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": planner.state_dict(), "optimizer": optimizer.state_dict(), "global_step": step, "config": cfg, "metrics": metrics}, path)


def finite_float(output: Dict[str, Any], key: str) -> float:
    value = output.get(key)
    if not isinstance(value, torch.Tensor):
        return 0.0
    if not torch.isfinite(value.detach()).all():
        raise RuntimeError(f"Non-finite metric {key}: {value}")
    return float(value.detach().float().item())


def current_gpu_memory(device: torch.device) -> Optional[int]:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def write_final_report(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# BiT-Drive Chunked Training Report",
        "",
        f"Output dir: `{summary['output_dir']}`",
        f"Global steps: {summary['global_step']}",
        f"Best loss: {summary.get('best_loss')}",
        f"Last loss: {summary.get('last_loss')}",
        f"Latest checkpoint: `{summary.get('latest_checkpoint')}`",
        f"Best checkpoint: `{summary.get('best_checkpoint')}`",
        f"Failure sampling enabled: {summary.get('failure_sampling_enabled')}",
        "",
        "This is supervised BiT Step 1-3 training only. No RL/GRPO is run by this script.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    cfg = load_yaml(args.config)
    if "lr_bit" in cfg:
        args.lr_bit = float(cfg["lr_bit"])
    if "lr_action_head" in cfg:
        args.lr_action_head = float(cfg["lr_action_head"])
    if bool(cfg.get("freeze_base_action_head", False)):
        args.freeze_base_action_head = True
    if args.use_base_traj_kd:
        cfg["bit_use_d5_conservative_loss"] = True
        if args.base_traj_kd_mode == "early_longitudinal":
            cfg.setdefault("bit_safe_kd_x_weight", 1.0)
            cfg.setdefault("bit_safe_kd_step_weight", 0.0)
            cfg.setdefault("bit_safe_kd_heading_weight", 0.0)
        elif args.base_traj_kd_mode == "early_step":
            cfg.setdefault("bit_safe_kd_x_weight", 1.0)
            cfg.setdefault("bit_safe_kd_step_weight", 0.5)
            cfg.setdefault("bit_safe_kd_heading_weight", 0.0)
        elif args.base_traj_kd_mode == "early_longitudinal_step_heading":
            cfg.setdefault("bit_safe_kd_x_weight", 1.0)
            cfg.setdefault("bit_safe_kd_step_weight", 0.5)
            cfg.setdefault("bit_safe_kd_heading_weight", 0.2)
    failure_options = resolve_failure_options(args, cfg)
    failure_index = None
    failure_metadata: Dict[str, Any] = {}
    if failure_options["enabled"]:
        if failure_options["path"] is None:
            raise ValueError("Failure sampling is enabled but no failure_index_path was provided.")
        failure_metadata = validate_failure_index_for_training(failure_options["path"])
        failure_index = load_failure_index(
            failure_options["path"],
            include_tags=failure_options["include_tags"],
            exclude_tags=failure_options["exclude_tags"],
        )
        if not failure_index:
            raise RuntimeError(
                f"Failure index {failure_options['path']} has no records after include/exclude tag filtering."
            )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_precision(args.precision, device)
    planner = build_planner(cfg, args.precision).to(device)
    load_report: Dict[str, Any] = {}
    if args.resume_from:
        load_report = shape_safe_load(planner, args.resume_from, strict_original=False)
    elif args.base_il_checkpoint:
        load_report = shape_safe_load(planner, args.base_il_checkpoint, strict_original=True)
    if dtype != torch.float32:
        planner = planner.to(dtype=dtype)
    set_trainable(planner, args.freeze_base_action_head)
    optimizer = optimizer_for(planner, args)
    counterfactual_labels, counterfactual_summary = load_counterfactual_labels(
        args.counterfactual_label_jsonl,
        safety_tags=tag_list(args.safety_tag_filter),
        dac_tags=tag_list(args.dac_tag_filter),
        max_age_seconds=args.max_counterfactual_label_age,
    )
    risk_labels = load_risk_labels(args.risk_label_jsonl)
    for sample_token, labels_for_token in counterfactual_labels.items():
        for label in labels_for_token:
            if "bit_risk_labels" in label:
                risk_labels.setdefault(sample_token, label["bit_risk_labels"])
                break
    scaler = torch.cuda.amp.GradScaler(enabled=(args.precision == "fp16" and device.type == "cuda"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoint_load.json").write_text(json.dumps(load_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "train_args.json").write_text(json.dumps(vars(args), default=str, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.counterfactual_label_jsonl is not None:
        (args.output_dir / "counterfactual_label_summary.json").write_text(
            json.dumps(counterfactual_summary, default=str, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (args.output_dir / "failure_sampling_options.json").write_text(
        json.dumps({**failure_options, "path": str(failure_options["path"]) if failure_options["path"] else None, "metadata": failure_metadata}, default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    chunks = chunk_dirs(args)
    sampling_reports: List[Dict[str, Any]] = []
    log_fp = (args.output_dir / "train_log.jsonl").open("w", encoding="utf-8")
    global_step = 0
    optimizer_step = 0
    pending = 0
    best_loss = math.inf
    last_loss: Optional[float] = None
    planner.train()
    stop = False
    try:
        while True:
            for chunk in chunks:
                try:
                    dataset = BitChunkDataset(
                        chunk,
                        max_samples=args.max_samples,
                        risk_labels=risk_labels,
                        require_risk_labels=args.require_risk_labels,
                        counterfactual_labels=counterfactual_labels,
                        require_counterfactual_labels=args.require_counterfactual_labels,
                    )
                except RuntimeError as exc:
                    if (args.require_risk_labels or args.require_counterfactual_labels) and "No records found" in str(exc):
                        print(f"Skipping {chunk.name}: no samples with required labels.")
                        continue
                    raise
                sampler = None
                shuffle = True
                batch_sampler = None
                if failure_options["enabled"]:
                    assert failure_index is not None
                    shuffle = False
                    chunk_report = {
                        "chunk": chunk.name,
                        **summarize_sampling_records(dataset.records, failure_index),
                    }
                    max_fraction = failure_options["max_failure_fraction_per_batch"]
                    if max_fraction is not None:
                        batch_sampler = build_failure_limited_batches(
                            dataset.records,
                            failure_index,
                            batch_size=args.batch_size,
                            max_failure_fraction=float(max_fraction),
                            seed=args.seed + global_step,
                        )
                        chunk_report.update(batch_failure_fraction_report(batch_sampler, dataset.records, failure_index))
                    else:
                        sampler = build_weighted_sampler(
                            dataset.records,
                            failure_options["path"],
                            failure_weights=failure_options["failure_weights"],
                            include_tags=failure_options["include_tags"],
                            exclude_tags=failure_options["exclude_tags"],
                        )
                        sampled_indices = list(iter(sampler))
                        sampled_batches = [
                            sampled_indices[index : index + args.batch_size]
                            for index in range(0, len(sampled_indices), args.batch_size)
                        ]
                        chunk_report.update(batch_failure_fraction_report(sampled_batches, dataset.records, failure_index))
                    sampling_reports.append(chunk_report)
                    (args.output_dir / "sampling_report.json").write_text(
                        json.dumps(
                            {
                                "failure_sampling_enabled": True,
                                "failure_index_path": str(failure_options["path"]),
                                "include_tags": failure_options["include_tags"],
                                "exclude_tags": failure_options["exclude_tags"],
                                "max_failure_fraction_per_batch": max_fraction,
                                "chunks": sampling_reports,
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                if batch_sampler is not None:
                    loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate, num_workers=0)
                else:
                    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler, collate_fn=collate, num_workers=0)
                for _epoch in range(args.epochs_per_chunk):
                    for batch_idx, (vl_features, action_input) in enumerate(loader):
                        step_start = time.time()
                        vl_features = vl_features.to(device=device, dtype=dtype)
                        for key, value in list(action_input.items()):
                            if isinstance(value, torch.Tensor):
                                action_input[key] = value.to(device=device, dtype=dtype)
                        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda" and dtype != torch.float32)):
                            output = planner(vl_features, action_input)
                            loss = output["loss"] / max(1, args.gradient_accumulation_steps)
                        scaler.scale(loss).backward()
                        pending += 1
                        grad_norm_value = None
                        if pending >= args.gradient_accumulation_steps:
                            scaler.unscale_(optimizer)
                            grad_norm = torch.nn.utils.clip_grad_norm_(planner.parameters(), 1.0)
                            grad_norm_value = float(grad_norm.detach().float().item())
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                            optimizer_step += 1
                            pending = 0
                        global_step += 1
                        record = {
                            "step": global_step,
                            "optimizer_step": optimizer_step,
                            "chunk": chunk.name,
                            "batch_index": batch_idx,
                            "total_loss": finite_float(output, "loss"),
                            "diffusion_loss": finite_float(output, "diffusion_loss"),
                            "bit_terminal_loss": finite_float(output, "bit_terminal_loss"),
                            "bit_path_loss": finite_float(output, "bit_path_loss"),
                            "bit_end_consistency_loss": finite_float(output, "bit_end_consistency_loss"),
                            "bit_reverse_loss": finite_float(output, "bit_reverse_loss"),
                            "bit_cycle_loss": finite_float(output, "bit_cycle_loss"),
                            "bit_used_gt_condition_rate": finite_float(output, "bit_used_gt_condition_rate"),
                            "bit_context_gate_value": finite_float(output, "bit_context_gate_value"),
                            "bit_action_gate_value": finite_float(output, "bit_action_gate_value"),
                            "bit_gt_condition_prob_current": finite_float(output, "bit_gt_condition_prob_current"),
                            "bit_condition_strength_context": finite_float(output, "bit_condition_strength_context"),
                            "bit_condition_strength_action": finite_float(output, "bit_condition_strength_action"),
                            "bit_base_longitudinal_loss": finite_float(output, "bit_base_longitudinal_loss"),
                            "bit_base_lateral_loss": finite_float(output, "bit_base_lateral_loss"),
                            "bit_risk_loss": finite_float(output, "bit_risk_loss"),
                            "bit_risk_bce_loss": finite_float(output, "bit_risk_bce_loss"),
                            "risk_bce_loss": finite_float(output, "risk_bce_loss"),
                            "bit_risk_zero_loss": finite_float(output, "bit_risk_zero_loss"),
                            "bit_risk_dac_loss": finite_float(output, "bit_risk_dac_loss"),
                            "bit_risk_nc_loss": finite_float(output, "bit_risk_nc_loss"),
                            "bit_risk_ttc_loss": finite_float(output, "bit_risk_ttc_loss"),
                            "d5_safe_kd_loss": finite_float(output, "d5_safe_kd_loss"),
                            "d5_dac_path_preserve_loss": finite_float(output, "d5_dac_path_preserve_loss"),
                            "d5_safety_mask_rate": finite_float(output, "d5_safety_mask_rate"),
                            "d5_hard_safety_mask_rate": finite_float(output, "d5_hard_safety_mask_rate"),
                            "d5_soft_safety_mask_rate": finite_float(output, "d5_soft_safety_mask_rate"),
                            "d5_dac_fix_mask_rate": finite_float(output, "d5_dac_fix_mask_rate"),
                            "bit_risk_zero_prob_mean": finite_float(output, "bit_risk_zero_prob_mean"),
                            "bit_risk_dac_prob_mean": finite_float(output, "bit_risk_dac_prob_mean"),
                            "bit_risk_nc_prob_mean": finite_float(output, "bit_risk_nc_prob_mean"),
                            "bit_risk_ttc_prob_mean": finite_float(output, "bit_risk_ttc_prob_mean"),
                            "grad_norm": grad_norm_value,
                            "lr_bit": args.lr_bit,
                            "lr_action_head": args.lr_action_head,
                            "gpu_memory": current_gpu_memory(device),
                            "step_time": round(time.time() - step_start, 4),
                        }
                        last_loss = record["total_loss"]
                        log_fp.write(json.dumps(record, sort_keys=True) + "\n")
                        log_fp.flush()
                        if global_step % max(1, args.log_every) == 0:
                            print(
                                f"step={global_step} loss={record['total_loss']:.6f} "
                                f"diff={record['diffusion_loss']:.6f} bit_t={record['bit_terminal_loss']:.6f} "
                                f"bit_p={record['bit_path_loss']:.6f}"
                            )
                        metrics = {"step": global_step, "loss": record["total_loss"], "best_loss": best_loss}
                        if record["total_loss"] < best_loss:
                            best_loss = record["total_loss"]
                            metrics["best_loss"] = best_loss
                            save_checkpoint(args.output_dir / "best.ckpt", planner, optimizer, global_step, cfg, metrics)
                        if args.save_every and global_step % args.save_every == 0:
                            save_checkpoint(args.output_dir / "latest.ckpt", planner, optimizer, global_step, cfg, metrics)
                        if args.num_steps is not None and global_step >= args.num_steps:
                            stop = True
                            break
                    if stop:
                        break
                if stop:
                    break
            if stop or args.num_steps is None:
                break
        if pending > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(planner.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        final_metrics = {"step": global_step, "loss": last_loss, "best_loss": best_loss}
        save_checkpoint(args.output_dir / "latest.ckpt", planner, optimizer, global_step, cfg, final_metrics)
        if not (args.output_dir / "best.ckpt").is_file():
            save_checkpoint(args.output_dir / "best.ckpt", planner, optimizer, global_step, cfg, final_metrics)
        summary = {
            "output_dir": str(args.output_dir),
            "global_step": global_step,
            "best_loss": best_loss,
            "last_loss": last_loss,
            "latest_checkpoint": str(args.output_dir / "latest.ckpt"),
            "best_checkpoint": str(args.output_dir / "best.ckpt"),
            "failure_sampling_enabled": bool(failure_options["enabled"]),
        }
        write_final_report(args.output_dir / "final_report.md", summary)
    finally:
        log_fp.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
