#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import iter_index, load_sample  # noqa: E402
from navsim.planning.script.run_training_recogdrive import ChunkCacheDataset, custom_collate_fn  # noqa: E402


REQUIRED_FEATURE_KEYS = (
    "history_trajectory",
    "high_command_one_hot",
    "last_hidden_state",
    "status_feature",
)
REQUIRED_TARGET_KEYS = ("trajectory",)
TRAIN_ONLY_TARGET_KEYS = ("jepa_target_tokens", "vggt_target_tokens")
EXPERT_CONTEXT_KEYS = ("jepa_context_tokens", "vggt_context_tokens")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check the official-aligned local chunk-cache loader without training.")
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path(os.getenv("CACHE_PATH") or os.getenv("TRAIN_CHUNK_CACHE_ROOT") or "cache/recogdrive_expert_chunks/full_v1"),
        help="Local chunk cache root or one chunk directory.",
    )
    parser.add_argument("--val-cache-path", type=Path, default=None, help="Optional separate validation chunk cache root.")
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("reports/official_aligned_local_loader_smoke.json"))
    parser.add_argument("--build-agent", action="store_true", help="Instantiate ReCogDriveAgent only; does not run training.")
    parser.add_argument("--official-log-split", action="store_true", help="Filter local chunks with official train_logs/val_logs.")
    return parser.parse_args()


def resolve_sample_path(chunk_dir: Path, record: Dict[str, Any]) -> Path:
    path = Path(record["path"])
    if path.is_file():
        return path
    if not path.is_absolute():
        candidate = chunk_dir / path
        if candidate.is_file():
            return candidate
    return path


def chunk_dirs(cache_path: Path, include_navtest: bool = False) -> List[Path]:
    if (cache_path / "index.jsonl").is_file():
        return [cache_path]
    return sorted(
        child for child in cache_path.iterdir()
        if child.is_dir()
        and (child / "index.jsonl").is_file()
        and (include_navtest or not child.name.startswith("navtest"))
    )


def read_records(cache_path: Path, include_navtest: bool = False) -> List[Tuple[Path, Dict[str, Any]]]:
    records: List[Tuple[Path, Dict[str, Any]]] = []
    for chunk_dir in chunk_dirs(cache_path, include_navtest=include_navtest):
        for record in iter_index(chunk_dir):
            records.append((chunk_dir, record))
    return records


def select_indices(total: int, count: int) -> List[int]:
    if total <= 0:
        return []
    count = min(count, total)
    if count == 1:
        return [0]
    return sorted({round(i * (total - 1) / (count - 1)) for i in range(count)})


def tensor_summary(tensor: torch.Tensor) -> Dict[str, Any]:
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype).replace("torch.", "")}


def summarize_batch(features: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, Any]:
    tensor_shapes: Dict[str, Any] = {}
    tensor_dtypes: Dict[str, Any] = {}
    for prefix, mapping in (("features", features), ("targets", targets)):
        for key, value in mapping.items():
            if isinstance(value, torch.Tensor):
                name = f"{prefix}.{key}"
                tensor_shapes[name] = list(value.shape)
                tensor_dtypes[name] = str(value.dtype).replace("torch.", "")
    return {"tensor_shapes": tensor_shapes, "tensor_dtypes": tensor_dtypes}


def command_index(sample: Dict[str, Any]) -> str:
    value = sample.get("high_command_one_hot")
    if isinstance(value, torch.Tensor) and value.numel() > 0:
        return str(int(torch.argmax(value.float()).item()))
    return "missing"


def token_set(records: Iterable[Tuple[Path, Dict[str, Any]]]) -> Counter:
    counter: Counter = Counter()
    for _, record in records:
        token = str(record.get("sample_token") or Path(str(record.get("path", ""))).stem)
        counter[token] += 1
    return counter


def maybe_build_agent() -> Dict[str, Any]:
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent

    agent = ReCogDriveAgent(
        trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
        vlm_path=None,
        cache_hidden_state=True,
        dit_type="small",
        vlm_size="small",
        sampling_method="ddim",
        lr=1e-4,
        grpo=False,
        use_expert_features=False,
        use_jepa=False,
        use_vggt=False,
    )
    return {
        "class": agent.__class__.__name__,
        "cache_hidden_state": bool(agent.cache_hidden_state),
        "use_expert_features": bool(agent.use_expert_features),
        "param_dtype_counts": dict(Counter(str(p.dtype).replace("torch.", "") for p in agent.parameters())),
    }


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    train_log_names = val_log_names = None
    if args.official_log_split:
        import yaml
        split_yaml = REPO_ROOT / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
        split_data = yaml.safe_load(split_yaml.read_text(encoding="utf-8")) or {}
        train_log_names = [str(item) for item in split_data.get("train_logs", [])]
        val_log_names = [str(item) for item in split_data.get("val_logs", [])]

    dataset = ChunkCacheDataset(str(args.cache_path), log_names=train_log_names, split_name="train" if args.official_log_split else "all")
    val_dataset = (
        ChunkCacheDataset(str(args.cache_path), log_names=val_log_names, split_name="val")
        if args.official_log_split
        else None
    )
    indices = select_indices(len(dataset), args.sample_count)
    samples = [dataset[i] for i in indices]
    sample_payloads: List[Dict[str, Any]] = []
    last_hidden_lengths: List[int] = []
    missing_keys: Counter = Counter()
    sample_key_rows: List[Dict[str, Any]] = []
    command_counts: Counter = Counter()
    log_counts: Counter = Counter()

    raw_records = [(chunk_dir, record) for chunk_dir, _, record in dataset.records]
    record_by_token: Dict[str, Dict[str, Any]] = {
        str(record.get("sample_token") or Path(str(record.get("path", ""))).stem): record
        for _, record in raw_records
    }

    for features, targets, token in samples:
        token = str(token)
        sample_path = None
        record = record_by_token.get(token)
        if record is not None:
            for chunk_dir, candidate in raw_records:
                if str(candidate.get("sample_token") or Path(str(candidate.get("path", ""))).stem) == token:
                    sample_path = resolve_sample_path(chunk_dir, candidate)
                    break
        raw_sample = load_sample(sample_path) if sample_path is not None and sample_path.is_file() else {}
        sample_payloads.append(raw_sample)
        for key in REQUIRED_FEATURE_KEYS:
            if key not in features:
                missing_keys[f"features.{key}"] += 1
        for key in REQUIRED_TARGET_KEYS:
            if key not in targets:
                missing_keys[f"targets.{key}"] += 1
        if isinstance(features.get("last_hidden_state"), torch.Tensor):
            last_hidden_lengths.append(int(features["last_hidden_state"].shape[0]))
        command_counts[command_index(raw_sample or features)] += 1
        if record is not None:
            log_counts[str(record.get("log_name") or "missing")] += 1
        sample_key_rows.append({
            "token": token,
            "sample_keys": sorted(raw_sample.keys()),
            "feature_keys": sorted(features.keys()),
            "target_keys": sorted(targets.keys()),
            "shapes": {
                key: list(value.shape)
                for key, value in {**features, **targets}.items()
                if isinstance(value, torch.Tensor)
            },
            "dtypes": {
                key: str(value.dtype).replace("torch.", "")
                for key, value in {**features, **targets}.items()
                if isinstance(value, torch.Tensor)
            },
        })

    batch = samples[: max(1, min(args.batch_size, len(samples)))]
    collated_features, collated_targets, collated_tokens = custom_collate_fn(batch)
    batch_summary = summarize_batch(collated_features, collated_targets)

    train_tokens = token_set(raw_records)
    duplicate_train_tokens = sum(count - 1 for count in train_tokens.values() if count > 1)
    val_overlap: Optional[int] = None
    val_unique_count: Optional[int] = None
    duplicate_val_tokens: Optional[int] = None
    if val_dataset is not None:
        val_records = [(chunk_dir, record) for chunk_dir, _, record in val_dataset.records]
        val_tokens = token_set(val_records)
        val_overlap = len(set(train_tokens) & set(val_tokens))
        val_unique_count = len(val_tokens)
        duplicate_val_tokens = sum(count - 1 for count in val_tokens.values() if count > 1)
    elif args.val_cache_path is not None:
        val_records = read_records(args.val_cache_path, include_navtest=True)
        val_tokens = token_set(val_records)
        val_overlap = len(set(train_tokens) & set(val_tokens))
        val_unique_count = len(val_tokens)
        duplicate_val_tokens = sum(count - 1 for count in val_tokens.values() if count > 1)

    length_stats = {
        "min": min(last_hidden_lengths) if last_hidden_lengths else None,
        "mean": sum(last_hidden_lengths) / len(last_hidden_lengths) if last_hidden_lengths else None,
        "max": max(last_hidden_lengths) if last_hidden_lengths else None,
    }
    smoke = {
        "cache_path": str(args.cache_path),
        "official_log_split": bool(args.official_log_split),
        "batch_size": len(batch),
        "sample_count": len(samples),
        "dataset_len": len(dataset),
        "train_dataset_report": dataset.report(),
        "val_dataset_report": val_dataset.report() if val_dataset is not None else None,
        "feature_keys": sorted(collated_features.keys()),
        "target_keys": sorted(collated_targets.keys()),
        **batch_summary,
        "tokens_list_type": type(collated_tokens).__name__,
        "tokens_list_len": len(collated_tokens),
        "last_hidden_state_lengths": last_hidden_lengths,
        "last_hidden_state_length_stats": length_stats,
        "missing_keys": dict(missing_keys),
        "train_only_target_keys_in_collated_features": [key for key in TRAIN_ONLY_TARGET_KEYS if key in collated_features],
        "expert_context_keys_in_collated_features": [key for key in EXPERT_CONTEXT_KEYS if key in collated_features],
        "duplicate_tokens_in_sample": len(collated_tokens) - len(set(map(str, collated_tokens))),
        "train_unique_sample_token_count": len(train_tokens),
        "train_duplicate_sample_token_count": duplicate_train_tokens,
        "train_val_overlap_if_available": val_overlap,
        "val_unique_sample_token_count_if_available": val_unique_count,
        "val_duplicate_sample_token_count_if_available": duplicate_val_tokens,
        "sample_key_rows": sample_key_rows,
        "sample_log_name_distribution": dict(log_counts),
        "sample_high_command_distribution": dict(command_counts),
    }
    if args.build_agent:
        smoke["agent"] = maybe_build_agent()

    args.output.write_text(json.dumps(smoke, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    print(json.dumps({
        "dataset_len": smoke["dataset_len"],
        "feature_keys": smoke["feature_keys"],
        "target_keys": smoke["target_keys"],
        "missing_keys": smoke["missing_keys"],
        "train_only_target_keys_in_collated_features": smoke["train_only_target_keys_in_collated_features"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
