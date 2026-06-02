#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.bit_safety_selector import SELECTOR_FEATURE_NAMES


HISTORY_FEATURE_NAMES = [
    "history_last_x",
    "history_last_y",
    "history_last_heading",
    "history_delta_x",
    "history_delta_y",
    "history_delta_heading",
    "history_step_length_mean",
    "history_step_length_max",
    "history_lateral_abs_max",
    "history_curvature_proxy",
]
STATUS_FEATURE_NAMES = [f"status_feature_{idx:02d}" for idx in range(8)]
COMMAND_FEATURE_NAMES = [f"high_command_{idx:02d}" for idx in range(4)]
SCENE_FEATURE_NAMES = STATUS_FEATURE_NAMES + COMMAND_FEATURE_NAMES + HISTORY_FEATURE_NAMES


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def load_metadata(input_jsonl: Path) -> Dict[str, Any]:
    metadata_path = input_jsonl.parent / "metadata.json"
    if metadata_path.is_file():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}


def chunk_dirs(root: Path, patterns: Sequence[str]) -> List[Path]:
    dirs: List[Path] = []
    for pattern in patterns:
        matches = sorted(path for path in root.glob(pattern) if path.is_dir())
        dirs.extend(matches)
    unique: List[Path] = []
    seen = set()
    for path in dirs:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def build_sample_index(root: Path, patterns: Sequence[str]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for chunk in chunk_dirs(root, patterns):
        index_path = chunk / "index.jsonl"
        if not index_path.is_file():
            continue
        for record in read_jsonl(index_path):
            token = record.get("sample_token")
            rel_path = record.get("path")
            if not token or not rel_path:
                continue
            index.setdefault(str(token), chunk / str(rel_path))
    return index


def tensor_to_float_list(value: Any, length: int) -> List[float]:
    if value is None:
        return [0.0] * length
    tensor = value.detach().float().cpu().flatten() if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.float32).flatten()
    out = tensor[:length].tolist()
    if len(out) < length:
        out.extend([0.0] * (length - len(out)))
    return [float(v) for v in out]


def curvature_proxy(history: torch.Tensor) -> float:
    if history.ndim != 2 or history.shape[0] < 3:
        return 0.0
    deltas = history[1:, :2] - history[:-1, :2]
    headings = torch.atan2(deltas[:, 1], deltas[:, 0])
    if headings.numel() < 2:
        return 0.0
    return float(torch.mean(torch.abs(headings[1:] - headings[:-1])).item())


def history_features(value: Any) -> Dict[str, float]:
    if value is None:
        return {name: 0.0 for name in HISTORY_FEATURE_NAMES}
    history = value.detach().float().cpu() if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.float32)
    if history.ndim != 2 or history.shape[-1] < 3 or history.shape[0] == 0:
        return {name: 0.0 for name in HISTORY_FEATURE_NAMES}
    first = history[0]
    last = history[-1]
    if history.shape[0] >= 2:
        steps = history[1:, :2] - history[:-1, :2]
        step_lengths = torch.linalg.norm(steps, dim=-1)
        step_mean = float(step_lengths.mean().item())
        step_max = float(step_lengths.max().item())
    else:
        step_mean = 0.0
        step_max = 0.0
    return {
        "history_last_x": float(last[0].item()),
        "history_last_y": float(last[1].item()),
        "history_last_heading": float(last[2].item()),
        "history_delta_x": float((last[0] - first[0]).item()),
        "history_delta_y": float((last[1] - first[1]).item()),
        "history_delta_heading": float((last[2] - first[2]).item()),
        "history_step_length_mean": step_mean,
        "history_step_length_max": step_max,
        "history_lateral_abs_max": float(torch.abs(history[:, 1]).max().item()),
        "history_curvature_proxy": curvature_proxy(history),
    }


def random_projection(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(input_dim, output_dim, generator=generator, dtype=torch.float32) / math.sqrt(float(input_dim))


def vlm_projection_features(sample: Dict[str, Any], projection: Optional[torch.Tensor]) -> Dict[str, float]:
    if projection is None:
        return {}
    hidden = sample.get("last_hidden_state")
    if hidden is None:
        return {f"vlm_mean_rp_{idx:03d}": 0.0 for idx in range(projection.shape[1])}
    tensor = hidden.detach().float().cpu() if isinstance(hidden, torch.Tensor) else torch.tensor(hidden, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[-1] != projection.shape[0]:
        return {f"vlm_mean_rp_{idx:03d}": 0.0 for idx in range(projection.shape[1])}
    projected = tensor.mean(dim=0).matmul(projection)
    return {f"vlm_mean_rp_{idx:03d}": float(projected[idx].item()) for idx in range(projected.shape[0])}


def scene_features(sample: Dict[str, Any], projection: Optional[torch.Tensor]) -> Dict[str, float]:
    features: Dict[str, float] = {}
    for idx, value in enumerate(tensor_to_float_list(sample.get("status_feature"), 8)):
        features[f"status_feature_{idx:02d}"] = value
    for idx, value in enumerate(tensor_to_float_list(sample.get("high_command_one_hot"), 4)):
        features[f"high_command_{idx:02d}"] = value
    features.update(history_features(sample.get("history_trajectory")))
    features.update(vlm_projection_features(sample, projection))
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment counterfactual rows with metric-free cache scene features.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-cache-root", type=Path, default=Path("/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1"))
    parser.add_argument("--chunk-name-pattern", action="append", default=None, help="Can be set multiple times. Defaults to train/navtest full chunks.")
    parser.add_argument("--vlm-rp-dim", type=int, default=0, help="Add deterministic random-projection features from VLM hidden-state mean.")
    parser.add_argument("--vlm-rp-seed", type=int, default=20260602)
    parser.add_argument("--copy-prediction-files", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    patterns = args.chunk_name_pattern or ["train_full_chunk_*", "navtest_full_chunk_*", "navval_full_chunk_*"]
    sample_index = build_sample_index(args.chunk_cache_root, patterns)
    if args.vlm_rp_dim < 0:
        raise ValueError("--vlm-rp-dim must be non-negative.")
    projection = random_projection(1536, args.vlm_rp_dim, args.vlm_rp_seed) if args.vlm_rp_dim > 0 else None
    vlm_names = [f"vlm_mean_rp_{idx:03d}" for idx in range(args.vlm_rp_dim)]
    feature_names = list(SELECTOR_FEATURE_NAMES) + SCENE_FEATURE_NAMES + vlm_names

    rows = read_jsonl(args.input_jsonl)
    missing = 0
    augmented: List[Dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        features = dict(row.get("features") or {})
        sample_path = sample_index.get(str(row.get("sample_token")))
        if sample_path is None or not sample_path.is_file():
            missing += 1
            features.update({name: 0.0 for name in SCENE_FEATURE_NAMES + vlm_names})
        else:
            sample = torch.load(sample_path, map_location="cpu")
            features.update(scene_features(sample, projection))
        row["features"] = features
        augmented.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "counterfactual_samples.jsonl", augmented)
    (args.output_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2) + "\n", encoding="utf-8")

    source_metadata = load_metadata(args.input_jsonl)
    metadata = {
        **source_metadata,
        "source_counterfactual_jsonl": str(args.input_jsonl),
        "scene_feature_augmentation": {
            "chunk_cache_root": str(args.chunk_cache_root),
            "chunk_name_pattern": patterns,
            "sample_index_count": len(sample_index),
            "missing_sample_count": missing,
            "vlm_rp_dim": args.vlm_rp_dim,
            "vlm_rp_seed": args.vlm_rp_seed,
            "feature_count": len(feature_names),
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name in ("label_distribution.json", "counterfactual_summary.md"):
        source = args.input_jsonl.parent / name
        if source.is_file():
            shutil.copy2(source, args.output_dir / name)
    if args.copy_prediction_files:
        for name in ("base_predictions.jsonl", "bit_predictions.jsonl"):
            source = args.input_jsonl.parent / name
            if source.is_file():
                shutil.copy2(source, args.output_dir / name)

    lines = [
        "# Counterfactual Scene Feature Augmentation",
        "",
        f"Source: `{args.input_jsonl}`",
        f"Rows: `{len(rows)}`",
        f"Sample index entries: `{len(sample_index)}`",
        f"Missing samples: `{missing}`",
        f"VLM random projection dim: `{args.vlm_rp_dim}`",
        f"Feature count: `{len(feature_names)}`",
        "",
        "Added metric-free ego status, high-level command, history, and optional current-frame VLM context projection features.",
    ]
    (args.output_dir / "scene_feature_augmentation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(rows), "missing": missing, "feature_count": len(feature_names)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
