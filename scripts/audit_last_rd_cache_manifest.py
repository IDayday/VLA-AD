#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import iter_index, load_sample


BASE_KEYS = ("history_trajectory", "high_command_one_hot", "last_hidden_state", "status_feature", "trajectory")
TOKEN_KEYS = (
    "jepa_context_tokens",
    "jepa_target_tokens",
    "vggt_context_tokens",
    "vggt_target_tokens",
    "vggt_geometry_tokens",
    "vggt_geometry_target_tokens",
    "vggt_depth_tokens",
    "vggt_pointmap_tokens",
    "vggt_camera_tokens",
)
RISK_KEYS = ("risk_labels", "generic_risk_labels", "drivable_risk_labels", "ttc_risk_labels", "comfort_risk_labels")
GEOMETRY_MODES = ("full_geometry", "patch_fallback", "no_geometry", "missing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a LaST-RD chunk cache without modifying it.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--chunk-name-pattern", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "reports" / "last_rd_cache_manifest.json")
    parser.add_argument("--future-jepa-loss-weight", type=float, default=0.0)
    parser.add_argument("--require-vggt-geometry", action="store_true")
    parser.add_argument("--risk-loss-weight", type=float, default=0.0)
    return parser.parse_args()


def chunk_dirs(cache_root: Path, pattern: Optional[str]) -> List[Path]:
    if (cache_root / "index.jsonl").is_file():
        return [cache_root]
    if pattern:
        dirs = sorted(path for path in cache_root.glob(pattern) if path.is_dir() and (path / "index.jsonl").is_file())
    else:
        dirs = sorted(path for path in cache_root.iterdir() if path.is_dir() and (path / "index.jsonl").is_file())
    if not dirs:
        raise FileNotFoundError(f"No chunk cache directories with index.jsonl found under {cache_root}.")
    return dirs


def resolve_sample_path(chunk_dir: Path, record: Dict[str, Any]) -> Path:
    raw = Path(record["path"])
    if raw.is_file():
        return raw
    if not raw.is_absolute():
        candidate = chunk_dir / raw
        if candidate.is_file():
            return candidate
    return raw


def iter_samples(cache_root: Path, pattern: Optional[str], max_samples: Optional[int]) -> Iterable[Tuple[Path, Path, Dict[str, Any]]]:
    count = 0
    for chunk_dir in chunk_dirs(cache_root, pattern):
        for record in iter_index(chunk_dir):
            yield chunk_dir, resolve_sample_path(chunk_dir, record), record
            count += 1
            if max_samples is not None and count >= max_samples:
                return


def coverage(count: int, total: int) -> Dict[str, Any]:
    return {"count": int(count), "coverage": float(count / total) if total else 0.0}


def tensor_error(key: str, value: Any) -> Optional[str]:
    if not isinstance(value, torch.Tensor):
        return f"{key}:not_tensor:{type(value).__name__}"
    if not torch.isfinite(value.float()).all():
        return f"{key}:non_finite"
    return None


def mode_from_sample(sample: Dict[str, Any]) -> str:
    raw = sample.get("vggt_geometry_mode")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str) and raw in GEOMETRY_MODES:
        return raw
    if "vggt_geometry_tokens" in sample:
        return "missing"
    if "vggt_context_tokens" in sample:
        return "patch_fallback"
    return "no_geometry"


def audit_cache(
    cache_root: Path,
    *,
    chunk_name_pattern: Optional[str] = None,
    max_samples: Optional[int] = None,
    future_jepa_loss_weight: float = 0.0,
    require_vggt_geometry: bool = False,
    risk_loss_weight: float = 0.0,
) -> Dict[str, Any]:
    base_counts = Counter()
    token_counts = Counter()
    risk_counts = Counter()
    geometry_modes = Counter({mode: 0 for mode in GEOMETRY_MODES})
    high_command_shapes = Counter()
    last_hidden_lengths: List[int] = []
    sample_tokens: List[str] = []
    errors: List[str] = []
    scanned = 0

    for _, sample_path, record in iter_samples(cache_root, chunk_name_pattern, max_samples):
        scanned += 1
        try:
            sample = load_sample(sample_path)
        except Exception as exc:
            errors.append(f"{sample_path}:load:{type(exc).__name__}:{exc}")
            continue
        sample_token = str(sample.get("sample_token") or record.get("sample_token") or sample_path.stem)
        sample_tokens.append(sample_token)

        for key in BASE_KEYS:
            if key in sample:
                base_counts[key] += 1
                err = tensor_error(key, sample[key])
                if err:
                    errors.append(f"{sample_path}:{err}")
            if key == "high_command_one_hot" and key in sample and isinstance(sample[key], torch.Tensor):
                high_command_shapes[str(tuple(sample[key].shape))] += 1
            if key == "last_hidden_state" and key in sample and isinstance(sample[key], torch.Tensor) and sample[key].ndim >= 1:
                last_hidden_lengths.append(int(sample[key].shape[0]))

        for key in TOKEN_KEYS:
            if key in sample:
                token_counts[key] += 1
                err = tensor_error(key, sample[key])
                if err:
                    errors.append(f"{sample_path}:{err}")

        for key in RISK_KEYS:
            if key in sample:
                risk_counts[key] += 1
                err = tensor_error(key, sample[key])
                if err:
                    errors.append(f"{sample_path}:{err}")

        geometry_modes[mode_from_sample(sample)] += 1

    token_dupes = Counter(sample_tokens)
    warnings: List[Dict[str, str]] = []
    if future_jepa_loss_weight > 0 and scanned and token_counts["jepa_target_tokens"] / scanned < 0.99:
        warnings.append({
            "level": "high_risk",
            "message": "future_jepa_loss_weight > 0 but jepa_target_tokens coverage is below 99%.",
        })
    if require_vggt_geometry and scanned and geometry_modes["full_geometry"] / scanned < 0.99:
        warnings.append({
            "level": "blocker",
            "message": "require_vggt_geometry=true but full_geometry coverage is below 99%.",
        })
    if risk_loss_weight > 0 and sum(risk_counts.values()) == 0:
        warnings.append({
            "level": "high_risk",
            "message": "risk_loss_weight > 0 but risk label coverage is zero.",
        })

    length_stats = {
        "min": min(last_hidden_lengths) if last_hidden_lengths else None,
        "mean": sum(last_hidden_lengths) / len(last_hidden_lengths) if last_hidden_lengths else None,
        "max": max(last_hidden_lengths) if last_hidden_lengths else None,
    }
    return {
        "cache_root": str(cache_root),
        "chunk_name_pattern": chunk_name_pattern,
        "num_samples_scanned": scanned,
        "required_base_key_coverage": {key: coverage(base_counts[key], scanned) for key in BASE_KEYS},
        "high_command_one_hot_shape_distribution": dict(sorted(high_command_shapes.items())),
        "last_hidden_state_length": length_stats,
        "teacher_token_coverage": {key: coverage(token_counts[key], scanned) for key in TOKEN_KEYS},
        "vggt_geometry_mode_distribution": dict(sorted(geometry_modes.items())),
        "risk_label_coverage": {key: coverage(risk_counts[key], scanned) for key in RISK_KEYS},
        "shape_dtype_finiteness_errors": errors,
        "num_shape_dtype_finiteness_errors": len(errors),
        "sample_token_duplicates": {
            "num_duplicate_tokens": sum(count - 1 for count in token_dupes.values() if count > 1),
            "examples": [token for token, count in token_dupes.items() if count > 1][:20],
        },
        "warnings": warnings,
        "pass": not any(item["level"] == "blocker" for item in warnings) and not errors,
    }


def main() -> int:
    args = parse_args()
    manifest = audit_cache(
        args.cache_root,
        chunk_name_pattern=args.chunk_name_pattern,
        max_samples=args.max_samples,
        future_jepa_loss_weight=args.future_jepa_loss_weight,
        require_vggt_geometry=args.require_vggt_geometry,
        risk_loss_weight=args.risk_loss_weight,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return 0 if manifest["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
