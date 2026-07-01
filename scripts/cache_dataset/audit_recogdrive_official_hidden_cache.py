#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


FEATURE_FILE = "internvl_feature.gz"
TARGET_FILE = "trajectory_target.gz"
MANIFEST_NAME = ".recogdrive_official_stage1_hidden_cache.json"
OFFICIAL_FEATURE_KEYS = {
    "history_trajectory",
    "high_command_one_hot",
    "last_hidden_state",
    "status_feature",
}
BANNED_KEY_FRAGMENTS = (
    "jepa",
    "vggt",
    "last_rd",
    "last_vla",
    "teacher",
    "two_expert",
    "risk",
    "vlm_text",
)


def _load_gzip_pickle(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a dict, got {type(payload).__name__}.")
    return payload


def _load_torch(path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a dict, got {type(payload).__name__}.")
    return payload


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _indexed_dirs(root: Path) -> List[Path]:
    if (root / "index.jsonl").is_file():
        return [root]
    dirs: List[Path] = []
    if not root.is_dir():
        return dirs
    dirs.extend(child for child in sorted(root.iterdir()) if child.is_dir() and (child / "index.jsonl").is_file())
    shard_root = root / "shards"
    if shard_root.is_dir():
        dirs.extend(child for child in sorted(shard_root.iterdir()) if child.is_dir() and (child / "index.jsonl").is_file())
    return dirs


def _looks_like_indexed(root: Path) -> bool:
    return bool(_indexed_dirs(root))


def _official_samples(root: Path, max_samples: int) -> Iterable[Tuple[Path, Dict[str, Any], Dict[str, Any]]]:
    seen = 0
    for log_dir in sorted(root.iterdir()):
        if seen >= max_samples:
            break
        if not log_dir.is_dir() or log_dir.name == "shards":
            continue
        for token_dir in sorted(log_dir.iterdir()):
            if seen >= max_samples:
                break
            if not token_dir.is_dir():
                continue
            feature_path = token_dir / FEATURE_FILE
            target_path = token_dir / TARGET_FILE
            if not feature_path.is_file() or not target_path.is_file():
                continue
            seen += 1
            yield token_dir, _load_gzip_pickle(feature_path), _load_gzip_pickle(target_path)


def _indexed_samples(root: Path, max_samples: int) -> Iterable[Tuple[Path, Dict[str, Any], Dict[str, Any]]]:
    seen = 0
    for index_dir in _indexed_dirs(root):
        if seen >= max_samples:
            break
        for record in _iter_jsonl(index_dir / "index.jsonl"):
            if seen >= max_samples:
                break
            raw_path = str(record.get("path") or "")
            if not raw_path:
                continue
            sample_path = Path(raw_path)
            if not sample_path.is_absolute():
                sample_path = index_dir / sample_path
            if not sample_path.is_file():
                continue
            payload = _load_torch(sample_path)
            target = {"trajectory": payload.get("trajectory")}
            seen += 1
            yield sample_path, payload, target


def _shape(value: Any) -> Optional[Tuple[int, ...]]:
    return tuple(value.shape) if isinstance(value, torch.Tensor) else None


def _validate_feature_target(sample_path: Path, features: Dict[str, Any], targets: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    missing = sorted(key for key in OFFICIAL_FEATURE_KEYS if key not in features)
    if missing:
        errors.append(f"{sample_path}: missing feature keys {missing}")

    extra_banned = sorted(key for key in features if any(fragment in key for fragment in BANNED_KEY_FRAGMENTS))
    if extra_banned:
        errors.append(f"{sample_path}: contains external/augmented keys {extra_banned}")

    expected_shapes = {
        "history_trajectory": (4, 3),
        "status_feature": (8,),
    }
    for key, expected in expected_shapes.items():
        value = features.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            errors.append(f"{sample_path}: {key} shape {_shape(value)} != {expected}")

    command = features.get("high_command_one_hot")
    if not isinstance(command, torch.Tensor) or tuple(command.shape) not in {(3,), (4,)}:
        errors.append(f"{sample_path}: high_command_one_hot shape {_shape(command)} is not [3] or legacy [4]")
    elif tuple(command.shape) == (4,) and float(command[-1].abs().item()) > 1e-6:
        errors.append(f"{sample_path}: legacy high_command_one_hot[3] is not zero")

    hidden = features.get("last_hidden_state")
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 2 or int(hidden.shape[-1]) != 1536:
        errors.append(f"{sample_path}: last_hidden_state shape {_shape(hidden)} is not [N,1536]")

    trajectory = targets.get("trajectory")
    if not isinstance(trajectory, torch.Tensor) or tuple(trajectory.shape) != (8, 3):
        errors.append(f"{sample_path}: trajectory shape {_shape(trajectory)} != (8, 3)")

    return errors


def _read_manifest(root: Path) -> Dict[str, Any]:
    path = root / MANIFEST_NAME
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _manifest_proves_official(manifest: Dict[str, Any], expected_vlm_path: Optional[str]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if not manifest:
        return False, ["missing manifest"]
    if manifest.get("official_recogdrive_stage1_hidden_cache") is not True:
        reasons.append("manifest flag official_recogdrive_stage1_hidden_cache is not true")
    if manifest.get("external_knowledge_injection") not in {False, "false", "False", 0}:
        reasons.append("manifest does not declare external_knowledge_injection=false")
    if str(manifest.get("stage1_train_mode", "")).lower() not in {"official", "frozen", "none"}:
        reasons.append(f"manifest stage1_train_mode={manifest.get('stage1_train_mode')!r}")
    if str(manifest.get("status", "")).lower() != "completed":
        reasons.append(f"manifest status={manifest.get('status')!r}")
    if expected_vlm_path and str(Path(str(manifest.get("vlm_path", "")))) != str(Path(expected_vlm_path)):
        reasons.append(f"manifest vlm_path={manifest.get('vlm_path')!r} != {expected_vlm_path!r}")
    return not reasons, reasons


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.cache_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Cache path does not exist: {root}")

    manifest = _read_manifest(root)
    manifest_official, manifest_reasons = _manifest_proves_official(manifest, args.expected_vlm_path)
    layout = "indexed" if _looks_like_indexed(root) else "official_gzip"
    iterator = _indexed_samples(root, args.max_samples) if layout == "indexed" else _official_samples(root, args.max_samples)

    errors: List[str] = []
    sample_reports: List[Dict[str, Any]] = []
    checked = 0
    for sample_path, features, targets in iterator:
        checked += 1
        errors.extend(_validate_feature_target(sample_path, features, targets))
        report = {
            "path": str(sample_path),
            "feature_keys": sorted(features.keys()),
            "last_hidden_state_shape": _shape(features.get("last_hidden_state")),
            "last_hidden_state_dtype": str(getattr(features.get("last_hidden_state"), "dtype", "")),
            "trajectory_shape": _shape(targets.get("trajectory")),
        }
        metadata = features.get("two_expert_metadata")
        if isinstance(metadata, dict):
            report["two_expert_metadata"] = {
                "stage1_train_mode": metadata.get("stage1_train_mode"),
                "loaded_lora_adapter": metadata.get("loaded_lora_adapter"),
                "vlm_checkpoint": metadata.get("vlm_checkpoint"),
                "source_stage1_checkpoint": metadata.get("source_stage1_checkpoint"),
            }
            if metadata.get("loaded_lora_adapter") or metadata.get("stage1_train_mode") not in {None, "frozen"}:
                errors.append(f"{sample_path}: indexed cache was produced with non-official stage1 metadata {report['two_expert_metadata']}")
        sample_reports.append(report)

    if checked == 0:
        errors.append(f"No samples found under {root} with layout={layout}")
    if checked < int(args.require_min_samples):
        errors.append(f"Only checked/found {checked} samples, below --require-min-samples={args.require_min_samples}")
    if args.require_official and not manifest_official:
        errors.append("Official source is not proven by manifest: " + "; ".join(manifest_reasons))

    return {
        "cache_path": str(root),
        "layout": layout,
        "checked_samples": checked,
        "max_samples": int(args.max_samples),
        "schema_ok": checked > 0 and not [e for e in errors if "Official source is not proven" not in e],
        "official_source_proven": manifest_official,
        "manifest": manifest,
        "manifest_reasons": manifest_reasons,
        "sample_reports": sample_reports[: int(args.report_samples)],
        "errors": errors,
        "ok": not errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a ReCogDrive official stage1 VLM hidden cache.")
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--expected-vlm-path", default="")
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--report-samples", type=int, default=3)
    parser.add_argument("--require-min-samples", type=int, default=1)
    parser.add_argument("--require-official", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = audit(args)
    except Exception as exc:
        summary = {"ok": False, "errors": [f"{type(exc).__name__}: {exc}"]}
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
