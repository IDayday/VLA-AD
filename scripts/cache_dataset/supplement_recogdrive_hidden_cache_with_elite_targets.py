#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch


FEATURE_FILE = "internvl_feature.gz"
TARGET_FILE = "trajectory_target.gz"
MANIFEST_NAME = ".recogdrive_official_stage1_hidden_cache.json"


def _dump_gzip_pickle(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=1) as f:
        pickle.dump(payload, f)
    os.replace(tmp, path)


def _load_gzip_pickle(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a dict, got {type(payload).__name__}.")
    return payload


def _load_elite_index(path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Elite target index must be a dict, got {type(payload).__name__}: {path}")
    tokens = payload.get("tokens")
    trajectories = payload.get("trajectories")
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise TypeError("Elite target index field 'tokens' must be list[str].")
    if not isinstance(trajectories, torch.Tensor) or trajectories.ndim != 3 or tuple(trajectories.shape[1:]) != (8, 3):
        raise ValueError("Elite target index field 'trajectories' must have shape [N,8,3].")
    if len(tokens) != int(trajectories.shape[0]):
        raise ValueError(f"tokens count {len(tokens)} != trajectories count {trajectories.shape[0]}")
    if len(set(tokens)) != len(tokens):
        raise ValueError("Elite target index contains duplicate tokens.")
    index = {str(token): trajectories[i].detach().float().cpu().contiguous() for i, token in enumerate(tokens)}
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    return index, {
        "path": str(path),
        "num_tokens": len(tokens),
        "summary": summary,
        "require_above_gt": payload.get("require_above_gt"),
        "selection_mode": payload.get("selection_mode"),
        "elite_buffer_root": payload.get("elite_buffer_root"),
        "source_cache_root": payload.get("cache_root"),
    }


def _iter_official_cache_records(root: Path) -> Iterable[Tuple[str, Path, Path, Path]]:
    for log_dir in sorted(root.iterdir()):
        if not log_dir.is_dir() or log_dir.name == "shards":
            continue
        for token_dir in sorted(log_dir.iterdir()):
            if not token_dir.is_dir():
                continue
            feature_path = token_dir / FEATURE_FILE
            target_path = token_dir / TARGET_FILE
            if feature_path.is_file() and target_path.is_file():
                yield token_dir.name, log_dir, feature_path, target_path


def _link_or_copy(src: Path, dst: Path, *, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported link mode: {mode}")


def _validate_base_target_shape(target_path: Path) -> None:
    payload = _load_gzip_pickle(target_path)
    trajectory = payload.get("trajectory")
    if not isinstance(trajectory, torch.Tensor) or tuple(trajectory.shape) != (8, 3):
        raise ValueError(f"{target_path} trajectory must have shape [8,3], got {getattr(trajectory, 'shape', None)}")


def build_overlay(args: argparse.Namespace) -> Dict[str, Any]:
    base = Path(args.base_cache_path)
    output = Path(args.output_cache_path)
    elite_path = Path(args.elite_target_index_path)
    if not base.is_dir():
        raise FileNotFoundError(f"Base cache does not exist: {base}")
    if not elite_path.is_file():
        raise FileNotFoundError(f"Elite target index does not exist: {elite_path}")
    if output.resolve() == base.resolve():
        raise ValueError("output-cache-path must differ from base-cache-path; do not overwrite the official hidden cache.")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output cache is non-empty: {output}. Pass --overwrite to update it.")

    elite_index, elite_metadata = _load_elite_index(elite_path)
    output.mkdir(parents=True, exist_ok=True)

    total = elite = fallback = skipped_existing = 0
    sample_reports: List[Dict[str, Any]] = []
    for token, log_dir, feature_path, target_path in _iter_official_cache_records(base):
        total += 1
        rel_log = log_dir.name
        out_token_dir = output / rel_log / token
        out_feature = out_token_dir / FEATURE_FILE
        out_target = out_token_dir / TARGET_FILE

        feature_preexists = out_feature.exists() or out_feature.is_symlink()
        target_preexists = out_target.exists() or out_target.is_symlink()
        if feature_preexists and target_preexists and not args.overwrite:
            skipped_existing += 1
            continue

        _link_or_copy(feature_path, out_feature, mode=args.feature_link_mode, overwrite=args.overwrite)
        if token in elite_index:
            if out_target.exists() or out_target.is_symlink():
                if args.overwrite:
                    out_target.unlink()
                else:
                    skipped_existing += 1
                    continue
            _dump_gzip_pickle(out_target, {"trajectory": elite_index[token].clone()})
            elite += 1
        else:
            if args.validate_fallback_targets and fallback < int(args.validate_fallback_targets):
                _validate_base_target_shape(target_path)
            _link_or_copy(target_path, out_target, mode=args.target_link_mode, overwrite=args.overwrite)
            fallback += 1

        if len(sample_reports) < int(args.report_samples):
            sample_reports.append(
                {
                    "token": token,
                    "log_name": rel_log,
                    "target_source": "elite_best_valid_above_gt" if token in elite_index else "gt_fallback",
                    "feature_path": str(out_feature),
                    "target_path": str(out_target),
                }
            )

        if args.max_records and total >= int(args.max_records):
            break

    if total == 0:
        raise FileNotFoundError(f"No official cache records with {FEATURE_FILE} and {TARGET_FILE} found under {base}")

    base_manifest_path = base / MANIFEST_NAME
    base_manifest: Dict[str, Any] = {}
    if base_manifest_path.is_file():
        with base_manifest_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            base_manifest = loaded

    manifest = {
        "official_recogdrive_stage1_hidden_cache": True,
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_cache_path": str(base),
        "output_cache_path": str(output),
        "vlm_path": args.expected_vlm_path or base_manifest.get("vlm_path", ""),
        "stage1_train_mode": "official",
        "external_knowledge_injection": False,
        "target_supplement": "elite_best_valid_above_gt_or_gt",
        "target_supplement_note": "VLM hidden features are linked from the official ReCogDrive stage1 cache; only trajectory_target.gz is replaced for tokens whose elite target is strictly above GT.",
        "feature_file": FEATURE_FILE,
        "target_file": TARGET_FILE,
        "num_records": total,
        "num_elite_targets": elite,
        "num_gt_fallback_targets": fallback,
        "num_skipped_existing": skipped_existing,
        "elite_target_index": elite_metadata,
        "base_manifest": base_manifest,
        "feature_link_mode": args.feature_link_mode,
        "target_link_mode": args.target_link_mode,
    }
    (output / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    return {
        "ok": True,
        "base_cache_path": str(base),
        "output_cache_path": str(output),
        "num_records_seen": total,
        "num_elite_targets_written": elite,
        "num_gt_fallback_targets_linked": fallback,
        "num_skipped_existing": skipped_existing,
        "elite_target_coverage_in_output": elite / max(1, elite + fallback),
        "elite_index": elite_metadata,
        "sample_reports": sample_reports,
        "manifest_path": str(output / MANIFEST_NAME),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an official ReCogDrive hidden-cache overlay with elite trajectory targets.")
    parser.add_argument("--base-cache-path", required=True)
    parser.add_argument("--output-cache-path", required=True)
    parser.add_argument("--elite-target-index-path", required=True)
    parser.add_argument("--expected-vlm-path", default="")
    parser.add_argument("--feature-link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--target-link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--report-samples", type=int, default=5)
    parser.add_argument("--validate-fallback-targets", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = build_overlay(args)
    except Exception as exc:
        summary = {"ok": False, "errors": [f"{type(exc).__name__}: {exc}"]}
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
