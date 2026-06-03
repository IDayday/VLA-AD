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


TEACHER_KEYS = ("teacher_trajectory", "teacher_trajectory_norm", "teacher_score", "gt_score", "oracle_best_of_k_score", "candidate_count")
GEOMETRY_KEYS = ("vggt_geometry_tokens", "vggt_context_tokens", "vggt_geometry_mode", "vggt_geometry_mode_code")
BASE_KEYS = ("history_trajectory", "high_command_one_hot", "last_hidden_state", "status_feature", "trajectory")
TOKEN_KEYS = ("jepa_context_tokens", "jepa_target_tokens", "vggt_context_tokens", "vggt_target_tokens")
GEOMETRY_MODE_TO_CODE = {"missing": -1, "no_geometry": 0, "patch_fallback": 1, "full_geometry": 2}
GEOMETRY_CODE_TO_MODE = {-1: "missing", 0: "no_geometry", 1: "patch_fallback", 2: "full_geometry"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Last-VLA teacher trajectory readiness for chunk caches.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--chunk-name-pattern", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "reports" / "last_vla_cache_manifest.json")
    parser.add_argument("--val-cache-root", type=Path, default=None)
    parser.add_argument("--val-chunk-name-pattern", default=None)
    parser.add_argument("--strict-teacher-traj-sft", action="store_true")
    parser.add_argument("--min-teacher-coverage", type=float, default=0.99)
    parser.add_argument("--strict-full-geometry", action="store_true")
    parser.add_argument("--allow-patch-fallback", action="store_true")
    parser.add_argument("--min-full-geometry-coverage", type=float, default=0.99)
    parser.add_argument("--geometry-teacher-dim", type=int, default=512)
    return parser.parse_args()


def chunk_dirs(cache_root: Path, pattern: Optional[str]) -> List[Path]:
    if (cache_root / "index.jsonl").is_file():
        return [cache_root]
    if pattern:
        dirs = []
        seen = set()
        for item in str(pattern).split(","):
            item = item.strip()
            if not item:
                continue
            for path in sorted(cache_root.glob(item)):
                if path.is_dir() and (path / "index.jsonl").is_file() and path not in seen:
                    dirs.append(path)
                    seen.add(path)
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


def geometry_mode_from_sample(sample: Dict[str, Any]) -> str:
    raw_code = sample.get("vggt_geometry_mode_code")
    if isinstance(raw_code, torch.Tensor) and raw_code.numel() >= 1:
        return GEOMETRY_CODE_TO_MODE.get(int(raw_code.detach().cpu().view(-1)[0].item()), "missing")
    if raw_code is not None:
        try:
            return GEOMETRY_CODE_TO_MODE.get(int(raw_code), "missing")
        except (TypeError, ValueError):
            return "missing"
    raw = sample.get("vggt_geometry_mode")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str) and raw in GEOMETRY_MODE_TO_CODE:
        return raw
    if "vggt_geometry_tokens" in sample:
        return "missing"
    if "vggt_context_tokens" in sample:
        return "patch_fallback"
    return "no_geometry"


def audit_cache(cache_root: Path, *, chunk_name_pattern: Optional[str], max_samples: Optional[int]) -> Dict[str, Any]:
    counts = Counter()
    geometry_counts = Counter({mode: 0 for mode in GEOMETRY_MODE_TO_CODE})
    geometry_shape_counts = Counter()
    sample_token_counts = Counter()
    errors: List[str] = []
    deltas: List[float] = []
    scanned = 0
    for _, sample_path, record in iter_samples(cache_root, chunk_name_pattern, max_samples):
        scanned += 1
        try:
            sample = load_sample(sample_path)
        except Exception as exc:
            errors.append(f"{sample_path}:load:{type(exc).__name__}:{exc}")
            continue
        sample_token = str(sample.get("sample_token") or record.get("sample_token") or sample_path.stem)
        sample_token_counts[sample_token] += 1
        for key in (*BASE_KEYS, *TOKEN_KEYS):
            if key in sample:
                counts[key] += 1
        has_teacher = isinstance(sample.get("teacher_trajectory"), torch.Tensor) or isinstance(sample.get("teacher_trajectory_norm"), torch.Tensor)
        if has_teacher:
            counts["teacher_trajectory_any"] += 1
        for key in TEACHER_KEYS:
            if key in sample:
                counts[key] += 1
                value = sample[key]
                if key in {"teacher_trajectory", "teacher_trajectory_norm"} and (not isinstance(value, torch.Tensor) or tuple(value.shape) != (8, 3)):
                    errors.append(f"{sample_path}:{key}:bad_shape:{tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__}")
                if key not in {"teacher_trajectory", "teacher_trajectory_norm"} and isinstance(value, torch.Tensor) and not torch.isfinite(value.float()).all():
                    errors.append(f"{sample_path}:{key}:non_finite")
        oracle = sample.get("oracle_best_of_k_score")
        gt = sample.get("gt_score")
        if isinstance(oracle, torch.Tensor) and isinstance(gt, torch.Tensor):
            deltas.append(float((oracle.float().view(-1)[0] - gt.float().view(-1)[0]).item()))
        for key in GEOMETRY_KEYS:
            if key in sample:
                counts[key] += 1
        geometry_mode = geometry_mode_from_sample(sample)
        geometry_counts[geometry_mode] += 1
        geometry_tokens = sample.get("vggt_geometry_tokens")
        if isinstance(geometry_tokens, torch.Tensor):
            geometry_shape_counts[str(tuple(geometry_tokens.shape))] += 1
        if "vggt_geometry_tokens" in sample and geometry_mode == "patch_fallback":
            # Explicitly allowed, but it must not be reported as full geometry.
            pass
    duplicate_tokens = {token: count for token, count in sample_token_counts.items() if count > 1}
    delta_tensor = torch.tensor(deltas, dtype=torch.float32) if deltas else torch.tensor([], dtype=torch.float32)
    return {
        "cache_root": str(cache_root),
        "scanned": scanned,
        "base_field_coverage": {key: coverage(counts[key], scanned) for key in BASE_KEYS},
        "teacher_token_coverage": {key: coverage(counts[key], scanned) for key in TOKEN_KEYS},
        "coverage": {key: coverage(counts[key], scanned) for key in ("teacher_trajectory_any", *TEACHER_KEYS)},
        "geometry_coverage": {key: coverage(counts[key], scanned) for key in GEOMETRY_KEYS},
        "vggt_geometry_mode_distribution": dict(sorted(geometry_counts.items())),
        "geometry_token_shape_distribution": dict(sorted(geometry_shape_counts.items())),
        "duplicate_sample_tokens": {"count": len(duplicate_tokens), "examples": list(duplicate_tokens)[:20]},
        "oracle_delta_vs_gt": {
            "count": len(deltas),
            "mean": float(delta_tensor.mean().item()) if deltas else None,
            "min": float(delta_tensor.min().item()) if deltas else None,
            "max": float(delta_tensor.max().item()) if deltas else None,
        },
        "ready_for_teacher_traj_sft": (counts["teacher_trajectory_any"] / scanned) >= 0.99 if scanned else False,
        "errors": errors[:200],
        "num_errors": len(errors),
    }


def collect_tokens(cache_root: Path, pattern: Optional[str], max_samples: Optional[int]) -> set[str]:
    tokens: set[str] = set()
    for _, sample_path, record in iter_samples(cache_root, pattern, max_samples):
        try:
            sample = load_sample(sample_path)
            tokens.add(str(sample.get("sample_token") or record.get("sample_token") or sample_path.stem))
        except Exception:
            tokens.add(str(record.get("sample_token") or sample_path.stem))
    return tokens


def main() -> int:
    args = parse_args()
    report = audit_cache(args.cache_root, chunk_name_pattern=args.chunk_name_pattern, max_samples=args.max_samples)
    teacher_cov = report["coverage"]["teacher_trajectory_any"]["coverage"]
    full_geometry_cov = (
        report["vggt_geometry_mode_distribution"].get("full_geometry", 0) / report["scanned"]
        if report["scanned"]
        else 0.0
    )
    expected_shape = str((12, int(args.geometry_teacher_dim)))
    shape_mismatch = [
        shape for shape, count in report["geometry_token_shape_distribution"].items()
        if count > 0 and shape != expected_shape
    ]
    report["ready_for_teacher_traj_sft"] = bool(teacher_cov >= float(args.min_teacher_coverage) and report["num_errors"] == 0)
    report["strict_full_geometry"] = bool(args.strict_full_geometry)
    report["allow_patch_fallback"] = bool(args.allow_patch_fallback)
    report["geometry_teacher_dim"] = int(args.geometry_teacher_dim)
    report["full_geometry_coverage"] = float(full_geometry_cov)
    report["geometry_shape_mismatch"] = shape_mismatch
    report["ready_for_strict_full_geometry"] = bool(
        full_geometry_cov >= float(args.min_full_geometry_coverage)
        and report["num_errors"] == 0
        and not shape_mismatch
        and (args.allow_patch_fallback or report["vggt_geometry_mode_distribution"].get("patch_fallback", 0) == 0)
    )
    report["geometry_high_risk_warning"] = bool(
        args.allow_patch_fallback and report["vggt_geometry_mode_distribution"].get("patch_fallback", 0) > 0
    )
    if args.val_cache_root is not None:
        train_tokens = collect_tokens(args.cache_root, args.chunk_name_pattern, args.max_samples)
        val_tokens = collect_tokens(args.val_cache_root, args.val_chunk_name_pattern, args.max_samples)
        overlap = sorted(train_tokens & val_tokens)
        report["train_val_overlap"] = {"count": len(overlap), "examples": overlap[:20]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict_teacher_traj_sft and not report["ready_for_teacher_traj_sft"]:
        return 2
    if args.strict_full_geometry and not report["ready_for_strict_full_geometry"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
