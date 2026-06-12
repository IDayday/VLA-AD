#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import load_sample, write_json  # noqa: E402
from scripts.last_vla_v2.two_expert_slot.two_expert_cache_utils import iter_indexed_records  # noqa: E402


def iter_cache_records(root: Optional[Path]) -> Iterable[Dict[str, Any]]:
    if root is None:
        return
    if not root.exists():
        raise FileNotFoundError(root)
    for _, sample_path, _ in iter_indexed_records(root):
        yield load_sample(sample_path)


def _finite_tensor(value: Any, key: str, errors: List[str]) -> Optional[torch.Tensor]:
    if not isinstance(value, torch.Tensor):
        errors.append(f"{key}:not_tensor")
        return None
    value = value.detach().float().cpu()
    if not torch.isfinite(value).all():
        errors.append(f"{key}:non_finite")
    return value


def audit_records(
    records: Iterable[Dict[str, Any]],
    *,
    expected_kind: str,
    require_strict: bool = False,
) -> Dict[str, Any]:
    total = 0
    failed = 0
    strict_count = 0
    fallback_count = 0
    duplicate_tokens: List[str] = []
    seen_tokens = set()
    errors: List[Dict[str, Any]] = []
    for record in records:
        total += 1
        token = str(record.get("sample_token") or f"record_{total:08d}")
        if token in seen_tokens:
            duplicate_tokens.append(token)
        seen_tokens.add(token)
        record_errors: List[str] = []
        if expected_kind == "jepa":
            value = _finite_tensor(record.get("jepa_dynamic_teacher_tokens"), "jepa_dynamic_teacher_tokens", record_errors)
            if value is not None and tuple(value.shape) != (3, 12, 1024):
                record_errors.append(f"jepa_dynamic_teacher_tokens:bad_shape:{tuple(value.shape)}")
            metadata = record.get("jepa_dynamic_teacher_metadata", {})
            strict_flag = bool(isinstance(metadata, dict) and metadata.get("strict_dynamic_teacher", False))
        elif expected_kind == "vggt":
            value = _finite_tensor(record.get("vggt_feature23_tokens"), "vggt_feature23_tokens", record_errors)
            if value is not None and (value.ndim != 2 or value.shape[0] != 12):
                record_errors.append(f"vggt_feature23_tokens:bad_shape:{tuple(value.shape)}")
            metadata = record.get("vggt_feature23_metadata", {})
            strict_flag = bool(isinstance(metadata, dict) and metadata.get("strict_geometry_teacher", False))
        else:
            raise ValueError(f"Unknown expected_kind={expected_kind!r}")
        strict_count += int(strict_flag)
        fallback_count += int(not strict_flag)
        if require_strict and not strict_flag:
            record_errors.append("strict_teacher_required_but_fallback_record")
        if record_errors:
            failed += 1
            if len(errors) < 20:
                errors.append({"sample_token": token, "errors": record_errors})
    return {
        "total_records": total,
        "failed_records": failed,
        "ok": failed == 0 and not duplicate_tokens,
        "strict_records": strict_count,
        "fallback_records": fallback_count,
        "duplicate_sample_tokens": duplicate_tokens[:20],
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit two_expert_slot JEPA/VGGT teacher cache schema.")
    parser.add_argument("--cache-root", type=Path, default=None, help="Cache containing both teacher keys.")
    parser.add_argument("--jepa-cache-root", type=Path, default=None)
    parser.add_argument("--vggt-cache-root", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--min-coverage", type=float, default=0.99)
    parser.add_argument("--expected-total", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    jepa_root = args.jepa_cache_root or args.cache_root
    vggt_root = args.vggt_cache_root or args.cache_root
    if jepa_root is None and vggt_root is None:
        raise ValueError("Provide --cache-root or at least one of --jepa-cache-root/--vggt-cache-root.")
    report: Dict[str, Any] = {}
    if jepa_root is not None:
        report["jepa"] = audit_records(iter_cache_records(jepa_root), expected_kind="jepa", require_strict=args.strict)
    if vggt_root is not None:
        report["vggt"] = audit_records(iter_cache_records(vggt_root), expected_kind="vggt", require_strict=args.strict)
    expected_total = int(args.expected_total) if args.expected_total is not None else None
    coverage_ok = True
    if expected_total and expected_total > 0:
        for key, item in report.items():
            item["coverage"] = float(item["total_records"]) / float(expected_total)
            if item["coverage"] < float(args.min_coverage):
                coverage_ok = False
                item["coverage_error"] = f"{item['coverage']:.4f} < {float(args.min_coverage):.4f}"
    ok = coverage_ok and all(bool(item.get("ok", False)) for item in report.values())
    report["ok"] = bool(ok)
    if args.json_out is not None:
        write_json(args.json_out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
