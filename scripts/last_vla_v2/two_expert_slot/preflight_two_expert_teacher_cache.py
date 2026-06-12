#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import load_sample, write_json  # noqa: E402
from scripts.last_vla_v2.two_expert_slot.two_expert_cache_utils import iter_indexed_records  # noqa: E402


def load_cache(root: Path, max_records: Optional[int]) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for _, path, record in iter_indexed_records(root, max_records=max_records):
        payload = load_sample(path)
        token = str(payload.get("sample_token") or record.get("sample_token") or path.stem)
        if token in mapping:
            raise ValueError(f"Duplicate sample_token {token} under {root}")
        mapping[token] = payload
    return mapping


def _is_finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor.detach().float()).all().item())


def _metadata_strict(payload: Dict[str, Any], key: str) -> bool:
    metadata = payload.get(key)
    return bool(metadata.get("strict_dynamic_teacher" if "jepa" in key else "strict_geometry_teacher", False)) if isinstance(metadata, dict) else False


def preflight(jepa_root: Path, vggt_root: Path, *, strict: bool, max_records: Optional[int] = None) -> Dict[str, Any]:
    jepa = load_cache(jepa_root, max_records)
    vggt = load_cache(vggt_root, max_records)
    intersection = sorted(set(jepa).intersection(vggt))
    errors: List[str] = []
    feature_dims = set()
    fallback_jepa = fallback_vggt = 0
    strict_jepa = strict_vggt = 0

    if not intersection:
        errors.append("empty_sample_token_intersection")
    for token in intersection:
        jepa_payload = jepa[token]
        vggt_payload = vggt[token]
        jt = jepa_payload.get("jepa_dynamic_teacher_tokens")
        vt = vggt_payload.get("vggt_feature23_tokens")
        if not isinstance(jt, torch.Tensor) or tuple(jt.shape) != (3, 12, 1024):
            errors.append(f"{token}:bad_jepa_shape:{tuple(jt.shape) if isinstance(jt, torch.Tensor) else type(jt).__name__}")
        elif not _is_finite(jt):
            errors.append(f"{token}:jepa_nonfinite")
        if not isinstance(vt, torch.Tensor) or vt.ndim != 2 or vt.shape[0] != 12:
            errors.append(f"{token}:bad_vggt_shape:{tuple(vt.shape) if isinstance(vt, torch.Tensor) else type(vt).__name__}")
        elif not _is_finite(vt):
            errors.append(f"{token}:vggt_nonfinite")
        else:
            feature_dims.add(int(vt.shape[-1]))
        j_strict = _metadata_strict(jepa_payload, "jepa_dynamic_teacher_metadata")
        v_strict = _metadata_strict(vggt_payload, "vggt_feature23_metadata")
        strict_jepa += int(j_strict)
        strict_vggt += int(v_strict)
        fallback_jepa += int(not j_strict)
        fallback_vggt += int(not v_strict)

    if len(feature_dims) > 1:
        errors.append(f"inconsistent_vggt_feature_dims:{sorted(feature_dims)}")
    if strict:
        if fallback_jepa:
            errors.append(f"strict_mode_jepa_fallback_count:{fallback_jepa}")
        if fallback_vggt:
            errors.append(f"strict_mode_vggt_fallback_count:{fallback_vggt}")

    teacher_status = "production_teacher_ok" if strict and not errors else "dev_fallback_not_for_final"
    return {
        "ok": not errors,
        "teacher_status": teacher_status,
        "strict_requested": bool(strict),
        "jepa_cache_root": str(jepa_root),
        "vggt_cache_root": str(vggt_root),
        "jepa_records": len(jepa),
        "vggt_records": len(vggt),
        "sample_token_intersection": len(intersection),
        "vggt_feature_dims": sorted(feature_dims),
        "num_jepa_strict": int(strict_jepa),
        "num_vggt_strict": int(strict_vggt),
        "num_jepa_legacy_fallback": int(fallback_jepa),
        "num_vggt_legacy_fallback": int(fallback_vggt),
        "errors": errors,
    }


def write_markdown(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# Two-Expert Teacher Cache Preflight",
        "",
        f"- OK: `{report['ok']}`",
        f"- Status: `{report['teacher_status']}`",
        f"- Strict requested: `{report['strict_requested']}`",
        f"- Intersection: `{report['sample_token_intersection']}`",
        f"- VGGT feature dims: `{report['vggt_feature_dims']}`",
        f"- JEPA fallback count: `{report['num_jepa_legacy_fallback']}`",
        f"- VGGT fallback count: `{report['num_vggt_legacy_fallback']}`",
        "",
        "Errors:",
    ]
    lines.extend(f"- `{item}`" for item in report["errors"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight two_expert_slot JEPA/VGGT teacher caches.")
    parser.add_argument("--jepa-cache-root", type=Path, required=True)
    parser.add_argument("--vggt-cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = preflight(args.jepa_cache_root, args.vggt_cache_root, strict=bool(args.strict), max_records=args.max_records)
    write_json(args.output_dir / "readiness.json", report)
    write_markdown(args.output_dir / "readiness.md", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
