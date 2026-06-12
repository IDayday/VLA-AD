#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return payload


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def validate_teacher(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if payload.get("ok") is not True:
        errors.append("teacher_preflight:not_ok")
    if payload.get("teacher_status") != "strict_production_teacher":
        errors.append(f"teacher_preflight:bad_status:{payload.get('teacher_status')}")
    if int(payload.get("num_jepa_legacy_fallback", -1)) != 0:
        errors.append(f"teacher_preflight:jepa_fallback:{payload.get('num_jepa_legacy_fallback')}")
    if int(payload.get("num_vggt_legacy_fallback", -1)) != 0:
        errors.append(f"teacher_preflight:vggt_fallback:{payload.get('num_vggt_legacy_fallback')}")
    return errors


def validate_internvl(payload: Dict[str, Any], threshold: float) -> List[str]:
    errors: List[str] = []
    if payload.get("ok") is not True:
        errors.append("internvl_smoke:not_ok")
    if payload.get("image_hidden_nonempty") is not True:
        errors.append("internvl_smoke:image_hidden_empty")
    if float(payload.get("image_sensitivity_delta", 0.0)) <= float(threshold):
        errors.append(f"internvl_smoke:image_delta:{payload.get('image_sensitivity_delta')}")
    if float(payload.get("slot_sensitivity_delta", 0.0)) <= float(threshold):
        errors.append(f"internvl_smoke:slot_delta:{payload.get('slot_sensitivity_delta')}")
    return errors


def validate_vggt(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if payload.get("ok") is not True:
        errors.append("vggt_smoke:not_ok")
    packed_shape = payload.get("packed_shape")
    if not isinstance(packed_shape, list) or not packed_shape or int(packed_shape[0]) != 12:
        errors.append(f"vggt_smoke:bad_packed_shape:{packed_shape}")
    if int(payload.get("feature_dim", 0)) <= 0:
        errors.append(f"vggt_smoke:bad_feature_dim:{payload.get('feature_dim')}")
    return errors


def validate_coverage(payload: Dict[str, Any], min_coverage: float) -> List[str]:
    errors: List[str] = []
    if payload.get("ok") is not True:
        errors.append("coverage_preflight:not_ok")
    coverage = float(payload.get("all_teacher_coverage", 0.0))
    if coverage < float(min_coverage):
        errors.append(f"coverage_preflight:low_all_teacher_coverage:{coverage:.6f}<{float(min_coverage):.6f}")
    return errors


def write_markdown(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# Two-Expert Final Readiness Gate",
        "",
        f"- Status: `{report['status']}`",
        f"- OK: `{report['ok']}`",
        f"- Threshold: `{report['threshold']}`",
        f"- Coverage OK: `{report.get('coverage_ok')}`",
        f"- All-teacher coverage: `{report.get('all_teacher_coverage')}`",
        f"- Min coverage: `{report.get('min_coverage')}`",
        "",
        "Errors:",
    ]
    if report["errors"]:
        lines.extend(f"- `{item}`" for item in report["errors"])
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final gate before two_expert_slot full training.")
    parser.add_argument("--teacher-preflight-json", type=Path, required=True)
    parser.add_argument("--internvl-smoke-json", type=Path, required=True)
    parser.add_argument("--vggt-smoke-json", type=Path, required=True)
    parser.add_argument("--coverage-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=1e-6)
    parser.add_argument("--min-coverage", type=float, default=0.99)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    teacher = load_json(args.teacher_preflight_json)
    internvl = load_json(args.internvl_smoke_json)
    vggt = load_json(args.vggt_smoke_json)
    coverage = load_json(args.coverage_json) if args.coverage_json is not None else None
    errors = []
    errors.extend(validate_teacher(teacher))
    errors.extend(validate_internvl(internvl, float(args.threshold)))
    errors.extend(validate_vggt(vggt))
    if coverage is not None:
        errors.extend(validate_coverage(coverage, float(args.min_coverage)))
    report = {
        "ok": not errors,
        "status": "READY" if not errors else "NOT_READY",
        "threshold": float(args.threshold),
        "min_coverage": float(args.min_coverage),
        "teacher_preflight_json": str(args.teacher_preflight_json),
        "internvl_smoke_json": str(args.internvl_smoke_json),
        "vggt_smoke_json": str(args.vggt_smoke_json),
        "coverage_json": str(args.coverage_json) if args.coverage_json is not None else None,
        "coverage_ok": bool(coverage and coverage.get("ok") is True and float(coverage.get("all_teacher_coverage", 0.0)) >= float(args.min_coverage)),
        "all_teacher_coverage": float(coverage.get("all_teacher_coverage", 0.0)) if coverage is not None else None,
        "errors": errors,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "readiness_gate.json", report)
    write_markdown(args.output_dir / "readiness_gate.md", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
