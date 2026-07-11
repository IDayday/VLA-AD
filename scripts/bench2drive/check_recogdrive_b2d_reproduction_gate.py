#!/usr/bin/env python3
"""Audit the provenance and contracts required for a ReCogDrive B2D reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence
from xml.etree import ElementTree as ET


class ReproductionGateError(RuntimeError):
    """Raised when a reproduction artifact is malformed or cannot be audited."""


READY_CONTRACT_STATUSES = {"ready_exact", "ready_public_proxy"}
CAMERA_ORDER = (
    "rgb_front",
    "rgb_front_left",
    "rgb_front_right",
    "rgb_back_left",
    "rgb_back_right",
    "rgb_back",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/bench2drive_recogdrive_reproduction_gate.json"),
    )
    parser.add_argument(
        "--target",
        choices=("artifacts", "stage1", "stage2", "stage3", "evaluation"),
        default="artifacts",
        help="Also enforce the selected stage contract after auditing artifacts.",
    )
    parser.add_argument("--traj-jsonl", type=Path, default=None)
    parser.add_argument("--qa-jsonl", type=Path, default=None)
    parser.add_argument("--raw-data-root", type=Path, default=None)
    parser.add_argument("--base-vlm-model", type=Path, default=None)
    parser.add_argument("--bench2drive-root", type=Path, default=None)
    parser.add_argument("--route-xml", type=Path, default=None)
    parser.add_argument("--carla-root", type=Path, default=None)
    parser.add_argument(
        "--full-hash",
        action="store_true",
        help="Hash the two JSONL files and 4 GB VLM weight in addition to size/schema checks.",
    )
    parser.add_argument(
        "--verify-all-images",
        action="store_true",
        help="Stat every image referenced by both official JSONL files; this is intentionally slow.",
    )
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def load_manifest(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReproductionGateError(f"cannot load manifest {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ReproductionGateError(f"unsupported reproduction manifest: {path}")
    return value


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path(spec: Mapping[str, Any], override: Optional[Path]) -> Path:
    return (override or Path(str(spec["local_path"]))).expanduser().resolve()


def _add(
    checks: list[Dict[str, Any]],
    name: str,
    ok: bool,
    detail: Any,
    *,
    category: str = "artifact",
) -> None:
    checks.append(
        {
            "name": name,
            "ok": bool(ok),
            "category": category,
            "detail": detail if isinstance(detail, str) else json.dumps(detail, sort_keys=True),
        }
    )


def _check_file(
    checks: list[Dict[str, Any]],
    name: str,
    path: Path,
    spec: Mapping[str, Any],
    *,
    full_hash: bool,
) -> bool:
    if not path.is_file():
        _add(checks, name, False, f"missing: {path}")
        return False
    expected_size = spec.get("size_bytes")
    actual_size = path.stat().st_size
    size_ok = expected_size is None or actual_size == int(expected_size)
    _add(
        checks,
        f"{name}.size",
        size_ok,
        {"path": str(path), "actual": actual_size, "expected": expected_size},
    )
    if full_hash and spec.get("sha256"):
        actual_hash = sha256_file(path)
        _add(
            checks,
            f"{name}.sha256",
            actual_hash == spec["sha256"],
            {"actual": actual_hash, "expected": spec["sha256"]},
        )
    return size_ok


def normalize_official_image_path(value: str) -> Path:
    normalized = value.replace("\\", "/")
    prefix = "./Bench2drive/v1/"
    if normalized.startswith(prefix):
        normalized = normalized[len(prefix) :]
    elif normalized.startswith("Bench2drive/v1/"):
        normalized = normalized[len("Bench2drive/v1/") :]
    else:
        raise ReproductionGateError(f"unexpected official image prefix: {value!r}")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ReproductionGateError(f"unsafe official image path: {value!r}")
    return Path(*pure.parts)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ReproductionGateError(f"{path}:{line_number} is not a JSON object")
                yield value
    except (OSError, json.JSONDecodeError) as exc:
        raise ReproductionGateError(f"cannot scan {path}: {exc}") from exc


def audit_official_jsonl(
    path: Path,
    raw_root: Path,
    expected: Mapping[str, Any],
    *,
    verify_all_images: bool,
    image_directory_cache: Optional[Dict[Path, set[str]]] = None,
) -> Dict[str, Any]:
    rows = 0
    clips: set[str] = set()
    images_per_record: Counter[int] = Counter()
    conversation_lengths: Counter[int] = Counter()
    roles: Counter[str] = Counter()
    cameras: set[str] = set()
    missing_images: list[str] = []
    checked_images = 0

    for record in _iter_jsonl(path):
        rows += 1
        images = record.get("image")
        conversations = record.get("conversations")
        if not isinstance(images, list) or not all(isinstance(item, str) for item in images):
            raise ReproductionGateError(f"row {rows} in {path} has an invalid image list")
        if not isinstance(conversations, list) or not all(isinstance(item, dict) for item in conversations):
            raise ReproductionGateError(f"row {rows} in {path} has invalid conversations")
        images_per_record[len(images)] += 1
        conversation_lengths[len(conversations)] += 1
        roles.update(str(item.get("from")) for item in conversations)

        relative_images = [normalize_official_image_path(item) for item in images]
        if not relative_images:
            raise ReproductionGateError(f"row {rows} in {path} has no images")
        clips.add(relative_images[0].parts[0])
        cameras.update(item.parts[-2] for item in relative_images)

        should_check = verify_all_images or rows == 1
        if should_check:
            for relative in relative_images:
                checked_images += 1
                candidate = raw_root / relative
                if image_directory_cache is None:
                    exists = candidate.is_file()
                else:
                    directory = candidate.parent
                    if directory not in image_directory_cache:
                        try:
                            image_directory_cache[directory] = set(os.listdir(directory))
                        except OSError:
                            image_directory_cache[directory] = set()
                    exists = candidate.name in image_directory_cache[directory]
                if not exists and len(missing_images) < 20:
                    missing_images.append(str(candidate))

    if rows == 0:
        raise ReproductionGateError(f"JSONL contains no records: {path}")

    summary = {
        "rows": rows,
        "clips": len(clips),
        "clip_names": clips,
        "images_per_record": dict(images_per_record),
        "conversation_min": min(conversation_lengths),
        "conversation_max": max(conversation_lengths),
        "roles": dict(roles),
        "cameras": sorted(cameras),
        "checked_images": checked_images,
        "cached_image_directories": len(image_directory_cache or {}),
        "missing_images": missing_images,
    }
    expected_images = int(expected["expected_images_per_record"])
    expected_roles = {str(key): int(value) for key, value in expected["expected_role_counts"].items()}
    summary["matches_expected"] = all(
        (
            rows == int(expected["expected_rows"]),
            len(clips) == int(expected["expected_clips"]),
            images_per_record == Counter({expected_images: rows}),
            summary["conversation_min"] == int(expected["expected_conversation_min"]),
            summary["conversation_max"] == int(expected["expected_conversation_max"]),
            dict(roles) == expected_roles,
            not missing_images,
        )
    )
    return summary


def _git_value(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReproductionGateError(f"cannot inspect git checkout {root}: {exc}") from exc
    return result.stdout.strip()


def _public_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "clip_names"}


def run_gate(args: argparse.Namespace) -> Dict[str, Any]:
    manifest = load_manifest(args.manifest)
    artifacts = manifest["artifacts"]
    checks: list[Dict[str, Any]] = []
    warnings: list[str] = []

    traj_spec = artifacts["traj_jsonl"]
    qa_spec = artifacts["qa_jsonl"]
    raw_spec = artifacts["raw_bench2drive_base"]
    vlm_spec = artifacts["base_vlm_model"]
    b2d_spec = artifacts["bench2drive_checkout"]
    route_spec = artifacts["route_xml"]
    carla_spec = artifacts["carla"]

    traj_path = _artifact_path(traj_spec, args.traj_jsonl)
    qa_path = _artifact_path(qa_spec, args.qa_jsonl)
    raw_root = _artifact_path(raw_spec, args.raw_data_root)
    vlm_path = _artifact_path(vlm_spec, args.base_vlm_model)
    b2d_root = _artifact_path(b2d_spec, args.bench2drive_root)
    route_path = _artifact_path(route_spec, args.route_xml)
    carla_root = _artifact_path(carla_spec, args.carla_root)

    traj_ok = _check_file(checks, "traj_jsonl", traj_path, traj_spec, full_hash=args.full_hash)
    qa_ok = _check_file(checks, "qa_jsonl", qa_path, qa_spec, full_hash=args.full_hash)
    vlm_ok = _check_file(checks, "base_vlm_model", vlm_path, vlm_spec, full_hash=args.full_hash)
    route_ok = _check_file(checks, "route_xml", route_path, route_spec, full_hash=args.full_hash)

    ignored = set(raw_spec.get("ignored_directories", []))
    if raw_root.is_dir():
        raw_clips = {path.name for path in raw_root.iterdir() if path.is_dir() and path.name not in ignored}
        _add(
            checks,
            "raw_bench2drive_base.clip_count",
            len(raw_clips) == int(raw_spec["expected_clips"]),
            {"path": str(raw_root), "actual": len(raw_clips), "expected": raw_spec["expected_clips"]},
        )
    else:
        raw_clips = set()
        _add(checks, "raw_bench2drive_base", False, f"missing: {raw_root}")

    traj_summary: Dict[str, Any] = {}
    qa_summary: Dict[str, Any] = {}
    image_directory_cache: Optional[Dict[Path, set[str]]] = {} if args.verify_all_images else None
    if traj_ok and raw_root.is_dir():
        traj_summary = audit_official_jsonl(
            traj_path,
            raw_root,
            traj_spec,
            verify_all_images=args.verify_all_images,
            image_directory_cache=image_directory_cache,
        )
        _add(checks, "traj_jsonl.schema", traj_summary["matches_expected"], _public_summary(traj_summary))
    if qa_ok and raw_root.is_dir():
        qa_summary = audit_official_jsonl(
            qa_path,
            raw_root,
            qa_spec,
            verify_all_images=args.verify_all_images,
            image_directory_cache=image_directory_cache,
        )
        _add(checks, "qa_jsonl.schema", qa_summary["matches_expected"], _public_summary(qa_summary))

    if traj_summary and qa_summary and raw_clips:
        traj_clips = set(traj_summary["clip_names"])
        qa_clips = set(qa_summary["clip_names"])
        relationship_ok = (
            traj_clips < qa_clips
            and qa_clips == raw_clips
            and len(qa_clips - traj_clips) == 50
        )
        _add(
            checks,
            "official_clip_relationship",
            relationship_ok,
            {
                "traj_clips": len(traj_clips),
                "qa_clips": len(qa_clips),
                "raw_clips": len(raw_clips),
                "trajectory_complement": len(qa_clips - traj_clips),
            },
        )
        expected_cameras = set(raw_spec["expected_cameras"])
        _add(
            checks,
            "official_multiview_cameras",
            set(traj_summary["cameras"]) == expected_cameras
            and set(qa_summary["cameras"]) == expected_cameras,
            {
                "traj": traj_summary["cameras"],
                "qa": qa_summary["cameras"],
                "expected": sorted(expected_cameras),
            },
        )

    config_path = Path(str(vlm_spec["config_path"])).expanduser().resolve()
    if vlm_ok and config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        hidden_size = config.get("llm_config", {}).get("hidden_size")
        image_size = config.get("force_image_size")
        _add(
            checks,
            "base_vlm_model.config",
            hidden_size == int(vlm_spec["expected_hidden_size"])
            and image_size == int(vlm_spec["expected_image_size"]),
            {"path": str(config_path), "hidden_size": hidden_size, "image_size": image_size},
        )
    else:
        _add(checks, "base_vlm_model.config", False, f"missing: {config_path}")

    if b2d_root.is_dir():
        head = _git_value(b2d_root, "rev-parse", "HEAD")
        dirty = bool(_git_value(b2d_root, "status", "--porcelain"))
        _add(
            checks,
            "bench2drive_checkout",
            head == b2d_spec["expected_commit"]
            and (not b2d_spec.get("require_clean", False) or not dirty),
            {
                "path": str(b2d_root),
                "head": head,
                "expected": b2d_spec["expected_commit"],
                "dirty": dirty,
            },
        )
    else:
        _add(checks, "bench2drive_checkout", False, f"missing: {b2d_root}")

    if route_ok:
        try:
            route_count = len(ET.parse(route_path).getroot().findall(".//route"))
        except (OSError, ET.ParseError) as exc:
            raise ReproductionGateError(f"cannot parse route XML {route_path}: {exc}") from exc
        _add(
            checks,
            "route_xml.route_count",
            route_count == int(route_spec["expected_routes"]),
            {"actual": route_count, "expected": route_spec["expected_routes"]},
        )

    version_file = Path(str(carla_spec["version_file"])).expanduser().resolve()
    if carla_root.is_dir() and version_file.is_file():
        version = version_file.read_text(encoding="utf-8").strip()
        _add(
            checks,
            "carla.version",
            version == carla_spec["expected_version"],
            {"path": str(carla_root), "actual": version, "expected": carla_spec["expected_version"]},
        )
    else:
        _add(checks, "carla.version", False, f"missing CARLA root/version: {carla_root}, {version_file}")

    contracts = manifest["stage_contracts"]
    contract: Optional[Mapping[str, Any]] = None
    if args.target != "artifacts":
        contract = contracts[args.target]
        contract_status = str(contract["status"])
        contract_ok = contract_status in READY_CONTRACT_STATUSES
        _add(
            checks,
            f"{args.target}.contract",
            contract_ok,
            {
                "status": contract_status,
                "classification": contract.get("classification"),
                "blocking_unknowns": contract.get("blocking_unknowns", []),
                "required_validation": contract.get("required_validation", []),
            },
            category="contract",
        )

    warnings.extend(
        [
            "The pinned Bench2Drive checkout is a clean current public proxy; the commit used for ReCogDrive Table 2 is unknown.",
            "A passing Stage1 gate means closest-public fidelity, not access to the private author Bench2Drive code.",
            "The public Stage3 reward is NAVSIM PDMS; no Bench2Drive or Pareto reward is admitted into the current baseline.",
        ]
    )
    ok = all(item["ok"] for item in checks)
    blocked_by_contract = any(not item["ok"] and item["category"] == "contract" for item in checks)
    result = {
        "ok": ok,
        "target": args.target,
        "classification": contract.get("classification") if contract else "artifact provenance audit",
        "manifest": str(args.manifest.resolve()),
        "full_hash": bool(args.full_hash),
        "verify_all_images": bool(args.verify_all_images),
        "blocked_by_contract": blocked_by_contract,
        "checks": checks,
        "warnings": warnings,
    }
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = run_gate(args)
    except (ReproductionGateError, OSError, ValueError, KeyError) as exc:
        print(f"reproduction gate failed: {exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if result["ok"]:
        return 0
    return 2 if result["blocked_by_contract"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
