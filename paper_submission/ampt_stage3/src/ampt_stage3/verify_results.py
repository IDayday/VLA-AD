"""Verify NAVSIM v1/v2 outputs against the archived paper protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from .checkpoint import inspect_checkpoint, sha256_artifact


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("expected"), Mapping):
        raise TypeError(f"invalid evaluation protocol: {path.name}")
    return payload


def _valid_mask(frame: pd.DataFrame) -> pd.Series:
    if "valid" not in frame:
        raise KeyError("NAVSIM v1 CSV is missing valid")
    values = frame["valid"]
    if values.dtype == bool:
        return values
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise ValueError("NAVSIM v1 valid column is not boolean")
    return normalized.isin({"true", "1"})


def verify_v1(csv_path: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    frame = pd.read_csv(csv_path)
    if "token" not in frame:
        raise KeyError("NAVSIM v1 CSV is missing token")
    frame = frame[frame["token"].astype(str).str.lower() != "average"].copy()
    if frame["token"].astype(str).duplicated().any():
        raise ValueError("NAVSIM v1 CSV contains duplicate scene tokens")
    valid = frame[_valid_mask(frame)]
    expected_count = int(protocol["expected_scored_scenes"])
    if len(valid) != expected_count:
        raise ValueError(f"NAVSIM v1 expected {expected_count} valid scenes, found {len(valid)}")
    actual = {}
    for name in protocol["expected"]:
        if name not in valid:
            raise KeyError(f"NAVSIM v1 CSV is missing {name}")
        values = pd.to_numeric(valid[name], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"NAVSIM v1 metric {name} contains NaN/Inf")
        actual[name] = float(values.mean())
    return _comparison(actual, protocol)


def verify_v2(summary_path: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise TypeError("NAVSIM v2 summary must be a JSON mapping")
    for key, expected in (
        ("num_predictions", protocol["expected_predictions"]),
        ("successful", protocol["expected_scored_scenes"]),
        ("failed", 0),
        ("extended_comfort_available", protocol["extended_comfort_available"]),
    ):
        if int(summary.get(key, -1)) != int(expected):
            raise ValueError(f"NAVSIM v2 {key}: expected {expected}, found {summary.get(key)}")
    revision = str(summary.get("official_navsim_revision", ""))
    if revision != str(protocol["official_scorer_revision"]):
        raise ValueError("NAVSIM v2 official scorer revision does not match the sealed protocol")
    actual = {name: float(summary[name]) for name in protocol["expected"]}
    if not all(np.isfinite(value) for value in actual.values()):
        raise ValueError("NAVSIM v2 summary contains NaN/Inf")
    return _comparison(actual, protocol)


def _comparison(actual: Mapping[str, float], protocol: Mapping[str, Any]) -> dict[str, Any]:
    tolerance = float(protocol["absolute_tolerance"])
    expected = {name: float(value) for name, value in protocol["expected"].items()}
    differences = {name: actual[name] - expected[name] for name in expected}
    passed = all(abs(value) <= tolerance for value in differences.values())
    return {
        "benchmark": protocol["benchmark"],
        "scenes": int(protocol["expected_scored_scenes"]),
        "actual": dict(actual),
        "expected": expected,
        "absolute_difference": differences,
        "absolute_tolerance": tolerance,
        "passed": passed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--v2-artifact", type=Path, required=True)
    parser.add_argument("--expected-v1-sha256", default="")
    parser.add_argument("--expected-v2-sha256", default="")
    parser.add_argument("--v1-csv", type=Path, required=True)
    parser.add_argument("--v2-summary", type=Path, required=True)
    parser.add_argument("--v1-protocol", type=Path, required=True)
    parser.add_argument("--v2-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    v1_checkpoint = inspect_checkpoint(
        args.v1_checkpoint, expected_sha256=args.expected_v1_sha256
    )
    v2_digest = sha256_artifact(args.v2_artifact)
    if args.expected_v2_sha256 and v2_digest != args.expected_v2_sha256.lower():
        raise ValueError("v2 evaluation artifact SHA-256 mismatch")
    v1_path, v2_path = args.v1_csv.resolve(), args.v2_summary.resolve()
    v1 = verify_v1(v1_path, _config(args.v1_protocol.resolve()))
    v2 = verify_v2(v2_path, _config(args.v2_protocol.resolve()))
    payload = {
        "schema_version": 1,
        "evaluation_artifacts": {
            "navsim_v1": {
                "artifact_id": v1_checkpoint.artifact_id,
                "sha256": v1_checkpoint.sha256,
            },
            "navsim_v2": {
                "artifact_id": f"artifact-{v2_digest[:12]}",
                "sha256": v2_digest,
            },
        },
        "inputs": {
            "navsim_v1_csv_sha256": _sha256(v1_path),
            "navsim_v2_summary_sha256": _sha256(v2_path),
        },
        "single_trajectory_inference": True,
        "test_time_scorer_or_reranking": False,
        "navsim_v1": v1,
        "navsim_v2": v2,
        "passed": bool(v1["passed"] and v2["passed"]),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": payload["passed"]}))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
