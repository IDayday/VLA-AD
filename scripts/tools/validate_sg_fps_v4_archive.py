#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import lzma
import math
import multiprocessing as mp
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.pareto_support import trajectory_snsad_distance


def _iter_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.glob("*.pkl.xz"))


def _load(path: Path) -> dict[str, Any]:
    with lzma.open(path, "rb") as file:
        record = pickle.load(file)
    if not isinstance(record, dict):
        raise TypeError(f"record must be a dict, got {type(record).__name__}.")
    return record


def _is_gt(source: str) -> bool:
    source = str(source).lower()
    return source == "gt" or source.startswith("gt:")


def _is_derived_external(source: str) -> bool:
    source = str(source).lower()
    external = source.startswith(("ddv2", "driveor", "drivor", "diffusiondrivev2"))
    return external and (":failure_expand_" in source or ":trust_region_" in source)


def _source_family(source: str) -> str:
    source = str(source).lower()
    if _is_gt(source):
        return "gt"
    if source.startswith(("ddv2", "diffusiondrivev2")):
        return "ddv2"
    if source.startswith(("driveor", "drivor")):
        return "driveor"
    if source.startswith("progress"):
        return "progress"
    if "lateral" in source:
        return "lateral"
    if source.startswith("timing"):
        return "timing"
    return "other"


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _array(record: Mapping[str, Any], key: str, count: int, dtype: Any) -> np.ndarray:
    if key not in record:
        raise KeyError(f"missing required field {key!r}")
    value = np.asarray(record[key], dtype=dtype).reshape(-1)
    if value.shape != (count,):
        raise ValueError(f"{key} has shape {value.shape}, expected {(count,)}")
    return value


def validate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    token = str(record.get("token", ""))
    version = int(record.get("version", 0))
    if version < 4:
        errors.append(f"version={version}, expected >=4")

    candidates = np.asarray(record.get("candidates"), dtype=np.float32)
    if candidates.ndim != 3 or candidates.shape[-1] != 3 or not np.isfinite(candidates).all():
        return {"token": token, "errors": [f"invalid candidates shape/content: {candidates.shape}"]}
    count = int(candidates.shape[0])
    sources = [str(source) for source in record.get("sources", [])]
    if len(sources) != count:
        return {"token": token, "errors": [f"source count={len(sources)}, candidate count={count}"]}
    gt_indices = [index for index, source in enumerate(sources) if _is_gt(source)]
    if len(gt_indices) != 1:
        return {"token": token, "errors": [f"GT candidate count={len(gt_indices)}, expected 1"]}
    gt_index = gt_indices[0]

    try:
        selected = np.asarray(record["support_indices"], dtype=np.int64).reshape(-1)
        if selected.size == 0 or selected.min() < 0 or selected.max() >= count:
            raise ValueError("support_indices must be non-empty and in range")
        mode_ids = _array(record, "mode_ids", count, np.int64)
        teacher_eligible = _array(record, "teacher_eligible_mask", count, np.bool_)
        source_conditioned = _array(record, "source_conditioned_mask", count, np.bool_)
        gt_ade = _array(record, "gt_relative_ade", count, np.float64)
        gt_fde = _array(record, "gt_relative_fde", count, np.float64)
        policy_reachability = _array(record, "policy_reachability_snsad", count, np.float64)
        objective_novelty = _array(record, "pareto_objective_novelty", count, np.float64)
        pareto = _array(record, "pareto_front_mask", count, np.bool_)
        rewards = _array(record, "rewards", count, np.float64)
    except (KeyError, ValueError) as exc:
        return {"token": token, "errors": [str(exc)]}

    metadata = dict(record.get("build_metadata", {}) or {})
    required_metadata = {
        "selection_strategy",
        "teacher_contract",
        "mode_selection_order",
        "support_top_m",
        "max_gt_ade_m",
        "max_gt_fde_m",
        "mode_distance_threshold",
        "max_gt_reward_drop",
        "pareto_eps",
        "raw_internal_candidates",
        "expand_external_candidates",
        "policy_reachability_required",
        "max_policy_snsad",
        "policy_samples_per_scene",
        "policy_checkpoint_sha256",
        "fs_norm_stats_sha256",
    }
    missing_metadata = sorted(required_metadata.difference(metadata))
    if missing_metadata:
        errors.append(f"missing build metadata: {missing_metadata}")
    if metadata.get("selection_strategy") != "mode_pareto_v4":
        errors.append(f"selection_strategy={metadata.get('selection_strategy')!r}")
    if metadata.get("teacher_contract") != "gt_anchor_plus_uniform_reachable_pareto_modes":
        errors.append(f"teacher_contract={metadata.get('teacher_contract')!r}")
    if metadata.get("mode_selection_order") != "scene_normalized_objective_fps_then_snsad":
        errors.append(f"mode_selection_order={metadata.get('mode_selection_order')!r}")
    if not bool(metadata.get("policy_reachability_required", False)):
        errors.append("policy_reachability_required=false")
    if int(metadata.get("policy_samples_per_scene", 0)) <= 0:
        errors.append("policy_samples_per_scene must be positive")
    if not str(metadata.get("policy_checkpoint_sha256", "")):
        errors.append("policy_checkpoint_sha256 is empty")
    if not str(metadata.get("fs_norm_stats_sha256", "")):
        errors.append("fs_norm_stats_sha256 is empty")

    top_m = int(metadata.get("support_top_m", 0))
    max_ade = float(metadata.get("max_gt_ade_m", math.inf))
    max_fde = float(metadata.get("max_gt_fde_m", math.inf))
    max_reward_drop = float(metadata.get("max_gt_reward_drop", math.inf))
    mode_threshold = float(metadata.get("mode_distance_threshold", 0.0))
    max_policy_snsad = float(metadata.get("max_policy_snsad", math.inf))
    if top_m <= 0 or selected.size > top_m:
        errors.append(f"selected count={selected.size} exceeds support_top_m={top_m}")
    if gt_index not in selected.tolist():
        errors.append("GT anchor is not selected")
    if not bool(np.all(teacher_eligible[selected])):
        errors.append("a selected candidate is not teacher eligible")
    selected_modes = mode_ids[selected]
    if len(set(selected_modes.tolist())) != selected.size or set(selected_modes.tolist()) != set(range(selected.size)):
        errors.append(f"selected mode IDs are not one-to-one contiguous IDs: {selected_modes.tolist()}")

    non_gt = np.asarray([index for index in selected if int(index) != gt_index], dtype=np.int64)
    if non_gt.size:
        if not bool(np.isfinite(objective_novelty[non_gt]).all()):
            errors.append("a selected non-GT candidate has non-finite objective novelty")
        elif bool(np.any(objective_novelty[non_gt] < -1e-8)):
            errors.append("a selected non-GT candidate has negative objective novelty")
        if not bool(np.all(pareto[non_gt])):
            errors.append("a selected non-GT candidate is outside the epsilon-Pareto front")
        if bool(np.any(gt_ade[non_gt] > max_ade + 1e-6)):
            errors.append("a selected non-GT candidate exceeds max_gt_ade_m")
        if bool(np.any(gt_fde[non_gt] > max_fde + 1e-6)):
            errors.append("a selected non-GT candidate exceeds max_gt_fde_m")
        if bool(np.any(rewards[non_gt] < rewards[gt_index] - max_reward_drop - 1e-6)):
            errors.append("a selected non-GT candidate exceeds max_gt_reward_drop")
        if not bool(np.isfinite(policy_reachability[non_gt]).all()):
            errors.append("a selected non-GT candidate has no finite current-policy reachability")
        elif bool(np.any(policy_reachability[non_gt] > max_policy_snsad + 1e-6)):
            errors.append("a selected non-GT candidate exceeds max_policy_snsad")
        if any(_is_derived_external(sources[int(index)]) for index in non_gt):
            errors.append("a derived external expansion is selected")

    min_pairwise_distance = math.inf
    for left_pos, left in enumerate(selected.tolist()):
        for right in selected[left_pos + 1 :].tolist():
            distance = trajectory_snsad_distance(candidates[left], candidates[right])
            min_pairwise_distance = min(min_pairwise_distance, distance)
            if distance + 1e-6 < mode_threshold:
                errors.append(
                    f"selected modes {left}/{right} have SNSAD={distance:.6f} below {mode_threshold:.6f}"
                )
    if not math.isfinite(min_pairwise_distance):
        min_pairwise_distance = 0.0

    selected_families = Counter(_source_family(sources[int(index)]) for index in selected)
    return {
        "token": token,
        "errors": errors,
        "candidate_count": count,
        "selected_count": int(selected.size),
        "non_gt_count": int(non_gt.size),
        "multimode": int(non_gt.size > 0),
        "raw_internal_candidates": int(bool(metadata.get("raw_internal_candidates", False))),
        "external_expansion_disabled": int(not bool(metadata.get("expand_external_candidates", True))),
        "policy_reachability_required": int(bool(metadata.get("policy_reachability_required", False))),
        "selected_source_conditioned": int(source_conditioned[selected].sum()),
        "selected_families": selected_families,
        "reward_delta": (rewards[non_gt] - rewards[gt_index]).tolist(),
        "gt_ade": gt_ade[non_gt].tolist(),
        "gt_fde": gt_fde[non_gt].tolist(),
        "policy_reachability": policy_reachability[non_gt].tolist(),
        "objective_novelty": objective_novelty[non_gt].tolist(),
        "min_pairwise_snsad": float(min_pairwise_distance),
    }


def _validate_path(path: Path) -> dict[str, Any]:
    try:
        return validate_record(_load(path))
    except Exception as exc:  # pragma: no cover - archive I/O error path
        return {"token": path.stem, "errors": [f"{path}: {exc}"]}


def validate_archive(
    root: Path,
    *,
    max_records: int = 0,
    workers: int = 1,
    require_raw_build: bool = True,
    min_multimode_scene_ratio: float = 0.25,
) -> dict[str, Any]:
    paths = _iter_paths(root)
    if max_records > 0:
        paths = paths[:max_records]
    rows: Iterable[dict[str, Any]]
    if workers > 1:
        with mp.Pool(workers) as pool:
            rows = list(pool.imap_unordered(_validate_path, paths, chunksize=64))
    else:
        rows = [_validate_path(path) for path in paths]

    totals: Counter[str] = Counter()
    source_families: Counter[str] = Counter()
    values: dict[str, list[float]] = defaultdict(list)
    error_examples: list[dict[str, Any]] = []
    for row in rows:
        totals["records"] += 1
        if row.get("errors"):
            totals["contract_errors"] += 1
            if len(error_examples) < 20:
                error_examples.append({"token": row.get("token", ""), "errors": row["errors"]})
            continue
        for key in (
            "candidate_count",
            "selected_count",
            "non_gt_count",
            "multimode",
            "raw_internal_candidates",
            "external_expansion_disabled",
            "policy_reachability_required",
            "selected_source_conditioned",
        ):
            totals[key] += int(row[key])
        source_families.update(row["selected_families"])
        values["reward_delta"].extend(row["reward_delta"])
        values["gt_ade"].extend(row["gt_ade"])
        values["gt_fde"].extend(row["gt_fde"])
        values["policy_reachability"].extend(row["policy_reachability"])
        values["objective_novelty"].extend(row["objective_novelty"])
        values["min_pairwise_snsad"].append(float(row["min_pairwise_snsad"]))

    valid_records = max(totals["records"] - totals["contract_errors"], 1)
    multimode_ratio = totals["multimode"] / valid_records
    raw_ratio = totals["raw_internal_candidates"] / valid_records
    expansion_disabled_ratio = totals["external_expansion_disabled"] / valid_records
    policy_reachability_ratio = totals["policy_reachability_required"] / valid_records
    hard_contract_pass = totals["records"] > 0 and totals["contract_errors"] == 0
    static_promotion_pass = (
        hard_contract_pass
        and multimode_ratio >= float(min_multimode_scene_ratio)
        and (not require_raw_build or raw_ratio == 1.0)
        and (not require_raw_build or expansion_disabled_ratio == 1.0)
        and policy_reachability_ratio == 1.0
    )
    return {
        "archive": str(root),
        "record_count": totals["records"],
        "contract_error_count": totals["contract_errors"],
        "error_examples": error_examples,
        "hard_contract_pass": hard_contract_pass,
        "static_promotion_pass": static_promotion_pass,
        "require_raw_build": bool(require_raw_build),
        "min_multimode_scene_ratio": float(min_multimode_scene_ratio),
        "candidate_count_per_scene": totals["candidate_count"] / valid_records,
        "selected_count_per_scene": totals["selected_count"] / valid_records,
        "non_gt_count_per_scene": totals["non_gt_count"] / valid_records,
        "multimode_scene_ratio": multimode_ratio,
        "raw_internal_candidate_build_ratio": raw_ratio,
        "external_expansion_disabled_ratio": expansion_disabled_ratio,
        "policy_reachability_required_ratio": policy_reachability_ratio,
        "selected_source_conditioned_ratio": totals["selected_source_conditioned"]
        / max(totals["selected_count"], 1),
        "selected_source_family_counts": dict(source_families),
        "non_gt_reward_delta": _stats(values["reward_delta"]),
        "non_gt_gt_relative_ade_m": _stats(values["gt_ade"]),
        "non_gt_gt_relative_fde_m": _stats(values["gt_fde"]),
        "non_gt_policy_reachability_snsad": _stats(values["policy_reachability"]),
        "non_gt_pareto_objective_novelty": _stats(values["objective_novelty"]),
        "min_pairwise_snsad_per_scene": _stats(values["min_pairwise_snsad"]),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            "# SG-FPS v4 archive validation",
            "",
            f"- Archive: `{report['archive']}`",
            f"- Records: {report['record_count']}",
            f"- Contract errors: {report['contract_error_count']}",
            f"- Hard contract: **{'PASS' if report['hard_contract_pass'] else 'FAIL'}**",
            f"- Static promotion gate: **{'PASS' if report['static_promotion_pass'] else 'FAIL'}**",
            f"- Mean selected trajectories: {report['selected_count_per_scene']:.4f}",
            f"- Multi-mode scene ratio: {report['multimode_scene_ratio']:.4f}",
            f"- Raw candidate build ratio: {report['raw_internal_candidate_build_ratio']:.4f}",
            f"- External expansion disabled ratio: {report['external_expansion_disabled_ratio']:.4f}",
            f"- Policy reachability required ratio: {report['policy_reachability_required_ratio']:.4f}",
            "",
            "Static promotion does not replace a Stage2 learnability smoke or Stage3 improvement probe.",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the SG-FPS v4 data and teacher contract.")
    parser.add_argument("--archive-path", required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--require-raw-build", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-multimode-scene-ratio", type=float, default=0.25)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    report = validate_archive(
        Path(args.archive_path),
        max_records=int(args.max_records),
        workers=max(int(args.workers), 1),
        require_raw_build=bool(args.require_raw_build),
        min_multimode_scene_ratio=float(args.min_multimode_scene_ratio),
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    if args.output_md:
        output = Path(args.output_md)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_markdown(report) + "\n", encoding="utf-8")
    if not args.report_only and not bool(report["static_promotion_pass"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
