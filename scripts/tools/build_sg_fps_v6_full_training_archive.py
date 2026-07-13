#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import lzma
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.pareto_support import (  # noqa: E402
    FullTrainingV6Config,
    promote_v5_record_to_v6,
    source_family,
    trajectory_snsad_distance,
)


def _paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.glob("*.pkl.xz"))


def _load(path: Path) -> dict[str, Any]:
    with lzma.open(path, "rb") as stream:
        record = pickle.load(stream)
    if not isinstance(record, dict):
        raise TypeError(f"record must be dict, got {type(record).__name__}: {path}")
    return record


def _save_atomic(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with lzma.open(temporary, "wb") as stream:
            pickle.dump(record, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _stats(record: dict[str, Any]) -> Counter:
    stats = Counter(records=1)
    indices = np.asarray(record["support_indices"], dtype=np.int64)
    mode_ids = np.asarray(record["mode_ids"], dtype=np.int64)
    selected_non_gt = indices[mode_ids[indices] > 0]
    stats["selected_non_gt"] += int(selected_non_gt.size)
    stats["scenes_with_non_gt"] += int(selected_non_gt.size > 0)
    stats["scenes_with_multiple_non_gt"] += int(selected_non_gt.size > 1)
    stats["eligible_non_gt"] += int(
        (
            np.asarray(record["teacher_eligible_mask"], dtype=np.bool_)
            & (mode_ids != 0)
        ).sum()
    )
    sources = [str(source) for source in record["sources"]]
    for index in selected_non_gt.tolist():
        stats[f"source:{source_family(sources[index])}"] += 1
        tier = int(np.asarray(record["teacher_tier"], dtype=np.int8)[index])
        stats[f"tier:{tier}"] += 1
    if selected_non_gt.size:
        candidates = np.asarray(record["candidates"], dtype=np.float32)
        pairs = [
            trajectory_snsad_distance(candidates[int(lhs)], candidates[int(rhs)])
            for left_pos, lhs in enumerate(indices.tolist())
            for rhs in indices.tolist()[left_pos + 1 :]
        ]
        if pairs:
            stats["pairwise_snsad_milli_sum"] += int(round(min(pairs) * 1000.0))
            stats["pairwise_snsad_scene_count"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Promote an SG-FPS v5 raw-candidate archive to the policy-independent "
            "v6 support distribution intended for random-initialized Stage2 training."
        )
    )
    parser.add_argument("--input-archive", required=True)
    parser.add_argument("--output-archive", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--support-top-m", type=int, default=4)
    parser.add_argument("--max-gt-ade-m", type=float, default=1.5)
    parser.add_argument("--max-gt-fde-m", type=float, default=4.0)
    parser.add_argument("--max-gt-reward-drop", type=float, default=0.05)
    parser.add_argument("--mode-distance-threshold", type=float, default=0.40)
    parser.add_argument("--pareto-eps", type=float, default=0.01)
    parser.add_argument(
        "--exclude-policy-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--exclude-derived-external",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.world_size <= 0 or args.rank < 0 or args.rank >= args.world_size:
        raise ValueError(f"invalid rank/world-size: {args.rank}/{args.world_size}")
    if args.support_top_m < 1:
        raise ValueError("support-top-m must include at least the GT anchor")
    cfg = FullTrainingV6Config(
        support_top_m=int(args.support_top_m),
        max_gt_ade_m=float(args.max_gt_ade_m),
        max_gt_fde_m=float(args.max_gt_fde_m),
        max_gt_reward_drop=float(args.max_gt_reward_drop),
        mode_distance_threshold=float(args.mode_distance_threshold),
        pareto_eps=float(args.pareto_eps),
        exclude_policy_candidates=bool(args.exclude_policy_candidates),
        exclude_derived_external=bool(args.exclude_derived_external),
    )
    input_root = Path(args.input_archive).expanduser()
    output_root = Path(args.output_archive).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    totals = Counter()
    errors: list[dict[str, str]] = []
    processed = 0
    for path_index, path in enumerate(_paths(input_root)):
        if path_index % args.world_size != args.rank:
            continue
        if args.max_records > 0 and processed >= args.max_records:
            break
        processed += 1
        destination = output_root / path.name
        if destination.exists() and not args.overwrite:
            totals["skipped_existing"] += 1
            continue
        try:
            record = _load(path)
            rebuilt = promote_v5_record_to_v6(record, cfg)
            metadata = dict(rebuilt["build_metadata"])
            metadata["promotion_source_archive"] = str(input_root)
            rebuilt["build_metadata"] = metadata
            _save_atomic(destination, rebuilt)
            totals.update(_stats(rebuilt))
        except Exception as exc:  # pragma: no cover - exercised by full archive jobs
            totals["errors"] += 1
            if len(errors) < 32:
                errors.append({"path": str(path), "error": str(exc)})

    summary = {
        "input_archive": str(input_root),
        "output_archive": str(output_root),
        "rank": int(args.rank),
        "world_size": int(args.world_size),
        "config": cfg.__dict__,
        "stats": dict(totals),
        "errors": errors,
    }
    if args.output_json:
        output_json = Path(args.output_json).expanduser()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if totals.get("errors", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
