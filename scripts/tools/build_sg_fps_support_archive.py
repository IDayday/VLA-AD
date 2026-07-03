#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import lzma
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.candidate_funnel import ExternalCandidateLoader, build_candidate_pool_for_scene
from navsim.agents.recogdrive.offline_rl_buffer import save_elite_record
from navsim.agents.recogdrive.pareto_support import (
    CandidateRecord,
    build_archive_record,
    select_feasible_pareto_support,
)
from navsim.agents.recogdrive.trajectory_feasibility import cheap_filter_mask, compute_feasibility_metrics


def _parse_source_roots(items: list[str]) -> dict[str, str]:
    roots = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Expected source=path, got {item!r}.")
        source, path = item.split("=", 1)
        roots[source] = path
    return roots


def _load_payload(path: Path) -> Any:
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu")
    if path.suffix == ".xz":
        with lzma.open(path, "rb") as f:
            return pickle.load(f)
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    raise ValueError(f"Unsupported cache file: {path}")


def _iter_cache_samples(cache_path: Path):
    if cache_path.is_file():
        payload = _load_payload(cache_path)
        if isinstance(payload, dict) and "samples" in payload and isinstance(payload["samples"], list):
            for sample in payload["samples"]:
                yield sample
        elif isinstance(payload, list):
            for sample in payload:
                yield sample
        else:
            yield payload
        return
    for path in sorted(cache_path.rglob("*.pt")):
        payload = _load_payload(path)
        if isinstance(payload, dict) and "samples" in payload and isinstance(payload["samples"], list):
            for sample in payload["samples"]:
                yield sample
        else:
            yield payload


def _traj(sample: dict[str, Any], *keys: str) -> np.ndarray | None:
    for key in keys:
        if key in sample:
            arr = np.asarray(sample[key], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[-1] == 3:
                return arr
    return None


def _record_with_labels(record: CandidateRecord, idx: int) -> CandidateRecord:
    comp = dict(record.components)
    feas = compute_feasibility_metrics(torch.as_tensor(record.trajectory, dtype=torch.float32).unsqueeze(0), {})
    feas_dict = {
        "feas_cost": float(feas.feas_cost[0]),
        "early_kink_rate": float(feas.early_kink_rate[0]),
        "tail_reverse_rate": float(feas.tail_reverse_rate[0]),
        "curvature_violation_rate": float(feas.curvature_violation_rate[0]),
    }
    return CandidateRecord(
        trajectory=record.trajectory,
        source=record.source,
        token=record.token,
        components=comp,
        reward=float(comp.get("pdms", record.reward)),
        feas=feas_dict,
        selection_score=float(record.selection_score if record.selection_score else comp.get("pdms", 0.0)),
        support_tag=record.support_tag,
        parent_id=record.parent_id or str(idx),
    )


def _require_verified(records: list[CandidateRecord], token: str) -> None:
    missing = [
        idx
        for idx, rec in enumerate(records)
        if "pdms" not in rec.components
        or "ego_progress" not in rec.components
        or "time_to_collision_within_bound" not in rec.components
        or "driving_direction_compliance" not in rec.components
    ]
    if missing:
        raise ValueError(
            f"Token {token}: {len(missing)} candidates lack evaluator components. "
            "SG-FPS support archive requires true evaluator labels; provide pre-labeled candidates "
            "or extend this script with a planner/evaluator context."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SG-FPS v3 feasible Pareto support archives.")
    parser.add_argument("--cache_path", required=True)
    parser.add_argument("--metric_cache_path", default="")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--stage2_checkpoint", default="")
    parser.add_argument("--current_policy_checkpoint", default="")
    parser.add_argument("--external_candidate_root", action="append", default=[], help="source=path, may be repeated")
    parser.add_argument("--max_candidates_per_scene", type=int, default=40)
    parser.add_argument("--support_top_m", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--missing_external_policy", choices=["error", "skip", "fallback_gt_il"], default="skip")
    parser.add_argument("--allow_unverified_components", action="store_true")
    args = parser.parse_args()

    cfg = {
        "max_candidates_per_scene": int(args.max_candidates_per_scene),
        "support_top_m": int(args.support_top_m),
    }
    output = Path(args.output_path)
    output.mkdir(parents=True, exist_ok=True)
    loader = ExternalCandidateLoader(_parse_source_roots(args.external_candidate_root))
    summary = Counter()
    tag_counts = Counter()

    if args.stage2_checkpoint or args.current_policy_checkpoint:
        print(
            "checkpoint sampling is not launched by this builder without an explicit planner context; "
            "using cache/external/pre-labeled candidates only."
        )

    for sample in _iter_cache_samples(Path(args.cache_path)):
        if not isinstance(sample, dict):
            continue
        token = str(sample.get("token", sample.get("scene_token", "")))
        if not token:
            continue
        gt = _traj(sample, "trajectory", "gt_trajectory")
        il = _traj(sample, "il_trajectory", "expert_trajectory", "pred_trajectory")
        external = loader.load(token)
        pool = build_candidate_pool_for_scene(token, gt, il, None, external, cfg)
        summary["candidate_count_before_filter"] += len(pool)
        if pool:
            trajs = torch.as_tensor(np.stack([rec.trajectory for rec in pool], axis=0), dtype=torch.float32)
            mask = cheap_filter_mask(trajs, cfg).cpu().numpy().astype(bool)
            pool = [rec for rec, keep in zip(pool, mask) if keep]
        summary["candidate_count_after_filter"] += len(pool)
        labeled = [_record_with_labels(rec, idx) for idx, rec in enumerate(pool)]
        if not args.allow_unverified_components:
            _require_verified(labeled, token)
        ref = labeled[0].components if labeled else {}
        selected = select_feasible_pareto_support(labeled, ref, cfg)
        for rec in selected:
            tag_counts[rec.support_tag] += 1
        record = build_archive_record(token, labeled, selected, ref=ref, cfg=cfg)
        save_elite_record(output, token, record)
        summary["scene_count"] += 1
        summary["valid_count"] += int(np.asarray(record["valid_mask"]).sum())
        summary["candidate_count"] += int(len(labeled))
        summary["pareto_front_count"] += int(np.asarray(record["pareto_front_mask"]).sum())

    denom = max(summary["candidate_count"], 1)
    report = {
        **dict(summary),
        "valid_ratio": float(summary["valid_count"]) / denom,
        "pareto_front_ratio": float(summary["pareto_front_count"]) / denom,
        "support_tag_counts": dict(tag_counts),
    }
    with open(output / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
