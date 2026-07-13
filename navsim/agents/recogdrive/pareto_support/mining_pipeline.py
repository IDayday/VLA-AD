from __future__ import annotations

import json
import lzma
import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .acquisition import select_candidates_for_evaluation
from .dataclasses import ParetoSupportArchive, TrajectoryCandidate
from ..offline_rl_buffer import save_elite_record
from .io import save_archive, should_skip_existing, stable_config_hash
from .metrics import merge_true_and_feasibility_metrics
from .operators import failure_conditioned_operators
from .pareto_archive import build_archive_record, build_support_set
from .path_speed import recombine_path_speed, select_path_bank, select_speed_bank
from .seed_collection import collect_seed_candidates, ensure_evaluator_verified


def load_payload(path: Path) -> Any:
    if path.name.endswith(".pkl.xz"):
        with lzma.open(path, "rb") as f:
            return pickle.load(f)
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    if path.suffix in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu", weights_only=False)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"Unsupported cache file: {path}")


def iter_cache_samples(cache_path: str | Path) -> Iterable[dict[str, Any]]:
    path = Path(cache_path)
    if path.is_file():
        payload = load_payload(path)
        if isinstance(payload, Mapping) and isinstance(payload.get("samples"), list):
            yield from payload["samples"]
        elif isinstance(payload, list):
            yield from payload
        elif isinstance(payload, Mapping):
            yield dict(payload)
        return
    for file_path in sorted(path.rglob("*.pt")):
        payload = load_payload(file_path)
        if isinstance(payload, Mapping) and isinstance(payload.get("samples"), list):
            yield from payload["samples"]
        elif isinstance(payload, Mapping):
            yield dict(payload)


def sample_to_features_targets(sample: Mapping[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    token = str(sample.get("token", sample.get("scene_token", "")))
    if not token:
        raise KeyError("Cache sample is missing token/scene_token.")
    features = {k: v for k, v in sample.items() if k not in {"trajectory", "gt_trajectory", "il_trajectory", "expert_trajectory"}}
    targets = {
        "trajectory": sample.get("trajectory", sample.get("gt_trajectory")),
        "gt_trajectory": sample.get("gt_trajectory", sample.get("trajectory")),
        "il_trajectory": sample.get("il_trajectory", sample.get("expert_trajectory", sample.get("pred_trajectory"))),
    }
    for key in ("trajectory_metrics", "il_metrics"):
        if key in sample:
            targets[key] = sample[key]
    return token, features, targets


def _path_speed_candidates(seeds: list[TrajectoryCandidate], cfg: Any) -> list[TrajectoryCandidate]:
    path_bank = select_path_bank(seeds, cfg)
    speed_bank = select_speed_bank(seeds, cfg)
    out = []
    for path_atom in path_bank:
        for speed_atom in speed_bank:
            try:
                out.append(recombine_path_speed(path_atom, speed_atom, cfg))
            except ValueError:
                continue
    return out


def _has_evaluator_components(candidate: TrajectoryCandidate) -> bool:
    metrics = candidate.true_metrics or {}
    required = (
        "pdms",
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "time_to_collision_within_bound",
        "ego_progress",
        "history_comfort",
        "driving_direction_compliance",
    )
    return all(key in metrics for key in required)


def mine_scene(
    sample: Mapping[str, Any],
    cfg: Any,
    metric_context: Any = None,
    scorer: Any = None,
    round_id: str = "all",
) -> ParetoSupportArchive:
    token, features, targets = sample_to_features_targets(sample)
    seeds = collect_seed_candidates(token, features, targets, cfg)
    if not seeds:
        raise RuntimeError(f"No seed candidates for scene {token}.")
    allow_unverified_smoke = bool(getattr(cfg, "allow_unverified_smoke", False)) if not isinstance(cfg, dict) else bool(cfg.get("allow_unverified_smoke", False))
    if metric_context is None:
        if not allow_unverified_smoke:
            missing = [candidate.candidate_id for candidate in seeds if not _has_evaluator_components(candidate)]
            if missing:
                raise RuntimeError(
                    f"Scene {token}: seed candidates lack evaluator components: {missing[:5]}. "
                    "Provide precomputed true metrics or run with a NAVSIM evaluator context."
                )
        for candidate in seeds:
            if candidate.true_metrics is None:
                candidate.true_metrics = merge_true_and_feasibility_metrics({}, candidate.trajectory, cfg)
    else:
        ensure_evaluator_verified(seeds, metric_context, cfg)
    ref = next((c for c in seeds if c.source == "recogdrive_stage3"), seeds[0])
    ref_metrics = ref.true_metrics or {}
    generated: list[TrajectoryCandidate] = []
    if round_id in {"1", "2", "all"}:
        generated.extend(_path_speed_candidates(seeds, cfg))
        op_candidates, op_stats = failure_conditioned_operators(seeds, ref_metrics, cfg)
        generated.extend(op_candidates)
    selected_for_eval = select_candidates_for_evaluation(generated, seeds, scorer, cfg)
    candidates = seeds + selected_for_eval
    if metric_context is None:
        if not allow_unverified_smoke:
            selected_for_eval = [candidate for candidate in selected_for_eval if _has_evaluator_components(candidate)]
            candidates = seeds + selected_for_eval
        for candidate in candidates:
            candidate.true_metrics = merge_true_and_feasibility_metrics(candidate.true_metrics or {}, candidate.trajectory, cfg)
    else:
        ensure_evaluator_verified(candidates, metric_context, cfg)
    archive = build_support_set(candidates, ref, cfg)
    archive.generated_candidates = generated
    archive.build_metadata.update({"round": round_id, "config_hash": stable_config_hash(cfg)})
    if round_id in {"1", "2", "all"}:
        archive.operator_stats.setdefault("generated_count", len(generated))
        archive.operator_stats.setdefault("true_evaluated_count", len(candidates))
    return archive


def mine_cache(
    cache_path: str | Path,
    output_archive_dir: str | Path,
    cfg: Any,
    metric_context: Any = None,
    scorer: Any = None,
    round_id: str = "all",
    limit_scenes: int = 0,
    rank: int = 0,
    world_size: int = 1,
    overwrite: bool = False,
    token_filter: set[str] | None = None,
) -> dict[str, Any]:
    output = Path(output_archive_dir)
    output.mkdir(parents=True, exist_ok=True)
    full_archive_root = output / "full_archive"
    full_archive_root.mkdir(parents=True, exist_ok=True)
    config_hash = stable_config_hash(cfg)
    summary = {
        "scene_count": 0,
        "skipped_existing": 0,
        "failed_count": 0,
        "avg_seed_count": 0.0,
        "avg_generated_count": 0.0,
        "avg_true_evaluated_count": 0.0,
        "avg_support_count": 0.0,
        "valid_rate": 0.0,
    }
    failures: list[dict[str, str]] = []
    seen = 0
    for idx, sample in enumerate(iter_cache_samples(cache_path)):
        if token_filter:
            try:
                token, _, _ = sample_to_features_targets(sample)
            except Exception:
                token = ""
            if str(token) not in token_filter:
                continue
        if idx % max(int(world_size), 1) != int(rank):
            continue
        if limit_scenes and seen >= int(limit_scenes):
            break
        seen += 1
        try:
            token, _, _ = sample_to_features_targets(sample)
            if should_skip_existing(full_archive_root, token, config_hash, overwrite=overwrite):
                summary["skipped_existing"] += 1
                continue
            archive = mine_scene(sample, cfg, metric_context=metric_context, scorer=scorer, round_id=round_id)
            save_archive(full_archive_root, archive, overwrite=True)
            legacy_all = [candidate.to_legacy_record() for candidate in archive.evaluated_candidates]
            legacy_selected = [candidate.to_legacy_record() for candidate in archive.support_set]
            elite_record = build_archive_record(
                token,
                legacy_all,
                legacy_selected,
                ref=archive.reference.get("ref_metrics", {}),
                cfg=cfg,
            )
            save_elite_record(output, token, elite_record)
            summary["scene_count"] += 1
            n = float(summary["scene_count"])
            summary["avg_seed_count"] += (len(archive.seed_candidates) - summary["avg_seed_count"]) / n
            summary["avg_generated_count"] += (len(archive.generated_candidates) - summary["avg_generated_count"]) / n
            summary["avg_true_evaluated_count"] += (len(archive.evaluated_candidates) - summary["avg_true_evaluated_count"]) / n
            summary["avg_support_count"] += (len(archive.support_set) - summary["avg_support_count"]) / n
            valid = sum(1 for c in archive.evaluated_candidates if c.valid_mask)
            total = max(len(archive.evaluated_candidates), 1)
            summary["valid_rate"] += ((valid / total) - summary["valid_rate"]) / n
        except Exception as exc:
            summary["failed_count"] += 1
            token = str(sample.get("token", sample.get("scene_token", f"idx{idx}"))) if isinstance(sample, Mapping) else f"idx{idx}"
            failures.append({"token": token, "error": repr(exc)})
    with open(output / f"global_summary_rank{rank}.json", "w", encoding="utf-8") as f:
        json.dump({**summary, "failures": failures[:50]}, f, indent=2, sort_keys=True)
    return summary
