from __future__ import annotations

import argparse
import json
import lzma
import pickle
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import iter_index, load_sample
from navsim.agents.recogdrive.offline_rl_buffer import REQUIRED_COMPONENT_KEYS, token_to_buffer_key
from navsim.agents.recogdrive.pareto_support import (
    DESCRIPTOR_NAMES,
    SUPPORT_SLOTS,
    select_adaptive_pareto_supports,
    trajectory_descriptor,
)
from navsim.planning.training.dataset import load_feature_target_from_pickle


BOOTSTRAP_SOURCE_MARKERS = ("stage3", "sota", "step21600", "grpo")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _iter_cache_entries(cache_root: Path) -> Iterable[Tuple[str, str, Path]]:
    if (cache_root / "index.jsonl").is_file():
        chunk_dirs = [cache_root]
    else:
        chunk_dirs = sorted(
            child
            for child in cache_root.iterdir()
            if child.is_dir() and (child / "index.jsonl").is_file() and not child.name.startswith("navtest")
        )
    if chunk_dirs:
        for chunk_dir in chunk_dirs:
            for record in iter_index(chunk_dir):
                token = str(record.get("sample_token") or Path(str(record.get("path", ""))).stem)
                log_name = str(record.get("log_name") or "")
                sample_path = Path(str(record["path"]))
                if not sample_path.is_absolute():
                    sample_path = chunk_dir / sample_path
                yield token, log_name, sample_path
        return

    for target_path in sorted(cache_root.glob("*/*/trajectory_target.gz")):
        yield target_path.parent.name, target_path.parent.parent.name, target_path


def _load_gt_trajectory(path: Path) -> torch.Tensor:
    if path.name == "trajectory_target.gz":
        payload = load_feature_target_from_pickle(path)
    else:
        payload = load_sample(path)
    traj = payload.get("trajectory")
    if not isinstance(traj, torch.Tensor):
        raise KeyError(f"Cache target {path} does not contain tensor field 'trajectory'.")
    if tuple(traj.shape) != (8, 3):
        raise ValueError(f"GT trajectory in {path} has shape {tuple(traj.shape)}, expected [8, 3].")
    return traj.detach().float()


def _load_buffer_record(buffer_root: Path, token: str) -> Optional[Dict[str, Any]]:
    path = buffer_root / f"{token_to_buffer_key(token)}.pkl.xz"
    if not path.is_file():
        return None
    with lzma.open(path, "rb") as f:
        record = pickle.load(f)
    if not isinstance(record, dict):
        raise TypeError(f"Elite buffer record must be a dict: {path}")
    return record


def _source_allowed(source: str, mode: str) -> bool:
    if mode == "bootstrap":
        return True
    source_l = str(source).lower()
    return not any(marker in source_l for marker in BOOTSTRAP_SOURCE_MARKERS)


def _filter_candidates(record: Dict[str, Any], mode: str) -> Dict[str, Any]:
    candidates = torch.as_tensor(np.asarray(record["candidates"], dtype=np.float32)).float()
    rewards = torch.as_tensor(np.asarray(record.get("rewards", record.get("selection_score")), dtype=np.float32)).float()
    sources = [str(source) for source in record.get("sources", ["unknown"] * candidates.shape[0])]
    if candidates.ndim != 3 or tuple(candidates.shape[1:]) != (8, 3):
        raise ValueError(f"record candidates must have shape [K, 8, 3], got {tuple(candidates.shape)}.")
    if rewards.shape != (candidates.shape[0],):
        raise ValueError(f"record rewards shape {tuple(rewards.shape)} does not match candidates.")
    allowed_idx = [idx for idx, source in enumerate(sources) if _source_allowed(source, mode)]
    if not allowed_idx:
        allowed_idx = list(range(candidates.shape[0]))
    idx_tensor = torch.tensor(allowed_idx, dtype=torch.long)
    components = {}
    raw_components = record.get("components", {})
    for key in REQUIRED_COMPONENT_KEYS:
        if key in raw_components:
            components[key] = torch.as_tensor(np.asarray(raw_components[key], dtype=np.float32))[idx_tensor].float()
    if "pdms" not in components:
        components["pdms"] = rewards[idx_tensor].float()
    valid_mask = record.get("valid_mask", None)
    if valid_mask is not None:
        valid_mask = torch.as_tensor(np.asarray(valid_mask, dtype=np.bool_))[idx_tensor].bool()
    else:
        valid_mask = torch.ones(len(allowed_idx), dtype=torch.bool)
    anchor_distance = record.get("anchor_distance", None)
    if anchor_distance is not None:
        anchor_distance = torch.as_tensor(np.asarray(anchor_distance, dtype=np.float32))[idx_tensor].float()
    return {
        "candidates": candidates[idx_tensor].contiguous(),
        "rewards": rewards[idx_tensor].float(),
        "sources": [sources[idx] for idx in allowed_idx],
        "components": components,
        "valid_mask": valid_mask,
        "anchor_distance": anchor_distance,
    }


def _first_source_index(sources: Sequence[str], names: Sequence[str]) -> Optional[int]:
    names_l = tuple(name.lower() for name in names)
    for idx, source in enumerate(sources):
        source_l = str(source).lower()
        if any(name in source_l for name in names_l):
            return idx
    return None


def _pdms_core(components: Dict[str, torch.Tensor]) -> torch.Tensor:
    ep = components["ego_progress"].float()
    ttc = components["time_to_collision_within_bound"].float()
    comfort = components["history_comfort"].float()
    return (5.0 * ep + 5.0 * ttc + 2.0 * comfort) / 12.0


def _pareto_front_mask(components: Dict[str, torch.Tensor], candidate_mask: torch.Tensor) -> torch.Tensor:
    objectives = torch.stack(
        (
            components["ego_progress"].float(),
            components["time_to_collision_within_bound"].float(),
            components["history_comfort"].float(),
        ),
        dim=1,
    )
    front = torch.zeros(objectives.shape[0], dtype=torch.bool)
    valid_idx = torch.nonzero(candidate_mask, as_tuple=False).view(-1)
    for idx in valid_idx.tolist():
        value = objectives[idx]
        dominated = False
        for other_idx in valid_idx.tolist():
            if other_idx == idx:
                continue
            other = objectives[other_idx]
            if bool((other >= value).all().item()) and bool((other > value).any().item()):
                dominated = True
                break
        front[idx] = not dominated
    return front


def _core_pareto_scores(filtered: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    components = filtered["components"]
    missing = sorted(set(REQUIRED_COMPONENT_KEYS).difference(components))
    if missing:
        raise KeyError(f"Buffer record missing component keys required for Core-Pareto support: {missing}")
    sources = filtered["sources"]
    gt_idx = _first_source_index(sources, ("gt", "ground_truth"))
    il_idx = _first_source_index(sources, ("il", "deterministic"))
    ref_candidates = [idx for idx in (gt_idx, il_idx) if idx is not None]
    if ref_candidates:
        pdms = components["pdms"].float()
        ref_idx = max(ref_candidates, key=lambda idx: float(pdms[idx].item()))
    else:
        ref_idx = int(torch.argmax(components["pdms"]).item())

    pdms = components["pdms"].float()
    core = _pdms_core(components)
    ep = components["ego_progress"].float()
    ttc = components["time_to_collision_within_bound"].float()
    nc = components["no_at_fault_collisions"].float()
    dac = components["drivable_area_compliance"].float()
    ddc = components["driving_direction_compliance"].float()
    valid_mask = filtered["valid_mask"].bool()

    ref_core = core[ref_idx]
    ref_ep = ep[ref_idx]
    ref_ttc = ttc[ref_idx]
    ref_ddc = ddc[ref_idx]
    nc_ok = nc >= 1.0
    dac_ok = dac >= 1.0
    ddc_ok = (ddc >= 0.95) | (ddc >= ref_ddc - 0.01)
    valid = valid_mask & nc_ok & dac_ok & ddc_ok
    ep_ok = ep >= ref_ep - 0.02
    slow_violation = (ref_ep - ep - 0.02).clamp(min=0.0)
    tradeoff_bad = (-(ep - ref_ep + ttc - ref_ttc) - 0.01).clamp(min=0.0)
    score = pdms + 0.3 * (core - ref_core) - 0.5 * slow_violation - 0.2 * tradeoff_bad
    pareto = _pareto_front_mask(components, valid & ep_ok)
    score = score + 0.2 * pareto.float()
    return score.float(), valid, ep_ok, core.float()


def _descriptor_stats(args: argparse.Namespace, entries: List[Tuple[str, str, Path]]) -> Tuple[torch.Tensor, torch.Tensor, Counter]:
    buffer_root = Path(args.elite_buffer_root)
    count = 0
    total = torch.zeros(len(DESCRIPTOR_NAMES), dtype=torch.float64)
    total_sq = torch.zeros(len(DESCRIPTOR_NAMES), dtype=torch.float64)
    stats = Counter()
    for row, (token, _, gt_path) in enumerate(entries, start=1):
        descriptors = []
        record = _load_buffer_record(buffer_root, token) if buffer_root.is_dir() else None
        if record is not None:
            try:
                filtered = _filter_candidates(record, args.source_mode)
                descriptors.append(trajectory_descriptor(filtered["candidates"]).double())
                stats["buffer_records"] += 1
            except Exception as exc:
                stats[f"descriptor_buffer_error:{type(exc).__name__}"] += 1
        if not descriptors:
            try:
                descriptors.append(trajectory_descriptor(_load_gt_trajectory(gt_path).unsqueeze(0)).double())
                stats["gt_descriptor_fallback"] += 1
            except Exception as exc:
                stats[f"descriptor_gt_error:{type(exc).__name__}"] += 1
                continue
        desc = torch.cat(descriptors, dim=0)
        total += desc.sum(dim=0)
        total_sq += (desc * desc).sum(dim=0)
        count += desc.shape[0]
        if args.progress_interval > 0 and row % args.progress_interval == 0:
            print(f"descriptor_pass={row}/{len(entries)} descriptor_count={count}", flush=True)
    if count <= 0:
        raise RuntimeError("No descriptors were collected; cannot build support index.")
    mean = total / float(count)
    var = (total_sq / float(count) - mean * mean).clamp(min=1e-8)
    std = var.sqrt().clamp(min=1e-4)
    stats["descriptor_count"] = count
    return mean.float(), std.float(), stats


def build_index(args: argparse.Namespace) -> Dict[str, Any]:
    cache_root = Path(args.cache_root)
    buffer_root = Path(args.elite_buffer_root)
    if not cache_root.is_dir():
        raise FileNotFoundError(f"cache root does not exist: {cache_root}")
    if args.source_mode not in {"clean", "bootstrap"}:
        raise ValueError("--source-mode must be clean or bootstrap")
    entries = list(_iter_cache_entries(cache_root))
    if args.max_records > 0:
        entries = entries[: int(args.max_records)]
    if not entries:
        raise FileNotFoundError(f"No cache entries found under {cache_root}")

    descriptor_mean, descriptor_std, descriptor_stats = _descriptor_stats(args, entries)
    tokens: List[str] = []
    support_trajs = []
    support_mask = []
    support_weights = []
    support_scores = []
    support_pdms = []
    support_core = []
    support_desc = []
    support_sources: List[List[str]] = []
    support_metadata: List[Dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    fallback_counts: Counter[str] = Counter()
    support_count_hist: Counter[str] = Counter()

    for row, (token, log_name, gt_path) in enumerate(entries, start=1):
        gt = None
        try:
            gt = _load_gt_trajectory(gt_path)
        except Exception as exc:
            fallback_counts[f"gt_load_error:{type(exc).__name__}"] += 1
        record = _load_buffer_record(buffer_root, token) if buffer_root.is_dir() else None
        try:
            if record is None:
                raise FileNotFoundError("missing_buffer_record")
            filtered = _filter_candidates(record, args.source_mode)
            score, valid, ep_ok, core = _core_pareto_scores(filtered)
            valid_positive = valid & ep_ok
            pdms = filtered["components"]["pdms"].float()
            nc_dac = (
                filtered["components"]["no_at_fault_collisions"].float()
                * filtered["components"]["drivable_area_compliance"].float()
            )
            gt_idx = _first_source_index(filtered["sources"], ("gt", "ground_truth"))
            il_idx = _first_source_index(filtered["sources"], ("il", "deterministic"))
            descriptors = trajectory_descriptor(filtered["candidates"])
            selection = select_adaptive_pareto_supports(
                filtered["candidates"],
                score,
                valid_positive,
                descriptors=descriptors,
                descriptor_mean=descriptor_mean,
                descriptor_std=descriptor_std,
                pdms=pdms,
                core=core,
                nc_dac=nc_dac,
                geometry_distance=filtered.get("anchor_distance"),
                sources=filtered["sources"],
                gt_candidate_index=gt_idx,
                deterministic_il_index=il_idx,
                quality_band=float(args.quality_band),
                min_descriptor_distance=float(args.min_descriptor_distance),
                best_weight=float(args.best_weight),
                gt_weight=float(args.gt_weight),
                other_weight=float(args.other_weight),
            )
            traj_slots = torch.zeros((SUPPORT_SLOTS, 8, 3), dtype=torch.float32)
            pdms_slots = torch.zeros(SUPPORT_SLOTS, dtype=torch.float32)
            core_slots = torch.zeros(SUPPORT_SLOTS, dtype=torch.float32)
            desc_slots = torch.zeros((SUPPORT_SLOTS, len(DESCRIPTOR_NAMES)), dtype=torch.float32)
            source_slots = [""] * SUPPORT_SLOTS
            for slot, idx_tensor in enumerate(selection.indices):
                idx = int(idx_tensor.item())
                if idx < 0:
                    continue
                traj_slots[slot] = filtered["candidates"][idx].float()
                pdms_slots[slot] = pdms[idx].float()
                core_slots[slot] = core[idx].float()
                desc_slots[slot] = descriptors[idx].float()
                source_slots[slot] = filtered["sources"][idx]
                source_counts[source_slots[slot]] += 1
            if selection.fallback_mode != "none":
                fallback_counts[selection.fallback_mode] += 1
        except Exception as exc:
            if gt is None:
                raise
            traj_slots = torch.zeros((SUPPORT_SLOTS, 8, 3), dtype=torch.float32)
            traj_slots[0] = gt.float()
            selection = type(
                "FallbackSelection",
                (),
                {
                    "mask": torch.tensor([True, False, False]),
                    "weights": torch.tensor([1.0, 0.0, 0.0]),
                    "scores": torch.zeros(SUPPORT_SLOTS),
                    "metadata": {"fallback_mode": f"gt_fallback_error:{type(exc).__name__}"},
                },
            )()
            pdms_slots = torch.zeros(SUPPORT_SLOTS, dtype=torch.float32)
            core_slots = torch.zeros(SUPPORT_SLOTS, dtype=torch.float32)
            desc_slots = torch.zeros((SUPPORT_SLOTS, len(DESCRIPTOR_NAMES)), dtype=torch.float32)
            desc_slots[0] = trajectory_descriptor(gt.unsqueeze(0))[0]
            source_slots = ["gt_fallback", "", ""]
            source_counts["gt_fallback"] += 1
            fallback_counts[f"gt_fallback_error:{type(exc).__name__}"] += 1

        tokens.append(str(token))
        support_trajs.append(traj_slots)
        support_mask.append(selection.mask.bool())
        support_weights.append(selection.weights.float())
        support_scores.append(selection.scores.float())
        support_pdms.append(pdms_slots.float())
        support_core.append(core_slots.float())
        support_desc.append(desc_slots.float())
        support_sources.append(source_slots)
        count = int(selection.mask.sum().item())
        support_count_hist[str(count)] += 1
        support_metadata.append(
            {
                "token": str(token),
                "log_name": str(log_name),
                **dict(selection.metadata),
                "sources": source_slots,
            }
        )
        if args.progress_interval > 0 and row % args.progress_interval == 0:
            print(f"selection_pass={row}/{len(entries)}", flush=True)

    token_to_row = {token: idx for idx, token in enumerate(tokens)}
    if len(token_to_row) != len(tokens):
        raise ValueError("Duplicate tokens encountered while building support index.")
    payload = {
        "version": 1,
        "method": "psi_drive_stage2_pareto_support",
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True).strip(),
        "git_commit": _git_commit(),
        "source_mode": str(args.source_mode),
        "cache_root": str(cache_root),
        "elite_buffer_root": str(buffer_root),
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "descriptor_mean": descriptor_mean,
        "descriptor_std": descriptor_std,
        "tokens": tokens,
        "token_to_row": token_to_row,
        "support_trajectories": torch.stack(support_trajs, dim=0).contiguous(),
        "support_mask": torch.stack(support_mask, dim=0).bool().contiguous(),
        "support_weights": torch.stack(support_weights, dim=0).float().contiguous(),
        "support_scores": torch.stack(support_scores, dim=0).float().contiguous(),
        "support_pdms": torch.stack(support_pdms, dim=0).float().contiguous(),
        "support_core": torch.stack(support_core, dim=0).float().contiguous(),
        "support_descriptors": torch.stack(support_desc, dim=0).float().contiguous(),
        "support_sources": support_sources,
        "support_metadata": support_metadata,
    }
    summary = {
        "version": 1,
        "source_mode": str(args.source_mode),
        "num_records": len(tokens),
        "unique_tokens": len(token_to_row),
        "support_count_hist": dict(sorted(support_count_hist.items())),
        "source_hist": dict(sorted(source_counts.items())),
        "fallback_counts": dict(sorted(fallback_counts.items())),
        "descriptor_stats": dict(sorted(descriptor_stats.items())),
        "quality_band": float(args.quality_band),
        "min_descriptor_distance": float(args.min_descriptor_distance),
        "weights": {
            "best": float(args.best_weight),
            "gt": float(args.gt_weight),
            "other": float(args.other_weight),
        },
        "git_commit": payload["git_commit"],
    }
    payload["summary"] = summary
    return payload


def _write_report(summary: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Stage2 Pareto Support Index Audit",
        "",
        f"- source_mode: `{summary['source_mode']}`",
        f"- records: `{summary['num_records']}`",
        f"- unique_tokens: `{summary['unique_tokens']}`",
        f"- quality_band: `{summary['quality_band']}`",
        f"- min_descriptor_distance: `{summary['min_descriptor_distance']}`",
        f"- git_commit: `{summary['git_commit']}`",
        "",
        "## Support Count",
        "",
    ]
    for key, value in summary["support_count_hist"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Source Histogram", ""])
    for key, value in summary["source_hist"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Fallback Counts", ""])
    for key, value in summary["fallback_counts"].items():
        lines.append(f"- {key}: {value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PSI-Drive Stage2 Pareto support index from navtrain candidates.")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--elite-buffer-root", required=True)
    parser.add_argument("--output-pt", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--output-report-md", required=True)
    parser.add_argument("--source-mode", choices=["clean", "bootstrap"], default="clean")
    parser.add_argument("--quality-band", type=float, default=0.02)
    parser.add_argument("--min-descriptor-distance", type=float, default=0.75)
    parser.add_argument("--best-weight", type=float, default=0.50)
    parser.add_argument("--gt-weight", type=float, default=0.20)
    parser.add_argument("--other-weight", type=float, default=0.30)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=5000)
    args = parser.parse_args()

    payload = build_index(args)
    output_pt = Path(args.output_pt)
    output_json = Path(args.output_summary_json)
    output_md = Path(args.output_report_md)
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_pt)
    output_json.write_text(json.dumps(payload["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(payload["summary"], output_md)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
