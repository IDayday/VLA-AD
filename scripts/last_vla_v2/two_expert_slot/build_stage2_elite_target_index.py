from __future__ import annotations

import argparse
import json
import lzma
import pickle
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import iter_index
from navsim.agents.recogdrive.offline_rl_buffer import token_to_buffer_key


def _cache_chunk_dirs(cache_root: Path) -> list[Path]:
    if (cache_root / "index.jsonl").is_file():
        return [cache_root]
    chunk_dirs = sorted(
        child for child in cache_root.iterdir()
        if child.is_dir()
        and (child / "index.jsonl").is_file()
        and not child.name.startswith("navtest")
    )
    if not chunk_dirs:
        raise FileNotFoundError(f"No indexed cache chunks found under {cache_root}")
    return chunk_dirs


def _iter_cache_records(cache_root: Path) -> Iterable[Dict[str, Any]]:
    for chunk_dir in _cache_chunk_dirs(cache_root):
        for record in iter_index(chunk_dir):
            yield record


def _select_best_valid(record: Dict[str, Any]) -> Tuple[int, float, str]:
    candidates = np.asarray(record["candidates"], dtype=np.float32)
    rewards = np.asarray(record["rewards"], dtype=np.float32)
    valid_mask = np.asarray(record.get("valid_mask", np.ones(rewards.shape, dtype=np.bool_)), dtype=np.bool_)
    if candidates.ndim != 3 or candidates.shape[1:] != (8, 3):
        raise ValueError(f"Elite candidates must have shape [K, 8, 3], got {candidates.shape}.")
    if rewards.shape != (candidates.shape[0],):
        raise ValueError(f"Elite rewards shape {rewards.shape} does not match K={candidates.shape[0]}.")
    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size > 0:
        idx = int(valid_idx[int(np.argmax(rewards[valid_idx]))])
    else:
        idx = int(np.argmax(rewards))
    sources = [str(source) for source in record.get("sources", [])]
    source = sources[idx] if idx < len(sources) else "unknown"
    return idx, float(rewards[idx]), source


def _load_record_fast(path: Path) -> Dict[str, Any]:
    with lzma.open(path, "rb") as f:
        record = pickle.load(f)
    if not isinstance(record, dict):
        raise TypeError(f"Elite buffer record must be a dict, got {type(record).__name__}: {path}")
    for key in ("candidates", "rewards", "gt_reward"):
        if key not in record:
            raise KeyError(f"Elite buffer record {path} is missing {key!r}.")
    return record


def _worker_select_target(task: Tuple[str, str, bool, float]) -> Tuple[str, str, Any]:
    token, record_path, require_above_gt, min_reward_margin = task
    try:
        elite = _load_record_fast(Path(record_path))
        idx, reward, source = _select_best_valid(elite)
        gt_reward = float(elite.get("gt_reward", 0.0))
        if bool(require_above_gt) and reward <= gt_reward + float(min_reward_margin):
            return ("fallback", "not_above_gt", token)
        candidates = np.asarray(elite["candidates"], dtype=np.float32)
        return (
            "selected",
            token,
            {
                "trajectory": np.asarray(candidates[idx], dtype=np.float32),
                "reward": float(reward),
                "gt_reward": float(gt_reward),
                "candidate_index": int(idx),
                "source": str(source),
            },
        )
    except Exception as exc:
        return ("fallback", f"error:{type(exc).__name__}", token)


def build_index(args: argparse.Namespace) -> Dict[str, Any]:
    cache_root = Path(args.cache_root)
    buffer_root = Path(args.elite_buffer_root)
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Cache root does not exist: {cache_root}")
    if not buffer_root.is_dir():
        raise FileNotFoundError(f"Elite buffer root does not exist: {buffer_root}")

    tasks: list[Tuple[str, str, bool, float]] = []
    fallback_counts: Counter[str] = Counter()

    total = 0
    buffer_hits = 0
    duplicate_cache_tokens = 0
    seen_cache_tokens: set[str] = set()
    for record in _iter_cache_records(cache_root):
        token = str(record.get("sample_token") or Path(str(record.get("path", ""))).stem)
        if not token:
            fallback_counts["missing_token"] += 1
            continue
        if token in seen_cache_tokens:
            duplicate_cache_tokens += 1
        seen_cache_tokens.add(token)
        total += 1

        record_path = buffer_root / f"{token_to_buffer_key(token)}.pkl.xz"
        if not record_path.is_file():
            fallback_counts["missing_buffer_record"] += 1
            continue
        buffer_hits += 1
        tasks.append((token, str(record_path), bool(args.require_above_gt), float(args.min_reward_margin)))

    tokens: list[str] = []
    trajectories: list[torch.Tensor] = []
    rewards: list[float] = []
    gt_rewards: list[float] = []
    candidate_indices: list[int] = []
    sources: list[str] = []
    source_counts: Counter[str] = Counter()

    num_workers = int(args.num_workers)
    if num_workers <= 1:
        results = map(_worker_select_target, tasks)
        for processed, result in enumerate(results, start=1):
            if processed % int(args.progress_interval) == 0:
                print(f"processed_buffer_records={processed}/{len(tasks)}", flush=True)
            status, key, payload = result
            if status == "selected":
                tokens.append(str(key))
                trajectories.append(torch.from_numpy(payload["trajectory"]).float())
                rewards.append(float(payload["reward"]))
                gt_rewards.append(float(payload["gt_reward"]))
                candidate_indices.append(int(payload["candidate_index"]))
                sources.append(str(payload["source"]))
                source_counts[str(payload["source"])] += 1
            else:
                fallback_counts[str(key)] += 1
    else:
        with Pool(processes=num_workers) as pool:
            results = pool.imap(_worker_select_target, tasks, chunksize=int(args.chunksize))
            for processed, result in enumerate(results, start=1):
                if processed % int(args.progress_interval) == 0:
                    print(f"processed_buffer_records={processed}/{len(tasks)}", flush=True)
                status, key, payload = result
                if status == "selected":
                    tokens.append(str(key))
                    trajectories.append(torch.from_numpy(payload["trajectory"]).float())
                    rewards.append(float(payload["reward"]))
                    gt_rewards.append(float(payload["gt_reward"]))
                    candidate_indices.append(int(payload["candidate_index"]))
                    sources.append(str(payload["source"]))
                    source_counts[str(payload["source"])] += 1
                else:
                    fallback_counts[str(key)] += 1

    if trajectories:
        trajectory_tensor = torch.stack(trajectories, dim=0).contiguous()
    else:
        trajectory_tensor = torch.empty((0, 8, 3), dtype=torch.float32)
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    gt_reward_tensor = torch.tensor(gt_rewards, dtype=torch.float32)
    candidate_index_tensor = torch.tensor(candidate_indices, dtype=torch.int64)
    reward_delta = reward_tensor - gt_reward_tensor if reward_tensor.numel() else torch.empty(0)

    summary = {
        "version": 1,
        "cache_root": str(cache_root),
        "elite_buffer_root": str(buffer_root),
        "selection_mode": "best_valid_reward",
        "require_above_gt": bool(args.require_above_gt),
        "min_reward_margin": float(args.min_reward_margin),
        "cache_record_count": total,
        "unique_cache_token_count": len(seen_cache_tokens),
        "duplicate_cache_token_count": duplicate_cache_tokens,
        "buffer_record_hit_count": buffer_hits,
        "selected_target_count": len(tokens),
        "fallback_count": total - len(tokens),
        "fallback_reasons": dict(sorted(fallback_counts.items())),
        "buffer_coverage": buffer_hits / max(1, total),
        "selected_target_coverage": len(tokens) / max(1, total),
        "selected_source_counts": dict(sorted(source_counts.items())),
        "reward_delta_mean": float(reward_delta.mean().item()) if reward_delta.numel() else None,
        "reward_delta_min": float(reward_delta.min().item()) if reward_delta.numel() else None,
        "reward_delta_max": float(reward_delta.max().item()) if reward_delta.numel() else None,
    }
    return {
        **summary,
        "tokens": tokens,
        "trajectories": trajectory_tensor,
        "rewards": reward_tensor,
        "gt_rewards": gt_reward_tensor,
        "candidate_indices": candidate_index_tensor,
        "sources": sources,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact Stage2 target index from an AWAC elite buffer.")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--elite-buffer-root", required=True)
    parser.add_argument("--output-pt", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--min-reward-margin", type=float, default=0.0)
    parser.add_argument("--require-above-gt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=128)
    parser.add_argument("--progress-interval", type=int, default=5000)
    args = parser.parse_args()

    payload = build_index(args)
    output_pt = Path(args.output_pt)
    output_json = Path(args.output_summary_json)
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_pt)
    output_json.write_text(json.dumps(payload["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
