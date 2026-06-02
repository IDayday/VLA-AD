from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import WeightedRandomSampler


DEFAULT_FAILURE_WEIGHTS = {
    "zero_score": 4.0,
    "dac_zero": 3.0,
    "nc_zero": 3.0,
    "ttc_zero": 2.0,
    "low_pdms_below_0_5": 2.0,
    "normal": 1.0,
}


def failure_index_metadata_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_suffix(".metadata.json")


def load_failure_index_metadata(path: str | Path) -> Dict[str, Any]:
    meta_path = failure_index_metadata_path(path)
    if meta_path.is_file():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def validate_failure_index_for_training(path: str | Path) -> Dict[str, Any]:
    metadata = load_failure_index_metadata(path)
    split = str(metadata.get("split", "")).lower()
    allowed = metadata.get("allowed_for_training")
    if split in {"navtest", "test", "val", "validation"} or allowed is False:
        raise RuntimeError(
            f"Refusing to use failure index for training because split={split!r} "
            f"allowed_for_training={allowed!r}: {path}"
        )
    if not metadata:
        raise RuntimeError(
            f"Failure index is missing metadata sidecar {failure_index_metadata_path(path)}. "
            "Rebuild it with scripts/build_bit_failure_index.py and an explicit training split."
        )
    return metadata


def _tag_allowed(
    tags: Sequence[str],
    include_tags: Optional[Sequence[str]],
    exclude_tags: Optional[Sequence[str]],
) -> bool:
    tag_set = {str(tag) for tag in tags}
    if include_tags and tag_set.isdisjoint({str(tag) for tag in include_tags}):
        return False
    if exclude_tags and not tag_set.isdisjoint({str(tag) for tag in exclude_tags}):
        return False
    return True


def load_failure_index(
    path: str | Path,
    *,
    include_tags: Optional[Sequence[str]] = None,
    exclude_tags: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"BiT failure index not found: {path}")
    records: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "__metadata__" in record:
                continue
            tags = [str(tag) for tag in (record.get("failure_tags") or ["normal"])]
            if not _tag_allowed(tags, include_tags, exclude_tags):
                continue
            for key in ("sample_token", "scene_token"):
                token = record.get(key)
                if token:
                    records[str(token)] = record
    return records


def sample_failure_weight(
    sample: Dict[str, Any],
    failure_index: Dict[str, Dict[str, Any]],
    failure_weights: Optional[Dict[str, float]] = None,
) -> float:
    weights = {**DEFAULT_FAILURE_WEIGHTS, **(failure_weights or {})}
    record = None
    for key in ("sample_token", "scene_token"):
        token = sample.get(key)
        if token is not None and str(token) in failure_index:
            record = failure_index[str(token)]
            break
    if record is None:
        return float(weights["normal"])
    if "sample_weight" in record:
        try:
            return max(float(record["sample_weight"]), 0.0)
        except (TypeError, ValueError):
            pass
    tags = record.get("failure_tags") or ["normal"]
    return max(float(weights.get(str(tag), weights["normal"])) for tag in tags)


def weights_for_records(
    records: Iterable[Dict[str, Any]],
    failure_index: Dict[str, Dict[str, Any]],
    failure_weights: Optional[Dict[str, float]] = None,
) -> List[float]:
    return [sample_failure_weight(record, failure_index, failure_weights) for record in records]


def build_weighted_sampler(
    records: Iterable[Dict[str, Any]],
    failure_index_path: str | Path,
    *,
    failure_weights: Optional[Dict[str, float]] = None,
    include_tags: Optional[Sequence[str]] = None,
    exclude_tags: Optional[Sequence[str]] = None,
    replacement: bool = True,
) -> WeightedRandomSampler:
    materialized = list(records)
    if not materialized:
        raise ValueError("Cannot build a failure-focused sampler for an empty record list.")
    failure_index = load_failure_index(failure_index_path, include_tags=include_tags, exclude_tags=exclude_tags)
    weights = torch.as_tensor(weights_for_records(materialized, failure_index, failure_weights), dtype=torch.double)
    if not torch.isfinite(weights).all() or float(weights.sum().item()) <= 0.0:
        raise ValueError("Failure sampler weights must be finite and have positive sum.")
    return WeightedRandomSampler(weights=weights, num_samples=len(materialized), replacement=replacement)


def matching_failure_record(
    sample: Dict[str, Any],
    failure_index: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    for key in ("sample_token", "scene_token"):
        token = sample.get(key)
        if token is not None and str(token) in failure_index:
            return failure_index[str(token)]
    return None


def is_failure_sample(sample: Dict[str, Any], failure_index: Dict[str, Dict[str, Any]]) -> bool:
    record = matching_failure_record(sample, failure_index)
    if record is None:
        return False
    tags = {str(tag) for tag in (record.get("failure_tags") or [])}
    return bool(tags and tags != {"normal"})


def summarize_sampling_records(
    records: Iterable[Dict[str, Any]],
    failure_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    materialized = list(records)
    normal = 0
    failure = 0
    tags: Counter[str] = Counter()
    for record in materialized:
        failure_record = matching_failure_record(record, failure_index)
        if failure_record is None:
            normal += 1
            tags["normal"] += 1
            continue
        failure_tags = [str(tag) for tag in (failure_record.get("failure_tags") or ["normal"])]
        if set(failure_tags) == {"normal"}:
            normal += 1
        else:
            failure += 1
        tags.update(failure_tags)
    return {
        "total_records": len(materialized),
        "normal_sample_count": normal,
        "failure_sample_count": failure,
        "tag_distribution": dict(sorted(tags.items())),
    }


def build_failure_limited_batches(
    records: Iterable[Dict[str, Any]],
    failure_index: Dict[str, Dict[str, Any]],
    *,
    batch_size: int,
    max_failure_fraction: float,
    seed: int = 0,
) -> List[List[int]]:
    materialized = list(records)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not 0.0 <= max_failure_fraction <= 1.0:
        raise ValueError("max_failure_fraction must be in [0, 1].")
    rng = random.Random(seed)
    failure_indices = [idx for idx, record in enumerate(materialized) if is_failure_sample(record, failure_index)]
    failure_set = set(failure_indices)
    normal_indices = [idx for idx, _record in enumerate(materialized) if idx not in failure_set]
    rng.shuffle(failure_indices)
    rng.shuffle(normal_indices)
    if not materialized:
        return []
    max_failures = min(int(math.floor(batch_size * max_failure_fraction)), batch_size)
    num_batches = int(math.ceil(len(materialized) / float(batch_size)))
    batches: List[List[int]] = []
    failure_cursor = 0
    normal_cursor = 0
    for _ in range(num_batches):
        batch: List[int] = []
        fail_slots = min(max_failures, len(failure_indices))
        for _slot in range(fail_slots):
            if not failure_indices:
                break
            batch.append(failure_indices[failure_cursor % len(failure_indices)])
            failure_cursor += 1
        while len(batch) < batch_size and normal_indices:
            batch.append(normal_indices[normal_cursor % len(normal_indices)])
            normal_cursor += 1
        while len(batch) < batch_size and failure_indices and len([idx for idx in batch if idx in failure_indices]) < max_failures:
            batch.append(failure_indices[failure_cursor % len(failure_indices)])
            failure_cursor += 1
        if batch:
            rng.shuffle(batch)
            batches.append(batch)
    return batches


def batch_failure_fraction_report(
    batches: Sequence[Sequence[int]],
    records: Sequence[Dict[str, Any]],
    failure_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    fractions: List[float] = []
    for batch in batches:
        if not batch:
            continue
        failure_count = sum(1 for idx in batch if is_failure_sample(records[idx], failure_index))
        fractions.append(failure_count / float(len(batch)))
    return {
        "num_batches": len(batches),
        "actual_batch_failure_fraction_average": float(sum(fractions) / len(fractions)) if fractions else 0.0,
        "actual_batch_failure_fraction_max": float(max(fractions)) if fractions else 0.0,
    }
