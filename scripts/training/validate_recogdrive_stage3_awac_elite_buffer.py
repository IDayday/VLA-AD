from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

from navsim.agents.recogdrive.offline_rl_buffer import REQUIRED_COMPONENT_KEYS


def _assert_close(token: str, field: str, observed: Any, expected: float, *, atol: float = 1e-5) -> None:
    observed_f = float(observed)
    if not np.isclose(observed_f, float(expected), rtol=0.0, atol=atol):
        raise ValueError(
            f"token={token} {field} mismatch: record={observed_f:.8f}, expected={float(expected):.8f}."
        )


def _load_records(buffer_dir: Path) -> Iterable[Dict[str, Any]]:
    import lzma
    import pickle

    for path in sorted(buffer_dir.glob("*.pkl.xz")):
        with lzma.open(path, "rb") as f:
            record = pickle.load(f)
        if not isinstance(record, dict):
            raise TypeError(f"Record must be a dict: {path}")
        yield record


def _validate_record(record: Dict[str, Any], *, strict_v2: bool) -> Dict[str, Any]:
    token = str(record.get("token", ""))
    candidates = np.asarray(record.get("candidates"))
    rewards = np.asarray(record.get("rewards"))
    anchor_distance = np.asarray(record.get("anchor_distance"))
    sources = list(record.get("sources", []))
    version = int(record.get("version", 1))
    if strict_v2 and version < 2:
        raise ValueError(f"token={token} has version={version}; strict v2 requires version >= 2.")
    if candidates.ndim != 3 or candidates.shape[-1] != 3:
        raise ValueError(f"token={token} invalid candidates shape {candidates.shape}.")
    k = candidates.shape[0]
    if k <= 0:
        raise ValueError(f"token={token} has no candidates.")
    if rewards.shape != (k,):
        raise ValueError(f"token={token} rewards shape {rewards.shape} does not match K={k}.")
    if anchor_distance.shape != (k,):
        raise ValueError(f"token={token} anchor_distance shape {anchor_distance.shape} does not match K={k}.")
    if len(sources) != k:
        raise ValueError(f"token={token} sources length {len(sources)} does not match K={k}.")
    components = record.get("components")
    if not isinstance(components, dict):
        raise TypeError(f"token={token} components must be a dict.")
    for key in REQUIRED_COMPONENT_KEYS:
        values = np.asarray(components.get(key))
        if values.shape != (k,):
            raise ValueError(f"token={token} component {key} shape {values.shape} does not match K={k}.")
        if not np.isfinite(values).all():
            raise ValueError(f"token={token} component {key} contains non-finite values.")
    if not np.isfinite(candidates).all() or not np.isfinite(rewards).all() or not np.isfinite(anchor_distance).all():
        raise ValueError(f"token={token} contains non-finite candidate/reward/anchor values.")
    if "valid_mask" not in record:
        if strict_v2:
            raise KeyError(f"token={token} is missing valid_mask.")
        valid_mask = np.zeros((k,), dtype=np.bool_)
    else:
        valid_mask = np.asarray(record["valid_mask"], dtype=np.bool_)
        if valid_mask.shape != (k,):
            raise ValueError(f"token={token} valid_mask shape {valid_mask.shape} does not match K={k}.")
    if "selection_score" not in record:
        if strict_v2:
            raise KeyError(f"token={token} is missing selection_score.")
        selection_score = rewards.astype(np.float32)
    else:
        selection_score = np.asarray(record["selection_score"], dtype=np.float32)
        if selection_score.shape != (k,):
            raise ValueError(f"token={token} selection_score shape {selection_score.shape} does not match K={k}.")
        if not np.isfinite(selection_score).all():
            raise ValueError(f"token={token} selection_score contains non-finite values.")

    if strict_v2:
        required_best_fields = (
            "best_raw_reward",
            "best_valid_reward",
            "has_valid_candidate",
            "best_valid_source",
        )
        missing_best_fields = [field for field in required_best_fields if field not in record]
        if missing_best_fields:
            raise KeyError(f"token={token} is missing v2 best-consistency fields: {missing_best_fields}.")

    raw_idx = int(np.argmax(rewards))
    expected_best_raw_reward = float(rewards[raw_idx])
    expected_has_valid_candidate = bool(valid_mask.any())
    if expected_has_valid_candidate:
        valid_indices = np.flatnonzero(valid_mask)
        best_valid_idx = int(valid_indices[int(np.argmax(rewards[valid_indices]))])
    else:
        best_valid_idx = raw_idx
    expected_best_valid_reward = float(rewards[best_valid_idx])
    expected_best_valid_source = str(sources[best_valid_idx])

    if "best_raw_reward" in record:
        _assert_close(token, "best_raw_reward", record["best_raw_reward"], expected_best_raw_reward)
    if "best_valid_reward" in record:
        _assert_close(token, "best_valid_reward", record["best_valid_reward"], expected_best_valid_reward)
    if "has_valid_candidate" in record and bool(record["has_valid_candidate"]) != expected_has_valid_candidate:
        raise ValueError(
            f"token={token} has_valid_candidate mismatch: record={bool(record['has_valid_candidate'])}, "
            f"expected={expected_has_valid_candidate}."
        )
    if "best_valid_source" in record and str(record["best_valid_source"]) != expected_best_valid_source:
        raise ValueError(
            f"token={token} best_valid_source mismatch: record={record['best_valid_source']!r}, "
            f"expected={expected_best_valid_source!r}."
        )

    return {
        "token": token,
        "version": version,
        "num_candidates": k,
        "valid_count": int(valid_mask.sum()),
        "valid_ratio": float(valid_mask.mean()),
        "has_valid_candidate": (
            expected_has_valid_candidate if "has_valid_candidate" not in record else bool(record["has_valid_candidate"])
        ),
        "gt_reward": float(record.get("gt_reward", np.nan)),
        "il_reward": float(record.get("il_reward", np.nan)),
        "best_raw_reward": float(record.get("best_raw_reward", expected_best_raw_reward)),
        "best_valid_reward": float(record.get("best_valid_reward", expected_best_valid_reward)),
        "best_valid_source": str(record.get("best_valid_source", expected_best_valid_source)),
    }


def validate_buffer(buffer_dir: Path, *, strict_v2: bool, allow_low_valid_ratio: bool) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    records = list(_load_records(buffer_dir))
    if not records:
        raise ValueError(f"No *.pkl.xz elite buffer records found under {buffer_dir}.")
    rows = [_validate_record(record, strict_v2=strict_v2) for record in records]
    valid_total = sum(row["valid_count"] for row in rows)
    candidate_total = sum(row["num_candidates"] for row in rows)
    valid_ratio = valid_total / max(1, candidate_total)
    if valid_ratio < 0.01 and not allow_low_valid_ratio:
        raise ValueError(
            f"Elite buffer valid_candidate_ratio is extremely low ({valid_ratio:.6f}). "
            "Use --allow-low-valid-ratio only for debugging."
        )
    source_counts = Counter(row["best_valid_source"] for row in rows)
    n = len(rows)
    summary = {
        "num_records": n,
        "mean_candidates_per_record": candidate_total / max(1, n),
        "valid_candidate_ratio": valid_ratio,
        "has_valid_candidate_ratio": sum(float(row["has_valid_candidate"]) for row in rows) / max(1, n),
        "mean_gt_reward": float(np.nanmean([row["gt_reward"] for row in rows])),
        "mean_il_reward": float(np.nanmean([row["il_reward"] for row in rows])),
        "mean_best_raw_reward": float(np.nanmean([row["best_raw_reward"] for row in rows])),
        "mean_best_valid_reward": float(np.nanmean([row["best_valid_reward"] for row in rows])),
        "pct_best_valid_above_gt": sum(float(row["best_valid_reward"] > row["gt_reward"]) for row in rows) / max(1, n),
        "pct_best_valid_above_il": sum(float(row["best_valid_reward"] > row["il_reward"]) for row in rows) / max(1, n),
        "best_valid_source_distribution": dict(source_counts),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer-dir", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--strict-v2", action="store_true")
    parser.add_argument("--allow-low-valid-ratio", action="store_true")
    args = parser.parse_args()

    rows, summary = validate_buffer(
        Path(args.buffer_dir),
        strict_v2=bool(args.strict_v2),
        allow_low_valid_ratio=bool(args.allow_low_valid_ratio),
    )
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with Path(args.summary_csv).open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "token",
                "version",
                "num_candidates",
                "valid_count",
                "valid_ratio",
                "has_valid_candidate",
                "gt_reward",
                "il_reward",
                "best_raw_reward",
                "best_valid_reward",
                "best_valid_source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
