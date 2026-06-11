#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch


EXPECTED_SHAPES = {
    "last_vla_h_dyn": (64, 1536),
    "last_vla_h_geo": (64, 1536),
    "last_vla_h_plan": (32, 1536),
}
TRAIN_TEACHER_KEYS = ("jepa_target_tokens", "cosmos_future_features", "vggt_geometry_tokens", "vggt_geometry_target_tokens")


def _is_tensor_like(value: Any) -> bool:
    return isinstance(value, (torch.Tensor, np.ndarray))


def _shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if isinstance(value, np.ndarray):
        return tuple(value.shape)
    raise TypeError(f"Expected tensor-like value, got {type(value).__name__}.")


def _finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value.float()).all().item())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value.astype(np.float32)).all())
    return False


def _load_one(path: Path) -> Any:
    if path.suffix in {".pt", ".pth", ".ckpt"}:
        return torch.load(path, map_location="cpu")
    if path.suffix in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            return pickle.load(f)
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return {key: data[key] for key in data.files}
    raise ValueError(f"Unsupported cache file suffix: {path}")


def iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    if path.is_file():
        loaded = _load_one(path)
        if isinstance(loaded, dict) and "records" in loaded and isinstance(loaded["records"], list):
            yield from loaded["records"]
        elif isinstance(loaded, list):
            yield from loaded
        elif isinstance(loaded, dict):
            yield loaded
        else:
            raise TypeError(f"Unsupported cache object in {path}: {type(loaded).__name__}")
        return
    for suffix in ("*.pt", "*.pth", "*.pkl", "*.pickle", "*.npz"):
        for item in sorted(path.rglob(suffix)):
            yield from iter_records(item)


def audit_record(
    record: Dict[str, Any],
    *,
    split: str,
    require_train_teachers: bool,
    seen_tokens: set[str],
) -> List[str]:
    errors: List[str] = []
    sample_token = str(record.get("sample_token", ""))
    if not sample_token:
        errors.append("missing sample_token")
    elif sample_token in seen_tokens:
        errors.append(f"duplicate sample_token={sample_token}")
    else:
        seen_tokens.add(sample_token)

    for key, expected in EXPECTED_SHAPES.items():
        if key not in record:
            errors.append(f"missing {key}")
            continue
        value = record[key]
        if not _is_tensor_like(value):
            errors.append(f"{key} is not tensor-like")
            continue
        if _shape(value) != expected:
            errors.append(f"{key} shape {_shape(value)} != {expected}")
        if not _finite(value):
            errors.append(f"{key} contains non-finite values")

    if "last_hidden_state" in record:
        value = record["last_hidden_state"]
        if not _is_tensor_like(value) or len(_shape(value)) != 2 or _shape(value)[-1] != 1536:
            errors.append("last_hidden_state must have shape [N_total, 1536]")
        elif not _finite(value):
            errors.append("last_hidden_state contains non-finite values")

    metadata = record.get("last_vla_slot_metadata")
    if metadata is None:
        errors.append("missing last_vla_slot_metadata")
    elif isinstance(metadata, dict):
        for name, count in (("dyn", 64), ("geo", 64), ("plan", 32)):
            key = f"{name}_count"
            if int(metadata.get(key, count)) != count:
                errors.append(f"slot metadata {key}={metadata.get(key)} != {count}")
    else:
        errors.append("last_vla_slot_metadata must be a dict")

    if split == "train" and require_train_teachers:
        if not any(key in record for key in TRAIN_TEACHER_KEYS):
            errors.append("train record has no teacher targets")
    if split != "train":
        forbidden = [key for key in TRAIN_TEACHER_KEYS if key in record]
        if forbidden and not bool(record.get("allow_eval_teacher_targets", False)):
            errors.append(f"eval record contains train-only teacher targets: {forbidden}")

    return errors


def audit_records(
    records: Iterable[Dict[str, Any]],
    *,
    split: str,
    require_train_teachers: bool = True,
    max_records: Optional[int] = None,
) -> Dict[str, Any]:
    seen_tokens: set[str] = set()
    failures: List[Dict[str, Any]] = []
    total = 0
    for record in records:
        total += 1
        errors = audit_record(record, split=split, require_train_teachers=require_train_teachers, seen_tokens=seen_tokens)
        if errors:
            failures.append({"sample_token": str(record.get("sample_token", "")), "errors": errors})
        if max_records is not None and total >= max_records:
            break
    return {
        "total_records": total,
        "failed_records": len(failures),
        "ok": len(failures) == 0,
        "failures": failures[:50],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit strict LaST-VLA latent cache schema.")
    parser.add_argument("cache_path", type=Path)
    parser.add_argument("--split", choices=["train", "val", "navtest", "eval"], default="train")
    parser.add_argument("--allow-missing-train-teachers", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    report = audit_records(
        iter_records(args.cache_path),
        split=args.split,
        require_train_teachers=not args.allow_missing_train_teachers,
        max_records=args.max_records,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
