#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import iter_index, load_sample, write_json  # noqa: E402


TRAIN_TEACHER_KEYS = ("jepa_dynamic_teacher_tokens", "vggt_feature23_tokens")


def iter_cache_records(root: Path, max_records: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    count = 0
    for row in iter_index(root):
        yield load_sample(Path(row["path"]))
        count += 1
        if max_records is not None and count >= max_records:
            return


def _finite_tensor(value: Any, key: str, errors: List[str]) -> Optional[torch.Tensor]:
    if not isinstance(value, torch.Tensor):
        errors.append(f"{key}:not_tensor")
        return None
    value = value.detach().float().cpu()
    if not torch.isfinite(value).all():
        errors.append(f"{key}:non_finite")
    return value


def audit_record(record: Dict[str, Any], *, split: str, vlm_hidden_dim: int, allow_eval_teacher_targets: bool) -> List[str]:
    errors: List[str] = []
    last_hidden = _finite_tensor(record.get("last_hidden_state"), "last_hidden_state", errors)
    if last_hidden is not None and (last_hidden.ndim != 2 or last_hidden.shape[-1] != vlm_hidden_dim):
        errors.append(f"last_hidden_state:bad_shape:{tuple(last_hidden.shape)}")
    h_dyn = _finite_tensor(record.get("two_expert_h_dyn"), "two_expert_h_dyn", errors)
    if h_dyn is not None and tuple(h_dyn.shape) != (3, 12, vlm_hidden_dim):
        errors.append(f"two_expert_h_dyn:bad_shape:{tuple(h_dyn.shape)}")
    h_geo = _finite_tensor(record.get("two_expert_h_geo"), "two_expert_h_geo", errors)
    if h_geo is not None and tuple(h_geo.shape) != (12, vlm_hidden_dim):
        errors.append(f"two_expert_h_geo:bad_shape:{tuple(h_geo.shape)}")
    metadata = record.get("two_expert_metadata")
    if not isinstance(metadata, dict):
        errors.append("two_expert_metadata:missing_or_not_dict")
    elif metadata.get("schema") != "two_expert_slot_hidden_cache_v1":
        errors.append("two_expert_metadata:bad_schema")
    if split in {"eval", "navtest", "val"} and not allow_eval_teacher_targets:
        leaked = [key for key in TRAIN_TEACHER_KEYS if key in record]
        if leaked:
            errors.append(f"eval_teacher_targets_present:{','.join(leaked)}")
    return errors


def audit_cache(
    root: Path,
    *,
    split: str,
    vlm_hidden_dim: int = 1536,
    allow_eval_teacher_targets: bool = False,
    max_records: Optional[int] = None,
) -> Dict[str, Any]:
    total = 0
    failed = 0
    duplicate_tokens: List[str] = []
    seen_tokens = set()
    failures: List[Dict[str, Any]] = []
    teacher_coverage = {key: 0 for key in TRAIN_TEACHER_KEYS}
    for record in iter_cache_records(root, max_records=max_records):
        total += 1
        token = str(record.get("sample_token") or f"record_{total:08d}")
        if token in seen_tokens:
            duplicate_tokens.append(token)
        seen_tokens.add(token)
        for key in TRAIN_TEACHER_KEYS:
            teacher_coverage[key] += int(key in record)
        errors = audit_record(
            record,
            split=split,
            vlm_hidden_dim=vlm_hidden_dim,
            allow_eval_teacher_targets=allow_eval_teacher_targets,
        )
        if errors:
            failed += 1
            if len(failures) < 20:
                failures.append({"sample_token": token, "errors": errors})
    coverage = {
        key: (float(count) / float(total) if total else 0.0)
        for key, count in teacher_coverage.items()
    }
    return {
        "total_records": total,
        "failed_records": failed,
        "ok": failed == 0 and not duplicate_tokens,
        "duplicate_sample_tokens": duplicate_tokens[:20],
        "teacher_coverage": coverage,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit two_expert_slot hidden cache schema.")
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("--split", choices=("train", "navtrain", "val", "navtest", "eval"), default="navtest")
    parser.add_argument("--vlm-hidden-dim", type=int, default=1536)
    parser.add_argument("--allow-eval-teacher-targets", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_cache(
        args.cache_root,
        split=args.split,
        vlm_hidden_dim=int(args.vlm_hidden_dim),
        allow_eval_teacher_targets=bool(args.allow_eval_teacher_targets),
        max_records=args.max_records,
    )
    if args.json_out is not None:
        write_json(args.json_out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
