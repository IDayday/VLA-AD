#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch


TRAIN_ONLY_TEACHER_KEYS = {
    "jepa_target_tokens",
    "cosmos_future_features",
    "vggt_geometry_tokens",
    "vggt_geometry_target_tokens",
}


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _save_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)


def strip_train_only_teachers(record: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in record.items() if key not in TRAIN_ONLY_TEACHER_KEYS}


def strict_cache_record_from_outputs(
    *,
    sample_token: str,
    last_hidden_state: torch.Tensor,
    h_dyn: torch.Tensor,
    h_geo: torch.Tensor,
    h_plan: torch.Tensor,
    base_record: Dict[str, Any],
    include_teacher_targets: bool,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "sample_token": sample_token,
        "last_hidden_state": last_hidden_state.detach().cpu(),
        "last_vla_h_dyn": h_dyn.detach().cpu(),
        "last_vla_h_geo": h_geo.detach().cpu(),
        "last_vla_h_plan": h_plan.detach().cpu(),
        "last_vla_slot_metadata": {
            "dyn_count": int(h_dyn.shape[0]),
            "geo_count": int(h_geo.shape[0]),
            "plan_count": int(h_plan.shape[0]),
            "vlm_hidden_dim": int(h_dyn.shape[-1]),
            "strict_last_vla_schema": "latent_slots_v1",
        },
    }
    for key in ("trajectory", "history_trajectory", "high_command_one_hot", "status_feature"):
        if key in base_record:
            record[key] = base_record[key]
    if include_teacher_targets:
        for key in TRAIN_ONLY_TEACHER_KEYS:
            if key in base_record:
                record[key] = base_record[key]
    return record


def iter_base_records(base_chunk_root: Path) -> Iterable[tuple[Path, Dict[str, Any]]]:
    suffixes = ("*.pkl", "*.pickle")
    for suffix in suffixes:
        for path in sorted(base_chunk_root.rglob(suffix)):
            loaded = _load_pickle(path)
            if isinstance(loaded, dict) and "records" in loaded and isinstance(loaded["records"], list):
                for record in loaded["records"]:
                    yield path, record
            elif isinstance(loaded, list):
                for record in loaded:
                    yield path, record
            elif isinstance(loaded, dict):
                yield path, loaded
            else:
                raise TypeError(f"Unsupported base cache object in {path}: {type(loaded).__name__}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build strict LaST-VLA latent-slot cache. This entrypoint is gated by RUN_CACHE=1 "
            "and expects a Stage1 strict slot/adapters checkpoint plus NAVSIM image inputs."
        )
    )
    parser.add_argument("--navsim-log-path", type=Path, default=None)
    parser.add_argument("--sensor-blobs-path", type=Path, default=None)
    parser.add_argument("--vlm-path", type=Path, default=None)
    parser.add_argument("--stage1-checkpoint", type=Path, default=None)
    parser.add_argument("--base-chunk-root", type=Path, default=None)
    parser.add_argument("--teacher-cache-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "navtest", "eval"], default="train")
    parser.add_argument("--include-teacher-targets", action="store_true")
    parser.add_argument("--allow-eval-teacher-targets", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    return parser


def _write_manifest(output_root: Path, payload: Dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "strict_last_vla_latent_cache_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def main() -> int:
    args = build_parser().parse_args()
    if args.split != "train" and args.include_teacher_targets and not args.allow_eval_teacher_targets:
        raise ValueError("Eval/navtest strict cache must not include future teacher targets unless --allow-eval-teacher-targets is set.")

    run_cache = os.getenv("RUN_CACHE", "0") == "1"
    if args.dry_run or not run_cache:
        _write_manifest(
            args.output_root,
            {
                "status": "dry_run" if args.dry_run else "blocked_by_RUN_CACHE_gate",
                "run_cache_env": os.getenv("RUN_CACHE", "0"),
                "split": args.split,
                "include_teacher_targets": bool(args.include_teacher_targets),
                "message": "Set RUN_CACHE=1 to run VLM latent-slot cache generation.",
            },
        )
        print(f"Strict latent cache generation not launched. Manifest written to {args.output_root}.")
        return 0

    required = {
        "navsim_log_path": args.navsim_log_path,
        "sensor_blobs_path": args.sensor_blobs_path,
        "vlm_path": args.vlm_path,
        "stage1_checkpoint": args.stage1_checkpoint,
    }
    missing = [name for name, value in required.items() if value is None or not Path(value).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs for strict latent cache generation: {missing}")

    # The actual online VLM pass is intentionally kept behind this narrow
    # entrypoint so launchers cannot accidentally run it. The project-specific
    # dataloader wiring should call ReCogDriveBackbone.forward_last_vla_latent_slots
    # and strict_cache_record_from_outputs for each sample.
    if args.base_chunk_root is not None and args.base_chunk_root.exists():
        count = 0
        preview: List[str] = []
        for _, record in iter_base_records(args.base_chunk_root):
            token = str(record.get("sample_token", f"record_{count:08d}"))
            preview.append(token)
            count += 1
            if args.max_records is not None and count >= args.max_records:
                break
        _write_manifest(
            args.output_root,
            {
                "status": "preflight_ready",
                "split": args.split,
                "base_records_seen": count,
                "preview_tokens": preview[:10],
                "stage1_checkpoint": str(args.stage1_checkpoint),
                "vlm_path": str(args.vlm_path),
                "schema": "strict_last_vla_latent_slots_v1",
            },
        )
        print("Strict latent cache preflight completed. Integrate online VLM loop to materialize chunks.")
        return 0

    raise NotImplementedError(
        "Strict latent cache generation requires the project dataloader loop over NAVSIM logs/sensor blobs. "
        "Use ReCogDriveBackbone.forward_last_vla_latent_slots and strict_cache_record_from_outputs; "
        "this script currently performs the gated preflight and schema-safe record writing helpers."
    )


if __name__ == "__main__":
    raise SystemExit(main())
