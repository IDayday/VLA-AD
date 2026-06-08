#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import atomic_torch_save, write_json  # noqa: E402
from scripts.build_recogdrive_chunk_cache import (  # noqa: E402
    frame_camera_path,
    get_cached_loader,
    normalize_path_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build exact high-cap JEPA overlay samples missing from base train chunks.")
    parser.add_argument("--base-chunk-root", type=Path, required=True)
    parser.add_argument("--chunk-name-pattern", default="train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*")
    parser.add_argument("--existing-jepa-cache-root", type=Path, required=True)
    parser.add_argument("--output-cache-root", type=Path, required=True)
    parser.add_argument("--output-prefix", default="missing_highcap_jepa_shard")
    parser.add_argument("--exclude-existing-prefix", action="append", default=None)
    parser.add_argument("--require-existing-sample-files", action="store_true")
    parser.add_argument("--navsim-root", type=Path, default=Path("/mnt/navsim"))
    parser.add_argument("--split", choices=("navtrain", "navtest"), default="navtrain")
    parser.add_argument("--jepa-model-path", type=Path, required=True)
    parser.add_argument("--num-jepa-tokens", type=int, default=128)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--loader-max-scenes", type=int, default=200000)
    parser.add_argument("--resolve-loader-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def chunk_dirs(root: Path, pattern: str) -> List[Path]:
    dirs: List[Path] = []
    seen: Set[Path] = set()
    for item in str(pattern).split(","):
        item = item.strip()
        if not item:
            continue
        for path in sorted(root.glob(item)):
            if path.is_dir() and (path / "index.jsonl").is_file() and path not in seen:
                dirs.append(path)
                seen.add(path)
    if not dirs:
        raise FileNotFoundError(f"No indexed chunks matching {pattern!r} under {root}")
    return dirs


def iter_index(index_path: Path) -> Iterable[Dict[str, Any]]:
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def token_from_record(record: Dict[str, Any]) -> str:
    return str(record.get("sample_token") or Path(record["path"]).stem)


def load_base_records(root: Path, pattern: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for chunk_dir in chunk_dirs(root, pattern):
        for record in iter_index(chunk_dir / "index.jsonl"):
            token = token_from_record(record)
            if token in seen:
                raise ValueError(f"Duplicate base sample_token {token} in {chunk_dir}")
            seen.add(token)
            records.append(
                {
                    "sample_token": token,
                    "scene_token": str(record.get("scene_token") or ""),
                    "log_name": record.get("log_name"),
                    "source_chunk": chunk_dir.name,
                }
            )
    return records


def indexed_dirs(root: Path) -> List[Path]:
    if (root / "index.jsonl").is_file():
        return [root]
    if not root.is_dir():
        return []
    return sorted(path for path in root.glob("*") if path.is_dir() and (path / "index.jsonl").is_file())


def load_existing_tokens(root: Path, exclude_prefixes: List[str], *, require_sample_files: bool) -> Set[str]:
    tokens: Set[str] = set()
    for chunk_dir in indexed_dirs(root):
        if any(chunk_dir.name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        for record in iter_index(chunk_dir / "index.jsonl"):
            token = token_from_record(record)
            if require_sample_files:
                sample_path = Path(record.get("path", ""))
                if not sample_path.is_absolute():
                    sample_path = chunk_dir / sample_path
                if not sample_path.is_file():
                    continue
            tokens.add(token)
    return tokens


def validate_jepa_sample(path: Path, expected_tokens: int) -> bool:
    if not path.is_file():
        return False
    payload = torch.load(path, map_location="cpu")
    for key in ("jepa_context_tokens", "jepa_target_tokens"):
        value = payload.get(key)
        if not isinstance(value, torch.Tensor):
            return False
        if tuple(value.shape) != (int(expected_tokens), 1024):
            return False
        if not torch.isfinite(value.float()).all():
            return False
    return True


def build_loader_records(args: argparse.Namespace, assigned: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    assigned_by_token = {item["sample_token"]: item for item in assigned}
    builder_args = normalize_path_args(
        argparse.Namespace(
            data_root=args.navsim_root,
            split=args.split,
            chunk_index=0,
            chunk_size=int(args.loader_max_scenes),
            chunk_start=0,
            chunk_stop=None,
            max_samples=int(args.loader_max_scenes),
            build_vlm_hidden=False,
            build_jepa=True,
            build_vggt=False,
            allow_missing_future_frames=False,
            log_every=args.log_every,
        )
    )
    cached = get_cached_loader(builder_args)
    loader = cached["loader"]
    scene_filter = cached["scene_filter"]
    blobs = cached["blobs"]
    h = scene_filter.num_history_frames
    records: Dict[str, Dict[str, Any]] = {}
    for loader_token in loader.tokens:
        frames = loader.scene_frames_dicts[loader_token]
        sample_token = str(frames[h - 1]["token"])
        if sample_token not in assigned_by_token:
            continue
        history_frames = frames[:h]
        future_frames = frames[h : h + 4]
        history_paths = [frame_camera_path(frame, blobs) for frame in history_frames[-4:]]
        future_paths = [frame_camera_path(frame, blobs) for frame in future_frames]
        required_paths = history_paths[-4:] + future_paths[:4]
        missing_paths = [path for path in required_paths if not Path(path).is_file()]
        if len(future_paths) < 4:
            raise RuntimeError(f"Sample {sample_token} has only {len(future_paths)} future frame(s); need 4 for JEPA target.")
        if missing_paths:
            raise FileNotFoundError(f"Sample {sample_token} missing required image(s): {missing_paths[:3]}")
        base_item = assigned_by_token[sample_token]
        records[sample_token] = {
            "sample_token": sample_token,
            "scene_token": str(frames[h - 1].get("scene_token") or base_item.get("scene_token") or ""),
            "log_name": frames[h - 1].get("log_name") or base_item.get("log_name"),
            "history_cam_f0": history_paths,
            "future_cam_f0": future_paths,
            "loader_token": loader_token,
            "source_chunk": base_item.get("source_chunk"),
        }
        if len(records) == len(assigned_by_token):
            break
    missing = sorted(set(assigned_by_token) - set(records))
    if missing:
        raise KeyError(f"{len(missing)} assigned sample_token(s) were not found in NAVSIM loader, first={missing[:10]}")
    return [records[item["sample_token"]] for item in assigned]


def main() -> int:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    exclude_prefixes = list(args.exclude_existing_prefix or [])
    if args.output_prefix not in exclude_prefixes:
        exclude_prefixes.append(args.output_prefix)

    base_records = load_base_records(args.base_chunk_root, args.chunk_name_pattern)
    existing_tokens = load_existing_tokens(
        args.existing_jepa_cache_root,
        exclude_prefixes,
        require_sample_files=bool(args.require_existing_sample_files),
    )
    missing_records = [item for item in base_records if item["sample_token"] not in existing_tokens]
    assigned = [item for idx, item in enumerate(missing_records) if idx % args.num_shards == args.shard_index]
    output_dir = args.output_cache_root / f"{args.output_prefix}_{args.shard_index:02d}_of_{args.num_shards:02d}"
    samples_dir = output_dir / "samples"

    print(
        json.dumps(
            {
                "base_records": len(base_records),
                "existing_tokens_excluding_output_prefix": len(existing_tokens),
                "missing_records": len(missing_records),
                "assigned_records": len(assigned),
                "output_dir": str(output_dir),
                "dry_run": bool(args.dry_run),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if args.dry_run:
        return 0

    records = build_loader_records(args, assigned)
    if args.resolve_loader_only:
        print(f"resolved {len(records)} assigned sample_token(s) from NAVSIM loader", flush=True)
        return 0
    from navsim.agents.recogdrive.expert_extractors.vjepa2_extractor import VJEPA2Extractor

    jepa = VJEPA2Extractor(
        args.jepa_model_path,
        device=args.device,
        precision=args.precision,
        num_tokens=int(args.num_jepa_tokens),
        strict_highcap_jepa=True,
    )
    index_records: List[Dict[str, Any]] = []
    written = 0
    skipped = 0
    samples_dir.mkdir(parents=True, exist_ok=True)
    for idx, record in enumerate(records, start=1):
        out_path = samples_dir / f"{record['sample_token']}.pt"
        if out_path.is_file() and not args.overwrite and validate_jepa_sample(out_path, args.num_jepa_tokens):
            skipped += 1
        else:
            payload = {
                "sample_token": record["sample_token"],
                "scene_token": record["scene_token"],
                "jepa_context_tokens": jepa.extract_context(record["history_cam_f0"][-4:]),
                "jepa_tokenizer_metadata": dict(getattr(jepa, "last_tokenizer_metadata", {}) or {}),
                "jepa_num_tokens": torch.tensor(int(args.num_jepa_tokens), dtype=torch.int64),
                "jepa_target_tokens": jepa.extract_target(record["future_cam_f0"][:4]),
            }
            if not validate_jepa_payload(payload, args.num_jepa_tokens):
                raise RuntimeError(f"Invalid JEPA payload for {record['sample_token']}")
            atomic_torch_save(payload, out_path)
            written += 1
        index_records.append(
            {
                "log_name": record.get("log_name"),
                "path": str(out_path.relative_to(output_dir)),
                "sample_token": record["sample_token"],
                "scene_token": record["scene_token"],
                "source_chunk": record.get("source_chunk"),
            }
        )
        if args.log_every and idx % args.log_every == 0:
            print(f"processed {idx}/{len(records)} written={written} skipped={skipped}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "index.jsonl").open("w", encoding="utf-8") as f:
        for record in index_records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    write_json(
        output_dir / "metadata.json",
        {
            "contains_jepa": True,
            "strict_highcap_jepa": True,
            "jepa_model_path": str(args.jepa_model_path),
            "jepa_num_tokens": int(args.num_jepa_tokens),
            "num_records": len(index_records),
            "num_written": int(written),
            "num_skipped": int(skipped),
            "shard_index": int(args.shard_index),
            "num_shards": int(args.num_shards),
            "token_shapes": {
                "jepa_context_tokens": [int(args.num_jepa_tokens), 1024],
                "jepa_target_tokens": [int(args.num_jepa_tokens), 1024],
            },
            "source_base_chunk_root": str(args.base_chunk_root),
            "source_existing_jepa_cache_root": str(args.existing_jepa_cache_root),
        },
    )
    print(f"wrote {len(index_records)} records to {output_dir}", flush=True)
    return 0


def validate_jepa_payload(payload: Dict[str, Any], expected_tokens: int) -> bool:
    for key in ("jepa_context_tokens", "jepa_target_tokens"):
        value = payload.get(key)
        if not isinstance(value, torch.Tensor):
            return False
        if tuple(value.shape) != (int(expected_tokens), 1024):
            return False
        if not torch.isfinite(value.float()).all():
            return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
