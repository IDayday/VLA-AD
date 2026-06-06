#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import atomic_torch_save, iter_index, load_sample, write_json  # noqa: E402
from navsim.agents.recogdrive.expert_extractors.vjepa2_extractor import VJEPA2Extractor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a V-JEPA2 overlay cache aligned to an existing chunk cache.")
    parser.add_argument("--base-chunk-root", type=Path, required=True)
    parser.add_argument("--chunk-name-pattern", default="navtest_full_chunk_*")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", "--navsim-root", dest="data_root", type=Path, default=Path("/mnt/navsim"))
    parser.add_argument("--split", default="navtest")
    parser.add_argument("--jepa-model-path", type=Path, required=True)
    parser.add_argument("--num-jepa-tokens", type=int, default=128)
    parser.add_argument("--strict-highcap-jepa", action="store_true")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--loader-max-scenes", type=int, default=None)
    parser.add_argument("--strict-coverage", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def _load_chunk_builder_helpers():
    path = REPO_ROOT / "scripts" / "build_recogdrive_chunk_cache.py"
    spec = importlib.util.spec_from_file_location("_recogdrive_chunk_cache_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def chunk_dirs(root: Path, pattern: str) -> List[Path]:
    if (root / "index.jsonl").is_file():
        return [root]
    dirs: List[Path] = []
    seen = set()
    for item in str(pattern).split(","):
        item = item.strip()
        if not item:
            continue
        for path in sorted(root.glob(item)):
            if path.is_dir() and (path / "index.jsonl").is_file() and path not in seen:
                dirs.append(path)
                seen.add(path)
    if not dirs:
        raise FileNotFoundError(f"No chunk directories matching {pattern!r} under {root}")
    return dirs


def resolve_sample_path(chunk_dir: Path, record: Dict[str, Any]) -> Path:
    raw = Path(record["path"])
    if raw.is_file():
        return raw
    return raw if raw.is_absolute() else chunk_dir / raw


def requested_samples(root: Path, pattern: str, max_samples: Optional[int]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen = set()
    for chunk_dir in chunk_dirs(root, pattern):
        for record in iter_index(chunk_dir):
            sample_path = resolve_sample_path(chunk_dir, record)
            token = record.get("sample_token")
            scene = record.get("scene_token")
            log_name = record.get("log_name")
            if token is None or scene is None:
                sample = load_sample(sample_path)
                token = token or sample.get("sample_token") or sample_path.stem
                scene = scene or sample.get("scene_token") or ""
                log_name = log_name or sample.get("log_name")
            token = str(token)
            if token in seen:
                raise ValueError(f"Duplicate sample_token in base cache: {token}")
            seen.add(token)
            row = {"sample_token": token, "scene_token": str(scene or ""), "source_chunk": chunk_dir.name}
            if log_name:
                row["log_name"] = str(log_name)
            rows.append(row)
            if max_samples is not None and len(rows) >= max_samples:
                return rows
    return rows


def build_loader(args: argparse.Namespace):
    helpers = _load_chunk_builder_helpers()
    data_root, openscene, maps = helpers.autodetect_data_paths(args.data_root)
    logs, blobs = helpers.split_paths(openscene, args.split)
    scene_filter, scene_filter_path = helpers.load_scene_filter(args.split, args.loader_max_scenes)
    from navsim.common.dataloader import SceneLoader

    loader = SceneLoader(
        data_path=logs,
        sensor_blobs_path=blobs,
        scene_filter=scene_filter,
        sensor_config=helpers.cam_f0_sensor_config(),
        load_image_path=True,
    )
    metadata = {
        "NAVSIM_DATA_ROOT": str(data_root),
        "OPENSCENE_DATA_ROOT": str(openscene),
        "NUPLAN_MAPS_ROOT": str(maps),
        "navsim_log_path": str(logs),
        "sensor_blobs_path": str(blobs),
        "scene_filter": str(scene_filter_path),
        "loader_tokens": len(loader.tokens),
        "loader_max_scenes": args.loader_max_scenes,
    }
    return helpers, loader, scene_filter, blobs, metadata


def build_frame_records(args: argparse.Namespace, requested: List[Dict[str, str]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    helpers, loader, scene_filter, blobs, metadata = build_loader(args)
    wanted = {row["sample_token"] for row in requested}
    records: Dict[str, Dict[str, Any]] = {}
    missing_images = 0
    short_future = 0
    for loader_token in loader.tokens:
        frames = loader.scene_frames_dicts[loader_token]
        h = int(scene_filter.num_history_frames)
        if len(frames) < h:
            continue
        sample_token = str(frames[h - 1]["token"])
        if sample_token not in wanted or sample_token in records:
            continue
        history_frames = frames[:h]
        future_frames = frames[h:h + 4]
        if len(future_frames) < 4:
            short_future += 1
            continue
        history_paths = [helpers.frame_camera_path(frame, blobs) for frame in history_frames[-4:]]
        future_paths = [helpers.frame_camera_path(frame, blobs) for frame in future_frames[:4]]
        required = [*history_paths, *future_paths]
        absent = [path for path in required if not Path(path).is_file()]
        if absent:
            missing_images += 1
            continue
        records[sample_token] = {
            "sample_token": sample_token,
            "scene_token": str(frames[h - 1].get("scene_token", "")),
            "log_name": str(frames[h - 1].get("log_name", "")),
            "history_cam_f0": history_paths,
            "future_cam_f0": future_paths,
            "loader_token": loader_token,
        }
        if len(records) == len(wanted):
            break
    metadata.update(
        {
            "requested_samples": len(requested),
            "matched_samples": len(records),
            "missing_requested_samples": sorted(wanted - set(records))[:100],
            "missing_requested_count": len(wanted - set(records)),
            "missing_image_records": missing_images,
            "short_future_records": short_future,
        }
    )
    return records, metadata


def validate_jepa_payload(payload: Dict[str, Any], num_tokens: int) -> None:
    for key in ("jepa_context_tokens", "jepa_target_tokens"):
        value = payload.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key} must be a tensor, got {type(value).__name__}.")
        if tuple(value.shape) != (int(num_tokens), 1024):
            raise ValueError(f"{key} expected shape ({num_tokens}, 1024), got {tuple(value.shape)}.")
        if not torch.isfinite(value.float()).all():
            raise ValueError(f"{key} contains non-finite values.")


def main() -> int:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards).")

    requested = requested_samples(args.base_chunk_root, args.chunk_name_pattern, args.max_samples)
    frame_records, loader_metadata = build_frame_records(args, requested)
    if args.strict_coverage and loader_metadata["missing_requested_count"]:
        raise RuntimeError(
            f"Could not map {loader_metadata['missing_requested_count']} requested sample tokens to NAVSIM frames."
        )

    extractor = VJEPA2Extractor(
        args.jepa_model_path,
        device=args.device,
        precision=args.precision,
        num_tokens=int(args.num_jepa_tokens),
        strict_highcap_jepa=bool(args.strict_highcap_jepa),
    )
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    index_records: List[Dict[str, Any]] = []
    errors: List[str] = []
    written = skipped_by_shard = missing = 0

    for idx, row in enumerate(requested):
        if idx % int(args.num_shards) != int(args.shard_index):
            skipped_by_shard += 1
            continue
        token = row["sample_token"]
        record = frame_records.get(token)
        if record is None:
            missing += 1
            errors.append(f"{token}:missing_frame_record")
            continue
        try:
            payload = {
                "sample_token": token,
                "scene_token": row.get("scene_token") or record.get("scene_token", ""),
                "jepa_context_tokens": extractor.extract_context(record["history_cam_f0"][-4:]),
                "jepa_target_tokens": extractor.extract_target(record["future_cam_f0"][:4]),
                "jepa_tokenizer_metadata": dict(getattr(extractor, "last_tokenizer_metadata", {}) or {}),
                "jepa_num_tokens": torch.tensor(int(args.num_jepa_tokens), dtype=torch.int64),
            }
            if row.get("log_name") or record.get("log_name"):
                payload["log_name"] = row.get("log_name") or record.get("log_name")
            validate_jepa_payload(payload, int(args.num_jepa_tokens))
            out_path = samples_dir / f"{token}.pt"
            atomic_torch_save(payload, out_path)
            index_record = {
                "sample_token": token,
                "scene_token": str(payload.get("scene_token", "")),
                "path": str(out_path.relative_to(args.output_dir)),
            }
            if payload.get("log_name"):
                index_record["log_name"] = str(payload["log_name"])
            index_records.append(index_record)
            written += 1
        except Exception as exc:
            errors.append(f"{token}:{type(exc).__name__}:{exc}")
            if args.strict_coverage:
                raise
        if args.log_every and (written + missing) % int(args.log_every) == 0:
            print(
                f"shard={args.shard_index}/{args.num_shards} processed={written + missing} "
                f"written={written} missing={missing}",
                flush=True,
            )

    with (args.output_dir / "index.jsonl").open("w", encoding="utf-8") as f:
        for item in index_records:
            f.write(json.dumps(item, sort_keys=True) + "\n")

    metadata = {
        "version": "recogdrive_jepa_overlay_from_chunks_v1",
        "source_base_chunk_root": str(args.base_chunk_root),
        "chunk_name_pattern": str(args.chunk_name_pattern),
        "split": str(args.split),
        "jepa_model_path": str(args.jepa_model_path),
        "precision": str(args.precision),
        "strict_highcap_jepa": bool(args.strict_highcap_jepa),
        "jepa_num_tokens": int(args.num_jepa_tokens),
        "contains_jepa": True,
        "token_shapes": {
            "jepa_context_tokens": [int(args.num_jepa_tokens), 1024],
            "jepa_target_tokens": [int(args.num_jepa_tokens), 1024],
        },
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "num_records": int(written),
        "num_written": int(written),
        "num_skipped_by_shard": int(skipped_by_shard),
        "num_missing_in_shard": int(missing),
        "errors": errors[:100],
        **loader_metadata,
    }
    write_json(args.output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    if args.strict_coverage and missing:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
