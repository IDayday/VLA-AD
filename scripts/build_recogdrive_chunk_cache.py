#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue as queue_lib
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import (  # noqa: E402
    CHUNK_VERSION,
    atomic_torch_save,
    validate_sample_payload,
    write_json,
)

SCENE_FILTER_DIR = REPO_ROOT / "navsim/planning/script/config/common/train_test_split/scene_filter"
_LOADER_CACHE: Dict[Tuple[str, str, Optional[int]], Dict[str, Any]] = {}
_PATH_ARG_NAMES = (
    "data_root",
    "project_root",
    "output_dir",
    "recogdrive_vlm_path",
    "jepa_model_path",
    "vggt_model_path",
)

def cam_f0_sensor_config():
    from navsim.common.dataclasses import SensorConfig
    return SensorConfig(
        cam_f0=True,
        cam_l0=False,
        cam_l1=False,
        cam_l2=False,
        cam_r0=False,
        cam_r1=False,
        cam_r2=False,
        cam_b0=False,
        lidar_pc=False,
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small ReCogDrive expert-token chunk cache on NAVSIM.")
    parser.add_argument("--data-root", "--navsim-root", dest="data_root", type=Path, default=Path(os.environ.get("NAVSIM_DATA_ROOT", "/mnt/navsim")))
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--split", choices=("navtrain", "navtest", "trainval", "test", "mini", "navmini"), required=True)
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--chunk-start", type=int, default=None, help="Optional absolute start offset in the split. Defaults to chunk_index * chunk_size.")
    parser.add_argument("--chunk-stop", type=int, default=None, help="Optional exclusive raw-token stop. Valid-fill scanning will not cross this boundary.")
    parser.add_argument("--allow-partial-final-chunk", action="store_true", help="Allow fewer than chunk_size valid records for the final chunk.")
    parser.add_argument("--strict-token-window", action="store_true", help="Do not scan beyond [chunk_start, chunk_start + chunk_size); useful for non-overlapping full-split caches.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-vlm-hidden", action="store_true")
    parser.add_argument("--build-jepa", action="store_true")
    parser.add_argument("--build-vggt", action="store_true")
    parser.add_argument("--require-vggt-geometry", action="store_true")
    parser.add_argument("--recogdrive-vlm-path", type=Path, default=None)
    parser.add_argument("--jepa-model-path", type=Path, default=None)
    parser.add_argument("--vggt-model-path", type=Path, default=None)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--workers", type=int, default=None, help="Total chunk-cache worker processes. Defaults to num_gpus * workers_per_gpu for CUDA.")
    parser.add_argument("--workers-per-gpu", type=int, default=1, help="CUDA worker processes per requested GPU when --workers is not set.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Skip samples that already exist; this is the default unless --overwrite is passed.")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--allow-missing-future-frames", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--finalize-existing", action="store_true", help="Write index/metadata from existing real sample .pt files and exit.")
    return parser.parse_args()

def dtype_from_precision(precision: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

def normalize_path_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in _PATH_ARG_NAMES:
        value = getattr(args, name, None)
        if value is not None and not isinstance(value, Path):
            setattr(args, name, Path(value))
    return args

def _optional_env_path(name: str) -> Optional[Path]:
    value = os.environ.get(name)
    return Path(value) if value else None

def autodetect_data_paths(data_root: Path) -> Tuple[Path, Path, Path]:
    candidates_open = [
        _optional_env_path("OPENSCENE_DATA_ROOT"),
        data_root / "openscene",
        data_root / "open_scene",
        data_root,
    ]
    openscene = next((path for path in candidates_open if path is not None and path.exists()), None)
    candidates_maps = [
        _optional_env_path("NUPLAN_MAPS_ROOT"),
        data_root / "maps",
        data_root / "nuplan_maps",
        data_root / "nuplan-maps-v1.0",
    ]
    maps = next((path for path in candidates_maps if path is not None and path.exists()), None)
    if openscene is None:
        raise FileNotFoundError(f"Could not find OPENSCENE_DATA_ROOT under {data_root}. Set OPENSCENE_DATA_ROOT explicitly.")
    if maps is None:
        raise FileNotFoundError(f"Could not find NUPLAN_MAPS_ROOT under {data_root}. Set NUPLAN_MAPS_ROOT explicitly.")
    os.environ["NAVSIM_DATA_ROOT"] = str(data_root)
    os.environ["OPENSCENE_DATA_ROOT"] = str(openscene)
    os.environ["NUPLAN_MAPS_ROOT"] = str(maps)
    # navsim.common.dataclasses reads this env var into a module global at import time.
    # Keep already-imported modules synchronized for scripts that auto-detect paths.
    module = sys.modules.get("navsim.common.dataclasses")
    if module is not None:
        setattr(module, "NUPLAN_MAPS_ROOT", str(maps))
    return data_root, openscene, maps

def _split_aliases(split: str) -> List[str]:
    aliases = {
        "navtrain": ["navtrain", "trainval"],
        "navtest": ["navtest", "test"],
        "navmini": ["navmini", "mini"],
    }
    return aliases.get(split, [split])

def split_paths(openscene: Path, split: str) -> Tuple[Path, Path]:
    aliases = _split_aliases(split)
    log_candidates = []
    blob_candidates = []
    for name in aliases:
        log_candidates.extend([
            openscene / "navsim_logs" / name,
            openscene / name / "navsim_logs",
            openscene / name,
            openscene / f"{name}_navsim_logs" / name,
            openscene / f"{name}_navsim_log" / name,
            openscene / f"{name}_navsim_logs",
            openscene / f"{name}_navsim_log",
        ])
        blob_override = _optional_env_path(f"NAVSIM_{name.upper()}_SENSOR_BLOBS_ROOT")
        if blob_override is not None:
            blob_candidates.append(blob_override)
        if name == "trainval":
            blob_candidates.extend([
                openscene / "trainval_all" / "trainval_sensor_blobs" / "trainval",
                openscene / "trainval_all" / "trainval_sensor_blobs",
            ])
        blob_candidates.extend([
            openscene / "sensor_blobs" / name,
            openscene / name / "sensor_blobs",
            openscene / "sensor_blobs",
            openscene / f"{name}_sensor_blobs" / name,
            openscene / f"{name}_navsim_sensor" / name,
            openscene / f"{name}_sensor_blobs",
            openscene / f"{name}_navsim_sensor",
        ])
    logs = next((path for path in log_candidates if path.exists()), None)
    blobs = next((path for path in blob_candidates if path.exists()), None)
    if logs is None:
        raise FileNotFoundError(f"Could not find NAVSIM log directory for split={split}. Tried: {log_candidates}")
    if blobs is None:
        raise FileNotFoundError(f"Could not find sensor blob directory for split={split}. Tried: {blob_candidates}")
    return logs, blobs

def load_scene_filter(split: str, max_samples: Optional[int]):
    from navsim.common.dataclasses import SceneFilter

    path = SCENE_FILTER_DIR / f"{split}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Scene filter config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw.pop("_target_", None)
    raw.pop("_convert_", None)
    if max_samples is not None:
        raw["max_scenes"] = max_samples
    return SceneFilter(**raw), path

def _sensor_blob_fallback_roots(sensor_blobs_path: Path) -> List[Path]:
    roots: List[Path] = []
    env_value = os.environ.get("NAVSIM_SENSOR_BLOBS_FALLBACK_ROOTS")
    if env_value:
        roots.extend(Path(item) for item in env_value.split(os.pathsep) if item)

    path_text = str(sensor_blobs_path)
    if path_text.endswith("/trainval_all/trainval_sensor_blobs/trainval") and len(sensor_blobs_path.parents) >= 3:
        roots.append(sensor_blobs_path.parents[2] / "trainval_sensor_blobs" / "trainval")
    elif path_text.endswith("/trainval_sensor_blobs/trainval") and "trainval_all" not in sensor_blobs_path.parts:
        roots.append(sensor_blobs_path.parent.parent / "trainval_all" / "trainval_sensor_blobs" / "trainval")

    unique: List[Path] = []
    seen = {str(sensor_blobs_path)}
    for root in roots:
        key = str(root)
        if key not in seen and root.exists():
            unique.append(root)
            seen.add(key)
    return unique

def frame_camera_path(frame: Dict[str, Any], sensor_blobs_path: Path, camera_key: str = "cam_f0") -> str:
    for raw_key, camera in frame["cams"].items():
        if raw_key.lower() == camera_key:
            rel_path = Path(camera["data_path"])
            primary = sensor_blobs_path / rel_path
            if primary.is_file():
                return str(primary)
            for fallback_root in _sensor_blob_fallback_roots(sensor_blobs_path):
                fallback = fallback_root / rel_path
                if fallback.is_file():
                    return str(fallback)
            return str(primary)
    raise KeyError(f"Camera {camera_key} not found. Available: {list(frame['cams'].keys())}")

def chunk_start_value(args: argparse.Namespace) -> int:
    if getattr(args, "chunk_start", None) is not None:
        return int(args.chunk_start)
    return args.chunk_index * args.chunk_size

def effective_loader_max_scenes(args: argparse.Namespace) -> Optional[int]:
    chunk_end = chunk_start_value(args) + args.chunk_size
    # Load a bounded amount of slack so missing sensor blobs can be skipped
    # without scanning the full NAVSIM split.
    needed = chunk_end + max(args.chunk_size, 512)
    if getattr(args, "chunk_stop", None) is not None:
        needed = min(max(needed, chunk_start_value(args)), int(args.chunk_stop))
    if args.max_samples is None:
        return needed
    return min(args.max_samples, needed)

def get_cached_loader(args: argparse.Namespace) -> Dict[str, Any]:
    loader_max_scenes = effective_loader_max_scenes(args)
    key = (str(args.data_root.resolve()), args.split, loader_max_scenes)
    cached = _LOADER_CACHE.get(key)
    if cached is not None:
        return cached
    data_root, openscene, maps = autodetect_data_paths(args.data_root)
    from navsim.common.dataloader import SceneLoader

    logs, blobs = split_paths(openscene, args.split)
    scene_filter, scene_filter_path = load_scene_filter(args.split, loader_max_scenes)
    loader = SceneLoader(
        data_path=logs,
        sensor_blobs_path=blobs,
        scene_filter=scene_filter,
        sensor_config=cam_f0_sensor_config(),
        load_image_path=True,
    )
    cached = {
        "loader": loader,
        "data_root": data_root,
        "openscene": openscene,
        "maps": maps,
        "logs": logs,
        "blobs": blobs,
        "scene_filter": scene_filter,
        "scene_filter_path": scene_filter_path,
    }
    _LOADER_CACHE[key] = cached
    return cached

def missing_required_image_paths(record: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    paths: List[str] = []
    if args.build_vlm_hidden and record["history_cam_f0"]:
        paths.append(record["history_cam_f0"][-1])
    if args.build_jepa:
        paths.extend(record["history_cam_f0"][-4:])
        if not args.allow_missing_future_frames:
            paths.extend(record["future_cam_f0"][:4])
    if args.build_vggt and record["history_cam_f0"]:
        paths.append(record["history_cam_f0"][-1])
    return [item for item in dict.fromkeys(paths) if not Path(item).is_file()]

def build_records(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cached = get_cached_loader(args)
    loader = cached["loader"]
    scene_filter = cached["scene_filter"]
    blobs = cached["blobs"]
    all_tokens = loader.tokens
    start = chunk_start_value(args)
    records: List[Dict[str, Any]] = []
    seen_sample_tokens: set[str] = set()
    scanned = start
    missing_image_records = 0
    duplicate_records = 0
    stop = len(all_tokens) if args.chunk_stop is None else min(len(all_tokens), int(args.chunk_stop))
    if stop < start:
        raise ValueError(f"--chunk-stop ({stop}) must be >= chunk_start ({start})")
    if args.strict_token_window:
        token_window = all_tokens[start:min(start + args.chunk_size, stop)]
    else:
        token_window = all_tokens[start:stop]
    for token in token_window:
        if len(records) >= args.chunk_size:
            break
        scanned += 1
        frames = loader.scene_frames_dicts[token]
        h = scene_filter.num_history_frames
        history_frames = frames[:h]
        future_frames = frames[h:h + 4]
        history_paths = [frame_camera_path(frame, blobs) for frame in history_frames[-4:]]
        future_paths = [frame_camera_path(frame, blobs) for frame in future_frames]
        required_paths = list(history_paths)
        if args.build_jepa:
            if len(future_paths) < 4:
                if args.allow_missing_future_frames:
                    missing_image_records += 1
                    continue
                raise ValueError(f"Missing future frames for {token}; need 4 for JEPA target.")
            required_paths.extend(future_paths[:4])
        if args.build_vggt and history_paths:
            required_paths.append(history_paths[-1])
        missing_paths = [path for path in required_paths if not Path(path).is_file()]
        if missing_paths:
            missing_image_records += 1
            if args.log_every and missing_image_records <= 20:
                print(f"skip token={token} missing required image(s): {missing_paths[:3]}")
            continue
        sample_token = frames[h - 1]["token"]
        if sample_token in seen_sample_tokens:
            duplicate_records += 1
            continue
        seen_sample_tokens.add(sample_token)
        records.append({
            "scene_token": frames[h - 1].get("scene_token", token),
            "sample_token": sample_token,
            "log_name": frames[h - 1]["log_name"],
            "history_cam_f0": history_paths,
            "future_cam_f0": future_paths,
            "loader_token": token,
        })
    if len(records) < args.chunk_size:
        if not args.allow_partial_final_chunk or not records:
            raise RuntimeError(
                f"Only found {len(records)} valid unique records for chunk_size={args.chunk_size}; "
                f"scanned={scanned - start}, missing_image_records={missing_image_records}, duplicate_records={duplicate_records}."
            )
        print(
            f"Using partial final chunk with {len(records)} valid records "
            f"for requested chunk_size={args.chunk_size}.",
            flush=True,
        )
    info = {
        "NAVSIM_DATA_ROOT": str(cached["data_root"]),
        "OPENSCENE_DATA_ROOT": str(cached["openscene"]),
        "NUPLAN_MAPS_ROOT": str(cached["maps"]),
        "navsim_log_path": str(cached["logs"]),
        "sensor_blobs_path": str(cached["blobs"]),
        "scene_filter": str(cached["scene_filter_path"]),
        "num_available": len(all_tokens),
        "chunk_start": start,
        "chunk_stop": args.chunk_stop,
        "chunk_end": scanned,
        "num_missing_image_records": missing_image_records,
        "num_duplicate_records": duplicate_records,
        "strict_token_window": bool(args.strict_token_window),
        "raw_window_size": args.chunk_size,
    }
    print(json.dumps(info, indent=2, sort_keys=True))
    return records, {**info, "scene_filter_config": asdict(scene_filter)}

def build_vlm_builder(args: argparse.Namespace):
    if not args.build_vlm_hidden:
        return None
    if args.recogdrive_vlm_path is None:
        raise ValueError("--build-vlm-hidden requires --recogdrive-vlm-path")
    from navsim.agents.recogdrive.recogdrive_features import ReCogDriveFeatureBuilder
    return ReCogDriveFeatureBuilder(
        cache_hidden_state=True,
        cache_mode=True,
        model_type="internvl",
        checkpoint_path=str(args.recogdrive_vlm_path),
        device=args.device,
    )

def build_teacher_extractors(args: argparse.Namespace):
    jepa = vggt = None
    if args.build_jepa:
        if args.jepa_model_path is None:
            raise ValueError("--build-jepa requires --jepa-model-path")
        from navsim.agents.recogdrive.expert_extractors.vjepa2_extractor import VJEPA2Extractor
        jepa = VJEPA2Extractor(args.jepa_model_path, device=args.device, precision=args.precision)
    if args.build_vggt:
        if args.vggt_model_path is None:
            raise ValueError("--build-vggt requires --vggt-model-path")
        from navsim.agents.recogdrive.expert_extractors.vggt_extractor import VGGTExtractor
        vggt = VGGTExtractor(
            args.vggt_model_path,
            device=args.device,
            precision=args.precision,
            require_geometry=bool(getattr(args, "require_vggt_geometry", False)),
        )
    return jepa, vggt

def navsim_training_fields(loader_token: str, args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    cached = get_cached_loader(args)
    loader = cached["loader"]
    from navsim.agents.recogdrive.recogdrive_features import ReCogDriveFeatureBuilder, TrajectoryTargetBuilder
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    scene = loader.get_scene_from_token(loader_token)
    agent_input = scene.get_agent_input()
    builder = ReCogDriveFeatureBuilder(cache_hidden_state=False, use_expert_features=False)
    features = builder.compute_features(agent_input)
    target = TrajectoryTargetBuilder(TrajectorySampling(time_horizon=4, interval_length=0.5)).compute_targets(scene)
    return {**features, **target}

def index_record_for_sample(args: argparse.Namespace, record: Dict[str, Any]) -> Dict[str, Any]:
    out_path = args.output_dir / "samples" / f"{record['sample_token']}.pt"
    return {
        "scene_token": record["scene_token"],
        "sample_token": record["sample_token"],
        "log_name": record.get("log_name"),
        "path": str(out_path.relative_to(args.output_dir)),
    }

def configure_worker_device(worker_id: int, args: argparse.Namespace) -> str:
    actual_device = args.device
    if args.device.startswith("cuda"):
        visible_gpus = torch.cuda.device_count()
        if visible_gpus <= 0:
            raise RuntimeError("--device cuda requested but torch reports no visible CUDA devices")
        usable_gpus = min(max(1, args.num_gpus), visible_gpus)
        gpu_id = worker_id % usable_gpus
        torch.cuda.set_device(gpu_id)
        actual_device = f"cuda:{gpu_id}"
        # Keep bare "cuda" for libraries that honor the current CUDA device.
        # This preserves the already-working single-GPU load path while sharding
        # workers across GPUs via torch.cuda.set_device().
        if args.device != "cuda":
            args.device = actual_device
    return actual_device

def set_current_worker_device(actual_device: str) -> None:
    if actual_device.startswith("cuda:"):
        torch.cuda.set_device(int(actual_device.split(":", 1)[1]))

def process_records_with_models(
    worker_id: int,
    args: argparse.Namespace,
    records: List[Dict[str, Any]],
    *,
    actual_device: str,
    vlm_builder: Any,
    jepa: Any,
    vggt: Any,
) -> Dict[str, Any]:
    normalize_path_args(args)
    set_current_worker_device(actual_device)
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    index_records = []
    written = skipped = missing_image_skipped = 0
    for i, record in enumerate(records):
        out_path = samples_dir / f"{record['sample_token']}.pt"
        index_record = index_record_for_sample(args, record)
        if out_path.is_file() and not args.overwrite:
            skipped += 1
            index_records.append(index_record)
            continue
        if args.build_jepa and len(record["future_cam_f0"]) < 4 and not args.allow_missing_future_frames:
            skipped += 1
            missing_image_skipped += 1
            print(f"[worker {worker_id}] skipping {record['sample_token']}: need 4 future frames for JEPA target", flush=True)
            continue
        required_images = []
        if args.build_vlm_hidden or args.build_jepa or args.build_vggt:
            required_images.extend(record["history_cam_f0"][-4:])
        if args.build_jepa:
            required_images.extend(record["future_cam_f0"][:4])
        missing_images = [image for image in required_images if not Path(image).is_file()]
        if missing_images:
            skipped += 1
            missing_image_skipped += 1
            print(f"[worker {worker_id}] skipping {record['sample_token']}: missing required image {missing_images[0]}", flush=True)
            continue

        payload: Dict[str, Any] = {
            "scene_token": record["scene_token"],
            "sample_token": record["sample_token"],
        }
        # Save planner-stage fields so chunk training does not need a global cache.
        payload.update(navsim_training_fields(record["loader_token"], args))
        if vlm_builder is not None:
            loader = get_cached_loader(args)["loader"]
            agent_input = loader.get_agent_input_from_token(record["loader_token"])
            if record.get("history_cam_f0"):
                agent_input.cameras[-1].cam_f0.image = Path(record["history_cam_f0"][-1])
            payload.update(vlm_builder.compute_features(agent_input))
        if jepa is not None:
            payload["jepa_context_tokens"] = jepa.extract_context(record["history_cam_f0"][-4:])
            if len(record["future_cam_f0"]) >= 4:
                payload["jepa_target_tokens"] = jepa.extract_target(record["future_cam_f0"][:4])
        if vggt is not None:
            current = record["history_cam_f0"][-1]
            geometry_payload = vggt.extract_with_geometry(current)
            payload["vggt_context_tokens"] = geometry_payload["vggt_context_tokens"]
            payload["vggt_geometry_tokens"] = geometry_payload["vggt_geometry_tokens"]
            payload["vggt_geometry_mode"] = geometry_payload["vggt_geometry_mode"]
            if "vggt_geometry_mode_code" in geometry_payload:
                payload["vggt_geometry_mode_code"] = geometry_payload["vggt_geometry_mode_code"]
            payload["vggt_target_tokens"] = vggt.extract_target(current)
            payload["vggt_geometry_target_tokens"] = payload["vggt_geometry_tokens"]
        if args.build_jepa or args.build_vggt:
            validate_sample_payload(payload, require_jepa=args.build_jepa, require_vggt=args.build_vggt, require_targets=True)
        atomic_torch_save(payload, out_path)
        index_records.append(index_record)
        written += 1
        if args.log_every and (i + 1) % args.log_every == 0:
            print(
                f"[worker {worker_id} {actual_device}] processed {i + 1}/{len(records)} "
                f"written={written} skipped={skipped}",
                flush=True,
            )

    return {
        "worker_id": worker_id,
        "device": actual_device,
        "num_assigned": len(records),
        "num_written": written,
        "num_skipped": skipped,
        "num_missing_image_skipped": missing_image_skipped,
        "index_records": index_records,
    }

class ChunkCacheWorkerRuntime:
    def __init__(self, worker_id: int, args_dict: Dict[str, Any]) -> None:
        self.worker_id = worker_id
        self.args = normalize_path_args(argparse.Namespace(**args_dict))
        self.actual_device = configure_worker_device(worker_id, self.args)
        self.vlm_builder = build_vlm_builder(self.args)
        self.jepa, self.vggt = build_teacher_extractors(self.args)

    def process(self, args_dict: Dict[str, Any], records: List[Dict[str, Any]]) -> Dict[str, Any]:
        args = normalize_path_args(argparse.Namespace(**args_dict))
        # Reuse the already-loaded model device. The per-chunk args only change
        # data/output window fields, not the model placement.
        args.device = self.args.device
        return process_records_with_models(
            self.worker_id,
            args,
            records,
            actual_device=self.actual_device,
            vlm_builder=self.vlm_builder,
            jepa=self.jepa,
            vggt=self.vggt,
        )

def process_records_shard(worker_id: int, args_dict: Dict[str, Any], records: List[Dict[str, Any]]) -> Dict[str, Any]:
    args = normalize_path_args(argparse.Namespace(**args_dict))
    actual_device = configure_worker_device(worker_id, args)
    vlm_builder = build_vlm_builder(args)
    jepa, vggt = build_teacher_extractors(args)
    return process_records_with_models(
        worker_id,
        args,
        records,
        actual_device=actual_device,
        vlm_builder=vlm_builder,
        jepa=jepa,
        vggt=vggt,
    )


def process_records_shard_entry(worker_id: int, args_dict: Dict[str, Any], records: List[Dict[str, Any]], queue: Any) -> None:
    try:
        queue.put({"ok": True, "result": process_records_shard(worker_id, args_dict, records)})
    except Exception:
        queue.put({"ok": False, "worker_id": worker_id, "traceback": traceback.format_exc()})

def choose_worker_count(args: argparse.Namespace, record_count: int) -> int:
    if record_count <= 0:
        return 0
    if args.workers is not None and args.workers <= 0:
        raise ValueError("--workers must be positive when provided")
    if args.workers_per_gpu <= 0:
        raise ValueError("--workers-per-gpu must be positive")
    if not args.device.startswith("cuda"):
        requested = args.workers if args.workers is not None else 1
        return min(requested, record_count)
    visible = torch.cuda.device_count()
    if visible <= 0:
        raise RuntimeError("--device cuda requested but torch reports no visible CUDA devices")
    usable_gpus = min(max(1, args.num_gpus), visible)
    requested = args.workers if args.workers is not None else usable_gpus * args.workers_per_gpu
    return min(requested, record_count)

def merge_worker_reports(records: List[Dict[str, Any]], worker_reports: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    order = {record["sample_token"]: i for i, record in enumerate(records)}
    index_records: List[Dict[str, Any]] = []
    for report in worker_reports:
        index_records.extend(report["index_records"])
    index_records.sort(key=lambda item: order.get(item["sample_token"], len(order)))
    worker_reports.sort(key=lambda item: item["worker_id"])
    return index_records, worker_reports

def process_records(args: argparse.Namespace, records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    worker_count = choose_worker_count(args, len(records))
    args_dict = vars(args).copy()
    if worker_count <= 1:
        report = process_records_shard(0, args_dict, records)
        return report["index_records"], [report]

    visible = torch.cuda.device_count() if args.device.startswith("cuda") else 0
    usable = min(max(1, args.num_gpus), visible) if visible else 0
    print(f"Launching {worker_count} chunk-cache workers across {usable} CUDA device(s).", flush=True)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = []
    for worker_id in range(worker_count):
        shard = records[worker_id::worker_count]
        process = ctx.Process(
            target=process_records_shard_entry,
            args=(worker_id, args_dict, shard, queue),
            name=f"chunk-cache-worker-{worker_id}",
        )
        process.start()
        processes.append(process)

    results = []
    failures = []
    while len(results) + len(failures) < len(processes):
        try:
            message = queue.get(timeout=10)
        except queue_lib.Empty:
            dead = [process for process in processes if process.exitcode not in (0, None)]
            if not dead:
                continue
            for process in dead:
                failures.append({
                    "worker_id": process.name,
                    "traceback": f"process exited with code {process.exitcode} before reporting",
                })
            break
        if message.get("ok"):
            results.append(message["result"])
        else:
            failures.append(message)
    for process in processes:
        process.join()
        if process.exitcode not in (0, None) and not any(str(item.get("worker_id")) == process.name for item in failures):
            failures.append({
                "worker_id": process.name,
                "traceback": f"process exited with code {process.exitcode}",
            })
    if failures:
        for failure in failures:
            print(f"Worker failure: {failure}", file=sys.stderr, flush=True)
        raise RuntimeError(f"{len(failures)} chunk-cache worker(s) failed")

    return merge_worker_reports(records, results)

def write_built_chunk_outputs(
    args: argparse.Namespace,
    info: Dict[str, Any],
    index_records: List[Dict[str, Any]],
    worker_reports: List[Dict[str, Any]],
) -> Dict[str, Any]:
    from datetime import datetime, timezone

    normalize_path_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "index.jsonl").open("w", encoding="utf-8") as f:
        for record in index_records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    written = sum(report["num_written"] for report in worker_reports)
    skipped = sum(report["num_skipped"] for report in worker_reports)
    missing_image_skipped = sum(report["num_missing_image_skipped"] for report in worker_reports)
    metadata = {
        "version": CHUNK_VERSION,
        "is_dummy": False,
        "contains_vlm_hidden": bool(args.build_vlm_hidden),
        "contains_jepa": bool(args.build_jepa),
        "contains_vggt": bool(args.build_vggt),
        "target_tokens_are_train_only": True,
        "recogdrive_vlm_path": str(args.recogdrive_vlm_path) if args.recogdrive_vlm_path else None,
        "jepa_model_path": str(args.jepa_model_path) if args.jepa_model_path else None,
        "vggt_model_path": str(args.vggt_model_path) if args.vggt_model_path else None,
        "token_shapes": {
            "last_hidden_state": [None, 1536] if args.build_vlm_hidden else None,
            "jepa_context_tokens": [12, 1024] if args.build_jepa else None,
            "jepa_target_tokens": [12, 1024] if args.build_jepa else None,
            "vggt_context_tokens": [12, 2048] if args.build_vggt else None,
            "vggt_target_tokens": [12, 2048] if args.build_vggt else None,
            "vggt_geometry_tokens": [12, 2048] if args.build_vggt else None,
            "vggt_geometry_target_tokens": [12, 2048] if args.build_vggt else None,
            "vggt_depth_tokens": [12, 2048] if args.build_vggt else None,
            "vggt_pointmap_tokens": [12, 2048] if args.build_vggt else None,
            "vggt_camera_tokens": [12, 2048] if args.build_vggt else None,
        },
        "contains_vggt_geometry": bool(args.build_vggt),
        "require_vggt_geometry": bool(getattr(args, "require_vggt_geometry", False)),
        "split": args.split,
        "chunk_index": args.chunk_index,
        "chunk_start": chunk_start_value(args),
        "chunk_size": args.chunk_size,
        "num_records": len(index_records),
        "strict_token_window": bool(args.strict_token_window),
        "raw_window_size": args.chunk_size,
        "num_written": written,
        "num_skipped": skipped,
        "num_missing_image_skipped": missing_image_skipped,
        "num_workers": len(worker_reports),
        "num_gpus_requested": args.num_gpus,
        "workers_requested": args.workers,
        "workers_per_gpu": args.workers_per_gpu,
        "worker_reports": [
            {key: value for key, value in report.items() if key != "index_records"}
            for report in worker_reports
        ],
        "created_time": datetime.now(timezone.utc).isoformat(),
        **info,
    }
    write_json(args.output_dir / "metadata.json", metadata)
    print(f"Wrote chunk cache metadata to {args.output_dir / 'metadata.json'}")
    return metadata


def finalize_existing_chunk(args: argparse.Namespace) -> int:
    from datetime import datetime, timezone

    samples_dir = args.output_dir / "samples"
    sample_files = sorted(samples_dir.glob("*.pt")) if samples_dir.is_dir() else []
    if not sample_files:
        raise RuntimeError(f"No existing sample files found under {samples_dir}")
    index_records = []
    contains_vlm = contains_jepa = contains_vggt = contains_geometry = False
    checked = 0
    for sample_path in sample_files:
        payload = torch.load(sample_path, map_location="cpu")
        contains_vlm = contains_vlm or "last_hidden_state" in payload
        contains_jepa = contains_jepa or "jepa_context_tokens" in payload
        contains_vggt = contains_vggt or "vggt_context_tokens" in payload
        contains_geometry = contains_geometry or any(
            key in payload for key in ("vggt_geometry_tokens", "vggt_depth_tokens", "vggt_pointmap_tokens", "vggt_camera_tokens")
        )
        validate_sample_payload(
            payload,
            require_jepa=args.build_jepa,
            require_vggt=args.build_vggt,
            require_targets=bool(args.build_jepa or args.build_vggt),
        )
        rel_path = sample_path.relative_to(args.output_dir)
        index_records.append({
            "scene_token": str(payload.get("scene_token", sample_path.stem)),
            "sample_token": str(payload.get("sample_token", sample_path.stem)),
            "log_name": payload.get("log_name"),
            "path": str(rel_path),
        })
        checked += 1
    with (args.output_dir / "index.jsonl").open("w", encoding="utf-8") as f:
        for record in index_records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    metadata = {
        "version": CHUNK_VERSION,
        "is_dummy": False,
        "contains_vlm_hidden": contains_vlm,
        "contains_jepa": contains_jepa,
        "contains_vggt": contains_vggt,
        "target_tokens_are_train_only": True,
        "recogdrive_vlm_path": str(args.recogdrive_vlm_path) if args.recogdrive_vlm_path else None,
        "jepa_model_path": str(args.jepa_model_path) if args.jepa_model_path else None,
        "vggt_model_path": str(args.vggt_model_path) if args.vggt_model_path else None,
        "token_shapes": {
            "last_hidden_state": [None, 1536] if contains_vlm else None,
            "jepa_context_tokens": [12, 1024] if contains_jepa else None,
            "jepa_target_tokens": [12, 1024] if contains_jepa else None,
            "vggt_context_tokens": [12, 2048] if contains_vggt else None,
            "vggt_target_tokens": [12, 2048] if contains_vggt else None,
            "vggt_geometry_tokens": [12, 2048] if contains_geometry else None,
            "vggt_geometry_target_tokens": [12, 2048] if contains_geometry else None,
            "vggt_depth_tokens": [12, 2048] if contains_geometry else None,
            "vggt_pointmap_tokens": [12, 2048] if contains_geometry else None,
            "vggt_camera_tokens": [12, 2048] if contains_geometry else None,
        },
        "contains_vggt_geometry": contains_geometry,
        "require_vggt_geometry": bool(getattr(args, "require_vggt_geometry", False)),
        "split": args.split,
        "chunk_index": args.chunk_index,
        "chunk_start": chunk_start_value(args),
        "chunk_stop": args.chunk_stop,
        "chunk_size": args.chunk_size,
        "num_records": len(index_records),
        "strict_token_window": bool(getattr(args, "strict_token_window", False)),
        "raw_window_size": args.chunk_size,
        "num_finalized_existing": checked,
        "created_time": datetime.now(timezone.utc).isoformat(),
        "finalized_from_existing_samples": True,
    }
    write_json(args.output_dir / "metadata.json", metadata)
    print(json.dumps({"finalized": checked, "metadata": str(args.output_dir / "metadata.json")}, indent=2))
    return 0

def main() -> int:
    args = normalize_path_args(parse_args())
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.finalize_existing:
        return finalize_existing_chunk(args)
    records, info = build_records(args)
    if not records:
        raise RuntimeError("No records selected for this chunk.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    index_records, worker_reports = process_records(args, records)
    write_built_chunk_outputs(args, info, index_records, worker_reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
