#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import yaml
from hydra.utils import instantiate

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import iter_index, load_sample  # noqa: E402
from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent  # noqa: E402
from navsim.common.dataloader import SceneLoader  # noqa: E402
from navsim.planning.training.dataset import (  # noqa: E402
    dump_feature_target_to_pickle,
    load_feature_target_from_pickle,
)


A0_FEATURE_KEYS = (
    "history_trajectory",
    "high_command_one_hot",
    "last_hidden_state",
    "status_feature",
)
A0_TARGET_KEYS = ("trajectory",)
RAW_TOKEN_FILTER: Optional[Set[str]] = None


def parse_args() -> argparse.Namespace:
    openscene_root = os.environ.get("OPENSCENE_DATA_ROOT")
    default_log_path = Path(openscene_root) / "navsim_logs/trainval" if openscene_root else None
    default_sensor_path = Path(openscene_root) / "sensor_blobs/trainval" if openscene_root else None
    maps_root_env = os.environ.get("NUPLAN_MAPS_ROOT")
    default_maps_root = Path(maps_root_env) if maps_root_env else Path("/mnt/navsim/maps")
    parser = argparse.ArgumentParser(
        description=(
            "Generate a lightweight official-format raw-data cache for selected chunk-cache tokens, "
            "then compare A0 training tensors token-by-token."
        )
    )
    parser.add_argument("--chunk-cache-root", type=Path, default=Path("cache/recogdrive_expert_chunks/full_v1"))
    parser.add_argument("--navsim-log-path", type=Path, default=default_log_path)
    parser.add_argument("--sensor-blobs-path", type=Path, default=default_sensor_path)
    parser.add_argument("--nuplan-maps-root", type=Path, default=default_maps_root)
    parser.add_argument("--output-cache-root", type=Path, default=Path("outputs/cache_parity_official_raw_cache"))
    parser.add_argument("--output-report", type=Path, default=Path("reports/cache_parity_raw_official.json"))
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument(
        "--scene-filter",
        type=str,
        default="navtrain",
        help="Scene-filter config name whose log_names are intersected with train/val logs. Use 'none' to disable.",
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--selection", choices=("head", "spread", "random"), default="spread")
    parser.add_argument("--tokens", type=str, default="", help="Comma-separated sample tokens to compare.")
    parser.add_argument("--tokens-file", type=Path, default=None, help="Optional newline-delimited sample tokens.")
    parser.add_argument(
        "--compare-hidden-state",
        action="store_true",
        help="Also regenerate and compare last_hidden_state with the official VLM feature builder.",
    )
    parser.add_argument(
        "--vlm-path",
        type=str,
        default=os.environ.get("RECOGDRIVE_VLM_PATH", ""),
        help="Required only with --compare-hidden-state; defaults to RECOGDRIVE_VLM_PATH.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for optional hidden-state regeneration. Defaults to CPU to keep audit/counting GPU-free.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Worker processes for raw NAVSIM split counting.",
    )
    parser.add_argument(
        "--count-raw-split",
        action="store_true",
        help="Additionally scan raw NAVSIM logs for the whole official train/val split and compare token sets.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_scene_filter_data(scene_filter_name: str = "navtrain") -> Dict[str, Any]:
    if not scene_filter_name or scene_filter_name.lower() == "none":
        return {}
    filter_path = (
        REPO_ROOT
        / "navsim/planning/script/config/common/train_test_split/scene_filter"
        / f"{scene_filter_name}.yaml"
    )
    return yaml.safe_load(filter_path.read_text(encoding="utf-8")) or {}


def load_scene_filter_tokens(scene_filter_name: str = "navtrain") -> Optional[Set[str]]:
    tokens = load_scene_filter_data(scene_filter_name).get("tokens")
    if tokens is None:
        return None
    return {str(item) for item in tokens}


def load_split_logs(scene_filter_name: str = "navtrain") -> Dict[str, List[str]]:
    path = REPO_ROOT / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    split_logs = {
        "train": [str(item) for item in data.get("train_logs", [])],
        "val": [str(item) for item in data.get("val_logs", [])],
    }
    if scene_filter_name and scene_filter_name.lower() != "none":
        filter_data = load_scene_filter_data(scene_filter_name)
        filter_logs = filter_data.get("log_names")
        if filter_logs is not None:
            allowed = {str(item) for item in filter_logs}
            split_logs = {
                split: [log_name for log_name in logs if log_name in allowed]
                for split, logs in split_logs.items()
            }
    return split_logs


def chunk_dirs(cache_root: Path) -> List[Path]:
    if (cache_root / "index.jsonl").is_file():
        return [cache_root]
    return sorted(
        child for child in cache_root.iterdir()
        if child.is_dir() and (child / "index.jsonl").is_file() and not child.name.startswith("navtest")
    )


def resolve_sample_path(chunk_dir: Path, record: Dict[str, Any]) -> Path:
    path = Path(record["path"])
    if path.is_file():
        return path
    if not path.is_absolute():
        candidate = chunk_dir / path
        if candidate.is_file():
            return candidate
    return path


def read_chunk_records(cache_root: Path, split_logs: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    train_logs = set(split_logs["train"])
    val_logs = set(split_logs["val"])
    records: List[Dict[str, Any]] = []
    for chunk_dir in chunk_dirs(cache_root):
        for record in iter_index(chunk_dir):
            log_name = str(record.get("log_name") or "")
            if log_name in train_logs:
                split = "train"
            elif log_name in val_logs:
                split = "val"
            else:
                split = "other"
            sample_path = resolve_sample_path(chunk_dir, record)
            token = str(record.get("sample_token") or sample_path.stem)
            records.append({
                "split": split,
                "log_name": log_name,
                "sample_token": token,
                "scene_token": str(record.get("scene_token") or ""),
                "chunk_dir": str(chunk_dir),
                "sample_path": str(sample_path),
            })
    return records


def explicit_tokens(args: argparse.Namespace) -> List[str]:
    tokens: List[str] = []
    if args.tokens:
        tokens.extend(token.strip() for token in args.tokens.split(",") if token.strip())
    if args.tokens_file is not None:
        tokens.extend(
            line.strip()
            for line in args.tokens_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    return tokens


def select_records(records: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    by_token = {record["sample_token"]: record for record in records if record["split"] == args.split}
    requested = explicit_tokens(args)
    if requested:
        missing = [token for token in requested if token not in by_token]
        if missing:
            raise KeyError(f"Requested tokens not found in chunk cache split={args.split}: {missing[:10]}")
        return [by_token[token] for token in requested]

    candidates = [record for record in records if record["split"] == args.split]
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    count = min(args.num_samples, len(candidates))
    if args.selection == "head":
        return candidates[:count]
    if args.selection == "random":
        rng = random.Random(args.seed)
        return rng.sample(candidates, count)
    if count == 1:
        return [candidates[0]]
    indices = sorted({round(i * (len(candidates) - 1) / (count - 1)) for i in range(count)})
    return [candidates[i] for i in indices]


def build_agent(args: argparse.Namespace) -> ReCogDriveAgent:
    if args.compare_hidden_state and not args.vlm_path:
        raise ValueError("--compare-hidden-state requires --vlm-path or RECOGDRIVE_VLM_PATH.")
    agent = ReCogDriveAgent(
        trajectory_sampling=instantiate({
            "_target_": "nuplan.planning.simulation.trajectory.trajectory_sampling.TrajectorySampling",
            "time_horizon": 4,
            "interval_length": 0.5,
        }),
        vlm_path=args.vlm_path if args.compare_hidden_state else None,
        cam_type="single",
        vlm_type="internvl",
        dit_type="small",
        sampling_method="ddim",
        cache_mode=False,
        cache_hidden_state=True,
        lr=1e-4,
        grpo=False,
        vlm_size="small",
        use_expert_features=False,
        use_jepa=False,
        use_vggt=False,
    )
    return agent


def build_feature_and_target_builders(agent: ReCogDriveAgent, args: argparse.Namespace):
    feature_builder = agent.get_feature_builders()[0]
    feature_builder.cache_hidden_state = bool(args.compare_hidden_state)
    feature_builder.cache_mode = bool(args.compare_hidden_state)
    feature_builder.device = torch.device(args.device)
    if args.compare_hidden_state and feature_builder.backbone is None:
        from navsim.agents.recogdrive.recogdrive_backbone import RecogDriveBackbone

        feature_builder.backbone = RecogDriveBackbone(
            model_type="internvl",
            checkpoint_path=args.vlm_path,
            device=args.device,
        )
    target_builder = agent.get_target_builders()[0]
    return feature_builder, target_builder


def make_scene_loader(records: Sequence[Dict[str, Any]], agent: ReCogDriveAgent, args: argparse.Namespace) -> SceneLoader:
    if args.navsim_log_path is None or not args.navsim_log_path.is_dir():
        raise FileNotFoundError(
            "--navsim-log-path is required and must exist. Set OPENSCENE_DATA_ROOT or pass the path explicitly."
        )
    if args.sensor_blobs_path is None or not args.sensor_blobs_path.is_dir():
        raise FileNotFoundError(
            "--sensor-blobs-path is required and must exist. Set OPENSCENE_DATA_ROOT or pass the path explicitly."
        )
    scene_filter = instantiate({
        "_target_": "navsim.common.dataclasses.SceneFilter",
        "_convert_": "all",
        "num_history_frames": 4,
        "num_future_frames": 10,
        "frame_interval": 1,
        "has_route": True,
        "max_scenes": None,
        "log_names": sorted({record["log_name"] for record in records}),
        "tokens": [record["sample_token"] for record in records],
    })
    return SceneLoader(
        sensor_blobs_path=args.sensor_blobs_path,
        data_path=args.navsim_log_path,
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True,
    )


def write_official_cache(
    record: Dict[str, Any],
    features: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[Path, Path]:
    token_dir = args.output_cache_root / record["log_name"] / record["sample_token"]
    token_dir.mkdir(parents=True, exist_ok=True)
    feature_path = token_dir / "internvl_feature.gz"
    target_path = token_dir / "trajectory_target.gz"
    if not args.overwrite and (feature_path.exists() or target_path.exists()):
        raise FileExistsError(f"Official cache exists for {record['sample_token']}; pass --overwrite: {token_dir}")
    dump_feature_target_to_pickle(feature_path, features)
    dump_feature_target_to_pickle(target_path, targets)
    return feature_path, target_path


def tensor_summary(tensor: torch.Tensor) -> Dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
    }


def compare_tensor(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "chunk": tensor_summary(a),
        "official": tensor_summary(b),
        "shape_equal": tuple(a.shape) == tuple(b.shape),
        "dtype_equal": a.dtype == b.dtype,
    }
    if tuple(a.shape) != tuple(b.shape):
        result.update({"max_abs_diff": None, "mean_abs_diff": None, "allclose_1e_6": False, "allclose_1e_4": False})
        return result
    af = a.detach().cpu().float()
    bf = b.detach().cpu().float()
    diff = (af - bf).abs()
    result.update({
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "allclose_1e_6": bool(torch.allclose(af, bf, atol=1e-6, rtol=1e-6)),
        "allclose_1e_4": bool(torch.allclose(af, bf, atol=1e-4, rtol=1e-4)),
    })
    return result


def decode_path_tensor(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().flatten().tolist()
    return "".join(chr(int(value)) for value in values)


def normalized_sensor_suffix(path: str) -> str:
    parts = Path(path).parts
    for marker in ("trainval", "test", "mini", "private_test_hard"):
        indices = [idx for idx, part in enumerate(parts) if part == marker]
        if indices:
            return "/".join(parts[indices[-1]:])
    return "/".join(parts[-3:])


def compare_image_path_tensor(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    chunk_path = decode_path_tensor(a)
    official_path = decode_path_tensor(b)
    chunk_suffix = normalized_sensor_suffix(chunk_path)
    official_suffix = normalized_sensor_suffix(official_path)
    return {
        "chunk": tensor_summary(a),
        "official": tensor_summary(b),
        "shape_equal": tuple(a.shape) == tuple(b.shape),
        "dtype_equal": a.dtype == b.dtype,
        "chunk_path": chunk_path,
        "official_path": official_path,
        "chunk_normalized_suffix": chunk_suffix,
        "official_normalized_suffix": official_suffix,
        "exact_path_equal": chunk_path == official_path,
        "normalized_path_equal": chunk_suffix == official_suffix,
    }


def compare_sample(
    chunk_sample: Dict[str, Any],
    official_features: Dict[str, torch.Tensor],
    official_targets: Dict[str, torch.Tensor],
    *,
    compare_hidden_state: bool,
) -> Dict[str, Any]:
    key_results: Dict[str, Any] = {}
    keys = ["history_trajectory", "high_command_one_hot", "status_feature", "trajectory"]
    if compare_hidden_state:
        keys.append("last_hidden_state")
    else:
        keys.append("image_path_tensor")
    for key in keys:
        if key == "trajectory":
            official_value = official_targets.get(key)
        else:
            official_value = official_features.get(key)
        chunk_value = chunk_sample.get(key)
        if not isinstance(chunk_value, torch.Tensor) or not isinstance(official_value, torch.Tensor):
            key_results[key] = {
                "present_in_chunk": isinstance(chunk_value, torch.Tensor),
                "present_in_official": isinstance(official_value, torch.Tensor),
            }
            continue
        if key == "image_path_tensor":
            key_results[key] = compare_image_path_tensor(chunk_value, official_value)
        else:
            key_results[key] = compare_tensor(chunk_value, official_value)
    return key_results


def split_list(input_list: List[Any], num_frames: int, frame_interval: int) -> List[List[Any]]:
    return [input_list[i : i + num_frames] for i in range(0, len(input_list), frame_interval)]


def init_raw_scan_worker(token_filter: Optional[Sequence[str]]) -> None:
    global RAW_TOKEN_FILTER
    RAW_TOKEN_FILTER = set(token_filter) if token_filter is not None else None


def scan_raw_log_pickle(task: Tuple[str, str, int, int, int, bool]) -> Dict[str, Any]:
    log_name, path_str, num_history_frames, num_future_frames, frame_interval, has_route = task
    num_frames = num_history_frames + num_future_frames
    tokens: List[str] = []
    with open(path_str, "rb") as f:
        scene_dict_list = pickle.load(f)
    for frame_list in split_list(scene_dict_list, num_frames, frame_interval):
        if len(frame_list) < num_frames:
            continue
        anchor = frame_list[num_history_frames - 1]
        if has_route and len(anchor.get("roadblock_ids", [])) == 0:
            continue
        token = anchor.get("token")
        if token is None:
            continue
        token = str(token)
        if RAW_TOKEN_FILTER is not None and token not in RAW_TOKEN_FILTER:
            continue
        tokens.append(token)
    return {
        "log_name": log_name,
        "tokens": tokens,
        "num_tokens": len(tokens),
    }


def count_raw_split_tokens(
    args: argparse.Namespace,
    split_logs: Dict[str, List[str]],
    chunk_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if not args.count_raw_split:
        return {}
    if args.navsim_log_path is None or not args.navsim_log_path.is_dir():
        raise FileNotFoundError("--count-raw-split requires --navsim-log-path.")
    out: Dict[str, Any] = {}
    workers = max(1, int(args.num_workers))
    scene_filter_tokens = load_scene_filter_tokens(args.scene_filter)
    scene_filter_token_list = sorted(scene_filter_tokens) if scene_filter_tokens is not None else None
    chunk_tokens_by_split = {
        split: {record["sample_token"] for record in chunk_records if record["split"] == split}
        for split in ("train", "val")
    }
    for split in ("train", "val"):
        tasks = []
        missing_log_files = []
        for log_name in split_logs[split]:
            path = args.navsim_log_path / f"{log_name}.pkl"
            if path.is_file():
                tasks.append((str(log_name), str(path), 4, 10, 1, True))
            else:
                missing_log_files.append(str(log_name))

        results: List[Dict[str, Any]] = []
        if workers == 1:
            init_raw_scan_worker(scene_filter_token_list)
            results = [scan_raw_log_pickle(task) for task in tasks]
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=init_raw_scan_worker,
                initargs=(scene_filter_token_list,),
            ) as executor:
                for item in executor.map(scan_raw_log_pickle, tasks, chunksize=max(1, len(tasks) // (workers * 4) or 1)):
                    results.append(item)

        tokens: List[str] = []
        log_counts: Dict[str, int] = {}
        for item in results:
            item_tokens = item["tokens"]
            tokens.extend(item_tokens)
            log_counts[item["log_name"]] = int(item["num_tokens"])
        raw_token_set = set(tokens)
        chunk_token_set = chunk_tokens_by_split[split]
        raw_duplicates = len(tokens) - len(raw_token_set)
        out[split] = {
            "raw_scene_loader_count": len(tokens),
            "raw_unique_token_count": len(raw_token_set),
            "raw_duplicate_token_count": raw_duplicates,
            "raw_log_file_count": len(tasks),
            "missing_log_file_count": len(missing_log_files),
            "missing_log_file_examples": missing_log_files[:10],
            "chunk_unique_token_count": len(chunk_token_set),
            "raw_minus_chunk_count": len(raw_token_set - chunk_token_set),
            "chunk_minus_raw_count": len(chunk_token_set - raw_token_set),
            "raw_chunk_overlap_count": len(raw_token_set & chunk_token_set),
            "top_raw_log_counts": sorted(log_counts.items(), key=lambda item: item[1], reverse=True)[:10],
            "num_workers": workers,
            "scene_filter_token_count": len(scene_filter_tokens) if scene_filter_tokens is not None else None,
        }
    return out


def status_from_comparison(sample_results: Sequence[Dict[str, Any]], *, compare_hidden_state: bool) -> str:
    required = ["history_trajectory", "high_command_one_hot", "status_feature", "trajectory"]
    if compare_hidden_state:
        required.append("last_hidden_state")
    else:
        required.append("image_path_tensor")
    for sample in sample_results:
        for key in required:
            item = sample.get("comparisons", {}).get(key, {})
            if key == "image_path_tensor":
                if not item.get("normalized_path_equal", False):
                    return "not_aligned"
                continue
            if not item.get("shape_equal", False):
                return "not_aligned"
            if key != "last_hidden_state" and not item.get("allclose_1e_6", False):
                return "not_aligned"
            if key == "last_hidden_state" and not item.get("allclose_1e_4", False):
                return "not_aligned"
    return "aligned"


def main() -> int:
    args = parse_args()
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_cache_root.mkdir(parents=True, exist_ok=True)
    if args.nuplan_maps_root is not None:
        if not args.nuplan_maps_root.is_dir():
            raise FileNotFoundError(f"--nuplan-maps-root does not exist: {args.nuplan_maps_root}")
        os.environ["NUPLAN_MAPS_ROOT"] = str(args.nuplan_maps_root)
        import navsim.common.dataclasses as navsim_dataclasses

        navsim_dataclasses.NUPLAN_MAPS_ROOT = str(args.nuplan_maps_root)

    split_logs = load_split_logs(args.scene_filter)
    all_chunk_records = read_chunk_records(args.chunk_cache_root, split_logs)
    scene_filter_tokens = load_scene_filter_tokens(args.scene_filter)
    chunk_counts = Counter(record["split"] for record in all_chunk_records)
    chunk_unique = {
        split: len({record["sample_token"] for record in all_chunk_records if record["split"] == split})
        for split in ("train", "val", "other")
    }
    selected = select_records(all_chunk_records, args)

    agent = build_agent(args)
    feature_builder, target_builder = build_feature_and_target_builders(agent, args)
    scene_loader = make_scene_loader(selected, agent, args)
    missing_in_raw = [record["sample_token"] for record in selected if record["sample_token"] not in set(scene_loader.tokens)]
    if missing_in_raw:
        raise KeyError(f"Selected tokens missing from raw SceneLoader: {missing_in_raw[:10]}")

    sample_results: List[Dict[str, Any]] = []
    for record in selected:
        token = record["sample_token"]
        scene = scene_loader.get_scene_from_token(token)
        agent_input = scene.get_agent_input()
        features = feature_builder.compute_features(agent_input)
        targets = target_builder.compute_targets(scene)
        feature_path, target_path = write_official_cache(record, features, targets, args)

        # Compare through the serialized official cache, not only in-memory builder output.
        official_features = load_feature_target_from_pickle(feature_path)
        official_targets = load_feature_target_from_pickle(target_path)
        chunk_sample = load_sample(Path(record["sample_path"]))
        comparisons = compare_sample(
            chunk_sample,
            official_features,
            official_targets,
            compare_hidden_state=args.compare_hidden_state,
        )
        sample_results.append({
            "sample_token": token,
            "log_name": record["log_name"],
            "chunk_sample_path": record["sample_path"],
            "official_feature_path": str(feature_path),
            "official_target_path": str(target_path),
            "chunk_keys": sorted(chunk_sample.keys()),
            "official_feature_keys": sorted(official_features.keys()),
            "official_target_keys": sorted(official_targets.keys()),
            "comparisons": comparisons,
        })

    report = {
        "status": status_from_comparison(sample_results, compare_hidden_state=args.compare_hidden_state),
        "chunk_cache_root": str(args.chunk_cache_root),
        "navsim_log_path": str(args.navsim_log_path),
        "sensor_blobs_path": str(args.sensor_blobs_path),
        "nuplan_maps_root": str(args.nuplan_maps_root),
        "output_cache_root": str(args.output_cache_root),
        "split": args.split,
        "scene_filter": args.scene_filter,
        "scene_filter_token_count": len(scene_filter_tokens) if scene_filter_tokens is not None else None,
        "split_log_counts": {split: len(logs) for split, logs in split_logs.items()},
        "selection": args.selection,
        "num_selected": len(selected),
        "compare_hidden_state": bool(args.compare_hidden_state),
        "chunk_record_counts": dict(chunk_counts),
        "chunk_unique_token_counts": chunk_unique,
        "raw_selected_scene_count": len(scene_loader.tokens),
        "raw_selected_unique_token_count": len(set(scene_loader.tokens)),
        "raw_full_split_counts": count_raw_split_tokens(args, split_logs, all_chunk_records),
        "samples": sample_results,
    }
    args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output_report)
    print(json.dumps({
        "status": report["status"],
        "num_selected": report["num_selected"],
        "compare_hidden_state": report["compare_hidden_state"],
        "chunk_record_counts": report["chunk_record_counts"],
        "raw_selected_scene_count": report["raw_selected_scene_count"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
