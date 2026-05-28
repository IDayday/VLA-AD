#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCENE_FILTER_DIR = REPO_ROOT / "navsim/planning/script/config/common/train_test_split/scene_filter"
KNOWN_SPLITS = ["navtrain", "navtest", "trainval", "test", "mini", "navmini", "warmup_test_e2e", "private_test_e2e"]


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
    parser = argparse.ArgumentParser(description="Inspect real NAVSIM data for ReCogDrive expert-token runs.")
    parser.add_argument("--navsim-root", type=Path, default=Path("/mnt/navsim"))
    parser.add_argument("--max-scenes", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("/mnt/project/VLA-AD/experiments/recogdrive_expert/navsim_inspection.md"))
    return parser.parse_args()


def _optional_env_path(name: str) -> Optional[Path]:
    value = os.environ.get(name)
    return Path(value) if value else None


def autodetect_data_paths(navsim_root: Path) -> Dict[str, Optional[Path]]:
    openscene_candidates = [
        _optional_env_path("OPENSCENE_DATA_ROOT"),
        navsim_root / "openscene",
        navsim_root / "open_scene",
        navsim_root,
    ]
    maps_candidates = [
        _optional_env_path("NUPLAN_MAPS_ROOT"),
        navsim_root / "maps",
        navsim_root / "nuplan_maps",
        navsim_root / "nuplan-maps-v1.0",
    ]
    openscene = next((p for p in openscene_candidates if p is not None and p.exists()), None)
    maps = next((p for p in maps_candidates if p is not None and p.exists()), None)
    if openscene:
        os.environ.setdefault("OPENSCENE_DATA_ROOT", str(openscene))
    if maps:
        os.environ.setdefault("NUPLAN_MAPS_ROOT", str(maps))
    os.environ.setdefault("NAVSIM_DATA_ROOT", str(navsim_root))
    return {"navsim_root": navsim_root, "openscene_root": openscene, "maps_root": maps}


def _split_aliases(split: str) -> List[str]:
    aliases = {
        "navtrain": ["navtrain", "trainval"],
        "navtest": ["navtest", "test"],
        "navmini": ["navmini", "mini"],
    }
    return aliases.get(split, [split])


def split_paths(openscene: Path, split: str) -> Tuple[Optional[Path], Optional[Path], List[str]]:
    log_candidates: List[Path] = []
    blob_candidates: List[Path] = []
    for name in _split_aliases(split):
        log_candidates.extend([
            openscene / "navsim_logs" / name,
            openscene / name / "navsim_logs",
            openscene / name,
            openscene / f"{name}_navsim_logs" / name,
            openscene / f"{name}_navsim_log" / name,
            openscene / f"{name}_navsim_logs",
            openscene / f"{name}_navsim_log",
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
    tried = [*(str(p) for p in log_candidates), *(str(p) for p in blob_candidates)]
    return logs, blobs, tried


def load_scene_filter(split: str, max_scenes: int):
    from navsim.common.dataclasses import SceneFilter

    path = SCENE_FILTER_DIR / f"{split}.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw.pop("_target_", None)
    raw.pop("_convert_", None)
    raw["max_scenes"] = max_scenes
    return SceneFilter(**raw), path


def frame_camera_path(frame: Dict[str, Any], blobs: Path, camera_key: str = "cam_f0") -> Optional[str]:
    for raw_key, camera in frame.get("cams", {}).items():
        if raw_key.lower() == camera_key:
            return str(blobs / camera["data_path"])
    return None


def inspect_split(split: str, roots: Dict[str, Optional[Path]], max_scenes: int) -> Dict[str, Any]:
    result: Dict[str, Any] = {"split": split, "ok": False}
    openscene = roots.get("openscene_root")
    if openscene is None:
        result["error"] = "openscene root not found"
        return result
    logs, blobs, tried = split_paths(openscene, split)
    result.update({"logs_path": str(logs) if logs else None, "sensor_blobs_path": str(blobs) if blobs else None, "tried_paths": tried})
    if logs is None or blobs is None:
        result["error"] = "split log path or sensor blob path not found"
        return result
    try:
        from navsim.common.dataloader import SceneLoader
        from navsim.common.dataclasses import SensorConfig
        from navsim.agents.recogdrive.recogdrive_features import ReCogDriveFeatureBuilder, TrajectoryTargetBuilder
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

        scene_filter, scene_filter_path = load_scene_filter(split, max_scenes)
        loader = SceneLoader(
            data_path=logs,
            sensor_blobs_path=blobs,
            scene_filter=scene_filter,
            sensor_config=cam_f0_sensor_config(),
            load_image_path=True,
        )
        result["scene_filter"] = str(scene_filter_path)
        result["num_scenes"] = len(loader)
        result["tokens"] = loader.tokens[:max_scenes]
        if not loader.tokens:
            result["error"] = "loader returned zero scenes"
            return result
        token = loader.tokens[0]
        frames = loader.scene_frames_dicts[token]
        h = scene_filter.num_history_frames
        result["example_scene_token"] = frames[h - 1].get("scene_token")
        result["example_sample_token"] = frames[h - 1].get("token")
        history_paths = [frame_camera_path(frame, blobs) for frame in frames[:h]]
        future_paths = [frame_camera_path(frame, blobs) for frame in frames[h:h + 4]]
        result["example_image_paths"] = {"history_cam_f0": history_paths, "future_cam_f0": future_paths}
        result["cam_f0_images_exist"] = all(p is not None and Path(p).is_file() for p in history_paths[-1:])
        result["history_frames_available"] = len([p for p in history_paths if p]) >= 4
        result["future_frames_available_for_jepa_target"] = len([p for p in future_paths if p]) >= 4
        agent_input = loader.get_agent_input_from_token(token)
        features = ReCogDriveFeatureBuilder(cache_hidden_state=False, use_expert_features=False).compute_features(agent_input)
        scene = loader.get_scene_from_token(token)
        target = TrajectoryTargetBuilder(TrajectorySampling(time_horizon=4, interval_length=0.5)).compute_targets(scene)
        result["ego_status_and_trajectory_accessible"] = "status_feature" in features and "trajectory" in target
        result["recogdrive_feature_builder_basic_ok"] = all(k in features for k in ["history_trajectory", "high_command_one_hot", "status_feature", "image_path_tensor"])
        result["evaluation_split_loadable"] = split == "navtest" and len(loader) > 0
        result["ok"] = True
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def available_split_dirs(root: Path) -> List[str]:
    names = set()
    for child in root.iterdir() if root.exists() else []:
        if child.is_dir():
            names.add(child.name)
            for grand in child.iterdir():
                if grand.is_dir():
                    names.add(grand.name)
    return sorted(names)


def write_md(path: Path, report: Dict[str, Any]) -> None:
    lines = ["# NAVSIM Real Data Inspection", ""]
    roots = report["roots"]
    for key, value in roots.items():
        lines.append(f"- {key}: `{value}`")
    lines.append(f"- available directory names: {', '.join(report['available_directories'])}")
    lines.append("")
    lines.append("## Splits")
    lines.append("")
    for split, item in report["splits"].items():
        lines.append(f"### {split}")
        lines.append(f"- ok: {item.get('ok')}")
        lines.append(f"- scenes/samples loaded: {item.get('num_scenes')}")
        lines.append(f"- cam_f0 images exist: {item.get('cam_f0_images_exist')}")
        lines.append(f"- history frames available: {item.get('history_frames_available')}")
        lines.append(f"- future frames available for JEPA target: {item.get('future_frames_available_for_jepa_target')}")
        lines.append(f"- ego status and trajectory target accessible: {item.get('ego_status_and_trajectory_accessible')}")
        lines.append(f"- feature builder basic ok: {item.get('recogdrive_feature_builder_basic_ok')}")
        lines.append(f"- evaluation split loadable: {item.get('evaluation_split_loadable')}")
        lines.append(f"- example scene token: `{item.get('example_scene_token')}`")
        lines.append(f"- example sample token: `{item.get('example_sample_token')}`")
        if item.get("example_image_paths"):
            lines.append("- example image paths:")
            for group, values in item["example_image_paths"].items():
                lines.append(f"  - {group}: {values}")
        if item.get("error"):
            lines.append(f"- error: `{item['error']}`")
        lines.append("")
    lines.append(f"Overall OK: {report['ok']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    roots = autodetect_data_paths(args.navsim_root)
    splits_to_check = [split for split in KNOWN_SPLITS if (SCENE_FILTER_DIR / f"{split}.yaml").is_file()]
    report = {
        "roots": {key: str(value) if value else None for key, value in roots.items()},
        "available_directories": available_split_dirs(args.navsim_root),
        "splits": {},
    }
    for split in splits_to_check:
        report["splits"][split] = inspect_split(split, roots, args.max_scenes)
    report["ok"] = bool(report["splits"].get("navtrain", {}).get("ok") and report["splits"].get("navtest", {}).get("ok"))
    write_md(args.output, report)
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
