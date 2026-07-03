#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import default_collate
from tqdm import tqdm


def _insert_external_navsim(ddv2_root: Path) -> None:
    root = str(ddv2_root.resolve())
    sys.path = [p for p in sys.path if str(Path(p).resolve()) != root] if root else sys.path
    sys.path.insert(0, root)
    os.chdir(root)


def _to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_device(v, device) for v in value)
    return value


def _load_log_names(path: str) -> list[str] | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("log_names", data.get("logs", data.get("items")))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list or a dict with log_names/logs/items.")
    return [str(x) for x in data]


def _resolve_navsim_split_paths(data_root: Path, data_split: str) -> tuple[Path, Path]:
    candidates = [
        (data_root / "navsim_logs" / data_split, data_root / "sensor_blobs" / data_split),
        (data_root / f"{data_split}_navsim_logs" / data_split, data_root / f"{data_split}_sensor_blobs" / data_split),
    ]
    for log_path, sensor_path in candidates:
        if log_path.is_dir() and sensor_path.is_dir():
            return log_path.resolve(), sensor_path.resolve()
    details = "\n".join(f"  logs={log_path} sensors={sensor_path}" for log_path, sensor_path in candidates)
    raise FileNotFoundError(f"Could not resolve NAVSIM split paths for data_root={data_root}, split={data_split}:\n{details}")


def _compose_cfg(args: argparse.Namespace):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    config_dir = args.ddv2_root / "navsim" / "planning" / "script" / "config" / "pdm_scoring"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="default_run_create_submission_pickle",
            overrides=[
                f"train_test_split={args.train_test_split}",
                f"agent={args.agent_name}",
            ],
        )
    OmegaConf.set_struct(cfg, False)
    with open_dict(cfg):
        cfg.agent.checkpoint_path = str(args.checkpoint)
        cfg.output_dir = str(args.output_dir)
        cfg.experiment_name = args.experiment_name
        cfg.team_name = args.team_name
        cfg.authors = args.authors
        cfg.email = args.email
        cfg.institution = args.institution
        cfg.country = args.country
        navsim_log_path, sensor_blobs_path = _resolve_navsim_split_paths(args.data_root, str(cfg.train_test_split.data_split))
        cfg.navsim_log_path = str(navsim_log_path)
        cfg.sensor_blobs_path = str(sensor_blobs_path)
        args.resolved_navsim_log_path = str(navsim_log_path)
        args.resolved_sensor_blobs_path = str(sensor_blobs_path)
        log_names = _load_log_names(args.log_names_json)
        if log_names is not None:
            cfg.train_test_split.scene_filter.log_names = log_names
        if args.max_scenes is not None:
            cfg.train_test_split.scene_filter.max_scenes = int(args.max_scenes)
    return cfg


def _write_submission(path: Path, output: dict[str, Any], args: argparse.Namespace) -> None:
    submission = {
        "team_name": args.team_name,
        "authors": args.authors,
        "email": args.email,
        "institution": args.institution,
        "country / region": args.country,
        "predictions": [output],
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(submission, f)
    tmp_path.replace(path)


def _load_existing_predictions(output_dir: Path) -> dict[str, Any]:
    for name in ("submission.pkl", "submission.partial.pkl"):
        path = output_dir / name
        if not path.exists():
            continue
        with open(path, "rb") as f:
            payload = pickle.load(f)
        predictions = payload.get("predictions", [{}])
        if predictions and isinstance(predictions[0], dict):
            return {str(k): v for k, v in predictions[0].items()}
    return {}


def _write_summary(path: Path, args: argparse.Namespace, output_count: int, failure_count: int, total_count: int | None, complete: bool) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "agent_name": args.agent_name,
                "checkpoint": str(args.checkpoint),
                "train_test_split": args.train_test_split,
                "prediction_count": int(output_count),
                "failure_count": int(failure_count),
                "total_count": int(total_count) if total_count is not None else None,
                "complete": bool(complete),
                "batch_size": args.batch_size,
                "data_root": str(args.data_root),
                "maps_root": str(args.maps_root),
                "resolved_navsim_log_path": str(getattr(args, "resolved_navsim_log_path", "")),
                "resolved_sensor_blobs_path": str(getattr(args, "resolved_sensor_blobs_path", "")),
            },
            f,
            indent=2,
            sort_keys=True,
        )


def _flush_batch(agent: Any, feature_batch: list[dict[str, Any]], token_batch: list[str], output: dict[str, Any], device: torch.device) -> None:
    from navsim.common.dataclasses import Trajectory

    features = default_collate(feature_batch)
    features = _to_device(features, device)
    with torch.inference_mode():
        predictions = agent.forward(features)
    if "trajectory" not in predictions:
        raise KeyError(f"agent.forward did not return trajectory; keys={sorted(predictions.keys())}")
    poses = predictions["trajectory"]
    if hasattr(poses, "poses"):
        poses = poses.poses
    poses_np = poses.detach().float().cpu().numpy()
    if poses_np.shape[0] != len(token_batch):
        raise ValueError(f"Batch output size mismatch: got {poses_np.shape[0]}, expected {len(token_batch)}")
    for token, pose in zip(token_batch, poses_np):
        output[str(token)] = Trajectory(pose)


def run(args: argparse.Namespace) -> None:
    args.ddv2_root = args.ddv2_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    args.data_root = args.data_root.resolve()
    args.maps_root = args.maps_root.resolve()
    if args.log_names_json:
        args.log_names_json = str(Path(args.log_names_json).resolve())
    _insert_external_navsim(args.ddv2_root)
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(args.maps_root))
    os.environ.setdefault("OPENSCENE_DATA_ROOT", str(args.data_root))

    from hydra.utils import instantiate
    from navsim.common.dataloader import SceneLoader

    cfg = _compose_cfg(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    agent = instantiate(cfg.agent)
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    input_loader = SceneLoader(
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        sensor_config=agent.get_sensor_config(),
    )
    agent.initialize()
    agent.eval()
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    agent.to(device)

    feature_builders = agent.get_feature_builders()
    output: dict[str, Any] = _load_existing_predictions(args.output_dir) if args.resume else {}
    feature_batch: list[dict[str, Any]] = []
    token_batch: list[str] = []
    failures: list[dict[str, str]] = []
    last_partial_count = len(output)

    total = len(input_loader) if hasattr(input_loader, "__len__") else None
    for token in tqdm(input_loader, total=total, desc="Generating DDV2 submission"):
        if str(token) in output:
            continue
        try:
            agent_input = input_loader.get_agent_input_from_token(token)
            features: dict[str, Any] = {}
            for builder in feature_builders:
                features.update(builder.compute_features(agent_input))
            feature_batch.append(features)
            token_batch.append(str(token))
            if len(feature_batch) >= args.batch_size:
                _flush_batch(agent, feature_batch, token_batch, output, device)
                feature_batch = []
                token_batch = []
                if args.partial_every > 0 and len(output) - last_partial_count >= args.partial_every:
                    _write_submission(args.output_dir / "submission.partial.pkl", output, args)
                    _write_summary(
                        args.output_dir / "summary.partial.json",
                        args,
                        output_count=len(output),
                        failure_count=len(failures),
                        total_count=total,
                        complete=False,
                    )
                    last_partial_count = len(output)
        except Exception as exc:  # pragma: no cover - exercised in long-running data jobs
            failures.append({"token": str(token), "error": repr(exc), "traceback": traceback.format_exc()})
    if feature_batch:
        _flush_batch(agent, feature_batch, token_batch, output, device)

    _write_submission(args.output_dir / "submission.pkl", output, args)
    _write_summary(
        args.output_dir / "summary.json",
        args,
        output_count=len(output),
        failure_count=len(failures),
        total_count=total,
        complete=True,
    )
    for stale_partial in ("submission.partial.pkl", "summary.partial.json"):
        try:
            (args.output_dir / stale_partial).unlink()
        except FileNotFoundError:
            pass
    if failures:
        with open(args.output_dir / "failures.jsonl", "w", encoding="utf-8") as f:
            for item in failures:
                f.write(json.dumps(item, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate batched DiffusionDriveV2 NAVSIM submission candidates.")
    parser.add_argument("--ddv2-root", type=Path, default=Path("/mnt/project/external/DiffusionDriveV2"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--agent-name", default="diffusiondrivev2_sel_agent")
    parser.add_argument("--train-test-split", default="navtrain")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", default="ddv2_candidates")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("OPENSCENE_DATA_ROOT", "/mnt/navsim")))
    parser.add_argument("--maps-root", type=Path, default=Path(os.environ.get("NUPLAN_MAPS_ROOT", "/mnt/navsim/maps")))
    parser.add_argument("--log-names-json", default="")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--partial-every", type=int, default=0, help="Write submission.partial.pkl every N new predictions; 0 disables.")
    parser.add_argument("--resume", action="store_true", help="Resume from submission.pkl or submission.partial.pkl in output-dir.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--team-name", default="SGFPS")
    parser.add_argument("--authors", default="SGFPS")
    parser.add_argument("--email", default="none@example.com")
    parser.add_argument("--institution", default="NA")
    parser.add_argument("--country", default="NA")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
