#!/usr/bin/env python3
"""Build a coherent GT/Stage2 reference cache for LFP-GRPO."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import pickle
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader, Dataset, Subset
from transformers.feature_extraction_utils import BatchFeature

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.recogdrive_agent import (
    EXPERT_FEATURE_KEYS,
    EXPERT_TARGET_FEATURE_KEYS,
    LAST_VLA_TARGET_KEYS,
    TWO_EXPERT_TARGET_KEYS,
)
from navsim.agents.recogdrive.stage3_metric_adapter import Stage3MetricAdapter
from navsim.agents.recogdrive.stage3_reference_cache import select_coherent_reference
from navsim.common.dataclasses import Trajectory
from navsim.planning.script.run_training_recogdrive_rl import (
    IndexedPtCacheDataset,
    custom_collate_fn,
)
from navsim.planning.training.dataset import CacheOnlyDataset


LOG = logging.getLogger("build_lfp_reference_cache")


class _TokenDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        tokens = getattr(dataset, "tokens", None)
        if tokens is None:
            raise AttributeError("Reference-cache dataset must expose tokens in index order.")
        self.tokens = [str(token) for token in tokens]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        if len(sample) == 3:
            return sample
        features, targets = sample
        return features, targets, self.tokens[index]


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _directory_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    root = path.expanduser().resolve()
    if root.is_file():
        return _sha256_file(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        stat = item.stat()
        digest.update(str(item.relative_to(root)).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _find_checkpoint_config(checkpoint: Path) -> Optional[Path]:
    candidates = []
    for parent in checkpoint.resolve().parents:
        candidates.extend(
            [
                parent / "code" / "hydra" / "config.yaml",
                parent / ".hydra" / "config.yaml",
                parent / "hydra" / "config.yaml",
                parent / "config.yaml",
            ]
        )
        if len(candidates) >= 40:
            break
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _load_training_config(args: argparse.Namespace) -> DictConfig:
    explicit = args.config_path.expanduser().resolve() if args.config_path else None
    inferred = explicit or _find_checkpoint_config(args.stage2_checkpoint)
    if inferred is not None:
        LOG.info("Loading resolved Stage2 config from %s", inferred)
        cfg = OmegaConf.load(inferred)
    else:
        config_dir = REPO_ROOT / "navsim" / "planning" / "script" / "config" / "training"
        LOG.warning(
            "No checkpoint-adjacent Hydra config found; composing repository defaults. "
            "Pass --config_path for non-default Stage2 architectures."
        )
        with initialize_config_dir(config_dir=str(config_dir), version_base=None, job_name="lfp_reference"):
            cfg = compose(config_name="default_training")
    if args.hydra_override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.hydra_override))
    if "agent" not in cfg:
        raise KeyError("Resolved configuration has no agent section.")
    return cfg


def _prepare_agent_config(cfg: DictConfig, args: argparse.Namespace) -> None:
    with open_dict(cfg.agent):
        cfg.agent.checkpoint_path = str(args.stage2_checkpoint)
        cfg.agent.reference_policy_checkpoint = str(args.stage2_checkpoint)
        cfg.agent.stage3_algorithm = "legacy"
        cfg.agent.stage3_objective = "none"
        cfg.agent.grpo = False
        cfg.agent.cache_hidden_state = True
        # Hidden states already exist; cache_mode=True would initialize the VLM
        # inside the feature builder even though no feature is recomputed.
        cfg.agent.cache_mode = False
        cfg.agent.allow_random_init = False
        cfg.agent.metric_cache_path = str(args.metric_cache_path)
        if args.vlm_path:
            cfg.agent.vlm_path = str(args.vlm_path)
        if args.fs_norm_stats_path:
            cfg.agent.fs_norm_stats_path = str(args.fs_norm_stats_path)
        if "lfp_grpo_cfg" in cfg.agent:
            cfg.agent.lfp_grpo_cfg.enabled = False


def _make_dataset(agent: Any, cache_path: Path) -> _TokenDataset:
    if IndexedPtCacheDataset.looks_like(str(cache_path)):
        dataset = IndexedPtCacheDataset(str(cache_path))
    else:
        dataset = CacheOnlyDataset(
            cache_path=str(cache_path),
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=None,
        )
    return _TokenDataset(dataset)


def _move_features(agent: Any, features: Dict[str, Any]) -> tuple[torch.Tensor, BatchFeature]:
    planner = agent.action_head
    parameter = next(planner.parameters())
    device, dtype = parameter.device, parameter.dtype
    features = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in features.items()
    }
    for key in EXPERT_FEATURE_KEYS:
        if key in features and isinstance(features[key], torch.Tensor):
            features[key] = features[key].to(dtype)
    agent._add_dummy_expert_features_if_needed(features, device, dtype)
    if "last_hidden_state" not in features:
        raise KeyError("Reference-cache construction requires cached last_hidden_state.")
    hidden = features["last_hidden_state"].to(device=device, dtype=dtype)
    history = features["history_trajectory"].to(device=device, dtype=dtype)
    status = features["status_feature"].to(device=device, dtype=dtype)
    command = features["high_command_one_hot"].to(device=device, dtype=dtype)
    history_flat = history.reshape(history.shape[0], -1)
    action_data: Dict[str, Any] = {
        "state": torch.cat((status, history_flat), dim=1),
        "his_traj": history_flat,
        "history_trajectory": history,
        "status_feature": status,
        "high_command_one_hot": command,
    }
    target_keys = set(EXPERT_TARGET_FEATURE_KEYS)
    if getattr(agent, "use_last_vla", False):
        target_keys.update(LAST_VLA_TARGET_KEYS)
    if getattr(agent, "use_two_expert_slots", False):
        target_keys.update(TWO_EXPERT_TARGET_KEYS)
    for key in EXPERT_FEATURE_KEYS:
        value = features.get(key)
        if isinstance(value, torch.Tensor) and key not in target_keys:
            action_data[key] = value.to(dtype)
    return hidden, BatchFeature(data=action_data)


def _deterministic_initial_noise(tokens: list[str], planner: Any, device: torch.device, dtype: torch.dtype):
    rows = []
    for token in tokens:
        seed = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "little") & 0x7FFF_FFFF
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        rows.append(
            torch.randn(
                planner.config.action_horizon,
                planner.config.action_dim,
                generator=generator,
                device=device,
                dtype=dtype,
            )
        )
    return torch.stack(rows)


def _canonical_row(adapter: Stage3MetricAdapter, metrics: Mapping[str, Any]) -> Dict[str, float]:
    canonical = adapter.canonicalize(metrics, batch_size=1, group_size=1)
    row = {
        "scalar": float(canonical.scalar.item()),
        "ep": float(canonical.ep.item()),
        "ttc": float(canonical.ttc.item()),
        "quality": float(canonical.quality.item()),
        "nc": float(canonical.nc.item()),
        "dac": float(canonical.dac.item()),
        "ddc": float(canonical.ddc_guard_value.item()),
    }
    if canonical.tlc is not None:
        row["tlc"] = float(canonical.tlc.item())
    return row


def _tensor_components_to_rows(
    adapter: Stage3MetricAdapter,
    components: Mapping[str, torch.Tensor],
    tokens: list[str],
) -> Dict[str, Dict[str, float]]:
    output = {}
    for index, token in enumerate(tokens):
        output[token] = _canonical_row(
            adapter,
            {key: value[index].reshape(1, 1) for key, value in components.items()},
        )
    return output


def _load_csv_metrics(
    path: Path,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, str], Dict[str, str]]:
    rows: Dict[str, Dict[str, Any]] = {}
    stage_types: Dict[str, str] = {}
    previous_tokens: Dict[str, str] = {}
    with path.expanduser().open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            token = str(row.get("token", ""))
            if not token or token.startswith(("average_", "extended_pdm_")):
                continue
            converted: Dict[str, Any] = {}
            for key, value in row.items():
                if value is None or value == "":
                    continue
                try:
                    converted[key] = float(value)
                except ValueError:
                    converted[key] = value
            rows[token] = converted
            frame = str(row.get("frame_type", "")).lower()
            stage_types[token] = "followup" if "synthetic" in frame else ("first" if frame else "unknown")
            previous = str(row.get("previous_token", "")).strip()
            if previous and previous.lower() != "nan":
                previous_tokens[token] = previous
    if not rows:
        raise ValueError(f"No token metric rows found in {path}.")
    return rows, stage_types, previous_tokens


def _write_submission(path: Path, trajectories: Mapping[str, np.ndarray]) -> None:
    predictions = {
        token: Trajectory(np.asarray(trajectory, dtype=np.float32))
        for token, trajectory in trajectories.items()
    }
    with path.open("wb") as handle:
        pickle.dump({"predictions": [predictions]}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _score_v2_submission(
    args: argparse.Namespace,
    trajectories: Mapping[str, np.ndarray],
    label: str,
    work_dir: Path,
) -> Path:
    if args.official_navsim_root is None:
        raise ValueError(
            "NAVSIM v2 cache construction requires --official_navsim_root, or both "
            "--gt_metrics_path and --stage2_metrics_path from the official scorer."
        )
    submission = work_dir / f"{label}_submission.pkl"
    output = work_dir / f"{label}_metrics.csv"
    _write_submission(submission, trajectories)
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "stage3" / "score_lfp_v2_one_stage.py"),
        "--submission_path",
        str(submission),
        "--navsim_root",
        str(args.official_navsim_root),
        "--metric_cache_path",
        str(args.metric_cache_path),
        "--output_path",
        str(output),
        "--config_name",
        args.official_config_name,
    ]
    if args.official_config_dir:
        command.extend(("--config_dir", str(args.official_config_dir)))
    for override in args.official_override:
        command.extend(("--override", override))
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=os.environ.copy())
    return output


def _is_feasible(row: Mapping[str, float], gt_ddc: float, benchmark: str, args: argparse.Namespace) -> bool:
    feasible = row["nc"] >= 1.0 and row["dac"] >= 1.0
    feasible = feasible and row["ddc"] >= float(gt_ddc) - float(args.ddc_gt_tolerance)
    if benchmark == "navsim_v2" and args.v2_require_tlc:
        feasible = feasible and row.get("tlc", float("-inf")) >= 1.0
    return bool(feasible)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("navsim_v1", "navsim_v2"), required=True)
    parser.add_argument("--stage2_checkpoint", "--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--cache_path", "--cache-path", type=Path, required=True)
    parser.add_argument("--metric_cache_path", "--metric-cache-path", type=Path, required=True)
    parser.add_argument("--output_path", "--output-path", type=Path, required=True)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=8)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=4)
    parser.add_argument("--max_scenes", "--max-scenes", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config_path", "--config-path", type=Path)
    parser.add_argument("--hydra_override", "--hydra-override", action="append", default=[])
    parser.add_argument("--vlm_path", "--vlm-path", type=Path)
    parser.add_argument("--fs_norm_stats_path", "--fs-norm-stats-path", type=Path)
    parser.add_argument("--gt_metrics_path", "--gt-metrics-path", type=Path)
    parser.add_argument("--stage2_metrics_path", "--stage2-metrics-path", type=Path)
    parser.add_argument("--official_navsim_root", "--official-navsim-root", type=Path)
    parser.add_argument("--official_config_dir", "--official-config-dir", type=Path)
    parser.add_argument("--official_config_name", default="default_run_pdm_score")
    parser.add_argument("--official_split", default="navtrain")
    parser.add_argument("--official_override", action="append", default=[])
    parser.add_argument("--ddc_gt_tolerance", type=float, default=0.01)
    parser.add_argument("--v2_require_tlc", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.stage2_checkpoint = args.stage2_checkpoint.expanduser().resolve()
    args.cache_path = args.cache_path.expanduser().resolve()
    args.metric_cache_path = args.metric_cache_path.expanduser().resolve()
    args.output_path = args.output_path.expanduser().resolve()
    if args.official_split and not any(
        override.startswith("train_test_split=") for override in args.official_override
    ):
        args.official_override.insert(0, f"train_test_split={args.official_split}")
    for path, label in (
        (args.stage2_checkpoint, "stage2 checkpoint"),
        (args.cache_path, "hidden cache"),
        (args.metric_cache_path, "metric cache"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if args.batch_size <= 0 or args.num_workers < 0 or args.max_scenes < 0:
        raise ValueError("batch_size must be positive; num_workers/max_scenes must be non-negative.")

    cfg = _load_training_config(args)
    _prepare_agent_config(cfg, args)
    agent = instantiate(cfg.agent)
    agent.initialize()
    device = torch.device(args.device)
    agent.to(device)
    agent.eval()
    planner = agent.action_head
    planner.eval()
    planner.config.grpo_cfg.metric_cache_path = str(args.metric_cache_path)

    dataset: Dataset = _make_dataset(agent, args.cache_path)
    if args.max_scenes:
        dataset = Subset(dataset, range(min(args.max_scenes, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=custom_collate_fn,
        pin_memory=device.type == "cuda",
    )
    adapter = Stage3MetricAdapter(args.benchmark)
    gt_trajectories: Dict[str, np.ndarray] = {}
    stage2_trajectories: Dict[str, np.ndarray] = {}
    gt_rows: Dict[str, Dict[str, float]] = {}
    stage2_rows: Dict[str, Dict[str, float]] = {}
    stage_types: Dict[str, str] = {}
    previous_tokens: Dict[str, str] = {}

    if args.benchmark == "navsim_v1":
        planner._init_stage3_oracle(planner.config.grpo_cfg)
    with torch.inference_mode():
        for batch_index, (features, targets, batch_tokens) in enumerate(loader):
            tokens = [str(token) for token in batch_tokens]
            hidden, action_input = _move_features(agent, features)
            parameter = next(planner.parameters())
            initial_noise = _deterministic_initial_noise(tokens, planner, parameter.device, parameter.dtype)
            prediction_output = planner.get_action(
                hidden,
                action_input,
                init_actions=initial_noise,
                deterministic=True,
            )
            prediction = prediction_output["pred_traj"]
            gt = targets["trajectory"].to(device=parameter.device, dtype=prediction.dtype)
            for index, token in enumerate(tokens):
                gt_trajectories[token] = gt[index].detach().float().cpu().numpy()
                stage2_trajectories[token] = prediction[index].detach().float().cpu().numpy()
            if args.benchmark == "navsim_v1":
                cache = planner._load_metric_cache_for_tokens(tokens)
                flat = torch.stack((gt, prediction), dim=1).reshape(-1, gt.shape[1], gt.shape[2])
                repeated_tokens = [token for token in tokens for _ in range(2)]
                _, components = planner.reward_fn(
                    flat,
                    repeated_tokens,
                    cache,
                    return_components=True,
                    strict_submetrics=True,
                    required_submetrics=(
                        "pdms",
                        "no_at_fault_collisions",
                        "drivable_area_compliance",
                        "time_to_collision_within_bound",
                        "ego_progress",
                        "history_comfort",
                        "driving_direction_compliance",
                    ),
                    missing_submetric_policy="error",
                    use_batched_pdm_scoring=True,
                    use_exact_array_pdm_state_conversion=True,
                    use_fast_pdm_scorer=True,
                )
                gt_components = {key: value[0::2] for key, value in components.items()}
                stage2_components = {key: value[1::2] for key, value in components.items()}
                gt_rows.update(_tensor_components_to_rows(adapter, gt_components, tokens))
                stage2_rows.update(_tensor_components_to_rows(adapter, stage2_components, tokens))
            if batch_index % 10 == 0:
                LOG.info("Generated deterministic references for %d scenes", len(stage2_trajectories))

    if args.benchmark == "navsim_v2":
        with tempfile.TemporaryDirectory(prefix="lfp-v2-reference-") as temporary:
            work_dir = Path(temporary)
            gt_metrics = args.gt_metrics_path or _score_v2_submission(args, gt_trajectories, "gt", work_dir)
            stage2_metrics = args.stage2_metrics_path or _score_v2_submission(
                args, stage2_trajectories, "stage2", work_dir
            )
            raw_gt, gt_stage_types, gt_previous = _load_csv_metrics(gt_metrics)
            raw_stage2, stage2_stage_types, stage2_previous = _load_csv_metrics(stage2_metrics)
            for token in stage2_trajectories:
                if token not in raw_gt or token not in raw_stage2:
                    raise KeyError(f"Official NAVSIM v2 metrics are missing token {token!r}.")
                gt_rows[token] = _canonical_row(adapter, raw_gt[token])
                stage2_rows[token] = _canonical_row(adapter, raw_stage2[token])
                stage_types[token] = gt_stage_types.get(token, stage2_stage_types.get(token, "unknown"))
                previous = gt_previous.get(token, stage2_previous.get(token))
                if previous:
                    previous_tokens[token] = previous

    if set(gt_rows) != set(stage2_trajectories) or set(stage2_rows) != set(stage2_trajectories):
        raise RuntimeError("Reference metric rows and deterministic trajectory tokens are inconsistent.")
    records: Dict[str, Any] = {}
    raw_ddc_count = 0
    for token in stage2_trajectories:
        gt_row = gt_rows[token]
        stage2_row = stage2_rows[token]
        gt_feasible = _is_feasible(gt_row, gt_row["ddc"], args.benchmark, args)
        stage2_feasible = _is_feasible(stage2_row, gt_row["ddc"], args.benchmark, args)
        record = select_coherent_reference(
            gt_row,
            stage2_row,
            gt_feasible=gt_feasible,
            stage2_feasible=stage2_feasible,
        )
        record.update(
            {
                "gt": gt_row,
                "stage2": stage2_row,
                "gt_feasible": gt_feasible,
                "stage2_feasible": stage2_feasible,
                "gt_trajectory": gt_trajectories[token].tolist(),
                "stage2_trajectory": stage2_trajectories[token].tolist(),
                "scene_stage_type": stage_types.get(token, "unknown"),
            }
        )
        previous_token = previous_tokens.get(token)
        if args.benchmark == "navsim_v2" and not previous_token:
            raise KeyError(
                f"NAVSIM v2 token {token!r} has no official adjacent previous token. "
                "LFP requires two-frame extended comfort for every training token; "
                "build from an adjacent-token split or provide scorer CSVs retaining previous_token."
            )
        if previous_token:
            if previous_token not in stage2_trajectories:
                raise KeyError(
                    f"v2 token {token!r} requires previous token {previous_token!r}, "
                    "which is absent from the deterministic Stage2 cache build."
                )
            record["previous_token"] = previous_token
            record["previous_stage2_trajectory"] = stage2_trajectories[previous_token].tolist()
        records[token] = record
        raw_ddc_count += int(adapter.last_field_mapping.get("ddc_guard_value", "").startswith("raw_"))

    metadata = {
        "version": 1,
        "benchmark": args.benchmark,
        "stage2_checkpoint_path": str(args.stage2_checkpoint),
        "stage2_checkpoint_sha256": _sha256_file(args.stage2_checkpoint),
        "metric_cache_path": str(args.metric_cache_path),
        "metric_cache_fingerprint": _directory_fingerprint(args.metric_cache_path),
        "scene_count": len(records),
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "component_field_mapping": dict(getattr(adapter, "last_field_mapping", {})),
        "raw_ddc_availability_ratio": raw_ddc_count / max(len(records), 1),
        "ddc_gt_tolerance": float(args.ddc_gt_tolerance),
        "v2_require_tlc": bool(args.v2_require_tlc),
    }
    if args.benchmark == "navsim_v2":
        metadata.update(
            {
                "one_stage_epdms": True,
                "extended_comfort_context": "frozen_stage2_previous_trajectory",
                "official_navsim_root": (
                    str(args.official_navsim_root.expanduser().resolve())
                    if args.official_navsim_root is not None
                    else "external_official_metric_csv"
                ),
            }
        )
    payload = {"metadata": metadata, "records": records}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.output_path.suffix == ".json":
        args.output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    else:
        torch.save(payload, args.output_path)
    LOG.info("Wrote %d coherent references to %s", len(records), args.output_path)


if __name__ == "__main__":
    main()
