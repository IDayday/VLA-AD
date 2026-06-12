from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.recogdrive.offline_rl_buffer import save_elite_record
from navsim.agents.recogdrive.recogdrive_agent import (
    EXPERT_FEATURE_KEYS,
    EXPERT_TARGET_FEATURE_KEYS,
    LAST_VLA_TARGET_KEYS,
    TWO_EXPERT_TARGET_KEYS,
)
from navsim.planning.script.run_training_recogdrive_rl import (
    TokenizedDataset,
    build_datasets,
    custom_collate_fn,
)
from navsim.planning.training.dataset import CacheOnlyDataset


logger = logging.getLogger(__name__)

CONFIG_PATH = "../../navsim/planning/script/config/training"
CONFIG_NAME = "default_training"


class TokenizedCacheOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: CacheOnlyDataset):
        self._dataset = dataset
        self._tokens = getattr(dataset, "tokens", None)
        if self._tokens is None:
            raise AttributeError("CacheOnlyDataset is missing tokens; cannot build token-keyed AWAC buffer.")

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]:
        sample = self._dataset[idx]
        if len(sample) == 3:
            return sample
        features, targets = sample
        return features, targets, self._tokens[idx]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _move_features_to_device(agent: AbstractAgent, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, BatchFeature]:
    action_head = agent.action_head
    action_device = next(action_head.parameters()).device
    model_dtype = next(action_head.parameters()).dtype
    features = {
        key: value.to(action_device) if isinstance(value, torch.Tensor) else value
        for key, value in features.items()
    }
    for key in EXPERT_FEATURE_KEYS:
        if key in features and isinstance(features[key], torch.Tensor):
            features[key] = features[key].to(model_dtype)
    agent._add_dummy_expert_features_if_needed(features, action_device, model_dtype)

    if "last_hidden_state" not in features:
        raise KeyError(
            "AWAC elite buffer builder currently requires cached last_hidden_state. "
            "Run with agent.cache_hidden_state=true and a ReCogDrive hidden-state cache."
        )
    last_hidden_state = features["last_hidden_state"].to(device=action_device, dtype=model_dtype)
    if last_hidden_state.ndim == 2:
        last_hidden_state = last_hidden_state.unsqueeze(0)

    history_trajectory = features["history_trajectory"].to(action_device)
    if history_trajectory.ndim == 2:
        history_trajectory = history_trajectory.unsqueeze(0)
    status_feature = features["status_feature"].to(action_device)
    if status_feature.ndim == 1:
        status_feature = status_feature.unsqueeze(0)
    high_command_one_hot = features["high_command_one_hot"].to(action_device)
    if high_command_one_hot.ndim == 1:
        high_command_one_hot = high_command_one_hot.unsqueeze(0)

    history_trajectory_reshaped = history_trajectory.view(history_trajectory.size(0), -1)
    input_state = torch.cat([status_feature, history_trajectory_reshaped], dim=1)
    action_input_data: Dict[str, Any] = {
        "state": input_state.to(model_dtype),
        "his_traj": history_trajectory_reshaped.to(model_dtype),
        "history_trajectory": history_trajectory.to(model_dtype),
        "status_feature": status_feature.to(model_dtype),
        "high_command_one_hot": high_command_one_hot.to(model_dtype),
    }

    target_feature_keys = set(EXPERT_TARGET_FEATURE_KEYS)
    if getattr(agent, "use_last_vla", False):
        target_feature_keys.update(LAST_VLA_TARGET_KEYS)
    if getattr(agent, "use_two_expert_slots", False):
        target_feature_keys.update(TWO_EXPERT_TARGET_KEYS)
    for key in EXPERT_FEATURE_KEYS:
        if key in features and isinstance(features[key], torch.Tensor) and key not in target_feature_keys:
            action_input_data[key] = features[key].to(model_dtype)
    return last_hidden_state, BatchFeature(data=action_input_data)


def _build_train_dataset(cfg: DictConfig, agent: AbstractAgent):
    if cfg.use_cache_without_dataset:
        dataset = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        return TokenizedCacheOnlyDataset(dataset)
    train_data, _ = build_datasets(cfg, agent)
    return TokenizedDataset(train_data)


def _component_value(components: Dict[str, torch.Tensor], key: str, batch_idx: int, candidate_idx: int) -> float:
    return float(components[key][batch_idx, candidate_idx].detach().cpu().item())


def _iter_summary_rows(
    tokens: List[str],
    awac_batch: Dict[str, Any],
) -> Iterable[Dict[str, Any]]:
    candidate_rewards = awac_batch["candidate_rewards"]
    candidate_components = awac_batch["candidate_components"]
    candidate_sources = awac_batch["candidate_sources"]
    for batch_idx, token in enumerate(tokens):
        best_idx = int(candidate_rewards[batch_idx].argmax().detach().cpu().item())
        gt_reward = float(awac_batch["gt_reward"][batch_idx].detach().cpu().item())
        il_reward = float(awac_batch["il_reward"][batch_idx].detach().cpu().item())
        best_reward = float(candidate_rewards[batch_idx, best_idx].detach().cpu().item())
        yield {
            "token": token,
            "gt_reward": gt_reward,
            "il_reward": il_reward,
            "best_reward": best_reward,
            "best_source": candidate_sources[best_idx],
            "best_minus_gt": best_reward - gt_reward,
            "best_minus_il": best_reward - il_reward,
            "pdms": _component_value(candidate_components, "pdms", batch_idx, best_idx),
            "nc": _component_value(candidate_components, "no_at_fault_collisions", batch_idx, best_idx),
            "dac": _component_value(candidate_components, "drivable_area_compliance", batch_idx, best_idx),
            "ttc": _component_value(candidate_components, "time_to_collision_within_bound", batch_idx, best_idx),
            "ep": _component_value(candidate_components, "ego_progress", batch_idx, best_idx),
            "comfort": _component_value(candidate_components, "history_comfort", batch_idx, best_idx),
            "ddc": _component_value(candidate_components, "driving_direction_compliance", batch_idx, best_idx),
            "tlc": _component_value(candidate_components, "traffic_light_compliance", batch_idx, best_idx),
        }


def _save_records(
    buffer_dir: Path,
    tokens: List[str],
    awac_batch: Dict[str, Any],
    dry_run: bool,
) -> None:
    selected_trajs = awac_batch["selected_trajs"].detach().cpu().float().numpy()
    selected_rewards = awac_batch["selected_rewards"].detach().cpu().float().numpy()
    selected_anchor_distance = awac_batch["selected_anchor_distance"].detach().cpu().float().numpy()
    selected_real_mask = awac_batch["selected_real_mask"].detach().cpu().bool().numpy()
    selected_components = {
        key: value.detach().cpu().float().numpy()
        for key, value in awac_batch["selected_components"].items()
    }
    selected_source_index = awac_batch["selected_source_index"].detach().cpu().long().numpy()
    candidate_rewards = awac_batch["candidate_rewards"].detach().cpu().float().numpy()
    candidate_sources = awac_batch["candidate_sources"]
    for batch_idx, token in enumerate(tokens):
        real_count = int(selected_real_mask[batch_idx].sum())
        source_indices = selected_source_index[batch_idx, :real_count]
        sources = [
            candidate_sources[int(index)] if int(index) >= 0 else "unknown"
            for index in source_indices
        ]
        global_best_idx = int(candidate_rewards[batch_idx].argmax())
        record = {
            "token": token,
            "candidates": selected_trajs[batch_idx, :real_count],
            "rewards": selected_rewards[batch_idx, :real_count],
            "components": {
                key: values[batch_idx, :real_count]
                for key, values in selected_components.items()
            },
            "sources": sources,
            "anchor_distance": selected_anchor_distance[batch_idx, :real_count],
            "gt_reward": float(awac_batch["gt_reward"][batch_idx].detach().cpu().item()),
            "il_reward": float(awac_batch["il_reward"][batch_idx].detach().cpu().item()),
            "best_reward": float(awac_batch["best_reward"][batch_idx].detach().cpu().item()),
            "best_source": candidate_sources[global_best_idx],
            "version": 1,
        }
        if not dry_run:
            save_elite_record(buffer_dir, token, record)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    torch.set_grad_enabled(False)

    out_root = Path(os.getenv("OUT_ROOT", str(cfg.output_dir))).expanduser()
    buffer_dir = Path(os.getenv("ELITE_BUFFER_DIR", str(out_root / "elite_buffer"))).expanduser()
    summary_csv = out_root / "awac_elite_buffer_summary.csv"
    dry_run = _env_flag("DRY_RUN", bool(cfg.get("dry_run", False)))
    max_scenes = _env_int("MAX_SCENES", int(cfg.get("max_scenes", 0)))
    batch_size = _env_int("BATCH_SIZE", int(cfg.dataloader.params.batch_size))

    cfg.agent.stage3_objective = "none"
    cfg.agent.offline_rl_enabled = True
    cfg.agent.offline_rl_build_candidates_online = True
    cfg.agent.offline_rl_elite_buffer_path = str(buffer_dir)
    if os.getenv("ONLINE_POLICY_SAMPLES"):
        cfg.agent.offline_rl_online_policy_samples = int(os.environ["ONLINE_POLICY_SAMPLES"])
    if os.getenv("ELITE_TOP_M"):
        cfg.agent.offline_rl_elite_top_m = int(os.environ["ELITE_TOP_M"])
    if os.getenv("IL_CHECKPOINT"):
        cfg.agent.checkpoint_path = os.environ["IL_CHECKPOINT"]
        cfg.agent.reference_policy_checkpoint = os.environ["IL_CHECKPOINT"]
    if os.getenv("VLM_PATH"):
        cfg.agent.vlm_path = os.environ["VLM_PATH"]
    if os.getenv("METRIC_CACHE_DIR"):
        cfg.agent.metric_cache_path = os.environ["METRIC_CACHE_DIR"]
    if os.getenv("CACHE_MODE"):
        cfg.agent.cache_mode = _env_flag("CACHE_MODE", bool(cfg.agent.get("cache_mode", False)))

    out_root.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        buffer_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building ReCogDrive agent for AWAC elite buffer generation.")
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    agent.action_head.eval()

    dataset = _build_train_dataset(cfg, agent)
    if max_scenes > 0:
        dataset = Subset(dataset, list(range(min(max_scenes, len(dataset)))))
    dataloader_params = dict(cfg.dataloader.params)
    dataloader_params["batch_size"] = batch_size
    dataloader_params["shuffle"] = False
    dataloader = DataLoader(dataset, collate_fn=custom_collate_fn, **dataloader_params)

    summary_fields = [
        "token",
        "gt_reward",
        "il_reward",
        "best_reward",
        "best_source",
        "best_minus_gt",
        "best_minus_il",
        "pdms",
        "nc",
        "dac",
        "ttc",
        "ep",
        "comfort",
        "ddc",
        "tlc",
    ]
    written = 0
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for features, targets, tokens in dataloader:
            tokens = [str(token) for token in tokens]
            last_hidden_state, action_inputs = _move_features_to_device(agent, features)
            action_device = next(agent.action_head.parameters()).device
            model_dtype = next(agent.action_head.parameters()).dtype
            action_inputs["action"] = targets["trajectory"].to(device=action_device, dtype=model_dtype)
            metric_cache = agent.action_head._load_metric_cache_for_tokens(tokens)
            awac_batch = agent.action_head._build_online_awac_candidates(
                last_hidden_state,
                action_inputs,
                tokens,
                metric_cache,
                agent.action_head.offline_rl_cfg,
            )
            _save_records(buffer_dir, tokens, awac_batch, dry_run)
            for row in _iter_summary_rows(tokens, awac_batch):
                writer.writerow(row)
            written += len(tokens)
            if written % 256 == 0:
                logger.info("Processed %d scenes for AWAC elite buffer.", written)

    logger.info("AWAC elite buffer summary written to %s", summary_csv)
    if dry_run:
        logger.info("DRY_RUN=true; elite records were not written.")
    else:
        logger.info("AWAC elite records written under %s", buffer_dir)


if __name__ == "__main__":
    main()
