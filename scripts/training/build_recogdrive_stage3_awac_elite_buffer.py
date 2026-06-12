from __future__ import annotations

import csv
import json
import logging
import os
from collections import Counter, defaultdict
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


def _source_bucket(source: str) -> str:
    if source == "gt":
        return "gt"
    if source == "il":
        return "il"
    if source.startswith("policy"):
        return "policy"
    if source.startswith("progress"):
        return "progress"
    if "lateral" in source:
        return "lateral"
    if source.startswith("timing"):
        return "timing"
    return "other"


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    mask_f = mask.to(dtype=values.dtype)
    return float(((values * mask_f).sum() / mask_f.sum().clamp(min=1.0)).detach().cpu().item())


def _iter_summary_rows(
    tokens: List[str],
    awac_batch: Dict[str, Any],
) -> Iterable[Dict[str, Any]]:
    candidate_sources = awac_batch["candidate_sources"]
    selected_rewards = awac_batch["selected_rewards"]
    selected_components = awac_batch["selected_components"]
    selected_real_mask = awac_batch["selected_real_mask"]
    selected_valid_mask = awac_batch["selected_valid_mask"]
    selected_source_index = awac_batch["selected_source_index"]
    fallback_candidate = awac_batch["fallback_candidate"]
    for batch_idx, token in enumerate(tokens):
        gt_reward = float(awac_batch["gt_reward"][batch_idx].detach().cpu().item())
        il_reward = float(awac_batch["il_reward"][batch_idx].detach().cpu().item())
        best_raw_reward = float(awac_batch["best_raw_reward"][batch_idx].detach().cpu().item())
        best_valid_reward = float(awac_batch["best_valid_reward"][batch_idx].detach().cpu().item())
        best_selected_reward = float(awac_batch["best_selected_reward"][batch_idx].detach().cpu().item())
        best_raw_idx = int(awac_batch["best_raw_index"][batch_idx].detach().cpu().item())
        best_valid_idx = int(awac_batch["best_valid_index"][batch_idx].detach().cpu().item())
        selected_sources = [
            candidate_sources[int(index)]
            for index in selected_source_index[batch_idx][selected_real_mask[batch_idx]].detach().cpu().tolist()
            if int(index) >= 0
        ]
        best_selected_source = "unknown"
        if selected_sources:
            selected_best_pos = int(selected_rewards[batch_idx].masked_fill(~selected_real_mask[batch_idx], -torch.inf).argmax().item())
            source_index = int(selected_source_index[batch_idx, selected_best_pos].detach().cpu().item())
            best_selected_source = candidate_sources[source_index] if source_index >= 0 else "unknown"
        real_mask = selected_real_mask[batch_idx]
        valid_mask = selected_valid_mask[batch_idx] & real_mask
        source_counts = Counter(_source_bucket(source) for source in selected_sources)
        source_total = max(1, len(selected_sources))
        yield {
            "token": token,
            "gt_reward": gt_reward,
            "il_reward": il_reward,
            "best_raw_reward": best_raw_reward,
            "best_valid_reward": best_valid_reward,
            "best_selected_reward": best_selected_reward,
            "best_raw_source": candidate_sources[best_raw_idx],
            "best_valid_source": candidate_sources[best_valid_idx],
            "best_selected_source": best_selected_source,
            "best_raw_minus_gt": best_raw_reward - gt_reward,
            "best_valid_minus_gt": best_valid_reward - gt_reward,
            "best_selected_minus_gt": best_selected_reward - gt_reward,
            "best_raw_minus_il": best_raw_reward - il_reward,
            "best_valid_minus_il": best_valid_reward - il_reward,
            "best_selected_minus_il": best_selected_reward - il_reward,
            "has_valid_candidate": bool(awac_batch["has_valid_candidate"][batch_idx].detach().cpu().item()),
            "selected_valid_ratio": float((valid_mask.float().sum() / real_mask.float().sum().clamp(min=1.0)).item()),
            "selected_mean_reward": _masked_mean(selected_rewards[batch_idx], real_mask),
            "selected_max_reward": float(selected_rewards[batch_idx].masked_fill(~real_mask, -torch.inf).max().item()),
            "selected_pdms_mean": _masked_mean(selected_components["pdms"][batch_idx], real_mask),
            "selected_nc_mean": _masked_mean(selected_components["no_at_fault_collisions"][batch_idx], real_mask),
            "selected_dac_mean": _masked_mean(selected_components["drivable_area_compliance"][batch_idx], real_mask),
            "selected_ttc_mean": _masked_mean(selected_components["time_to_collision_within_bound"][batch_idx], real_mask),
            "selected_ep_mean": _masked_mean(selected_components["ego_progress"][batch_idx], real_mask),
            "selected_comfort_mean": _masked_mean(selected_components["history_comfort"][batch_idx], real_mask),
            "selected_ddc_mean": _masked_mean(selected_components["driving_direction_compliance"][batch_idx], real_mask),
            "selected_tlc_mean": _masked_mean(selected_components["traffic_light_compliance"][batch_idx], real_mask),
            "source_gt_ratio": source_counts["gt"] / source_total,
            "source_il_ratio": source_counts["il"] / source_total,
            "source_policy_ratio": source_counts["policy"] / source_total,
            "source_progress_ratio": source_counts["progress"] / source_total,
            "source_lateral_ratio": source_counts["lateral"] / source_total,
            "source_timing_ratio": source_counts["timing"] / source_total,
            "fallback_candidate": bool(fallback_candidate[batch_idx].detach().cpu().item()),
            "record_version": 2,
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
    selected_selection_score = awac_batch["selected_selection_score"].detach().cpu().float().numpy()
    selected_real_mask = awac_batch["selected_real_mask"].detach().cpu().bool().numpy()
    selected_valid_mask = awac_batch["selected_valid_mask"].detach().cpu().bool().numpy()
    selected_components = {
        key: value.detach().cpu().float().numpy()
        for key, value in awac_batch["selected_components"].items()
    }
    selected_source_index = awac_batch["selected_source_index"].detach().cpu().long().numpy()
    candidate_sources = awac_batch["candidate_sources"]
    for batch_idx, token in enumerate(tokens):
        real_count = int(selected_real_mask[batch_idx].sum())
        source_indices = selected_source_index[batch_idx, :real_count]
        sources = [
            candidate_sources[int(index)] if int(index) >= 0 else "unknown"
            for index in source_indices
        ]
        row_rewards = selected_rewards[batch_idx, :real_count]
        row_valid_mask = selected_valid_mask[batch_idx, :real_count]
        best_raw_pos = int(row_rewards.argmax()) if real_count > 0 else 0
        valid_positions = row_valid_mask.nonzero()[0]
        has_valid = bool(valid_positions.size > 0)
        if has_valid:
            best_valid_pos = int(valid_positions[int(row_rewards[valid_positions].argmax())])
        else:
            best_valid_pos = best_raw_pos
        best_selected_pos = best_raw_pos
        best_raw_reward = float(row_rewards[best_raw_pos]) if real_count > 0 else 0.0
        best_valid_reward = float(row_rewards[best_valid_pos]) if real_count > 0 else 0.0
        best_selected_reward = float(row_rewards[best_selected_pos]) if real_count > 0 else 0.0
        best_raw_source = sources[best_raw_pos] if sources else "unknown"
        best_valid_source = sources[best_valid_pos] if sources else "unknown"
        best_selected_source = sources[best_selected_pos] if sources else "unknown"
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
            "valid_mask": selected_valid_mask[batch_idx, :real_count],
            "selection_score": selected_selection_score[batch_idx, :real_count],
            "gt_reward": float(awac_batch["gt_reward"][batch_idx].detach().cpu().item()),
            "il_reward": float(awac_batch["il_reward"][batch_idx].detach().cpu().item()),
            "best_reward": best_valid_reward,
            "best_source": best_valid_source,
            "best_raw_reward": best_raw_reward,
            "best_valid_reward": best_valid_reward,
            "best_selected_reward": best_selected_reward,
            "best_raw_source": best_raw_source,
            "best_valid_source": best_valid_source,
            "best_selected_source": best_selected_source,
            "has_valid_candidate": has_valid,
            "version": 2,
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
    summary_json = out_root / "awac_elite_buffer_summary.json"
    dry_run = _env_flag("DRY_RUN", bool(cfg.get("dry_run", False)))
    max_scenes = _env_int("MAX_SCENES", int(cfg.get("max_scenes", 0)))
    batch_size = _env_int("BATCH_SIZE", int(cfg.dataloader.params.batch_size))
    shard_index = _env_int("SHARD_INDEX", 0)
    shard_count = _env_int("SHARD_COUNT", 1)
    if shard_count <= 0:
        raise ValueError(f"SHARD_COUNT must be positive, got {shard_count}.")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"SHARD_INDEX must be in [0, SHARD_COUNT), got {shard_index}/{shard_count}.")
    if not _env_flag("RUN_STAGE3", False) and not _env_flag("RUN_TRAIN", False):
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "stage3_awac_elite_buffer_dry_run.txt").write_text(
            "Stage3 AWAC/IQL buffer generation is disabled by default. "
            "Set RUN_STAGE3=1 or RUN_TRAIN=1 to execute.\n",
            encoding="utf-8",
        )
        logger.info("Stage3 AWAC/IQL buffer generation is dry-run only; no model or dataset was loaded.")
        return

    cfg.agent.stage3_objective = "none"
    cfg.agent.offline_rl_enabled = True
    cfg.agent.offline_rl_build_candidates_online = True
    cfg.agent.offline_rl_elite_buffer_path = str(buffer_dir)
    cfg.agent.offline_rl_strict_reward_submetrics = _env_flag("AWAC_STRICT_REWARD_SUBMETRICS", True)
    cfg.agent.offline_rl_missing_submetric_policy = os.getenv("AWAC_MISSING_SUBMETRIC_POLICY", "error")
    cfg.agent.offline_rl_require_buffer_valid_mask = _env_flag("AWAC_REQUIRE_BUFFER_VALID_MASK", True)
    cfg.agent.offline_rl_allow_v1_buffer_recompute_valid_mask = _env_flag(
        "AWAC_ALLOW_V1_BUFFER_RECOMPUTE_VALID_MASK",
        True,
    )
    cfg.agent.offline_rl_select_valid_topk_only = _env_flag("AWAC_SELECT_VALID_TOPK_ONLY", True)
    cfg.agent.offline_rl_train_invalid_fallback_candidates = _env_flag(
        "AWAC_TRAIN_INVALID_FALLBACK_CANDIDATES",
        False,
    )
    cfg.agent.offline_rl_fallback_invalid_candidate_weight = float(
        os.getenv("AWAC_FALLBACK_INVALID_CANDIDATE_WEIGHT", "0.0")
    )
    cfg.agent.offline_rl_use_final_heading_guard = _env_flag("AWAC_USE_FINAL_HEADING_GUARD", True)
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
    if shard_count > 1:
        shard_indices = list(range(shard_index, len(dataset), shard_count))
        logger.info(
            "Applying AWAC elite buffer shard %d/%d with %d scenes from %d input scenes.",
            shard_index,
            shard_count,
            len(shard_indices),
            len(dataset),
        )
        dataset = Subset(dataset, shard_indices)
    dataloader_params = dict(cfg.dataloader.params)
    dataloader_params["batch_size"] = batch_size
    dataloader_params["shuffle"] = False
    if int(dataloader_params.get("num_workers", 0)) <= 0:
        dataloader_params.pop("prefetch_factor", None)
    dataloader = DataLoader(dataset, collate_fn=custom_collate_fn, **dataloader_params)

    summary_fields = [
        "token",
        "gt_reward",
        "il_reward",
        "best_raw_reward",
        "best_valid_reward",
        "best_selected_reward",
        "best_raw_source",
        "best_valid_source",
        "best_selected_source",
        "best_raw_minus_gt",
        "best_valid_minus_gt",
        "best_selected_minus_gt",
        "best_raw_minus_il",
        "best_valid_minus_il",
        "best_selected_minus_il",
        "has_valid_candidate",
        "selected_valid_ratio",
        "selected_mean_reward",
        "selected_max_reward",
        "selected_pdms_mean",
        "selected_nc_mean",
        "selected_dac_mean",
        "selected_ttc_mean",
        "selected_ep_mean",
        "selected_comfort_mean",
        "selected_ddc_mean",
        "selected_tlc_mean",
        "source_gt_ratio",
        "source_il_ratio",
        "source_policy_ratio",
        "source_progress_ratio",
        "source_lateral_ratio",
        "source_timing_ratio",
        "fallback_candidate",
        "record_version",
    ]
    written = 0
    aggregate: Dict[str, Any] = {
        "num_scenes": 0,
        "sums": defaultdict(float),
        "best_valid_sources": Counter(),
    }
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
                aggregate["num_scenes"] += 1
                for key in (
                    "gt_reward",
                    "il_reward",
                    "best_raw_reward",
                    "best_valid_reward",
                    "selected_valid_ratio",
                ):
                    aggregate["sums"][key] += float(row[key])
                aggregate["sums"]["best_raw_above_gt"] += float(row["best_raw_reward"] > row["gt_reward"])
                aggregate["sums"]["best_valid_above_gt"] += float(row["best_valid_reward"] > row["gt_reward"])
                aggregate["sums"]["best_valid_above_il"] += float(row["best_valid_reward"] > row["il_reward"])
                aggregate["sums"]["has_valid_candidate"] += float(bool(row["has_valid_candidate"]))
                aggregate["best_valid_sources"][row["best_valid_source"]] += 1
            written += len(tokens)
            if written % 256 == 0:
                logger.info("Processed %d scenes for AWAC elite buffer.", written)

    n = max(1, int(aggregate["num_scenes"]))
    global_summary = {
        "num_scenes": int(aggregate["num_scenes"]),
        "mean_gt_reward": aggregate["sums"]["gt_reward"] / n,
        "mean_il_reward": aggregate["sums"]["il_reward"] / n,
        "mean_best_raw_reward": aggregate["sums"]["best_raw_reward"] / n,
        "mean_best_valid_reward": aggregate["sums"]["best_valid_reward"] / n,
        "pct_best_raw_above_gt": aggregate["sums"]["best_raw_above_gt"] / n,
        "pct_best_valid_above_gt": aggregate["sums"]["best_valid_above_gt"] / n,
        "pct_best_valid_above_il": aggregate["sums"]["best_valid_above_il"] / n,
        "has_valid_candidate_ratio": aggregate["sums"]["has_valid_candidate"] / n,
        "best_valid_source_distribution": dict(aggregate["best_valid_sources"]),
        "selected_valid_ratio_mean": aggregate["sums"]["selected_valid_ratio"] / n,
    }
    summary_json.write_text(json.dumps(global_summary, indent=2, sort_keys=True), encoding="utf-8")

    logger.info("AWAC elite buffer summary written to %s", summary_csv)
    logger.info("AWAC elite buffer global summary written to %s", summary_json)
    if dry_run:
        logger.info("DRY_RUN=true; elite records were not written.")
    else:
        logger.info("AWAC elite records written under %s", buffer_dir)


if __name__ == "__main__":
    main()
