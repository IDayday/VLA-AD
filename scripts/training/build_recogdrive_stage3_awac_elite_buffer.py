from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader, Subset
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.recogdrive.offline_rl_buffer import (
    REQUIRED_COMPONENT_KEYS,
    elite_record_exists,
    load_elite_record,
    save_elite_record,
)
from navsim.agents.recogdrive.candidate_funnel import (
    ExternalCandidateLoader,
    failure_conditioned_expand,
    pareto_nms,
    trust_region_control_expand,
)
from navsim.agents.recogdrive.pareto_support import (
    CandidateRecord,
    build_archive_record,
    select_feasible_pareto_support,
    support_semantic_pass,
)
from navsim.agents.recogdrive.recogdrive_agent import (
    EXPERT_FEATURE_KEYS,
    EXPERT_TARGET_FEATURE_KEYS,
    LAST_VLA_TARGET_KEYS,
    TWO_EXPERT_TARGET_KEYS,
)
from navsim.agents.recogdrive.trajectory_feasibility import compute_feasibility_metrics
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


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_list_float(name: str, default: Tuple[float, ...]) -> Tuple[float, ...]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    if value.strip().lower() in {"none", "null", "off", "false", "[]"}:
        return ()
    items = value.replace(",", " ").split()
    return tuple(float(item) for item in items)


def _file_sha256(path: str) -> str:
    path_obj = Path(path).expanduser()
    if not path_obj.is_file():
        raise FileNotFoundError(path_obj)
    digest = hashlib.sha256()
    with path_obj.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_token_manifest(path: str) -> set[str]:
    if not path:
        return set()
    manifest = Path(path).expanduser()
    if not manifest.exists():
        raise FileNotFoundError(f"TOKEN_MANIFEST does not exist: {manifest}")
    tokens: set[str] = set()
    if manifest.suffix.lower() == ".json":
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            raw_items = payload.get("tokens", payload.get("scene_tokens", payload.get("items", [])))
        else:
            raw_items = payload
        for item in raw_items:
            if isinstance(item, dict):
                token = item.get("token", item.get("scene_token"))
            else:
                token = item
            if token is not None and str(token).strip():
                tokens.add(str(token).strip())
        return tokens
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens.add(line.split(",", 1)[0].strip())
    return tokens


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


def _filter_batch_by_indices(batch: Dict[str, Any], indices: List[int], batch_len: int) -> Dict[str, Any]:
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    filtered: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_len:
            filtered[key] = value.index_select(0, index_tensor.to(value.device))
        elif isinstance(value, list) and len(value) == batch_len:
            filtered[key] = [value[index] for index in indices]
        elif isinstance(value, tuple) and len(value) == batch_len:
            filtered[key] = tuple(value[index] for index in indices)
        else:
            filtered[key] = value
    return filtered


def _existing_record_is_usable(buffer_dir: Path, token: str, validate_existing: bool) -> bool:
    if not elite_record_exists(buffer_dir, token):
        return False
    if not validate_existing:
        return True
    try:
        load_elite_record(buffer_dir, token)
    except Exception as exc:  # pragma: no cover - defensive path for interrupted writes
        logger.warning("Existing AWAC record for token=%s is invalid and will be recomputed: %s", token, exc)
        return False
    return True


def _dataset_token_at(dataset: torch.utils.data.Dataset, index: int) -> str:
    if isinstance(dataset, Subset):
        return _dataset_token_at(dataset.dataset, int(dataset.indices[index]))
    tokens = getattr(dataset, "_tokens", None)
    if tokens is None:
        tokens = getattr(dataset, "tokens", None)
    if tokens is None:
        wrapped = getattr(dataset, "_dataset", None)
        if wrapped is not None:
            return _dataset_token_at(wrapped, index)
    if tokens is None:
        raise AttributeError(f"Dataset {type(dataset).__name__} does not expose token metadata.")
    return str(tokens[index])


def _prefilter_missing_existing_records(
    dataset: torch.utils.data.Dataset,
    buffer_dir: Path,
    validate_existing: bool,
) -> tuple[torch.utils.data.Dataset, int]:
    missing_indices: List[int] = []
    skipped = 0
    for index in range(len(dataset)):
        token = _dataset_token_at(dataset, index)
        if _existing_record_is_usable(buffer_dir, token, validate_existing):
            skipped += 1
        else:
            missing_indices.append(index)
    if skipped == 0:
        return dataset, 0
    return Subset(dataset, missing_indices), skipped


def _filter_dataset_by_tokens(
    dataset: torch.utils.data.Dataset,
    token_set: set[str],
) -> torch.utils.data.Dataset:
    if not token_set:
        return dataset
    keep_indices: List[int] = []
    for index in range(len(dataset)):
        if _dataset_token_at(dataset, index) in token_set:
            keep_indices.append(index)
    logger.info(
        "Applied TOKEN_MANIFEST filter: kept %d/%d dataset scenes.",
        len(keep_indices),
        len(dataset),
    )
    return Subset(dataset, keep_indices)


def _resolve_build_logs(cfg: DictConfig, build_log_split: str):
    split = str(build_log_split).strip().lower()
    if split in {"train", "navtrain"}:
        return cfg.train_logs
    if split in {"val", "valid", "validation", "navval"}:
        return cfg.val_logs
    if split in {"train_val", "trainval", "all"}:
        return list(cfg.train_logs) + list(cfg.val_logs)
    raise ValueError(
        "BUILD_LOG_SPLIT must be one of train, val, train_val; "
        f"got {build_log_split!r}."
    )


def _build_train_dataset(cfg: DictConfig, agent: AbstractAgent, build_log_split: str):
    log_names = _resolve_build_logs(cfg, build_log_split)
    if cfg.use_cache_without_dataset:
        dataset = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=log_names,
        )
        return TokenizedCacheOnlyDataset(dataset)
    if build_log_split not in {"train", "navtrain"}:
        raise ValueError(
            "Non-cache support build for val/train_val is not implemented; "
            "use cached ReCogDrive hidden states with use_cache_without_dataset=true."
        )
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


def _record_best_valid_reward(record: Dict[str, Any]) -> float:
    if "best_valid_reward" in record:
        return float(record["best_valid_reward"])
    rewards = np.asarray(record["rewards"], dtype=np.float32)
    valid_mask = np.asarray(record.get("valid_mask", np.zeros_like(rewards, dtype=np.bool_)), dtype=np.bool_)
    if rewards.size == 0:
        return 0.0
    if bool(valid_mask.any()):
        return float(rewards[valid_mask].max())
    return float(rewards.max())


def _candidate_digest(candidate: np.ndarray) -> str:
    rounded = np.round(np.asarray(candidate, dtype=np.float32), decimals=3)
    return hashlib.sha1(rounded.tobytes()).hexdigest()


def _merge_elite_records(
    existing: Dict[str, Any],
    new_record: Dict[str, Any],
    *,
    keep_top_k: int,
    keep_support: bool,
) -> Dict[str, Any]:
    """Merge two v2 selected-elite records without letting a weaker pass overwrite a stronger one."""
    if str(existing.get("token", "")) != str(new_record["token"]):
        raise ValueError(
            f"Cannot merge AWAC elite records with different tokens: "
            f"{existing.get('token')!r} vs {new_record['token']!r}."
        )
    if int(existing.get("version", 1)) < 2 or "valid_mask" not in existing:
        return new_record

    keep_top_k = max(1, int(keep_top_k))
    candidates = np.concatenate(
        [
            np.asarray(existing["candidates"], dtype=np.float32),
            np.asarray(new_record["candidates"], dtype=np.float32),
        ],
        axis=0,
    )
    rewards = np.concatenate(
        [
            np.asarray(existing["rewards"], dtype=np.float32),
            np.asarray(new_record["rewards"], dtype=np.float32),
        ],
        axis=0,
    )
    anchor_distance = np.concatenate(
        [
            np.asarray(existing["anchor_distance"], dtype=np.float32),
            np.asarray(new_record["anchor_distance"], dtype=np.float32),
        ],
        axis=0,
    )
    valid_mask = np.concatenate(
        [
            np.asarray(existing["valid_mask"], dtype=np.bool_),
            np.asarray(new_record["valid_mask"], dtype=np.bool_),
        ],
        axis=0,
    )
    selection_score = np.concatenate(
        [
            np.asarray(existing.get("selection_score", existing["rewards"]), dtype=np.float32),
            np.asarray(new_record.get("selection_score", new_record["rewards"]), dtype=np.float32),
        ],
        axis=0,
    )
    sources = [str(source) for source in existing["sources"]] + [str(source) for source in new_record["sources"]]
    components = {
        key: np.concatenate(
            [
                np.asarray(existing["components"][key], dtype=np.float32),
                np.asarray(new_record["components"][key], dtype=np.float32),
            ],
            axis=0,
        )
        for key in REQUIRED_COMPONENT_KEYS
    }

    best_by_digest: Dict[str, int] = {}
    for idx in range(candidates.shape[0]):
        digest = _candidate_digest(candidates[idx])
        if sources[idx] in {"gt", "il"}:
            digest = f"{sources[idx]}:{digest}"
        prev = best_by_digest.get(digest)
        if prev is None:
            best_by_digest[digest] = idx
            continue
        current_valid = bool(valid_mask[idx])
        previous_valid = bool(valid_mask[prev])
        if current_valid and not previous_valid:
            best_by_digest[digest] = idx
        elif current_valid == previous_valid and float(selection_score[idx]) > float(selection_score[prev]):
            best_by_digest[digest] = idx
    unique_idx = np.asarray(sorted(best_by_digest.values()), dtype=np.int64)

    candidates = candidates[unique_idx]
    rewards = rewards[unique_idx]
    anchor_distance = anchor_distance[unique_idx]
    valid_mask = valid_mask[unique_idx]
    selection_score = selection_score[unique_idx]
    sources = [sources[int(idx)] for idx in unique_idx.tolist()]
    components = {key: value[unique_idx] for key, value in components.items()}

    selected_indices: List[int] = []
    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size > 0:
        score = np.where(np.isfinite(selection_score[valid_idx]), selection_score[valid_idx], rewards[valid_idx])
        order = valid_idx[np.argsort(score)[::-1]]
        selected_indices.extend(int(idx) for idx in order[:keep_top_k])
    else:
        order = np.argsort(rewards)[::-1]
        selected_indices.extend(int(idx) for idx in order[:keep_top_k])

    if keep_support:
        for support_source in ("gt", "il"):
            support_idx = [idx for idx, source in enumerate(sources) if source == support_source]
            if support_idx:
                best_support = max(support_idx, key=lambda idx: float(rewards[idx]))
                selected_indices.append(int(best_support))

    if len(selected_indices) < keep_top_k:
        for idx in np.argsort(rewards)[::-1]:
            selected_indices.append(int(idx))
            if len(set(selected_indices)) >= keep_top_k:
                break

    deduped_indices: List[int] = []
    seen: set[int] = set()
    for idx in selected_indices:
        if idx not in seen:
            deduped_indices.append(idx)
            seen.add(idx)
    selected = np.asarray(deduped_indices, dtype=np.int64)

    rewards_sel = rewards[selected]
    valid_sel = valid_mask[selected]
    raw_pos = int(np.argmax(rewards_sel)) if rewards_sel.size > 0 else 0
    has_valid = bool(valid_sel.any())
    if has_valid:
        valid_positions = np.flatnonzero(valid_sel)
        valid_pos = int(valid_positions[int(np.argmax(rewards_sel[valid_positions]))])
    else:
        valid_pos = raw_pos
    selected_pos = raw_pos
    selected_sources = [sources[int(idx)] for idx in selected.tolist()]

    merged = {
        "token": str(new_record["token"]),
        "candidates": candidates[selected],
        "rewards": rewards_sel,
        "components": {key: value[selected] for key, value in components.items()},
        "sources": selected_sources,
        "anchor_distance": anchor_distance[selected],
        "valid_mask": valid_sel,
        "selection_score": selection_score[selected],
        "gt_reward": float(new_record.get("gt_reward", existing.get("gt_reward", 0.0))),
        "il_reward": float(new_record.get("il_reward", existing.get("il_reward", 0.0))),
        "best_reward": float(rewards_sel[valid_pos]) if rewards_sel.size > 0 else 0.0,
        "best_source": selected_sources[valid_pos] if selected_sources else "unknown",
        "best_raw_reward": float(rewards_sel[raw_pos]) if rewards_sel.size > 0 else 0.0,
        "best_valid_reward": float(rewards_sel[valid_pos]) if rewards_sel.size > 0 else 0.0,
        "best_selected_reward": float(rewards_sel[selected_pos]) if rewards_sel.size > 0 else 0.0,
        "best_raw_source": selected_sources[raw_pos] if selected_sources else "unknown",
        "best_valid_source": selected_sources[valid_pos] if selected_sources else "unknown",
        "best_selected_source": selected_sources[selected_pos] if selected_sources else "unknown",
        "has_valid_candidate": has_valid,
        "version": 2,
    }
    return merged


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
    merge_existing_records: bool,
    merge_keep_top_k: int,
    merge_keep_support: bool,
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
            if merge_existing_records and elite_record_exists(buffer_dir, token):
                try:
                    existing = load_elite_record(buffer_dir, token)
                    record = _merge_elite_records(
                        existing,
                        record,
                        keep_top_k=merge_keep_top_k,
                        keep_support=merge_keep_support,
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not merge existing AWAC elite record for token=%s; writing new record: %s",
                        token,
                        exc,
                    )
            save_elite_record(buffer_dir, token, record)


def _parse_external_candidate_roots(value: str) -> Dict[str, str]:
    roots: Dict[str, str] = {}
    if not value:
        return roots
    for item in value.replace(",", " ").split():
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "EXTERNAL_CANDIDATE_ROOTS must be a whitespace/comma separated list of source=path entries; "
                f"got {item!r}."
            )
        source, path = item.split("=", 1)
        roots[str(source)] = str(path)
    return roots


def _feasibility_dict(traj: np.ndarray) -> Dict[str, float]:
    metrics = compute_feasibility_metrics(torch.as_tensor(traj, dtype=torch.float32).unsqueeze(0), {})
    return {
        "feas_cost": float(metrics.feas_cost[0].detach().cpu().item()),
        "early_kink_rate": float(metrics.early_kink_rate[0].detach().cpu().item()),
        "tail_reverse_rate": float(metrics.tail_reverse_rate[0].detach().cpu().item()),
        "curvature_violation_rate": float(metrics.curvature_violation_rate[0].detach().cpu().item()),
    }


def _candidate_records_from_awac_row(
    token: str,
    awac_batch: Dict[str, Any],
    batch_idx: int,
    *,
    use_raw_candidates: bool = False,
) -> List[CandidateRecord]:
    candidate_sources = awac_batch["candidate_sources"]
    if use_raw_candidates:
        trajs = awac_batch["candidate_trajs"][batch_idx].detach().cpu().float().numpy()
        rewards = awac_batch["candidate_rewards"][batch_idx].detach().cpu().float().numpy()
        selection_score = awac_batch["candidate_selection_score"][batch_idx].detach().cpu().float().numpy()
        components = {
            key: value[batch_idx].detach().cpu().float().numpy()
            for key, value in awac_batch["candidate_components"].items()
        }
        source_indices = np.arange(len(candidate_sources), dtype=np.int64)
        real_count = len(candidate_sources)
    else:
        real_mask = awac_batch["selected_real_mask"][batch_idx].detach().cpu().bool().numpy()
        real_count = int(real_mask.sum())
        source_indices = awac_batch["selected_source_index"][batch_idx, :real_count].detach().cpu().long().numpy()
        trajs = awac_batch["selected_trajs"][batch_idx, :real_count].detach().cpu().float().numpy()
        rewards = awac_batch["selected_rewards"][batch_idx, :real_count].detach().cpu().float().numpy()
        selection_score = awac_batch["selected_selection_score"][batch_idx, :real_count].detach().cpu().float().numpy()
        components = {
            key: value[batch_idx, :real_count].detach().cpu().float().numpy()
            for key, value in awac_batch["selected_components"].items()
        }
    records: List[CandidateRecord] = []
    for idx in range(real_count):
        source_index = int(source_indices[idx])
        source = candidate_sources[source_index] if source_index >= 0 else "unknown"
        traj = np.asarray(trajs[idx], dtype=np.float32)
        comp = {key: float(values[idx]) for key, values in components.items()}
        records.append(
            CandidateRecord(
                trajectory=traj,
                source=str(source),
                token=str(token),
                components=comp,
                reward=float(rewards[idx]),
                feas=_feasibility_dict(traj),
                selection_score=float(selection_score[idx]),
            )
        )
    return records


def _score_external_records(
    action_head,
    token: str,
    external_records: List[CandidateRecord],
    metric_cache: Dict[str, Any],
    cfg,
    expected_shape: Tuple[int, int],
) -> List[CandidateRecord]:
    valid_records: List[CandidateRecord] = []
    for record in external_records:
        traj = np.asarray(record.trajectory, dtype=np.float32)
        if traj.shape != expected_shape:
            logger.warning(
                "Skipping external candidate token=%s source=%s with shape %s, expected %s.",
                token,
                record.source,
                traj.shape,
                expected_shape,
            )
            continue
        if not np.isfinite(traj).all():
            logger.warning("Skipping non-finite external candidate token=%s source=%s.", token, record.source)
            continue
        valid_records.append(record)
    if not valid_records:
        return []
    trajs = torch.as_tensor(
        np.stack([np.asarray(record.trajectory, dtype=np.float32) for record in valid_records], axis=0),
        dtype=torch.float32,
        device=next(action_head.parameters()).device,
    )
    rewards, components = action_head._score_candidate_trajectories(
        trajs.unsqueeze(0),
        [str(token)],
        metric_cache,
        cfg,
    )
    out: List[CandidateRecord] = []
    rewards_np = rewards[0].detach().cpu().float().numpy()
    components_np = {
        key: value[0].detach().cpu().float().numpy()
        for key, value in components.items()
    }
    for idx, record in enumerate(valid_records):
        traj = np.asarray(record.trajectory, dtype=np.float32)
        comp = {key: float(values[idx]) for key, values in components_np.items()}
        reward = float(rewards_np[idx])
        out.append(
            CandidateRecord(
                trajectory=traj,
                source=str(record.source),
                token=str(token),
                components=comp,
                reward=reward,
                feas=_feasibility_dict(traj),
                selection_score=reward,
                parent_id=record.parent_id,
            )
        )
    return out


def _expand_external_records(
    records: List[CandidateRecord],
    *,
    cfg,
    max_per_scene: int,
) -> List[CandidateRecord]:
    """Build structured variants around DDV2/DriveOR candidates before evaluator scoring."""
    if not records:
        return []
    expanded: List[CandidateRecord] = []
    for record in records:
        expanded.append(record)
        for candidate in failure_conditioned_expand(record, {}, cfg):
            candidate.source = f"{record.source}:{candidate.source}"
            candidate.parent_id = candidate.parent_id or record.parent_id or record.source
            expanded.append(candidate)
        for candidate in trust_region_control_expand(record, cfg):
            candidate.source = f"{record.source}:{candidate.source}"
            candidate.parent_id = candidate.parent_id or record.parent_id or record.source
            expanded.append(candidate)
    if max_per_scene > 0:
        expanded = pareto_nms(expanded, None, {"max_candidates_per_scene": int(max_per_scene)})
    return expanded


def _prefilter_external_anchor_records(
    records: List[CandidateRecord],
    *,
    ref_traj: np.ndarray | None,
    sg_cfg: Dict[str, Any],
) -> Tuple[List[CandidateRecord], int]:
    if not records or ref_traj is None or not bool(sg_cfg.get("support_semantic_enable", False)):
        return records, 0
    kept: List[CandidateRecord] = []
    dropped = 0
    for record in records:
        if support_semantic_pass(record.trajectory, ref_traj, sg_cfg):
            kept.append(record)
        else:
            dropped += 1
    return kept, dropped


def _save_sg_fps_v3_records(
    buffer_dir: Path,
    tokens: List[str],
    awac_batch: Dict[str, Any],
    action_head,
    metric_cache: Dict[str, Any],
    cfg,
    external_loader: ExternalCandidateLoader | None,
    dry_run: bool,
    support_top_m: int,
    expand_external_candidates: bool,
    external_expansion_max_per_scene: int,
    use_raw_internal_candidates: bool,
    archive_version: int,
    build_provenance: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "scene_count": 0,
        "candidate_count": 0,
        "external_candidate_count": 0,
        "external_anchor_semantic_dropped": 0,
        "support_tag_counts": Counter(),
    }
    H = int(awac_batch["selected_trajs"].shape[-2])
    D = int(awac_batch["selected_trajs"].shape[-1])
    sg_cfg = {
        "support_top_m": int(support_top_m),
        "support_selection_strategy": os.getenv("SG_FPS_SELECTION_STRATEGY", "quality_pareto"),
        "fp_ddc_min_absolute": float(getattr(cfg, "ddc_min_absolute", 0.95)),
        "fp_ddc_ref_tolerance": float(getattr(cfg, "ddc_max_relative_drop", 0.01)),
        "fpv3_ddc_min_absolute": _env_float("SG_FPS_DDC_MIN_ABSOLUTE", float(getattr(cfg, "ddc_min_absolute", 0.95))),
        "fpv3_ddc_drop_tolerance": _env_float("SG_FPS_DDC_DROP_TOLERANCE", float(getattr(cfg, "ddc_max_relative_drop", 0.01))),
        "support_ddc_gate_mode": os.getenv("SG_FPS_DDC_GATE_MODE", "ref_relative"),
        "support_relax_ddc_when_ref_below_min": _env_flag("SG_FPS_RELAX_DDC_WHEN_REF_BELOW_MIN", True),
        "fpv3_feas_cost_max": _env_float("SG_FPS_FEAS_COST_MAX", 0.10),
        "support_feas_gate_mode": os.getenv("SG_FPS_FEAS_GATE_MODE", "relax_ref_above_max"),
        "support_feas_cost_tolerance": _env_float("SG_FPS_FEAS_COST_TOLERANCE", 0.03),
        "fpv3_comfort_min": _env_float("SG_FPS_COMFORT_MIN", 0.95),
        "support_comfort_gate_mode": os.getenv("SG_FPS_COMFORT_GATE_MODE", "ref_relative"),
        "support_comfort_drop_tolerance": _env_float("SG_FPS_COMFORT_DROP_TOLERANCE", 0.05),
        "fp_comfort_min": 0.95,
        "support_feas_max": 1.0,
        "support_quality_enable": _env_flag("SG_FPS_ENABLE_TRAIN_QUALITY_GATE", True),
        "support_min_non_gt_reward": _env_float("SG_FPS_MIN_NON_GT_PDMS", 0.90),
        "support_reward_gate_mode": os.getenv("SG_FPS_REWARD_GATE_MODE", "absolute_or_gt_improver"),
        "support_min_non_gt_improver_reward": _env_float("SG_FPS_MIN_NON_GT_IMPROVER_PDMS", 0.70),
        "support_gt_improver_margin": _env_float("SG_FPS_GT_IMPROVER_MARGIN", 0.05),
        "support_gt_improver_ref_max_reward": _env_float("SG_FPS_GT_IMPROVER_REF_MAX_PDMS", 0.90),
        "support_max_first_xy_error_m": _env_float("SG_FPS_MAX_FIRST_XY_ERROR_M", 1.0),
        "support_max_first_heading_error_rad": _env_float("SG_FPS_MAX_FIRST_HEADING_ERROR_RAD", 0.8),
        "support_max_xy_turn_rad": _env_float("SG_FPS_MAX_XY_TURN_RAD", 1.2),
        "support_max_early_xy_turn_rad": _env_float("SG_FPS_MAX_EARLY_XY_TURN_RAD", 1.0),
        "support_max_step_m": _env_float("SG_FPS_MAX_STEP_M", 12.0),
        "support_semantic_enable": _env_flag("SG_FPS_ENABLE_SEMANTIC_GATE", False),
        "support_allow_turn_class_mismatch": _env_flag("SG_FPS_ALLOW_TURN_CLASS_MISMATCH", False),
        "support_max_semantic_final_heading_error_rad": _env_float("SG_FPS_MAX_SEMANTIC_FINAL_HEADING_ERROR_RAD", 0.75),
        "support_max_semantic_path_angle_error_rad": _env_float("SG_FPS_MAX_SEMANTIC_PATH_ANGLE_ERROR_RAD", 0.75),
        "support_max_semantic_endpoint_lateral_error_m": _env_float("SG_FPS_MAX_SEMANTIC_ENDPOINT_LATERAL_ERROR_M", 4.0),
        "support_keep_gt_by_default": _env_flag("SG_FPS_KEEP_GT_BY_DEFAULT", True),
        "support_gt_keep_min_reward": _env_float("SG_FPS_GT_KEEP_MIN_REWARD", 0.85),
        "support_gt_replace_margin": _env_float("SG_FPS_GT_REPLACE_MARGIN", 0.05),
        "support_include_il_anchor": _env_flag("SG_FPS_INCLUDE_IL_ANCHOR", False),
        "support_high_pdms_threshold": _env_float("SG_FPS_HIGH_PDMS_THRESHOLD", 0.95),
        "support_top_pdms_count": _env_int("SG_FPS_TOP_PDMS_COUNT", 3),
        "support_pareto_count": _env_int("SG_FPS_PARETO_COUNT", 5),
        "support_score_diversity_weight": _env_float("SG_FPS_SCORE_DIVERSITY_WEIGHT", 0.8),
        "support_min_trajectory_diversity_score": _env_float("SG_FPS_MIN_TRAJECTORY_DIVERSITY_SCORE", 0.05),
        "support_archive_version": int(archive_version),
        "support_v4_max_gt_ade_m": _env_float("SG_FPS_V4_MAX_GT_ADE_M", 1.5),
        "support_v4_max_gt_fde_m": _env_float("SG_FPS_V4_MAX_GT_FDE_M", 4.0),
        "support_v4_mode_distance_threshold": _env_float("SG_FPS_V4_MODE_DISTANCE_THRESHOLD", 0.25),
        "support_v4_reward_gain_cap": _env_float("SG_FPS_V4_REWARD_GAIN_CAP", 0.05),
        "support_v4_max_gt_reward_drop": _env_float("SG_FPS_V4_MAX_GT_REWARD_DROP", 0.05),
        "support_v4_pareto_eps": _env_float("SG_FPS_V4_PARETO_EPS", 0.01),
        "support_v4_exclude_derived_external": _env_flag("SG_FPS_V4_EXCLUDE_DERIVED_EXTERNAL", True),
        "support_v4_require_policy_reachability": _env_flag(
            "SG_FPS_V4_REQUIRE_POLICY_REACHABILITY",
            False,
        ),
        "support_v4_max_policy_snsad": _env_float("SG_FPS_V4_MAX_POLICY_SNSAD", 0.50),
        "support_build_metadata": {
            "raw_internal_candidates": bool(use_raw_internal_candidates),
            "expand_external_candidates": bool(expand_external_candidates),
            "external_expansion_max_per_scene": int(external_expansion_max_per_scene),
            **dict(build_provenance or {}),
        },
    }
    summary["build_config"] = dict(sg_cfg)
    for batch_idx, token in enumerate(tokens):
        candidates = _candidate_records_from_awac_row(
            str(token),
            awac_batch,
            batch_idx,
            use_raw_candidates=use_raw_internal_candidates,
        )
        ref_traj = next(
            (np.asarray(record.trajectory, dtype=np.float32) for record in candidates if record.source == "gt"),
            None,
        )
        external_scored: List[CandidateRecord] = []
        if external_loader is not None:
            external_raw = external_loader.load(str(token))
            external_raw, semantic_dropped = _prefilter_external_anchor_records(
                external_raw,
                ref_traj=ref_traj,
                sg_cfg=sg_cfg,
            )
            summary["external_anchor_semantic_dropped"] += int(semantic_dropped)
            if expand_external_candidates:
                external_raw = _expand_external_records(
                    external_raw,
                    cfg=cfg,
                    max_per_scene=int(external_expansion_max_per_scene),
                )
            external_scored = _score_external_records(
                action_head,
                str(token),
                external_raw,
                metric_cache,
                cfg,
                expected_shape=(H, D),
            )
            candidates.extend(external_scored)
        if not candidates:
            continue
        ref_record = next((record for record in candidates if record.source == "gt"), candidates[0])
        selected = select_feasible_pareto_support(candidates, ref_record.components, sg_cfg)
        record = build_archive_record(
            str(token),
            candidates,
            selected,
            ref=ref_record.components,
            cfg=sg_cfg,
        )
        if not dry_run:
            save_elite_record(buffer_dir, str(token), record)
        summary["scene_count"] += 1
        summary["candidate_count"] += len(candidates)
        summary["external_candidate_count"] += len(external_scored)
        summary["support_tag_counts"].update(tag for tag in record["support_tags"] if tag)
    return summary


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
    skip_existing_records = _env_flag("SKIP_EXISTING_RECORDS", False)
    validate_existing_records = _env_flag("VALIDATE_EXISTING_RECORDS", True)
    prefilter_existing_records = _env_flag("PREFILTER_EXISTING_RECORDS", True)
    merge_existing_records = _env_flag("MERGE_EXISTING_RECORDS", False)
    merge_keep_top_k = _env_int("MERGE_KEEP_TOP_K", _env_int("ELITE_TOP_M", 8))
    merge_keep_support = _env_flag("MERGE_KEEP_SUPPORT", True)
    write_sg_fps_v3 = _env_flag("WRITE_SG_FPS_V3", _env_flag("SG_FPS_V3", False))
    sg_fps_support_top_m = _env_int("SG_FPS_SUPPORT_TOP_M", _env_int("ELITE_TOP_M", 12))
    external_candidate_roots = _parse_external_candidate_roots(os.getenv("EXTERNAL_CANDIDATE_ROOTS", ""))
    external_loader = ExternalCandidateLoader(external_candidate_roots) if external_candidate_roots else None
    expand_external_candidates = _env_flag("SG_FPS_EXPAND_EXTERNAL_CANDIDATES", True)
    external_expansion_max_per_scene = _env_int("SG_FPS_EXTERNAL_EXPANSION_MAX_PER_SCENE", 32)
    use_raw_internal_candidates = _env_flag("SG_FPS_USE_RAW_INTERNAL_CANDIDATES", False)
    sg_fps_archive_version = _env_int("SG_FPS_ARCHIVE_VERSION", 3)
    require_policy_reachability = _env_flag("SG_FPS_V4_REQUIRE_POLICY_REACHABILITY", False)
    policy_samples_per_scene = _env_int("ONLINE_POLICY_SAMPLES", 0)
    policy_checkpoint_path = os.getenv("POLICY_CHECKPOINT", "").strip()
    fs_norm_stats_path = os.getenv("FS_NORM_STATS_PATH", "").strip()
    build_log_split = os.getenv("BUILD_LOG_SPLIT", "train").strip().lower()
    token_manifest = os.getenv("TOKEN_MANIFEST", os.getenv("TOKEN_LIST_PATH", "")).strip()
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

    if sg_fps_archive_version >= 4 and require_policy_reachability:
        if policy_samples_per_scene <= 0 or not _env_flag("ONLINE_USE_CURRENT_POLICY", False):
            raise ValueError(
                "A reachable SG-FPS v4 build requires ONLINE_POLICY_SAMPLES>0 and "
                "ONLINE_USE_CURRENT_POLICY=true."
            )
        if not policy_checkpoint_path or not Path(policy_checkpoint_path).expanduser().is_file():
            raise ValueError(
                "A reachable SG-FPS v4 build requires POLICY_CHECKPOINT to identify the Stage2 "
                "policy that defines the learning frontier."
            )
        if _env_flag("USE_FS_NORM", False) and not fs_norm_stats_path:
            raise ValueError("USE_FS_NORM=true requires FS_NORM_STATS_PATH for policy proposal sampling.")

    build_provenance: Dict[str, Any] = {
        "policy_samples_per_scene": int(policy_samples_per_scene),
        "policy_checkpoint_path": str(Path(policy_checkpoint_path).expanduser().resolve())
        if policy_checkpoint_path
        else "",
        "policy_checkpoint_sha256": (
            os.getenv("POLICY_CHECKPOINT_SHA256", "").strip()
            or (_file_sha256(policy_checkpoint_path) if policy_checkpoint_path else "")
        ),
        "fs_norm_stats_path": str(Path(fs_norm_stats_path).expanduser().resolve()) if fs_norm_stats_path else "",
        "fs_norm_stats_sha256": (
            os.getenv("FS_NORM_STATS_SHA256", "").strip()
            or (_file_sha256(fs_norm_stats_path) if fs_norm_stats_path else "")
        ),
        "build_seed": int(cfg.get("seed", 0)),
    }

    with open_dict(cfg.agent):
        cfg.agent.stage3_objective = "none"
        cfg.agent.offline_rl_enabled = True
        cfg.agent.offline_rl_build_candidates_online = True
        cfg.agent.offline_rl_elite_buffer_path = str(buffer_dir)
        cfg.agent.offline_rl_init_reference_policy = _env_flag("ONLINE_USE_OLD_POLICY", True)
        cfg.agent.offline_rl_keep_il_candidate = _env_flag("ONLINE_USE_OLD_POLICY", True)
        cfg.agent.offline_rl_strict_reward_submetrics = _env_flag("AWAC_STRICT_REWARD_SUBMETRICS", True)
        cfg.agent.offline_rl_missing_submetric_policy = os.getenv("AWAC_MISSING_SUBMETRIC_POLICY", "error")
        cfg.agent.offline_rl_use_batched_pdm_scoring = _env_flag("AWAC_USE_BATCHED_PDM_SCORING", True)
        cfg.agent.offline_rl_use_exact_array_pdm_state_conversion = _env_flag(
            "AWAC_USE_EXACT_ARRAY_PDM_STATE_CONVERSION",
            True,
        )
        cfg.agent.offline_rl_use_fast_pdm_scorer = _env_flag("AWAC_USE_FAST_PDM_SCORER", True)
        cfg.agent.offline_rl_pdm_batch_chunk_size = int(os.getenv("AWAC_PDM_BATCH_CHUNK_SIZE", "0"))
        cfg.agent.offline_rl_pdm_shadow_check = _env_flag("AWAC_PDM_SHADOW_CHECK", True)
        cfg.agent.offline_rl_pdm_shadow_max_samples = int(os.getenv("AWAC_PDM_SHADOW_MAX_SAMPLES", "4"))
        cfg.agent.offline_rl_pdm_shadow_max_abs_diff = float(os.getenv("AWAC_PDM_SHADOW_MAX_ABS_DIFF", "0.0"))
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
        if os.getenv("ONLINE_USE_CURRENT_POLICY") is not None:
            cfg.agent.offline_rl_online_use_current_policy = _env_flag("ONLINE_USE_CURRENT_POLICY", True)
        if os.getenv("ONLINE_USE_OLD_POLICY") is not None:
            cfg.agent.offline_rl_online_use_old_policy = _env_flag("ONLINE_USE_OLD_POLICY", True)
        if os.getenv("ONLINE_USE_GT") is not None:
            cfg.agent.offline_rl_online_use_gt = _env_flag("ONLINE_USE_GT", True)
        if os.getenv("PERTURB_GT") is not None:
            cfg.agent.offline_rl_perturb_gt = _env_flag("PERTURB_GT", True)
        if os.getenv("PERTURB_IL") is not None:
            cfg.agent.offline_rl_perturb_il = _env_flag("PERTURB_IL", True)
        cfg.agent.offline_rl_progress_endpoint_deltas_m = _env_list_float(
            "OFFLINE_RL_PROGRESS_ENDPOINT_DELTAS_M",
            tuple(float(x) for x in cfg.agent.offline_rl_progress_endpoint_deltas_m),
        )
        cfg.agent.offline_rl_progress_speed_scales = _env_list_float(
            "OFFLINE_RL_PROGRESS_SPEED_SCALES",
            tuple(float(x) for x in cfg.agent.offline_rl_progress_speed_scales),
        )
        cfg.agent.offline_rl_progress_time_gammas = _env_list_float(
            "OFFLINE_RL_PROGRESS_TIME_GAMMAS",
            tuple(float(x) for x in cfg.agent.offline_rl_progress_time_gammas),
        )
        cfg.agent.offline_rl_lateral_offsets_m = _env_list_float(
            "OFFLINE_RL_LATERAL_OFFSETS_M",
            tuple(float(x) for x in cfg.agent.offline_rl_lateral_offsets_m),
        )
        cfg.agent.offline_rl_endpoint_lateral_offsets_m = _env_list_float(
            "OFFLINE_RL_ENDPOINT_LATERAL_OFFSETS_M",
            tuple(float(x) for x in cfg.agent.offline_rl_endpoint_lateral_offsets_m),
        )
        cfg.agent.offline_rl_timing_slow_first_scales = _env_list_float(
            "OFFLINE_RL_TIMING_SLOW_FIRST_SCALES",
            tuple(float(x) for x in cfg.agent.offline_rl_timing_slow_first_scales),
        )
        cfg.agent.offline_rl_timing_delay_strengths = _env_list_float(
            "OFFLINE_RL_TIMING_DELAY_STRENGTHS",
            tuple(float(x) for x in cfg.agent.offline_rl_timing_delay_strengths),
        )
        if os.getenv("ELITE_TOP_M"):
            cfg.agent.offline_rl_elite_top_m = int(os.environ["ELITE_TOP_M"])
        if write_sg_fps_v3:
            cfg.agent.offline_rl_elite_top_m = max(int(cfg.agent.offline_rl_elite_top_m), int(sg_fps_support_top_m))
        if os.getenv("IL_CHECKPOINT"):
            cfg.agent.checkpoint_path = os.environ["IL_CHECKPOINT"]
            cfg.agent.reference_policy_checkpoint = os.environ["IL_CHECKPOINT"]
        if os.getenv("POLICY_CHECKPOINT"):
            cfg.agent.checkpoint_path = os.environ["POLICY_CHECKPOINT"]
            cfg.agent.allow_random_init = False
        if os.getenv("USE_FS_NORM") is not None:
            cfg.agent.use_fs_norm = _env_flag("USE_FS_NORM", False)
        if fs_norm_stats_path:
            cfg.agent.fs_norm_stats_path = fs_norm_stats_path
            cfg.agent.fs_norm_min_version = _env_int("FS_NORM_MIN_VERSION", 2)
            cfg.agent.fs_norm_require_archive_match = False
            cfg.agent.fs_norm_output_clip = _env_float("FS_NORM_OUTPUT_CLIP", 12.0)
            cfg.agent.fs_norm_output_clip_mode = os.getenv("FS_NORM_OUTPUT_CLIP_MODE", "stats_bounds")
        if os.getenv("USE_PLANNING_TOKEN_ADAPTER") is not None:
            cfg.agent.use_planning_token_adapter = _env_flag("USE_PLANNING_TOKEN_ADAPTER", False)
        cfg.agent.planning_num_tokens = _env_int("PLANNING_NUM_TOKENS", int(cfg.agent.planning_num_tokens))
        cfg.agent.planning_num_heads = _env_int("PLANNING_NUM_HEADS", int(cfg.agent.planning_num_heads))
        cfg.agent.planning_condition_layers = os.getenv(
            "PLANNING_CONDITION_LAYERS",
            str(cfg.agent.planning_condition_layers),
        )
        cfg.agent.planning_gate_init = _env_float("PLANNING_GATE_INIT", float(cfg.agent.planning_gate_init))
        cfg.agent.planning_context_gate_init = _env_float(
            "PLANNING_CONTEXT_GATE_INIT",
            float(cfg.agent.planning_context_gate_init),
        )
        cfg.agent.planning_condition_dropout = _env_float(
            "PLANNING_CONDITION_DROPOUT",
            float(cfg.agent.planning_condition_dropout),
        )
        if os.getenv("USE_JEPA") is not None:
            cfg.agent.use_jepa = _env_flag("USE_JEPA", False)
        if os.getenv("USE_VGGT") is not None:
            cfg.agent.use_vggt = _env_flag("USE_VGGT", False)
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

    dataset = _build_train_dataset(cfg, agent, build_log_split)
    if token_manifest:
        dataset = _filter_dataset_by_tokens(dataset, _load_token_manifest(token_manifest))
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
    prefiltered_existing = 0
    if skip_existing_records and prefilter_existing_records and not dry_run:
        try:
            dataset, prefiltered_existing = _prefilter_missing_existing_records(
                dataset,
                buffer_dir,
                validate_existing_records,
            )
            logger.info(
                "Prefiltered %d existing AWAC elite records before DataLoader; %d scenes remain.",
                prefiltered_existing,
                len(dataset),
            )
        except AttributeError as exc:
            logger.warning(
                "Could not prefilter existing AWAC records before DataLoader (%s); "
                "falling back to per-batch filtering.",
                exc,
            )
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
    skipped_existing = int(prefiltered_existing)
    aggregate: Dict[str, Any] = {
        "num_scenes": 0,
        "sums": defaultdict(float),
        "best_valid_sources": Counter(),
        "sg_fps": {
            "scene_count": 0,
            "candidate_count": 0,
            "external_candidate_count": 0,
            "external_anchor_semantic_dropped": 0,
            "support_tag_counts": Counter(),
            "build_config": {},
        },
    }
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for features, targets, tokens in dataloader:
            tokens = [str(token) for token in tokens]
            if skip_existing_records and not dry_run:
                missing_indices = [
                    index
                    for index, token in enumerate(tokens)
                    if not _existing_record_is_usable(buffer_dir, token, validate_existing_records)
                ]
                skipped_existing += len(tokens) - len(missing_indices)
                if not missing_indices:
                    if skipped_existing % 256 == 0:
                        logger.info("Skipped %d existing AWAC elite records.", skipped_existing)
                    continue
                if len(missing_indices) != len(tokens):
                    batch_len = len(tokens)
                    features = _filter_batch_by_indices(features, missing_indices, batch_len)
                    targets = _filter_batch_by_indices(targets, missing_indices, batch_len)
                    tokens = [tokens[index] for index in missing_indices]
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
            if write_sg_fps_v3:
                sg_summary = _save_sg_fps_v3_records(
                    buffer_dir,
                    tokens,
                    awac_batch,
                    agent.action_head,
                    metric_cache,
                    agent.action_head.offline_rl_cfg,
                    external_loader,
                    dry_run,
                    support_top_m=sg_fps_support_top_m,
                    expand_external_candidates=expand_external_candidates,
                    external_expansion_max_per_scene=external_expansion_max_per_scene,
                    use_raw_internal_candidates=use_raw_internal_candidates,
                    archive_version=sg_fps_archive_version,
                    build_provenance=build_provenance,
                )
                aggregate["sg_fps"]["scene_count"] += int(sg_summary["scene_count"])
                aggregate["sg_fps"]["candidate_count"] += int(sg_summary["candidate_count"])
                aggregate["sg_fps"]["external_candidate_count"] += int(sg_summary["external_candidate_count"])
                aggregate["sg_fps"]["external_anchor_semantic_dropped"] += int(
                    sg_summary.get("external_anchor_semantic_dropped", 0)
                )
                aggregate["sg_fps"]["support_tag_counts"].update(sg_summary["support_tag_counts"])
                if not aggregate["sg_fps"]["build_config"]:
                    aggregate["sg_fps"]["build_config"] = dict(sg_summary.get("build_config", {}))
            else:
                _save_records(
                    buffer_dir,
                    tokens,
                    awac_batch,
                    dry_run,
                    merge_existing_records=merge_existing_records,
                    merge_keep_top_k=merge_keep_top_k,
                    merge_keep_support=merge_keep_support,
                )
            for row in _iter_summary_rows(tokens, awac_batch):
                if write_sg_fps_v3:
                    row["record_version"] = int(sg_fps_archive_version)
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
        "record_version": int(sg_fps_archive_version) if write_sg_fps_v3 else 2,
        "sg_fps_v3": {
            "enabled": bool(write_sg_fps_v3),
            "scene_count": int(aggregate["sg_fps"]["scene_count"]),
            "candidate_count": int(aggregate["sg_fps"]["candidate_count"]),
            "external_candidate_count": int(aggregate["sg_fps"]["external_candidate_count"]),
            "external_anchor_semantic_dropped": int(aggregate["sg_fps"]["external_anchor_semantic_dropped"]),
            "support_tag_counts": dict(aggregate["sg_fps"]["support_tag_counts"]),
            "external_candidate_roots": external_candidate_roots,
            "build_config": aggregate["sg_fps"]["build_config"],
        },
    }
    summary_json.write_text(json.dumps(global_summary, indent=2, sort_keys=True), encoding="utf-8")

    logger.info("AWAC elite buffer summary written to %s", summary_csv)
    logger.info("AWAC elite buffer global summary written to %s", summary_json)
    if dry_run:
        logger.info("DRY_RUN=true; elite records were not written.")
    else:
        label = f"SG-FPS v{sg_fps_archive_version} support" if write_sg_fps_v3 else "AWAC elite"
        logger.info("%s records written under %s", label, buffer_dir)
    if skip_existing_records:
        logger.info("Skipped %d existing AWAC elite records.", skipped_existing)
    if merge_existing_records:
        logger.info(
            "Merged existing AWAC elite records with keep_top_k=%d keep_support=%s.",
            merge_keep_top_k,
            merge_keep_support,
        )


if __name__ == "__main__":
    main()
