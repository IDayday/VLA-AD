from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch


DESCRIPTOR_NAMES = (
    "x_final",
    "y_final",
    "heading_final",
    "mean_speed",
    "terminal_speed",
    "early_progress_ratio",
)
SUPPORT_SLOTS = 3
FREE_BUCKET_ID = SUPPORT_SLOTS
MISSING_BUCKET_ID = -1


@dataclass(frozen=True)
class ParetoSupportSelection:
    indices: torch.Tensor
    mask: torch.Tensor
    weights: torch.Tensor
    scores: torch.Tensor
    descriptors: torch.Tensor
    fallback_mode: str
    metadata: Dict[str, Any]


def _ensure_traj_tensor(traj: torch.Tensor, *, require_finite: bool = True) -> torch.Tensor:
    if not isinstance(traj, torch.Tensor):
        raise TypeError(f"trajectory must be a torch.Tensor, got {type(traj).__name__}.")
    if traj.shape[-2:] != (8, 3):
        raise ValueError(f"trajectory must end with shape [8, 3], got {tuple(traj.shape)}.")
    if not torch.is_floating_point(traj):
        traj = traj.float()
    if require_finite and not torch.isfinite(traj).all():
        raise ValueError("trajectory contains non-finite values.")
    return traj


def trajectory_descriptor(traj: torch.Tensor, interval_length: float = 0.5) -> torch.Tensor:
    """Compute PSI-Drive physical trajectory descriptors.

    The input is expected to be in physical ego-frame coordinates, not the
    normalized DiT action range.
    """
    traj = _ensure_traj_tensor(traj)
    if float(interval_length) <= 0.0:
        raise ValueError("interval_length must be positive.")
    xy = traj[..., :, :2]
    delta_xy = xy[..., 1:, :] - xy[..., :-1, :]
    speeds = torch.linalg.norm(delta_xy, dim=-1) / float(interval_length)
    mean_speed = speeds.mean(dim=-1)
    terminal_speed = speeds[..., -1]
    final = traj[..., -1, :]
    final_x = final[..., 0]
    early_x = traj[..., traj.shape[-2] // 2 - 1, 0]
    default_ratio = torch.full_like(final_x, 0.5)
    denominator = torch.where(final_x.abs() > 1e-4, final_x, torch.ones_like(final_x))
    early_ratio = torch.where(final_x.abs() > 1e-4, early_x / denominator, default_ratio)
    early_ratio = early_ratio.clamp(min=-5.0, max=5.0)
    descriptor = torch.stack(
        (
            final[..., 0],
            final[..., 1],
            final[..., 2],
            mean_speed,
            terminal_speed,
            early_ratio,
        ),
        dim=-1,
    )
    if not torch.isfinite(descriptor).all():
        raise ValueError("trajectory descriptor contains non-finite values.")
    return descriptor


def normalize_descriptors(descriptor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if descriptor.shape[-1] != len(DESCRIPTOR_NAMES):
        raise ValueError(f"descriptor last dimension must be {len(DESCRIPTOR_NAMES)}, got {descriptor.shape[-1]}.")
    mean = mean.to(device=descriptor.device, dtype=descriptor.dtype)
    std = std.to(device=descriptor.device, dtype=descriptor.dtype)
    if mean.shape != (len(DESCRIPTOR_NAMES),) or std.shape != (len(DESCRIPTOR_NAMES),):
        raise ValueError(
            f"descriptor mean/std must have shape [{len(DESCRIPTOR_NAMES)}], got {tuple(mean.shape)} and {tuple(std.shape)}."
        )
    if not torch.isfinite(descriptor).all() or not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("descriptor, mean, and std must be finite.")
    return (descriptor - mean) / std.clamp(min=1e-6)


def _finite_candidate_mask(candidates: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(candidates).flatten(1).all(dim=1)


def _source_priority(source: str) -> int:
    priorities = {
        "gt": 0,
        "ground_truth": 0,
        "il": 1,
        "stage2_deterministic": 1,
        "deterministic_il": 1,
        "stage2_stochastic": 2,
        "policy": 2,
        "structured_perturbation": 3,
        "sota_step21600": 4,
        "bootstrap": 4,
    }
    return priorities.get(str(source), 99)


def _rank_key(
    idx: int,
    scores: torch.Tensor,
    pdms: Optional[torch.Tensor],
    core: Optional[torch.Tensor],
    nc_dac: Optional[torch.Tensor],
    geometry_distance: Optional[torch.Tensor],
    sources: Optional[Sequence[str]],
) -> tuple:
    score_value = float(scores[idx].item())
    pdms_value = float(pdms[idx].item()) if pdms is not None else score_value
    core_value = float(core[idx].item()) if core is not None else score_value
    nc_dac_value = float(nc_dac[idx].item()) if nc_dac is not None else 0.0
    geometry_value = float(geometry_distance[idx].item()) if geometry_distance is not None else 0.0
    source_value = sources[idx] if sources is not None and idx < len(sources) else ""
    return (
        -score_value,
        -pdms_value,
        -core_value,
        -nc_dac_value,
        geometry_value,
        _source_priority(source_value),
        int(idx),
    )


def _support_weights(
    selected: Sequence[int],
    gt_candidate_index: Optional[int],
    *,
    best_weight: float,
    gt_weight: float,
    other_weight: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    weights = torch.zeros(SUPPORT_SLOTS, dtype=dtype)
    if not selected:
        return weights
    if len(selected) == 1:
        weights[0] = 1.0
        return weights

    selected_list = [int(idx) for idx in selected]
    best_slot = 0
    gt_slot = selected_list.index(int(gt_candidate_index)) if gt_candidate_index in selected_list else None
    weights[best_slot] += max(float(best_weight), 0.0)
    if gt_slot is None:
        weights[best_slot] += max(float(gt_weight), 0.0)
    else:
        weights[gt_slot] += max(float(gt_weight), 0.0)
    other_slots = [slot for slot in range(len(selected_list)) if slot != best_slot and slot != gt_slot]
    if other_slots:
        share = max(float(other_weight), 0.0) / float(len(other_slots))
        for slot in other_slots:
            weights[slot] += share
    else:
        weights[best_slot] += max(float(other_weight), 0.0)
    total = weights.sum()
    if total <= 0.0:
        weights[: len(selected_list)] = 1.0 / float(len(selected_list))
    else:
        weights = weights / total
    return weights


def select_adaptive_pareto_supports(
    candidates: torch.Tensor,
    scores: torch.Tensor,
    valid_positive_mask: torch.Tensor,
    *,
    descriptors: Optional[torch.Tensor] = None,
    descriptor_mean: Optional[torch.Tensor] = None,
    descriptor_std: Optional[torch.Tensor] = None,
    pdms: Optional[torch.Tensor] = None,
    core: Optional[torch.Tensor] = None,
    nc_dac: Optional[torch.Tensor] = None,
    geometry_distance: Optional[torch.Tensor] = None,
    sources: Optional[Sequence[str]] = None,
    gt_candidate_index: Optional[int] = None,
    deterministic_il_index: Optional[int] = None,
    quality_band: float = 0.02,
    min_descriptor_distance: float = 0.75,
    max_supports: int = SUPPORT_SLOTS,
    best_weight: float = 0.50,
    gt_weight: float = 0.20,
    other_weight: float = 0.30,
) -> ParetoSupportSelection:
    candidates = _ensure_traj_tensor(candidates, require_finite=False)
    if candidates.ndim != 3:
        raise ValueError(f"candidates must have shape [K, 8, 3], got {tuple(candidates.shape)}.")
    if int(max_supports) < 1 or int(max_supports) > SUPPORT_SLOTS:
        raise ValueError(f"max_supports must be in [1, {SUPPORT_SLOTS}], got {max_supports}.")
    k = candidates.shape[0]
    scores = torch.as_tensor(scores, dtype=torch.float32, device=candidates.device).view(-1)
    valid_positive_mask = torch.as_tensor(valid_positive_mask, dtype=torch.bool, device=candidates.device).view(-1)
    if scores.shape != (k,) or valid_positive_mask.shape != (k,):
        raise ValueError(f"scores and valid_positive_mask must have shape [K={k}].")
    if descriptors is None:
        descriptors = trajectory_descriptor(candidates)
    else:
        descriptors = torch.as_tensor(descriptors, dtype=torch.float32, device=candidates.device)
    if descriptors.shape != (k, len(DESCRIPTOR_NAMES)):
        raise ValueError(f"descriptors must have shape [K, {len(DESCRIPTOR_NAMES)}], got {tuple(descriptors.shape)}.")
    if descriptor_mean is None:
        descriptor_mean = descriptors.mean(dim=0)
    if descriptor_std is None:
        descriptor_std = descriptors.std(dim=0, unbiased=False).clamp(min=1e-6)
    norm_desc = normalize_descriptors(descriptors, descriptor_mean, descriptor_std)

    finite_mask = _finite_candidate_mask(candidates) & torch.isfinite(scores) & torch.isfinite(norm_desc).all(dim=1)
    valid_positive_mask = valid_positive_mask & finite_mask
    valid_indices = torch.nonzero(valid_positive_mask, as_tuple=False).view(-1).tolist()
    fallback_mode = "none"
    if valid_indices:
        best_score = max(float(scores[idx].item()) for idx in valid_indices)
        band = [idx for idx in valid_indices if float(scores[idx].item()) >= best_score - float(quality_band)]
        ordered = sorted(
            band,
            key=lambda idx: _rank_key(idx, scores, pdms, core, nc_dac, geometry_distance, sources),
        )
        selected: List[int] = [int(ordered[0])]
        remaining = [idx for idx in ordered[1:] if idx != selected[0]]
        while remaining and len(selected) < int(max_supports):
            selected_desc = norm_desc[torch.tensor(selected, device=norm_desc.device)]
            scored_remaining = []
            for idx in remaining:
                dist = torch.linalg.norm(norm_desc[idx : idx + 1] - selected_desc, dim=-1).min()
                scored_remaining.append((float(dist.item()), idx))
            dist_value, idx = max(
                scored_remaining,
                key=lambda item: (
                    item[0],
                    -_rank_key(item[1], scores, pdms, core, nc_dac, geometry_distance, sources)[0],
                    -item[1],
                ),
            )
            if dist_value < float(min_descriptor_distance):
                break
            selected.append(int(idx))
            remaining = [item for item in remaining if item != idx]
    else:
        selected = []
        for fallback_idx, mode in (
            (gt_candidate_index, "gt_fallback"),
            (deterministic_il_index, "deterministic_il_fallback"),
        ):
            if fallback_idx is None:
                continue
            idx = int(fallback_idx)
            if 0 <= idx < k and bool(finite_mask[idx].item()):
                selected = [idx]
                fallback_mode = mode
                break
        if not selected:
            finite_indices = torch.nonzero(finite_mask, as_tuple=False).view(-1).tolist()
            if finite_indices:
                selected = [int(finite_indices[0])]
                fallback_mode = "first_finite_fallback"
            else:
                raise ValueError("No finite candidate is available for Pareto support selection.")

    indices = torch.full((SUPPORT_SLOTS,), -1, dtype=torch.long)
    mask = torch.zeros((SUPPORT_SLOTS,), dtype=torch.bool)
    out_scores = torch.zeros((SUPPORT_SLOTS,), dtype=torch.float32)
    out_desc = torch.zeros((SUPPORT_SLOTS, len(DESCRIPTOR_NAMES)), dtype=torch.float32)
    for slot, idx in enumerate(selected[:SUPPORT_SLOTS]):
        indices[slot] = int(idx)
        mask[slot] = True
        out_scores[slot] = scores[int(idx)].detach().float().cpu()
        out_desc[slot] = descriptors[int(idx)].detach().float().cpu()
    weights = _support_weights(
        selected[:SUPPORT_SLOTS],
        gt_candidate_index,
        best_weight=best_weight,
        gt_weight=gt_weight,
        other_weight=other_weight,
        dtype=torch.float32,
    )
    weights = weights * mask.float()
    if weights.sum() > 0:
        weights = weights / weights.sum()
    metadata = {
        "selected_count": int(mask.sum().item()),
        "quality_band": float(quality_band),
        "min_descriptor_distance": float(min_descriptor_distance),
        "fallback_mode": fallback_mode,
        "selected_indices": [int(idx) for idx in selected[:SUPPORT_SLOTS]],
    }
    return ParetoSupportSelection(
        indices=indices,
        mask=mask,
        weights=weights,
        scores=out_scores,
        descriptors=out_desc,
        fallback_mode=fallback_mode,
        metadata=metadata,
    )


def load_pareto_support_index(path: str | Path, *, require_schema: bool = True) -> Dict[str, Any]:
    index_path = Path(path)
    if not index_path.is_file():
        raise FileNotFoundError(f"Pareto support index not found: {index_path}")
    payload = torch.load(index_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Pareto support index must be a dict, got {type(payload).__name__}.")
    if "token_to_row" not in payload:
        tokens = payload.get("tokens")
        if isinstance(tokens, list):
            payload["token_to_row"] = {str(token): idx for idx, token in enumerate(tokens)}
    if require_schema:
        required = (
            "tokens",
            "token_to_row",
            "support_trajectories",
            "support_mask",
            "support_weights",
            "support_scores",
            "descriptor_mean",
            "descriptor_std",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(f"Pareto support index missing required fields: {missing}")
        tokens = payload["tokens"]
        token_to_row = payload["token_to_row"]
        if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
            raise TypeError("Pareto support index field 'tokens' must be list[str].")
        if not isinstance(token_to_row, Mapping):
            raise TypeError("Pareto support index field 'token_to_row' must be a mapping.")
        if len(set(tokens)) != len(tokens):
            raise ValueError("Pareto support index contains duplicate tokens.")
        n = len(tokens)
        shape_checks = {
            "support_trajectories": (n, SUPPORT_SLOTS, 8, 3),
            "support_mask": (n, SUPPORT_SLOTS),
            "support_weights": (n, SUPPORT_SLOTS),
            "support_scores": (n, SUPPORT_SLOTS),
            "descriptor_mean": (len(DESCRIPTOR_NAMES),),
            "descriptor_std": (len(DESCRIPTOR_NAMES),),
        }
        for key, expected in shape_checks.items():
            value = payload[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Pareto support index field {key!r} must be a tensor.")
            if tuple(value.shape) != expected:
                raise ValueError(f"Pareto support index field {key!r} shape {tuple(value.shape)} != {expected}.")
            if key != "support_mask" and not torch.isfinite(value.float()).all():
                raise ValueError(f"Pareto support index field {key!r} contains non-finite values.")
        mask = payload["support_mask"].bool()
        if (mask.sum(dim=1) < 1).any() or (mask.sum(dim=1) > SUPPORT_SLOTS).any():
            raise ValueError("Each support index row must contain 1 to 3 supports.")
        weight_sum = (payload["support_weights"].float() * mask.float()).sum(dim=1)
        if not torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-4):
            raise ValueError("Support weights must sum to 1 over valid entries.")
    return payload


def lookup_support_batch(
    tokens: Sequence[str],
    support_index: Mapping[str, Any],
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    missing_policy: str = "zeros",
) -> Dict[str, torch.Tensor]:
    if missing_policy not in {"zeros", "error"}:
        raise ValueError("missing_policy must be 'zeros' or 'error'.")
    token_to_row = support_index.get("token_to_row", {})
    if not isinstance(token_to_row, Mapping):
        raise TypeError("support_index['token_to_row'] must be a mapping.")
    n = len(tokens)
    trajs = torch.zeros((n, SUPPORT_SLOTS, 8, 3), dtype=torch.float32)
    mask = torch.zeros((n, SUPPORT_SLOTS), dtype=torch.bool)
    weights = torch.zeros((n, SUPPORT_SLOTS), dtype=torch.float32)
    scores = torch.zeros((n, SUPPORT_SLOTS), dtype=torch.float32)
    missing = torch.zeros((n,), dtype=torch.bool)
    source = support_index["support_trajectories"]
    source_mask = support_index["support_mask"].bool()
    source_weights = support_index["support_weights"].float()
    source_scores = support_index["support_scores"].float()
    for batch_idx, token in enumerate(tokens):
        row = token_to_row.get(str(token))
        if row is None:
            if missing_policy == "error":
                raise KeyError(f"Token {token!r} is missing from Pareto support index.")
            missing[batch_idx] = True
            continue
        row = int(row)
        trajs[batch_idx] = source[row].float()
        mask[batch_idx] = source_mask[row]
        weights[batch_idx] = source_weights[row]
        scores[batch_idx] = source_scores[row]
    if device is not None:
        trajs = trajs.to(device=device, dtype=dtype or trajs.dtype)
        mask = mask.to(device=device)
        weights = weights.to(device=device, dtype=dtype or weights.dtype)
        scores = scores.to(device=device, dtype=dtype or scores.dtype)
        missing = missing.to(device=device)
    elif dtype is not None:
        trajs = trajs.to(dtype=dtype)
        weights = weights.to(dtype=dtype)
        scores = scores.to(dtype=dtype)
    return {
        "support_trajectories": trajs,
        "support_mask": mask,
        "support_weights": weights,
        "support_scores": scores,
        "support_missing_mask": missing,
    }


def assign_support_buckets(
    trajectories: torch.Tensor,
    support_trajectories: torch.Tensor,
    support_mask: torch.Tensor,
    descriptor_mean: torch.Tensor,
    descriptor_std: torch.Tensor,
    *,
    free_distance: float = 1.5,
    interval_length: float = 0.5,
) -> Dict[str, torch.Tensor]:
    if trajectories.ndim != 4 or trajectories.shape[-2:] != (8, 3):
        raise ValueError(f"trajectories must have shape [B, G, 8, 3], got {tuple(trajectories.shape)}.")
    if support_trajectories.ndim != 4 or support_trajectories.shape[1:] != (SUPPORT_SLOTS, 8, 3):
        raise ValueError(
            f"support_trajectories must have shape [B, {SUPPORT_SLOTS}, 8, 3], got {tuple(support_trajectories.shape)}."
        )
    if support_trajectories.shape[0] != trajectories.shape[0]:
        raise ValueError("support batch size must match trajectory batch size.")
    support_mask = support_mask.to(device=trajectories.device, dtype=torch.bool)
    if support_mask.shape != support_trajectories.shape[:2]:
        raise ValueError(f"support_mask shape {tuple(support_mask.shape)} does not match support slots.")
    B, G = trajectories.shape[:2]
    traj_desc = trajectory_descriptor(trajectories.reshape(B * G, 8, 3), interval_length=interval_length).reshape(
        B, G, -1
    )
    support_desc = trajectory_descriptor(
        support_trajectories.reshape(B * SUPPORT_SLOTS, 8, 3),
        interval_length=interval_length,
    ).reshape(B, SUPPORT_SLOTS, -1)
    traj_norm = normalize_descriptors(traj_desc, descriptor_mean, descriptor_std)
    support_norm = normalize_descriptors(support_desc, descriptor_mean, descriptor_std)
    distance = torch.linalg.norm(traj_norm[:, :, None, :] - support_norm[:, None, :, :], dim=-1)
    inf = torch.full_like(distance, float("inf"))
    masked_distance = torch.where(support_mask[:, None, :], distance, inf)
    min_distance, nearest = masked_distance.min(dim=-1)
    has_support = support_mask.any(dim=1)
    bucket = nearest.to(dtype=torch.long)
    bucket = torch.where(min_distance > float(free_distance), torch.full_like(bucket, FREE_BUCKET_ID), bucket)
    bucket = torch.where(has_support[:, None], bucket, torch.full_like(bucket, MISSING_BUCKET_ID))
    return {
        "bucket_id": bucket,
        "distance": distance,
        "min_distance": min_distance,
        "has_support": has_support,
        "free_bucket_id": torch.tensor(FREE_BUCKET_ID, device=trajectories.device, dtype=torch.long),
        "missing_bucket_id": torch.tensor(MISSING_BUCKET_ID, device=trajectories.device, dtype=torch.long),
    }
