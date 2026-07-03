from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass
class PDASMetrics:
    success_rate: torch.Tensor
    learnability: torch.Tensor
    reward_std: torch.Tensor
    bon_gap: torch.Tensor
    bucket_count: torch.Tensor
    bucket_diversity: torch.Tensor
    regression_risk: torch.Tensor
    weight: torch.Tensor


def _cfg_value(cfg: Any, name: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _component(components: Mapping[str, torch.Tensor], name: str, ref: torch.Tensor) -> torch.Tensor:
    value = components.get(name)
    if value is None:
        return torch.zeros_like(ref)
    return value.to(device=ref.device, dtype=ref.dtype)


def compute_phenotype_buckets(trajs, components, feas, ref, cfg) -> torch.Tensor:
    if trajs.ndim != 4:
        raise ValueError(f"trajs must have shape [B, G, H, 3], got {tuple(trajs.shape)}.")
    B, G = trajs.shape[:2]
    device = trajs.device
    ep = _component(components, "ego_progress", trajs[..., 0, 0])
    ttc = _component(components, "time_to_collision_within_bound", trajs[..., 0, 0])
    ref_ep = ref.get("ref_ep", ep.mean(dim=1)).to(device=device, dtype=trajs.dtype)
    endpoint_y = trajs[..., -1, 1]
    fast = ep > ref_ep[:, None] + float(_cfg_value(cfg, "fp_progress_fast_margin", 0.02))
    slow = ep < ref_ep[:, None] - float(_cfg_value(cfg, "fp_progress_slow_margin", 0.02))
    progress_bucket = torch.where(fast, torch.full_like(ep, 2, dtype=torch.long), torch.where(slow, torch.zeros_like(ep, dtype=torch.long), torch.ones_like(ep, dtype=torch.long)))
    lateral_threshold = float(_cfg_value(cfg, "fp_lateral_bucket_threshold_m", 0.5))
    lateral_bucket = torch.where(endpoint_y > lateral_threshold, torch.full((B, G), 2, device=device, dtype=torch.long), torch.where(endpoint_y < -lateral_threshold, torch.zeros((B, G), device=device, dtype=torch.long), torch.ones((B, G), device=device, dtype=torch.long)))
    ttc_bucket = (ttc >= float(_cfg_value(cfg, "fp_ttc_high_threshold", 0.95))).to(torch.long)
    feas_cost = feas.get("feas_cost") if isinstance(feas, Mapping) else None
    if feas_cost is None:
        feas_bucket = torch.zeros((B, G), device=device, dtype=torch.long)
    else:
        feas_bucket = (feas_cost.to(device=device, dtype=trajs.dtype) <= float(_cfg_value(cfg, "fp_feas_bucket_threshold", 0.1))).to(torch.long)
    return progress_bucket * 18 + lateral_bucket * 6 + ttc_bucket * 2 + feas_bucket


def compute_pdas_metrics(rewards_matrix, components_matrix, trajs_matrix, ref, cfg) -> PDASMetrics:
    if rewards_matrix.ndim != 2:
        raise ValueError(f"rewards_matrix must have shape [B, G], got {tuple(rewards_matrix.shape)}.")
    B, G = rewards_matrix.shape
    dtype = rewards_matrix.dtype
    device = rewards_matrix.device
    threshold = float(_cfg_value(cfg, "pdas_success_threshold", 0.0))
    if "ref_pdms" in ref:
        ref_reward = ref["ref_pdms"].to(device=device, dtype=dtype)
        success = rewards_matrix > ref_reward[:, None] + threshold
        regression = rewards_matrix < ref_reward[:, None] - float(_cfg_value(cfg, "pdas_regression_margin", 0.01))
    else:
        success = rewards_matrix > rewards_matrix.mean(dim=1, keepdim=True)
        regression = rewards_matrix < rewards_matrix.mean(dim=1, keepdim=True)
    success_rate = success.float().mean(dim=1)
    learnability = success_rate * (1.0 - success_rate)
    reward_std = rewards_matrix.std(dim=1, unbiased=False)
    bon_gap = rewards_matrix.max(dim=1).values - rewards_matrix.mean(dim=1)
    buckets = compute_phenotype_buckets(trajs_matrix, components_matrix, {}, ref, cfg)
    bucket_count_values = []
    for row in buckets:
        bucket_count_values.append(float(torch.unique(row).numel()))
    bucket_count = torch.tensor(bucket_count_values, device=device, dtype=dtype)
    bucket_diversity = bucket_count / max(float(G), 1.0)
    regression_risk = regression.float().mean(dim=1)

    eps = float(_cfg_value(cfg, "pdas_eps", 1e-3))
    alpha = float(_cfg_value(cfg, "pdas_alpha", 1.0))
    beta = float(_cfg_value(cfg, "pdas_beta", 1.0))
    gamma = float(_cfg_value(cfg, "pdas_gamma", 1.0))
    lambda_bucket = float(_cfg_value(cfg, "pdas_lambda_bucket", 0.5))
    lambda_regression = float(_cfg_value(cfg, "pdas_lambda_regression", 0.5))
    reward_std_norm = reward_std / reward_std.mean().clamp(min=eps)
    bon_gap_norm = bon_gap / bon_gap.mean().clamp(min=eps)
    weight = (
        (eps + learnability).pow(alpha)
        * (eps + reward_std_norm).pow(beta)
        * (eps + bon_gap_norm).pow(gamma)
        * (1.0 + lambda_bucket * bucket_diversity)
        * (1.0 + lambda_regression * regression_risk)
    )
    weight = weight / weight.mean().clamp(min=eps)
    return PDASMetrics(
        success_rate=success_rate,
        learnability=learnability,
        reward_std=reward_std,
        bon_gap=bon_gap,
        bucket_count=bucket_count,
        bucket_diversity=bucket_diversity,
        regression_risk=regression_risk,
        weight=weight,
    )
