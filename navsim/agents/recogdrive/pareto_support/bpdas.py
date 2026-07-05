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


def compute_phenotype_buckets(trajs: torch.Tensor, components: dict[str, torch.Tensor], feas: dict[str, torch.Tensor], ref: dict[str, torch.Tensor], cfg: Any = None) -> torch.Tensor:
    ep = components.get("ego_progress", components.get("ep", torch.zeros(trajs.shape[:-2], device=trajs.device)))
    ddc = components.get("driving_direction_compliance", components.get("ddc", torch.ones_like(ep)))
    feas_cost = feas.get("feas_cost", torch.zeros_like(ep))
    ref_ep = ref.get("ref_ep", ref.get("ego_progress", ref.get("ep", torch.zeros(ep.shape[0], device=trajs.device, dtype=ep.dtype))))
    ref_ep = ref_ep.to(device=trajs.device, dtype=ep.dtype)
    if ref_ep.ndim == 1:
        ref_ep = ref_ep[:, None]
    final_y = trajs[..., -1, 1]
    progress_bucket = torch.where(ep < ref_ep - 0.02, 0, torch.where(ep > ref_ep + 0.02, 2, 1))
    lateral_bucket = torch.where(final_y < -0.5, 0, torch.where(final_y > 0.5, 2, 1))
    ddc_bucket = (ddc >= float(_cfg_value(cfg, "fpv3_ddc_min_absolute", 0.95))).to(torch.long)
    feas_bucket = (feas_cost <= float(_cfg_value(cfg, "fpv3_feas_cost_max", 0.10))).to(torch.long)
    return progress_bucket * 18 + lateral_bucket * 6 + ddc_bucket * 2 + feas_bucket


def compute_pdas_metrics(
    rewards_matrix: torch.Tensor,
    components_matrix: dict[str, torch.Tensor],
    trajs_matrix: torch.Tensor,
    ref: dict[str, torch.Tensor],
    cfg: Any = None,
) -> PDASMetrics:
    rewards = rewards_matrix
    if rewards.ndim != 2:
        raise ValueError(f"rewards_matrix must have shape [B, G], got {tuple(rewards.shape)}.")
    ref_reward = ref.get("reward", torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype))
    delta = float(_cfg_value(cfg, "pdas_success_delta", 0.01))
    success = rewards > ref_reward[:, None] + delta
    success_rate = success.to(rewards.dtype).mean(dim=1)
    learnability = 4.0 * success_rate * (1.0 - success_rate)
    reward_std = rewards.std(dim=1, unbiased=False)
    bon_gap = rewards.max(dim=1).values - rewards.mean(dim=1)
    components = {k: v for k, v in components_matrix.items()}
    feas = {"feas_cost": components.pop("feas_cost", torch.zeros_like(rewards))}
    buckets = compute_phenotype_buckets(trajs_matrix, components, feas, ref, cfg)
    bucket_count = torch.tensor(
        [float(torch.unique(row).numel()) for row in buckets],
        device=rewards.device,
        dtype=rewards.dtype,
    )
    max_bucket = float(_cfg_value(cfg, "pdas_max_bucket_count", 36.0))
    bucket_diversity = bucket_count / max_bucket
    pdms = components_matrix.get("pdms", rewards)
    ddc = components_matrix.get("driving_direction_compliance", torch.ones_like(rewards))
    ref_pdms = ref.get("pdms", ref_reward)
    ref_ddc = ref.get("driving_direction_compliance", torch.ones_like(ref_reward))
    regression_risk = (((ref_pdms[:, None] > 0.0) & (pdms <= 0.0)) | (ddc < ref_ddc[:, None] - 0.01)).to(rewards.dtype).amax(dim=1)
    eps = float(_cfg_value(cfg, "pdas_eps", 0.05))
    alpha = float(_cfg_value(cfg, "pdas_alpha", 1.0))
    beta = float(_cfg_value(cfg, "pdas_beta", 1.0))
    gamma = float(_cfg_value(cfg, "pdas_gamma", 0.5))
    lambda_bucket = float(_cfg_value(cfg, "pdas_lambda_bucket", 0.2))
    lambda_reg = float(_cfg_value(cfg, "pdas_lambda_reg", 0.5))
    reward_std_norm = (reward_std / float(_cfg_value(cfg, "pdas_sigma0", 0.1))).clamp(0.0, 1.0)
    bon_gap_norm = (bon_gap / float(_cfg_value(cfg, "pdas_delta0", 0.1))).clamp(0.0, 1.0)
    weight = (
        (eps + learnability).pow(alpha)
        * (eps + reward_std_norm).pow(beta)
        * (eps + bon_gap_norm).pow(gamma)
        * (1.0 + lambda_bucket * bucket_diversity)
        * (1.0 + lambda_reg * regression_risk)
    )
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
