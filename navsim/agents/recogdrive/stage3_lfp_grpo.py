from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Literal, Mapping

import torch
import torch.distributed as dist

from .stage3_metric_adapter import CanonicalMetricBatch, Stage3MetricAdapter
from .stage3_reference_cache import ReferenceBatch


@dataclass
class LFPGRPOConfig:
    enabled: bool = False
    benchmark: Literal["navsim_v1", "navsim_v2"] = "navsim_v1"
    reference_cache_path: str = ""

    # Kept isolated because this repository and NAVSIM v2 share the navsim package name.
    v2_official_navsim_root: str = ""
    v2_official_config_dir: str = ""
    v2_official_config_name: str = "default_run_pdm_score"
    v2_official_overrides: tuple[str, ...] = ("train_test_split=navtrain",)
    v2_evaluator_timeout_s: float = 600.0

    require_nc: bool = True
    require_dac: bool = True
    ddc_guard_mode: Literal["gt_relative"] = "gt_relative"
    ddc_gt_tolerance: float = 0.01
    v2_require_tlc: bool = True

    ep_reference_tolerance: float = 0.02
    ttc_positive_credit_guard: bool = False
    ttc_reference_tolerance: float = 0.01
    reference_margin_weight: float = 0.20
    reference_margin_scale: float = 0.05
    pareto_gate_enabled: bool = True
    progress_gate_enabled: bool = True
    all_infeasible_rescue: bool = False
    gradient_checkpointing: bool = False

    global_std_floor: float = 0.05
    advantage_clip: float = 3.0

    curriculum_enabled: bool = True
    curriculum_warmup_epochs: int = 1
    frontier_fast_ema: float = 0.80
    frontier_slow_ema: float = 0.98
    frontier_progress_weight: float = 0.50
    frontier_priority_exponent: float = 0.50
    frontier_uniform_ratio: float = 0.20
    frontier_priority_eps: float = 1e-3
    frontier_priority_quantile_cap: float = 0.95
    frontier_state_path: str = ""

    reference_kl_coeff: float = 0.005

    def validate(self) -> None:
        if self.benchmark not in {"navsim_v1", "navsim_v2"}:
            raise ValueError("lfp_grpo_cfg.benchmark must be navsim_v1 or navsim_v2.")
        if self.ddc_guard_mode != "gt_relative":
            raise ValueError("LFP-GRPO only supports ddc_guard_mode='gt_relative'.")
        if self.enabled and not self.reference_cache_path:
            raise ValueError("LFP-GRPO requires lfp_grpo_cfg.reference_cache_path.")
        if self.enabled and self.benchmark == "navsim_v2" and not self.v2_official_navsim_root:
            raise ValueError(
                "NAVSIM v2 LFP-GRPO requires lfp_grpo_cfg.v2_official_navsim_root "
                "for official one-stage EPDMS scoring."
            )
        if self.v2_evaluator_timeout_s <= 0.0:
            raise ValueError("lfp_grpo_cfg.v2_evaluator_timeout_s must be positive.")
        for name in (
            "ddc_gt_tolerance",
            "ep_reference_tolerance",
            "ttc_reference_tolerance",
            "reference_margin_weight",
            "global_std_floor",
            "advantage_clip",
            "frontier_progress_weight",
            "frontier_priority_exponent",
            "frontier_priority_eps",
            "reference_kl_coeff",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"lfp_grpo_cfg.{name} must be non-negative.")
        if self.reference_margin_scale <= 0.0:
            raise ValueError("lfp_grpo_cfg.reference_margin_scale must be positive.")
        if not 0.0 < self.global_std_floor:
            raise ValueError("lfp_grpo_cfg.global_std_floor must be positive.")
        if not 0.0 < self.advantage_clip:
            raise ValueError("lfp_grpo_cfg.advantage_clip must be positive.")
        if self.curriculum_warmup_epochs < 0:
            raise ValueError("lfp_grpo_cfg.curriculum_warmup_epochs must be non-negative.")
        for name in ("frontier_fast_ema", "frontier_slow_ema"):
            value = float(getattr(self, name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"lfp_grpo_cfg.{name} must be in [0, 1).")
        if not 0.0 <= self.frontier_uniform_ratio <= 1.0:
            raise ValueError("lfp_grpo_cfg.frontier_uniform_ratio must be in [0, 1].")
        if not 0.0 < self.frontier_priority_quantile_cap <= 1.0:
            raise ValueError("lfp_grpo_cfg.frontier_priority_quantile_cap must be in (0, 1].")


def coerce_lfp_grpo_config(value: Any) -> LFPGRPOConfig:
    if value is None:
        return LFPGRPOConfig()
    if isinstance(value, LFPGRPOConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("lfp_grpo_cfg must be a mapping or LFPGRPOConfig.")
    valid = {item.name for item in fields(LFPGRPOConfig)}
    unknown = sorted(set(value) - valid)
    if unknown:
        raise TypeError(f"Unknown lfp_grpo_cfg fields: {unknown}.")
    return LFPGRPOConfig(**dict(value))


def validate_lfp_config_exclusivity(
    stage3_algorithm: str,
    grpo_cfg: Any,
    offline_cfg: Any = None,
    *,
    stage3_objective: str = "grpo",
) -> None:
    """Rejects every legacy Stage3 mechanism that could alter LFP credit."""
    if str(stage3_algorithm) != "lfp_grpo":
        return

    conflicts: list[str] = []

    def enabled(obj: Any, name: str, default: Any = False) -> bool:
        return bool(getattr(obj, name, default))

    def positive(obj: Any, name: str, default: float = 0.0) -> bool:
        return float(getattr(obj, name, default)) != 0.0

    if str(stage3_objective) == "grpo_replay":
        conflicts.append("stage3_objective=grpo_replay")
    sampling_std = float(getattr(grpo_cfg, "min_sampling_denoising_std", 0.04))
    logprob_std = float(getattr(grpo_cfg, "min_logprob_denoising_std", sampling_std))
    if sampling_std <= 0.0 or logprob_std <= 0.0:
        raise ValueError("LFP-GRPO requires positive reverse-transition standard-deviation floors.")
    if abs(sampling_std - logprob_std) > 1e-12:
        raise ValueError(
            "LFP-GRPO exact on-policy transitions require min_sampling_denoising_std "
            "and min_logprob_denoising_std to be identical; "
            f"got {sampling_std} and {logprob_std}."
        )
    if not enabled(grpo_cfg, "use_trajectory_level_objective", True):
        conflicts.append("use_trajectory_level_objective")
    if str(getattr(grpo_cfg, "trajectory_logprob_reduce", "discounted_mean")) != "discounted_mean":
        conflicts.append("trajectory_logprob_reduce")
    for name in (
        "use_gspo_ratio",
        "behavior_policy_sample",
        "use_core_pareto_grpo",
        "use_feasible_pareto_grpo",
        "fp_use_pdas",
        "core_pareto_use_phenotype_bucket_grpo",
        "core_pareto_use_adaptive_dual",
        "core_pareto_buffer_bonus_enabled",
        "use_dynamic_group_weight",
        "use_diversity_reward",
    ):
        if enabled(grpo_cfg, name):
            conflicts.append(name)
    for name in ("bc_coeff_start", "bc_coeff_end"):
        if positive(grpo_cfg, name):
            conflicts.append(name)

    if offline_cfg is not None:
        for name in (
            "grpo_buffer_guidance_enabled",
        ):
            if enabled(offline_cfg, name):
                conflicts.append(name)
        for name in (
            "grpo_buffer_reward_bonus_weight",
            "grpo_buffer_distill_loss_weight",
            "grpo_buffer_distill_loss_weight_start",
            "grpo_buffer_distill_loss_weight_end",
            "grpo_buffer_preference_dpo_loss_weight",
            "grpo_buffer_preference_dpo_loss_weight_start",
            "grpo_self_imitation_loss_weight",
            "grpo_self_imitation_loss_weight_start",
            "grpo_self_imitation_loss_weight_end",
            "bc_loss_weight",
            "bc_loss_weight_start",
            "bc_loss_weight_end",
        ):
            if positive(offline_cfg, name):
                conflicts.append(name)

    if conflicts:
        joined = ", ".join(sorted(set(conflicts)))
        raise ValueError(
            "stage3_algorithm='lfp_grpo' is a simple on-policy objective and conflicts with: "
            f"{joined}. Disable every listed field explicitly."
        )


@dataclass
class GlobalMoments:
    mean: torch.Tensor
    std: torch.Tensor
    count: torch.Tensor


@dataclass
class LFPAdvantageOutput:
    advantages: torch.Tensor
    pareto_front: torch.Tensor
    feasible: torch.Tensor
    progress_ok: torch.Tensor
    ttc_ok: torch.Tensor
    energy: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]


def compute_pareto_front_mask(
    objectives: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if objectives.ndim != 3 or objectives.shape[-1] != 3:
        raise ValueError(f"objectives must have shape [B, G, 3], got {tuple(objectives.shape)}.")
    if valid_mask.shape != objectives.shape[:2]:
        raise ValueError("valid_mask shape must match objectives[:2].")
    candidate_i = objectives[:, :, None, :]
    candidate_j = objectives[:, None, :, :]
    all_ge = (candidate_j >= candidate_i - float(eps)).all(dim=-1)
    any_gt = (candidate_j > candidate_i + float(eps)).any(dim=-1)
    valid_pair = valid_mask[:, :, None] & valid_mask[:, None, :]
    dominated = (all_ge & any_gt & valid_pair).any(dim=2)
    return valid_mask & ~dominated


def distributed_masked_moments(
    values: torch.Tensor,
    mask: torch.Tensor,
    std_floor: float,
) -> GlobalMoments:
    if values.shape != mask.shape:
        raise ValueError("values and mask must have identical shapes.")
    if std_floor <= 0.0:
        raise ValueError("std_floor must be positive.")
    values64 = values.detach().to(torch.float64)
    mask64 = mask.detach().to(device=values.device, dtype=torch.float64)
    finite_mask = mask64 * torch.isfinite(values64).to(torch.float64)
    count = finite_mask.sum()
    total = (torch.where(torch.isfinite(values64), values64, torch.zeros_like(values64)) * finite_mask).sum()
    total_sq = (
        torch.where(torch.isfinite(values64), values64.square(), torch.zeros_like(values64)) * finite_mask
    ).sum()
    stats = torch.stack((count, total, total_sq))
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    count, total, total_sq = stats.unbind()
    safe_count = count.clamp_min(1.0)
    mean64 = torch.where(count > 0.0, total / safe_count, torch.zeros_like(total))
    variance64 = (total_sq / safe_count - mean64.square()).clamp_min(0.0)
    std64 = torch.sqrt(variance64)
    std64 = torch.where(count >= 2.0, std64.clamp_min(float(std_floor)), std64.new_tensor(float(std_floor)))
    dtype = values.dtype if values.is_floating_point() else torch.float32
    return GlobalMoments(
        mean=mean64.to(device=values.device, dtype=dtype).detach(),
        std=std64.to(device=values.device, dtype=dtype).detach(),
        count=count.to(device=values.device, dtype=dtype).detach(),
    )


def group_center(
    score: torch.Tensor,
    feasible_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    if score.ndim != 2 or score.shape != feasible_mask.shape:
        raise ValueError("score and feasible_mask must both have shape [B, G].")
    feasible_float = feasible_mask.to(score)
    feasible_count = feasible_float.sum(dim=1)
    feasible_mean = (score * feasible_float).sum(dim=1) / feasible_count.clamp_min(1.0)
    all_mean = score.mean(dim=1)
    baseline = torch.where(feasible_count >= 2.0, feasible_mean, all_mean)
    has_feasible = feasible_count > 0.0
    normalization_mask = feasible_mask & has_feasible[:, None]
    diagnostics = {
        "feasible_count_mean": feasible_count.float().mean().to(score),
        "feasible_baseline_ratio": (feasible_count >= 2.0).float().mean().to(score),
    }
    return score - baseline[:, None], baseline, normalization_mask, diagnostics


def compute_bidirectional_pareto_advantage_energy(
    advantages: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if advantages.ndim != 2:
        raise ValueError("advantages must have shape [B, G].")
    positive = (advantages > float(eps)).float().mean(dim=1)
    negative = (advantages < -float(eps)).float().mean(dim=1)
    magnitude = advantages.abs().mean(dim=1)
    energy = 4.0 * positive * negative * magnitude
    return energy, {
        "positive_fraction": positive,
        "negative_fraction": negative,
        "advantage_magnitude": magnitude,
    }


def compute_lfp_advantages(
    metrics: CanonicalMetricBatch,
    reference: ReferenceBatch,
    cfg: LFPGRPOConfig,
) -> LFPAdvantageOutput:
    cfg.validate()
    if metrics.scalar.ndim != 2:
        raise ValueError("Canonical LFP metrics must have shape [B, G].")
    batch_size = metrics.scalar.shape[0]
    if reference.scalar.shape != (batch_size,):
        raise ValueError("Reference scalar must have shape [B].")

    adapter = Stage3MetricAdapter(cfg.benchmark)
    reference_margin = float(cfg.reference_margin_weight) * torch.clamp(
        (metrics.scalar - reference.scalar[:, None].to(metrics.scalar)) / float(cfg.reference_margin_scale),
        -1.0,
        1.0,
    )
    score = metrics.scalar + reference_margin
    feasible = adapter.feasible_mask(metrics, reference.gt_ddc, cfg)
    progress_ok = metrics.ep >= reference.ep[:, None].to(metrics.ep) - float(cfg.ep_reference_tolerance)
    if not bool(cfg.progress_gate_enabled):
        progress_ok = torch.ones_like(progress_ok)
    if bool(cfg.ttc_positive_credit_guard):
        ttc_ok = metrics.ttc >= (
            reference.ttc[:, None].to(metrics.ttc) - float(cfg.ttc_reference_tolerance)
        )
    else:
        ttc_ok = torch.ones_like(progress_ok)
    eligible = feasible & progress_ok & ttc_ok
    pareto = (
        compute_pareto_front_mask(adapter.pareto_objectives(metrics), eligible)
        if bool(cfg.pareto_gate_enabled)
        else eligible
    )

    centered, baseline, norm_mask, center_diag = group_center(score, feasible)
    moments = distributed_masked_moments(centered, norm_mask, float(cfg.global_std_floor))
    z = (centered - moments.mean) / moments.std

    advantages = torch.zeros_like(z)
    positive_eligible = feasible & progress_ok & ttc_ok & pareto
    nonpositive_eligible = feasible & (~progress_ok | ~ttc_ok | ~pareto)
    advantages[positive_eligible] = z[positive_eligible]
    advantages[nonpositive_eligible] = torch.minimum(
        z[nonpositive_eligible],
        torch.zeros_like(z[nonpositive_eligible]),
    )
    has_feasible = feasible.any(dim=1)
    advantages[(~feasible) & has_feasible[:, None]] = -1.0
    all_infeasible_mask = (~has_feasible)[:, None].expand_as(advantages)
    if bool(cfg.all_infeasible_rescue):
        rescue = torch.minimum(z, torch.zeros_like(z))
        advantages[all_infeasible_mask] = rescue[all_infeasible_mask]
    else:
        advantages[all_infeasible_mask] = 0.0

    pre_clip = advantages.clone()
    advantages = advantages.clamp(-float(cfg.advantage_clip), float(cfg.advantage_clip)).detach()
    energy, energy_diag = compute_bidirectional_pareto_advantage_energy(advantages)
    eps = 1e-8
    dominated = eligible & ~pareto
    all_infeasible = ~has_feasible
    diagnostics: Dict[str, torch.Tensor] = {
        "lfp_scalar_mean": metrics.scalar.mean().detach(),
        "lfp_ref_scalar_mean": reference.scalar.to(metrics.scalar).mean().detach(),
        "lfp_delta_scalar_mean": (metrics.scalar - reference.scalar[:, None].to(metrics.scalar)).mean().detach(),
        "lfp_ep_mean": metrics.ep.mean().detach(),
        "lfp_ttc_mean": metrics.ttc.mean().detach(),
        "lfp_quality_mean": metrics.quality.mean().detach(),
        "lfp_nc_mean": metrics.nc.mean().detach(),
        "lfp_dac_mean": metrics.dac.mean().detach(),
        "lfp_ddc_mean": metrics.ddc_guard_value.mean().detach(),
        "lfp_feasible_ratio": feasible.float().mean().detach(),
        "lfp_progress_ok_ratio": progress_ok.float().mean().detach(),
        "lfp_ttc_ok_ratio": ttc_ok.float().mean().detach(),
        "lfp_pareto_front_ratio": pareto.float().mean().detach(),
        "lfp_dominated_ratio": dominated.float().mean().detach(),
        "lfp_positive_advantage_ratio": (advantages > eps).float().mean().detach(),
        "lfp_negative_advantage_ratio": (advantages < -eps).float().mean().detach(),
        "lfp_zero_advantage_ratio": (advantages.abs() <= eps).float().mean().detach(),
        "lfp_all_infeasible_group_ratio": all_infeasible.float().mean().detach(),
        "lfp_all_infeasible_rescue_ratio": (
            ((advantages < -eps) & all_infeasible_mask).float().mean().detach()
        ),
        "lfp_advantage_clip_ratio": (pre_clip.abs() > float(cfg.advantage_clip)).float().mean().detach(),
        "lfp_global_centered_mean": moments.mean.detach(),
        "lfp_global_centered_std": moments.std.detach(),
        "lfp_global_normalization_count": moments.count.detach(),
        "lfp_frontier_energy_mean": energy.mean().detach(),
        "lfp_frontier_energy_p50": torch.quantile(energy.float(), 0.5).to(energy).detach(),
        "lfp_frontier_energy_p90": torch.quantile(energy.float(), 0.9).to(energy).detach(),
        "lfp_positive_fraction_mean": energy_diag["positive_fraction"].mean().to(energy).detach(),
        "lfp_negative_fraction_mean": energy_diag["negative_fraction"].mean().to(energy).detach(),
        "lfp_advantage_magnitude_mean": energy_diag["advantage_magnitude"].mean().to(energy).detach(),
        "lfp_group_baseline_mean": baseline.mean().detach(),
        "lfp_reference_fallback_ratio": reference.fallback_mask.float().mean().to(energy).detach(),
        "lfp_reference_source_code": reference.selected_source_code.float().mean().to(energy).detach(),
        **{f"lfp_{key}": value.detach() for key, value in center_diag.items()},
    }
    if metrics.tlc is not None:
        diagnostics["lfp_tlc_mean"] = metrics.tlc.mean().detach()
    for key, value in metrics.diagnostics.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            diagnostics[f"lfp_metric_{key}"] = value.detach().to(energy)
    return LFPAdvantageOutput(
        advantages=advantages,
        pareto_front=pareto,
        feasible=feasible,
        progress_ok=progress_ok,
        ttc_ok=ttc_ok,
        energy=energy.detach(),
        diagnostics=diagnostics,
    )


def trajectory_reinforce_loss(advantages: torch.Tensor, trajectory_logp: torch.Tensor) -> torch.Tensor:
    flat_advantages = advantages.reshape(-1).detach()
    if trajectory_logp.shape != flat_advantages.shape:
        raise ValueError(
            f"trajectory_logp must have shape {tuple(flat_advantages.shape)}, got {tuple(trajectory_logp.shape)}."
        )
    return -(flat_advantages * trajectory_logp).mean()
