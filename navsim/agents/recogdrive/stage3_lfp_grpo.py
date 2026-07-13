from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Literal, Mapping, Optional

import torch
import torch.distributed as dist

from .stage3_metric_adapter import CanonicalMetricBatch, Stage3MetricAdapter
from .stage3_policy_geometry import (
    apply_group_credit_gate,
    compute_group_credit_active_mask,
)
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
    quality_positive_credit_guard: bool = False
    quality_reference_tolerance: float = 0.02
    reference_pareto_gate_enabled: bool = False
    reference_margin_weight: float = 0.20
    reference_margin_scale: float = 0.05
    pareto_gate_enabled: bool = True
    progress_gate_enabled: bool = True
    all_infeasible_rescue: bool = False
    gradient_checkpointing: bool = False

    advantage_normalization: Literal["global_std", "scene_group_std"] = "global_std"
    global_std_floor: float = 0.05
    advantage_clip: float = 3.0

    # Representation-aware exploration and low-information credit suppression.
    fs_transition_std_enabled: bool = False
    fs_endpoint_std_x_m: float = 0.0
    fs_endpoint_std_y_m: float = 0.0
    fs_endpoint_std_heading_rad: float = 0.0
    min_group_pairwise_ade_m: float = 0.0
    min_group_scalar_span: float = 0.0
    advantage_deadband: float = 0.0

    curriculum_enabled: bool = True
    curriculum_warmup_epochs: int = 1
    frontier_fast_ema: float = 0.80
    frontier_slow_ema: float = 0.98
    frontier_progress_weight: float = 0.50
    frontier_priority_exponent: float = 0.50
    frontier_uniform_ratio: float = 0.20
    frontier_priority_eps: float = 1e-3
    frontier_priority_quantile_cap: float = 0.95
    frontier_safety_first: bool = False
    frontier_tradeoff_weight: float = 0.0
    frontier_tradeoff_eps: float = 0.01
    frontier_diversity_capacity_enabled: bool = False
    frontier_diversity_capacity_cache_path: str = ""
    frontier_diversity_gap_weight: float = 0.05
    frontier_diversity_dispersion_floor: float = 0.05
    frontier_state_path: str = ""

    reference_kl_coeff: float = 0.005

    def validate(self) -> None:
        if self.benchmark not in {"navsim_v1", "navsim_v2"}:
            raise ValueError("lfp_grpo_cfg.benchmark must be navsim_v1 or navsim_v2.")
        if self.ddc_guard_mode != "gt_relative":
            raise ValueError("LFP-GRPO only supports ddc_guard_mode='gt_relative'.")
        if self.advantage_normalization not in {"global_std", "scene_group_std"}:
            raise ValueError(
                "lfp_grpo_cfg.advantage_normalization must be 'global_std' or "
                "'scene_group_std'."
            )
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
            "quality_reference_tolerance",
            "reference_margin_weight",
            "global_std_floor",
            "advantage_clip",
            "fs_endpoint_std_x_m",
            "fs_endpoint_std_y_m",
            "fs_endpoint_std_heading_rad",
            "min_group_pairwise_ade_m",
            "min_group_scalar_span",
            "advantage_deadband",
            "frontier_progress_weight",
            "frontier_priority_exponent",
            "frontier_priority_eps",
            "frontier_tradeoff_weight",
            "frontier_tradeoff_eps",
            "frontier_diversity_gap_weight",
            "frontier_diversity_dispersion_floor",
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
        if not 0.0 < self.frontier_diversity_dispersion_floor:
            raise ValueError(
                "lfp_grpo_cfg.frontier_diversity_dispersion_floor must be positive."
            )
        if self.frontier_diversity_capacity_enabled:
            if not self.curriculum_enabled:
                raise ValueError(
                    "frontier_diversity_capacity_enabled requires curriculum_enabled=true."
                )
            if not self.frontier_diversity_capacity_cache_path:
                raise ValueError(
                    "frontier_diversity_capacity_enabled requires "
                    "frontier_diversity_capacity_cache_path."
                )
        if self.fs_transition_std_enabled:
            for name in (
                "fs_endpoint_std_x_m",
                "fs_endpoint_std_y_m",
                "fs_endpoint_std_heading_rad",
            ):
                if float(getattr(self, name)) <= 0.0:
                    raise ValueError(
                        f"lfp_grpo_cfg.{name} must be positive when "
                        "fs_transition_std_enabled=True."
                    )
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
        if not 0.0 <= self.frontier_tradeoff_weight <= 1.0:
            raise ValueError("lfp_grpo_cfg.frontier_tradeoff_weight must be in [0, 1].")


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
    quality_ok: torch.Tensor
    reference_pareto_ok: torch.Tensor
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


def compute_reference_dominance_mask(
    objectives: torch.Tensor,
    reference_objectives: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Returns candidates Pareto-dominated by their scene's coherent reference."""
    if objectives.ndim != 3 or objectives.shape[-1] != 3:
        raise ValueError(f"objectives must have shape [B, G, 3], got {tuple(objectives.shape)}.")
    if reference_objectives.shape != (objectives.shape[0], objectives.shape[-1]):
        raise ValueError(
            "reference_objectives must have shape [B, 3], got "
            f"{tuple(reference_objectives.shape)}."
        )
    reference = reference_objectives[:, None, :].to(objectives)
    all_ge = (reference >= objectives - float(eps)).all(dim=-1)
    any_gt = (reference > objectives + float(eps)).any(dim=-1)
    return all_ge & any_gt


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


def normalize_group_centered_scores(
    centered: torch.Tensor,
    normalization_mask: torch.Tensor,
    global_moments: GlobalMoments,
    *,
    mode: Literal["global_std", "scene_group_std"],
    std_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale scene-centered scores globally or by each scene's feasible spread."""
    if centered.ndim != 2 or normalization_mask.shape != centered.shape:
        raise ValueError("centered and normalization_mask must have identical [B, G] shapes.")
    if mode not in {"global_std", "scene_group_std"}:
        raise ValueError(f"Unsupported advantage normalization mode: {mode!r}.")
    if std_floor <= 0.0:
        raise ValueError("std_floor must be positive.")

    mask = normalization_mask.to(centered)
    count = mask.sum(dim=1)
    variance = (centered.square() * mask).sum(dim=1) / count.clamp_min(1.0)
    scene_std = variance.clamp_min(0.0).sqrt()
    scene_std = torch.where(
        count >= 2.0,
        scene_std.clamp_min(float(std_floor)),
        scene_std.new_full(scene_std.shape, float(std_floor)),
    ).detach()
    if mode == "scene_group_std":
        normalized = centered / scene_std[:, None]
    else:
        normalized = (centered - global_moments.mean) / global_moments.std
    return torch.nan_to_num(normalized), scene_std


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


def compute_safety_first_frontier_energy(
    pareto_energy: torch.Tensor,
    advantages: torch.Tensor,
    feasible_mask: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Use safety learning energy until a rollout group is fully feasible."""
    if advantages.ndim != 2 or feasible_mask.shape != advantages.shape:
        raise ValueError("advantages and feasible_mask must have identical [B, G] shapes.")
    if pareto_energy.shape != (advantages.shape[0],):
        raise ValueError("pareto_energy must have shape [B].")

    safe_fraction = feasible_mask.float().mean(dim=1)
    unsafe_fraction = 1.0 - safe_fraction
    magnitude = advantages.abs().mean(dim=1)
    mixed_safety_energy = 4.0 * safe_fraction * unsafe_fraction * magnitude
    all_feasible = feasible_mask.all(dim=1)
    all_infeasible = ~feasible_mask.any(dim=1)
    mixed = ~(all_feasible | all_infeasible)
    safety_energy = torch.where(mixed, mixed_safety_energy, magnitude)
    energy = torch.where(all_feasible, pareto_energy, safety_energy)
    return energy, {
        "safe_fraction": safe_fraction,
        "unsafe_fraction": unsafe_fraction,
        "safety_energy": safety_energy,
        "all_feasible": all_feasible.float(),
        "mixed_feasibility": mixed.float(),
        "all_infeasible": all_infeasible.float(),
    }


def compute_pareto_tradeoff_intensity(
    objectives: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 0.01,
) -> torch.Tensor:
    """Fraction of valid pairs with gains and losses on different objectives."""
    if objectives.ndim != 3 or objectives.shape[-1] != 3:
        raise ValueError(f"objectives must have shape [B, G, 3], got {tuple(objectives.shape)}.")
    if valid_mask.shape != objectives.shape[:2]:
        raise ValueError("valid_mask shape must match objectives[:2].")
    if eps < 0.0:
        raise ValueError("eps must be non-negative.")

    group_size = objectives.shape[1]
    delta = objectives[:, :, None, :] - objectives[:, None, :, :]
    mixed_tradeoff = (delta > float(eps)).any(dim=-1) & (delta < -float(eps)).any(dim=-1)
    upper_triangle = torch.triu(
        torch.ones((group_size, group_size), device=objectives.device, dtype=torch.bool),
        diagonal=1,
    )
    valid_pair = valid_mask[:, :, None] & valid_mask[:, None, :] & upper_triangle[None]
    count = (mixed_tradeoff & valid_pair).to(objectives).sum(dim=(1, 2))
    denominator = valid_pair.to(objectives).sum(dim=(1, 2))
    return torch.where(denominator > 0.0, count / denominator.clamp_min(1.0), torch.zeros_like(count))


def compute_lfp_advantages(
    metrics: CanonicalMetricBatch,
    reference: ReferenceBatch,
    cfg: LFPGRPOConfig,
    *,
    group_pairwise_ade_m: Optional[torch.Tensor] = None,
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
    if bool(cfg.quality_positive_credit_guard):
        quality_ok = metrics.quality >= (
            reference.quality[:, None].to(metrics.quality) - float(cfg.quality_reference_tolerance)
        )
    else:
        quality_ok = torch.ones_like(progress_ok)
    objectives = adapter.pareto_objectives(metrics)
    reference_objectives = torch.stack(
        (
            reference.ep.to(metrics.ep),
            reference.ttc.to(metrics.ttc),
            reference.quality.to(metrics.quality),
        ),
        dim=-1,
    )
    reference_dominated = compute_reference_dominance_mask(objectives, reference_objectives)
    reference_pareto_ok = (
        ~reference_dominated
        if bool(cfg.reference_pareto_gate_enabled)
        else torch.ones_like(progress_ok)
    )
    eligible = feasible & progress_ok & ttc_ok & quality_ok & reference_pareto_ok
    pareto = (
        compute_pareto_front_mask(objectives, eligible)
        if bool(cfg.pareto_gate_enabled)
        else eligible
    )

    scalar_span = metrics.scalar.max(dim=1).values - metrics.scalar.min(dim=1).values
    active_group, _ = compute_group_credit_active_mask(
        group_pairwise_ade_m,
        scalar_span,
        min_pairwise_ade_m=float(cfg.min_group_pairwise_ade_m),
        min_scalar_span=float(cfg.min_group_scalar_span),
    )
    centered, baseline, norm_mask, center_diag = group_center(score, feasible)
    norm_mask = norm_mask & active_group[:, None]
    moments = distributed_masked_moments(centered, norm_mask, float(cfg.global_std_floor))
    z, scene_group_std = normalize_group_centered_scores(
        centered,
        norm_mask,
        moments,
        mode=cfg.advantage_normalization,
        std_floor=float(cfg.global_std_floor),
    )

    advantages = torch.zeros_like(z)
    positive_eligible = feasible & progress_ok & ttc_ok & quality_ok & reference_pareto_ok & pareto
    nonpositive_eligible = feasible & (
        ~progress_ok | ~ttc_ok | ~quality_ok | ~reference_pareto_ok | ~pareto
    )
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
    gate_output = apply_group_credit_gate(
        advantages,
        group_pairwise_ade_m,
        scalar_span,
        min_pairwise_ade_m=float(cfg.min_group_pairwise_ade_m),
        min_scalar_span=float(cfg.min_group_scalar_span),
        advantage_deadband=float(cfg.advantage_deadband),
    )
    advantages = gate_output.advantages
    base_energy, energy_diag = compute_bidirectional_pareto_advantage_energy(advantages)
    pareto_tradeoff_intensity = compute_pareto_tradeoff_intensity(
        objectives,
        feasible,
        eps=float(cfg.frontier_tradeoff_eps),
    )
    alternative_tradeoff_diagnostics: Dict[str, torch.Tensor] = {}
    if cfg.benchmark == "navsim_v1":
        for name, alternative_objectives in (
            (
                "ep_ttc_ddc",
                torch.stack((metrics.ep, metrics.ttc, metrics.ddc_guard_value), dim=-1),
            ),
            (
                "ep_scalar_ddc",
                torch.stack((metrics.ep, metrics.scalar, metrics.ddc_guard_value), dim=-1),
            ),
        ):
            alternative_intensity = compute_pareto_tradeoff_intensity(
                alternative_objectives,
                feasible,
                eps=float(cfg.frontier_tradeoff_eps),
            )
            alternative_front = compute_pareto_front_mask(alternative_objectives, eligible)
            alternative_tradeoff_diagnostics.update(
                {
                    f"lfp_v1_{name}_tradeoff_intensity_mean": alternative_intensity.mean().detach(),
                    f"lfp_v1_{name}_pareto_front_ratio": alternative_front.float().mean().detach(),
                }
            )
    tradeoff_weight = float(cfg.frontier_tradeoff_weight)
    tradeoff_multiplier = (1.0 - tradeoff_weight) + tradeoff_weight * pareto_tradeoff_intensity
    pareto_energy = base_energy * tradeoff_multiplier
    safety_first_energy, safety_energy_diag = compute_safety_first_frontier_energy(
        pareto_energy,
        advantages,
        feasible,
    )
    energy = safety_first_energy if bool(cfg.frontier_safety_first) else pareto_energy
    eps = 1e-8
    dominated = eligible & ~pareto
    all_infeasible = ~has_feasible
    ttc_fail = feasible & ~ttc_ok
    quality_fail = feasible & ~quality_ok
    reference_dominated_feasible = feasible & reference_dominated
    progress_fail = feasible & ~progress_ok
    unsafe = ~feasible
    positive_advantage = advantages > eps
    scalar_delta = metrics.scalar - reference.scalar[:, None].to(metrics.scalar)
    quality_delta = metrics.quality - reference.quality[:, None].to(metrics.quality)

    def masked_advantage_mean(mask: torch.Tensor) -> torch.Tensor:
        mask_float = mask.to(advantages)
        return (
            (advantages * mask_float).sum() / mask_float.sum().clamp_min(1.0)
        ).detach()

    def masked_zero_credit_ratio(mask: torch.Tensor) -> torch.Tensor:
        mask_float = mask.to(advantages)
        return (
            ((advantages.abs() <= eps).to(advantages) * mask_float).sum()
            / mask_float.sum().clamp_min(1.0)
        ).detach()

    def conditional_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        denominator_float = denominator.to(advantages)
        return (
            numerator.to(advantages).sum() / denominator_float.sum().clamp_min(1.0)
        ).detach()

    def masked_value_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_float = mask.to(values)
        return ((values * mask_float).sum() / mask_float.sum().clamp_min(1.0)).detach()

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
        "lfp_quality_ok_ratio": quality_ok.float().mean().detach(),
        "lfp_reference_pareto_ok_ratio": reference_pareto_ok.float().mean().detach(),
        "lfp_reference_dominated_ratio": reference_dominated_feasible.float().mean().detach(),
        "lfp_quality_reference_regression_ratio": (
            quality_delta < -float(cfg.quality_reference_tolerance)
        ).float().mean().detach(),
        "lfp_pareto_front_ratio": pareto.float().mean().detach(),
        "lfp_dominated_ratio": dominated.float().mean().detach(),
        "lfp_positive_advantage_ratio": (advantages > eps).float().mean().detach(),
        "lfp_negative_advantage_ratio": (advantages < -eps).float().mean().detach(),
        "lfp_zero_advantage_ratio": (advantages.abs() <= eps).float().mean().detach(),
        "lfp_advantage_mean": advantages.mean().detach(),
        "lfp_advantage_std": advantages.std(unbiased=False).detach(),
        "lfp_positive_eligible_advantage_mean": masked_advantage_mean(positive_eligible),
        "lfp_ttc_fail_ratio": ttc_fail.float().mean().detach(),
        "lfp_ttc_fail_advantage_mean": masked_advantage_mean(ttc_fail),
        "lfp_ttc_fail_zero_credit_ratio": masked_zero_credit_ratio(ttc_fail),
        "lfp_quality_fail_ratio": quality_fail.float().mean().detach(),
        "lfp_quality_fail_advantage_mean": masked_advantage_mean(quality_fail),
        "lfp_quality_fail_zero_credit_ratio": masked_zero_credit_ratio(quality_fail),
        "lfp_reference_dominated_advantage_mean": masked_advantage_mean(reference_dominated_feasible),
        "lfp_reference_dominated_zero_credit_ratio": masked_zero_credit_ratio(reference_dominated_feasible),
        "lfp_positive_advantage_below_ref_scalar_ratio": conditional_ratio(
            positive_advantage & (scalar_delta < 0.0), positive_advantage
        ),
        "lfp_positive_advantage_below_ref_quality_ratio": conditional_ratio(
            positive_advantage & (quality_delta < 0.0), positive_advantage
        ),
        "lfp_positive_advantage_reference_dominated_ratio": conditional_ratio(
            positive_advantage & reference_dominated, positive_advantage
        ),
        "lfp_positive_advantage_scalar_delta_mean": masked_value_mean(scalar_delta, positive_advantage),
        "lfp_positive_advantage_quality_delta_mean": masked_value_mean(quality_delta, positive_advantage),
        "lfp_progress_fail_ratio": progress_fail.float().mean().detach(),
        "lfp_progress_fail_advantage_mean": masked_advantage_mean(progress_fail),
        "lfp_progress_fail_zero_credit_ratio": masked_zero_credit_ratio(progress_fail),
        "lfp_unsafe_advantage_mean": masked_advantage_mean(unsafe),
        "lfp_all_infeasible_group_ratio": all_infeasible.float().mean().detach(),
        "lfp_all_infeasible_rescue_ratio": (
            ((advantages < -eps) & all_infeasible_mask).float().mean().detach()
        ),
        "lfp_advantage_clip_ratio": (pre_clip.abs() > float(cfg.advantage_clip)).float().mean().detach(),
        "lfp_global_centered_mean": moments.mean.detach(),
        "lfp_global_centered_std": moments.std.detach(),
        "lfp_global_normalization_count": moments.count.detach(),
        "lfp_scene_group_std_mean": scene_group_std.mean().detach(),
        "lfp_advantage_normalization_scene_group": energy.new_tensor(
            float(cfg.advantage_normalization == "scene_group_std")
        ),
        "lfp_frontier_energy_mean": energy.mean().detach(),
        "lfp_frontier_base_energy_mean": base_energy.mean().detach(),
        "lfp_frontier_pareto_energy_mean": pareto_energy.mean().detach(),
        "lfp_frontier_safety_energy_mean": safety_energy_diag["safety_energy"].mean().detach(),
        "lfp_frontier_safety_first_enabled": energy.new_tensor(
            float(bool(cfg.frontier_safety_first))
        ),
        "lfp_all_feasible_group_ratio": safety_energy_diag["all_feasible"].mean().detach(),
        "lfp_mixed_feasibility_group_ratio": safety_energy_diag["mixed_feasibility"].mean().detach(),
        "lfp_unsafe_fraction_mean": safety_energy_diag["unsafe_fraction"].mean().detach(),
        "lfp_pareto_tradeoff_intensity_mean": pareto_tradeoff_intensity.mean().detach(),
        "lfp_pareto_tradeoff_intensity_p50": torch.quantile(
            pareto_tradeoff_intensity.float(), 0.5
        ).to(energy).detach(),
        "lfp_frontier_tradeoff_multiplier_mean": tradeoff_multiplier.mean().detach(),
        "lfp_frontier_energy_p50": torch.quantile(energy.float(), 0.5).to(energy).detach(),
        "lfp_frontier_energy_p90": torch.quantile(energy.float(), 0.9).to(energy).detach(),
        "lfp_positive_fraction_mean": energy_diag["positive_fraction"].mean().to(energy).detach(),
        "lfp_negative_fraction_mean": energy_diag["negative_fraction"].mean().to(energy).detach(),
        "lfp_advantage_magnitude_mean": energy_diag["advantage_magnitude"].mean().to(energy).detach(),
        "lfp_group_baseline_mean": baseline.mean().detach(),
        "lfp_reference_fallback_ratio": reference.fallback_mask.float().mean().to(energy).detach(),
        "lfp_reference_source_code": reference.selected_source_code.float().mean().to(energy).detach(),
        **alternative_tradeoff_diagnostics,
        **gate_output.diagnostics,
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
        quality_ok=quality_ok,
        reference_pareto_ok=reference_pareto_ok,
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
