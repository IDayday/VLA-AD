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
