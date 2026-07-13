from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

import navsim.agents.recogdrive.recogdrive_diffusion_planner as planner_module
from navsim.agents.recogdrive.offline_rl_buffer import REQUIRED_COMPONENT_KEYS
from navsim.agents.recogdrive.support_aligned_diversity import (
    SupportAlignedDiversityConfig,
    density_balanced_support_weights,
    trajectory_distance_matrix,
)
from navsim.agents.recogdrive.recogdrive_diffusion_planner import (
    GRPOConfig,
    OfflineRLConfig,
    ReCogDriveDiffusionPlanner,
    _apply_dpsi_residual_budget,
    _build_asmi_weights,
    _compute_dpsi_target_hardness_diagnostics,
    _compute_dpsi_mode_balance,
    _compute_dpsi_support_profile,
    _sample_asmi_targets,
)


def test_dpsi_target_hardness_separates_probability_and_residual_mass():
    losses = torch.tensor([[1.0, 9.0, 0.0]])
    weights = torch.tensor([[0.8, 0.2, 0.0]])
    sources = torch.tensor([[1, 7, 0]])

    diagnostics = _compute_dpsi_target_hardness_diagnostics(losses, weights, sources)

    assert diagnostics["gt_target_loss"].item() == pytest.approx(1.0)
    assert diagnostics["non_gt_target_loss"].item() == pytest.approx(9.0)
    assert diagnostics["non_gt_to_gt_loss_ratio"].item() == pytest.approx(9.0)
    assert diagnostics["non_gt_target_weight_ratio"].item() == pytest.approx(0.2)
    assert diagnostics["non_gt_residual_mass_ratio"].item() == pytest.approx(0.6 / 1.4)


def test_dpsi_target_hardness_handles_missing_source_without_nan():
    diagnostics = _compute_dpsi_target_hardness_diagnostics(
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1, 7]]),
    )

    assert all(torch.isfinite(value) for value in diagnostics.values())
    assert diagnostics["non_gt_to_gt_loss_ratio"].item() == 0.0
    assert diagnostics["non_gt_residual_mass_ratio"].item() == 0.0


def test_dpsi_residual_budget_caps_proxy_and_preserves_scene_mass():
    losses = torch.tensor([[1.0, 9.0], [4.0, 1.0]])
    weights = torch.tensor([[0.8, 0.2], [0.6, 0.4]])
    sources = torch.tensor([[1, 7], [1, 4]])

    adjusted, diagnostics = _apply_dpsi_residual_budget(
        losses,
        weights,
        sources,
        enabled=True,
        non_gt_residual_mass_cap=0.30,
    )

    torch.testing.assert_close(adjusted.sum(dim=1), weights.sum(dim=1))
    assert diagnostics["non_gt_residual_mass_ratio_pre_budget"].item() > 0.30
    assert diagnostics["non_gt_residual_mass_ratio_post_budget"].item() <= 0.30 + 1e-6
    assert diagnostics["residual_budget_active_ratio"].item() == 0.5
    assert diagnostics["residual_budget_unanchored_ratio"].item() == 0.0


def test_dpsi_residual_budget_flag_off_is_exactly_identity():
    losses = torch.tensor([[1.0, 9.0]])
    weights = torch.tensor([[0.8, 0.2]])
    sources = torch.tensor([[1, 7]])

    adjusted, diagnostics = _apply_dpsi_residual_budget(
        losses,
        weights,
        sources,
        enabled=False,
        non_gt_residual_mass_cap=0.30,
    )

    assert adjusted is weights
    assert torch.equal(adjusted, weights)
    assert diagnostics["residual_budget_enabled"].item() == 0.0
    assert diagnostics["non_gt_residual_mass_ratio_pre_budget"].item() == pytest.approx(
        diagnostics["non_gt_residual_mass_ratio_post_budget"].item()
    )


def test_dpsi_residual_budget_rejects_active_scene_without_gt():
    with pytest.raises(ValueError, match="positive-weight GT target"):
        _apply_dpsi_residual_budget(
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[0.5, 0.5]]),
            torch.tensor([[7, 4]]),
            enabled=True,
            non_gt_residual_mass_cap=0.38,
        )


def _record(k: int = 10) -> dict:
    components = {key: np.ones((k,), dtype=np.float32) for key in REQUIRED_COMPONENT_KEYS}
    components["ego_progress"] = np.linspace(0.4, 0.9, k, dtype=np.float32)
    return {
        "token": "tok",
        "candidates": np.arange(k * 8 * 3, dtype=np.float32).reshape(k, 8, 3),
        "rewards": np.linspace(0.1, 1.0, k, dtype=np.float32),
        "anchor_distance": np.arange(k, dtype=np.float32),
        "sources": ["ddv2"] * k,
        "support_tags": [""] * k,
        "components": components,
        "valid_mask": np.ones((k,), dtype=np.bool_),
        "selection_score": np.linspace(0.0, 1.0, k, dtype=np.float32),
        "support_indices": [1, 3, 5],
        "gt_reward": 0.6,
        "il_reward": 0.5,
        "best_raw_reward": 1.0,
        "best_valid_reward": 1.0,
        "best_selected_reward": 0.6,
        "best_raw_source": "ddv2",
        "best_valid_source": "ddv2",
        "best_selected_source": "gt",
        "has_valid_candidate": True,
        "version": 3,
    }


def test_dpsi_support_indices_filter():
    record = _record()
    record["sources"][5] = "gt"
    record["support_tags"][1] = "vector_pareto"
    record["support_tags"][5] = "gt_anchor"
    planner = ReCogDriveDiffusionPlanner.__new__(ReCogDriveDiffusionPlanner)

    def load(self, root, token, cfg):
        return record

    planner._load_elite_record_cached = MethodType(load, planner)
    action_input = BatchFeature(data={"action": torch.zeros((1, 8, 3), dtype=torch.float32)})
    cfg = OfflineRLConfig(enabled=True, support_archive_path="/tmp/support", use_dpsi=True)
    batch = planner._load_awac_buffer_candidates(action_input, ["tok"], {}, cfg)

    assert batch["selected_trajs"].shape[1] == 3
    assert torch.equal(batch["selected_source_code"][0], torch.tensor([7, 7, 1]))
    assert batch["selected_support_weight"][0, 0].item() == pytest.approx(cfg.dpsi_weight_vector_pareto)
    assert batch["selected_support_weight"][0, 1].item() == 0.0
    assert batch["selected_support_weight"][0, 2].item() == pytest.approx(cfg.dpsi_weight_gt)
    expected_distance = torch.linalg.vector_norm(
        batch["selected_trajs"][0, :, :, :2] - batch["selected_trajs"][0, 2, :, :2].unsqueeze(0),
        dim=-1,
    ).mean(dim=-1)
    assert torch.allclose(batch["selected_anchor_distance"][0], expected_distance)
    assert batch["selected_anchor_distance"][0, 2].item() == 0.0


def test_residual_budget_injects_gt_before_support_filter_without_changing_baseline():
    record = _record()
    record["sources"][7] = "gt"
    record["support_tags"][7] = "gt_anchor"
    planner = ReCogDriveDiffusionPlanner.__new__(ReCogDriveDiffusionPlanner)

    def load(self, root, token, cfg):
        return record

    planner._load_elite_record_cached = MethodType(load, planner)
    action_input = BatchFeature(data={"action": torch.zeros((1, 8, 3), dtype=torch.float32)})

    baseline_cfg = OfflineRLConfig(enabled=True, support_archive_path="/tmp/support", use_dpsi=True)
    baseline = planner._load_awac_buffer_candidates(action_input, ["tok"], {}, baseline_cfg)
    assert baseline["selected_trajs"].shape[1] == 3
    assert not (baseline["selected_source_code"] == 1).any()
    assert baseline["dpsi_gt_anchor_injected_ratio"].item() == 0.0

    budget_cfg = OfflineRLConfig(
        enabled=True,
        support_archive_path="/tmp/support",
        use_dpsi=True,
        dpsi_pair_target_randomness=True,
        dpsi_residual_budget_enabled=True,
    )
    budget = planner._load_awac_buffer_candidates(action_input, ["tok"], {}, budget_cfg)
    assert budget["selected_trajs"].shape[1] == 4
    assert torch.equal(budget["selected_source_code"][0], torch.tensor([7, 7, 7, 1]))
    assert budget["dpsi_gt_anchor_injected_ratio"].item() == 1.0
    assert budget["selected_anchor_distance"][0, -1].item() == 0.0


def test_dpsi_tag_weights():
    cfg = OfflineRLConfig()
    weight = ReCogDriveDiffusionPlanner._dpsi_weight_for_tag
    assert weight("gt_anchor", "gt", cfg) == pytest.approx(cfg.dpsi_weight_gt)
    assert weight("vector_pareto", "ddv2", cfg) == pytest.approx(cfg.dpsi_weight_vector_pareto)
    assert weight("diversity_max", "ddv2", cfg) == pytest.approx(cfg.dpsi_weight_diversity)
    assert weight("fallback_best", "ddv2", cfg) == pytest.approx(cfg.dpsi_weight_fallback)
    assert weight("", "ddv2", cfg) == 0.0
    assert weight("unknown", "ddv2", cfg) == pytest.approx(cfg.dpsi_weight_unknown)
    assert weight("unknown", "failure_expand", cfg) == pytest.approx(cfg.dpsi_weight_unknown)


def _traj_row(offsets: list[float]) -> torch.Tensor:
    trajs = torch.zeros((1, len(offsets), 8, 3), dtype=torch.float32)
    for i, offset in enumerate(offsets):
        trajs[0, i, :, 0] = torch.linspace(0, 5 + offset, 8)
        trajs[0, i, :, 1] = offset
    return trajs


def _profile_and_weights(trajs, rewards, source_code, gt, il, cfg, epoch=40):
    real = torch.ones_like(rewards, dtype=torch.bool)
    valid = torch.ones_like(rewards, dtype=torch.bool)
    support_weight = torch.ones_like(rewards)
    profile, _ = _compute_dpsi_support_profile(trajs, rewards, real, source_code, gt, il, valid, cfg, epoch)
    weights, _ = _build_asmi_weights(rewards, real, valid, source_code, support_weight, gt, il, profile, cfg)
    return profile, weights


def test_asmi_single_gt_scene():
    cfg = OfflineRLConfig()
    trajs = _traj_row([0.0])
    rewards = torch.tensor([[1.0]])
    source = torch.tensor([[1]])
    profile, weights = _profile_and_weights(trajs, rewards, source, torch.tensor([1.0]), torch.tensor([1.0]), cfg)
    assert profile["support_count"].item() == 1
    assert profile["beta_profile"].item() == 0.0
    assert weights.sum().item() == pytest.approx(1.0)
    assert weights[0, 0].item() == pytest.approx(1.0)


def test_asmi_multimodal_scene():
    cfg = OfflineRLConfig()
    trajs = _traj_row([-3.0, -1.5, 0.0, 1.5, 3.0, 4.5])
    rewards = torch.tensor([[0.65, 0.72, 0.80, 0.88, 0.90, 0.86]])
    source = torch.tensor([[1, 2, 7, 4, 5, 6]])
    profile, weights = _profile_and_weights(trajs, rewards, source, torch.tensor([0.60]), torch.tensor([0.62]), cfg)
    assert profile["beta_profile"].item() > 0.0
    assert weights[0, 2:].sum().item() > 0.0
    assert weights.sum().item() == pytest.approx(profile["scene_weight"].item())


def test_mode_balance_removes_duplicate_vote_advantage():
    trajs = _traj_row([0.0, 0.0, 4.0])
    stats = _compute_dpsi_mode_balance(
        trajs,
        torch.ones((1, 3), dtype=torch.bool),
        bandwidth=0.4,
    )

    probability = stats["mode_density_weights"][0]
    assert probability.sum().item() == pytest.approx(1.0)
    assert probability[:2].sum().item() == pytest.approx(probability[2].item(), rel=0.05)
    assert stats["effective_mode_count"].item() == pytest.approx(2.0, rel=0.05)
    assert stats["mode_capacity"].item() == pytest.approx(0.5, rel=0.05)


def test_mode_balance_matches_snsad_density_contract():
    trajs = _traj_row([-1.5, 0.0, 0.0, 3.0])
    stats = _compute_dpsi_mode_balance(
        trajs,
        torch.ones((1, 4), dtype=torch.bool),
        bandwidth=0.4,
    )
    config = SupportAlignedDiversityConfig(density_bandwidth=0.4)
    numpy_distance = trajectory_distance_matrix(trajs[0].numpy(), trajs[0].numpy(), config)
    numpy_weights = density_balanced_support_weights(numpy_distance, bandwidth=0.4)

    assert np.allclose(stats["mode_density_weights"][0].numpy(), numpy_weights, atol=1e-6)


def test_legacy_target_distribution_does_not_run_mode_balance(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("legacy path must not compute mode-density statistics")

    monkeypatch.setattr(planner_module, "_compute_dpsi_mode_balance", fail_if_called)
    cfg = OfflineRLConfig(dpsi_target_distribution="legacy")
    trajs = _traj_row([-1.0, 0.0, 1.0])
    rewards = torch.tensor([[0.8, 0.9, 1.0]])
    mask = torch.ones_like(rewards, dtype=torch.bool)
    source = torch.tensor([[1, 7, 4]])

    profile, diag = _compute_dpsi_support_profile(
        trajs,
        rewards,
        mask,
        source,
        torch.tensor([0.8]),
        torch.tensor([0.8]),
        mask,
        cfg,
        current_epoch=40,
    )

    assert profile["effective_mode_count"].item() == 0.0
    assert diag["dpsi_mode_balance_enabled"].item() == 0.0


def test_mode_balanced_beta_is_independent_of_source_provenance():
    cfg = OfflineRLConfig(
        dpsi_target_distribution="mode_balanced",
        dpsi_beta_warmup_epochs=0,
        dpsi_high_gt_reward=1.1,
    )
    trajs = _traj_row([-3.0, -1.5, 0.0, 1.5, 3.0])
    rewards = torch.full((1, 5), 0.98)
    real = torch.ones_like(rewards, dtype=torch.bool)
    valid = torch.ones_like(rewards, dtype=torch.bool)
    gt = torch.tensor([0.98])
    il = torch.tensor([0.98])
    single_source = torch.tensor([[1, 7, 7, 7, 7]])
    varied_sources = torch.tensor([[1, 7, 4, 5, 6]])

    single, _ = _compute_dpsi_support_profile(
        trajs, rewards, real, single_source, gt, il, valid, cfg, current_epoch=0
    )
    varied, _ = _compute_dpsi_support_profile(
        trajs, rewards, real, varied_sources, gt, il, valid, cfg, current_epoch=0
    )

    assert single["source_entropy"].item() != pytest.approx(varied["source_entropy"].item())
    assert single["beta_profile"].item() == pytest.approx(varied["beta_profile"].item())
    assert single["beta_profile"].item() > 0.5


def test_mode_balanced_distribution_stays_active_for_saturated_gt():
    cfg = OfflineRLConfig(
        dpsi_target_distribution="mode_balanced",
        dpsi_beta_warmup_epochs=0,
        dpsi_high_gt_reward=0.95,
        dpsi_use_source_weight=False,
    )
    trajs = _traj_row([-3.0, -1.5, 0.0, 1.5, 3.0])
    rewards = torch.ones((1, 5))
    source = torch.tensor([[1, 7, 4, 5, 6]])
    real = torch.ones_like(rewards, dtype=torch.bool)
    valid = torch.ones_like(rewards, dtype=torch.bool)
    support_weight = torch.ones_like(rewards)
    gt = torch.tensor([1.0])
    il = torch.tensor([1.0])

    profile, diag = _compute_dpsi_support_profile(
        trajs, rewards, real, source, gt, il, valid, cfg, current_epoch=0
    )
    weights, weight_diag = _build_asmi_weights(
        rewards, real, valid, source, support_weight, gt, il, profile, cfg
    )

    assert profile["high_gt_saturated"].item()
    assert profile["beta_profile"].item() > 0.5
    assert diag["dpsi_mode_balance_enabled"].item() == 1.0
    assert weight_diag["dpsi_gt_weight_ratio"].item() < 0.6
    assert weight_diag["dpsi_effective_target_count_mean"].item() > 2.5
    assert weights.sum().item() == pytest.approx(profile["scene_weight"].item())


def test_mode_balanced_beta_and_scene_mass_ignore_reward_improver_shortcuts():
    cfg = OfflineRLConfig(
        dpsi_target_distribution="mode_balanced",
        dpsi_beta_max=0.25,
        dpsi_beta_warmup_epochs=0,
        dpsi_scene_normalize_weights=True,
        dpsi_strong_improver_margin=0.01,
        dpsi_improver_scene_weight_gain=2.0,
    )
    trajs = _traj_row([0.0, 1.5, 3.0])
    rewards = torch.tensor([[0.5, 0.9, 1.0]])
    source = torch.tensor([[1, 7, 4]])
    mask = torch.ones_like(rewards, dtype=torch.bool)

    profile, diagnostics = _compute_dpsi_support_profile(
        trajs,
        rewards,
        mask,
        source,
        torch.tensor([0.5]),
        torch.tensor([0.5]),
        mask,
        cfg,
        current_epoch=0,
    )

    expected_beta = cfg.dpsi_beta_max * profile["mode_capacity"]
    assert profile["strong_improver"].item()
    assert profile["beta_profile"].item() == pytest.approx(expected_beta.item())
    assert profile["scene_weight"].item() == 1.0
    assert diagnostics["dpsi_scene_uniform_weighting"].item() == 1.0


def test_mode_balance_rejects_nonpositive_bandwidth():
    with pytest.raises(ValueError, match="bandwidth"):
        _compute_dpsi_mode_balance(
            _traj_row([0.0, 1.0]),
            torch.ones((1, 2), dtype=torch.bool),
            bandwidth=0.0,
        )


def test_mode_balanced_systematic_sampler_is_unbiased_for_target_distribution():
    torch.manual_seed(7)
    batch_size = 2048
    cfg = OfflineRLConfig(
        dpsi_target_distribution="mode_balanced",
        dpsi_target_sample_m=4,
        dpsi_target_sample_m_after_warmup=4,
        dpsi_beta_warmup_epochs=0,
    )
    trajs = _traj_row([0.0, 1.0, 2.0]).expand(batch_size, -1, -1, -1).clone()
    weights = torch.tensor([0.50, 0.25, 0.25]).expand(batch_size, -1).clone()
    real = torch.ones((batch_size, 3), dtype=torch.bool)
    valid = torch.ones_like(real)
    source = torch.tensor([1, 7, 4]).expand(batch_size, -1).clone()
    rewards = torch.tensor([1.0, 0.9, 0.8]).expand(batch_size, -1).clone()

    _, sampled_weights, sampled_mask, diag = _sample_asmi_targets(
        trajs,
        weights,
        real,
        valid,
        source,
        rewards,
        cfg,
        current_epoch=0,
        training=True,
    )

    assert sampled_mask.all()
    assert torch.allclose(sampled_weights, torch.full_like(sampled_weights, 0.25))
    assert diag["dpsi_unbiased_systematic_sampling"].item() == 1.0
    assert diag["dpsi_sampled_gt_weight_ratio"].item() == pytest.approx(0.5, abs=0.02)
    assert diag["dpsi_sampled_unique_target_ratio"].item() == pytest.approx(0.75)


def test_residual_budget_systematic_sampler_keeps_gt_anchor():
    torch.manual_seed(19)
    batch_size = 256
    cfg = OfflineRLConfig(
        dpsi_target_distribution="mode_balanced",
        dpsi_pair_target_randomness=True,
        dpsi_residual_budget_enabled=True,
        dpsi_target_sample_m=2,
        dpsi_target_sample_m_after_warmup=2,
        dpsi_beta_warmup_epochs=0,
    )
    trajs = _traj_row([0.0, 1.0, 2.0]).expand(batch_size, -1, -1, -1).clone()
    weights = torch.tensor([0.05, 0.475, 0.475]).expand(batch_size, -1).clone()
    real = torch.ones((batch_size, 3), dtype=torch.bool)
    source = torch.tensor([1, 7, 4]).expand(batch_size, -1).clone()
    rewards = torch.tensor([1.0, 0.9, 0.8]).expand(batch_size, -1).clone()

    _, _, _, diagnostics = _sample_asmi_targets(
        trajs,
        weights,
        real,
        real,
        source,
        rewards,
        cfg,
        current_epoch=0,
        training=True,
    )
    sampled_source = diagnostics["_sampled_target_source_code"]

    assert (sampled_source == 1).any(dim=1).all()
    assert diagnostics["dpsi_residual_budget_forced_gt_ratio"].item() > 0.8
    assert diagnostics["dpsi_unbiased_systematic_sampling"].item() == 0.0


def test_residual_budget_config_requires_paired_mode_balanced_targets():
    cfg = OfflineRLConfig(
        enabled=True,
        use_dpsi=True,
        dpsi_target_distribution="mode_balanced",
        dpsi_residual_budget_enabled=True,
        dpsi_pair_target_randomness=False,
    )
    with pytest.raises(ValueError, match="paired difficulty estimates"):
        ReCogDriveDiffusionPlanner._validate_offline_rl_config(cfg)


def test_asmi_high_gt_saturated():
    cfg = OfflineRLConfig()
    trajs = _traj_row([-1.0, 0.0, 1.0, 2.0])
    rewards = torch.tensor([[1.0, 0.98, 0.97, 0.96]])
    source = torch.tensor([[1, 7, 4, 5]])
    profile, _ = _profile_and_weights(trajs, rewards, source, torch.tensor([1.0]), torch.tensor([0.98]), cfg)
    assert profile["high_gt_saturated"].item()
    assert profile["beta_profile"].item() == 0.0


def test_asmi_low_gt_strong_improver():
    cfg = OfflineRLConfig()
    trajs = _traj_row([0.0, 2.0])
    rewards = torch.tensor([[0.50, 0.56]])
    source = torch.tensor([[1, 7]])
    profile, _ = _profile_and_weights(trajs, rewards, source, torch.tensor([0.50]), torch.tensor([0.49]), cfg)
    assert profile["low_support"].item()
    assert profile["beta_profile"].item() >= 0.4
    assert profile["scene_weight"].item() > 1.0


def test_dpsi_target_sampling():
    cfg = OfflineRLConfig()
    M = 12
    trajs = _traj_row([float(i) for i in range(M)])
    rewards = torch.linspace(0.1, 0.9, M).view(1, M)
    rewards[0, 7] = 1.0
    weights = torch.ones((1, M), dtype=torch.float32) / M
    real = torch.ones((1, M), dtype=torch.bool)
    valid = torch.ones((1, M), dtype=torch.bool)
    source = torch.zeros((1, M), dtype=torch.long)
    source[0, 0] = 1
    out_trajs, out_weights, out_mask, _ = _sample_asmi_targets(
        trajs, weights, real, valid, source, rewards, cfg, current_epoch=0, training=False
    )
    ids = set(out_trajs[0, :, 0, 1].round().long().tolist())
    assert out_mask.all().item()
    assert 0 in ids
    assert 7 in ids
    assert out_trajs.shape[1] == 4
    assert out_weights.sum().item() == pytest.approx(weights.sum().item())


def test_asmi_production_settings_keep_support_active_from_epoch0():
    cfg = OfflineRLConfig(
        dpsi_empty_tag_zero=False,
        dpsi_weight_unknown=0.55,
        dpsi_beta_warmup_epochs=0,
        dpsi_high_gt_reward=1.1,
        dpsi_target_sample_m=4,
        dpsi_target_sample_m_after_warmup=4,
        dpsi_force_anchor_target=True,
        dpsi_force_best_target=True,
    )
    M = 8
    trajs = _traj_row([float(i) for i in range(M)])
    rewards = torch.tensor([[1.0, 1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94]])
    source = torch.tensor([[1, 7, 4, 5, 6, 8, 10, 7]])
    real = torch.ones_like(rewards, dtype=torch.bool)
    valid = torch.ones_like(rewards, dtype=torch.bool)
    support_weight = torch.tensor([[0.45, 0.9, 0.55, 0.6, 0.55, 0.55, 0.55, 0.55]])

    profile, _ = _compute_dpsi_support_profile(
        trajs,
        rewards,
        real,
        source,
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        valid,
        cfg,
        current_epoch=0,
    )
    weights, diag = _build_asmi_weights(
        rewards,
        real,
        valid,
        source,
        support_weight,
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        profile,
        cfg,
    )
    out_trajs, out_weights, out_mask, sample_diag = _sample_asmi_targets(
        trajs,
        weights,
        real,
        valid,
        source,
        rewards,
        cfg,
        current_epoch=0,
        training=False,
    )

    assert profile["beta_profile"].item() > 0.0
    assert diag["dpsi_pareto_weight_mean"].item() > 0.0
    assert diag["dpsi_gt_weight_ratio"].item() < 0.9
    assert sample_diag["dpsi_sampled_target_count_mean"].item() == pytest.approx(4.0)
    assert out_mask.sum().item() == 4
    assert out_weights.sum().item() == pytest.approx(weights.sum().item())
    sampled_offsets = set(out_trajs[0, out_mask[0], 0, 1].round().long().tolist())
    assert 0 in sampled_offsets
    assert len(sampled_offsets) >= 4


def test_dpsi_single_scene_level_target_sampling():
    cfg = OfflineRLConfig(
        dpsi_target_sample_m=1,
        dpsi_target_sample_m_after_warmup=1,
        dpsi_force_anchor_target=False,
        dpsi_force_best_target=False,
        dpsi_beta_warmup_epochs=0,
    )
    M = 6
    trajs = _traj_row([float(i) for i in range(M)])
    rewards = torch.linspace(0.1, 0.9, M).view(1, M)
    weights = torch.ones((1, M), dtype=torch.float32) / M
    real = torch.ones((1, M), dtype=torch.bool)
    valid = torch.ones((1, M), dtype=torch.bool)
    source = torch.zeros((1, M), dtype=torch.long)
    out_trajs, out_weights, out_mask, diag = _sample_asmi_targets(
        trajs, weights, real, valid, source, rewards, cfg, current_epoch=0, training=False
    )
    assert out_trajs.shape == (1, 1, 8, 3)
    assert out_mask.sum().item() == 1
    assert out_weights.sum().item() == pytest.approx(weights.sum().item())
    assert diag["dpsi_sampled_target_count_mean"].item() == pytest.approx(1.0)


def _fp_planner() -> ReCogDriveDiffusionPlanner:
    planner = ReCogDriveDiffusionPlanner.__new__(ReCogDriveDiffusionPlanner)
    cfg = GRPOConfig()
    for key, value in cfg.__dict__.items():
        if isinstance(value, (int, float, bool, str)):
            setattr(planner, key, value)
    planner.fp_use_bucketed_advantage = True
    planner.fp_use_pdas = True
    planner.pdas_lambda_coverage = 0.2
    planner.pdas_min_support_count_for_coverage = 4
    planner.config = SimpleNamespace(
        geo_curvature_weight=1.0,
        geo_reverse_weight=1.0,
        geo_tail_reverse_weight=2.0,
        geo_early_kink_weight=2.0,
        geo_jerk_weight=0.2,
    )
    return planner


def _fp_inputs(B=1, G=4):
    rewards = torch.full((B, G), 0.8)
    components = {key: torch.ones((B, G)) for key in REQUIRED_COMPONENT_KEYS}
    components["pdms"] = rewards
    components["ego_progress"] = torch.full((B, G), 0.5)
    components["time_to_collision_within_bound"] = torch.ones((B, G))
    components["history_comfort"] = torch.ones((B, G))
    components["driving_direction_compliance"] = torch.ones((B, G))
    trajs = torch.zeros((B, G, 8, 3))
    trajs[..., 0] = torch.linspace(0, 5, 8)
    ref = {"ref_pdms": torch.full((B,), 0.75), "ref_ep": torch.full((B,), 0.5), "ref_ttc": torch.ones(B), "ref_ddc": torch.ones(B)}
    return rewards, components, trajs, ref


def test_bucket_fallback():
    planner = _fp_planner()
    rewards, components, trajs, ref = _fp_inputs()
    _, _, aux = planner._compute_feasible_pareto_advantages(rewards, components, trajs, ref)
    assert aux["fp_unique_bucket_count_mean"].item() == pytest.approx(1.0)
    assert aux["fp_bucketed_active_ratio"].item() == 0.0
    assert aux["fp_bucket_fallback_ratio"].item() == 1.0


def test_pdas_coverage_gap():
    planner = _fp_planner()
    rewards, components, trajs, ref = _fp_inputs()
    support_trajs = trajs.clone()
    support_trajs[0, :, -1, 1] = torch.tensor([-1.0, 0.0, 1.0, -1.5])
    support_components = {key: value.clone() for key, value in components.items()}
    support_components["ego_progress"][0] = torch.tensor([0.3, 0.5, 0.7, 0.8])
    support_batch = {
        "selected_trajs": support_trajs,
        "selected_components": support_components,
        "selected_real_mask": torch.ones((1, 4), dtype=torch.bool),
        "selected_valid_mask": torch.ones((1, 4), dtype=torch.bool),
    }
    _, _, aux = planner._compute_feasible_pareto_advantages(rewards, components, trajs, ref, support_batch=support_batch)
    assert aux["pdas_archive_bucket_count"].item() >= 4.0
    assert aux["pdas_sampled_bucket_count"].item() == pytest.approx(1.0)
    assert aux["pdas_support_coverage_gap"].item() > 0.0

    support_batch["selected_real_mask"][0, 3] = False
    _, _, low_aux = planner._compute_feasible_pareto_advantages(rewards, components, trajs, ref, support_batch=support_batch)
    assert low_aux["pdas_support_coverage_gap"].item() == 0.0
