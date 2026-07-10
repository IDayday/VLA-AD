from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.offline_rl_buffer import REQUIRED_COMPONENT_KEYS
from navsim.agents.recogdrive.recogdrive_diffusion_planner import (
    GRPOConfig,
    OfflineRLConfig,
    ReCogDriveDiffusionPlanner,
    _build_asmi_weights,
    _compute_dpsi_support_profile,
    _sample_asmi_targets,
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
