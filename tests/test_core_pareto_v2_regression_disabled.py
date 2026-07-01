from __future__ import annotations

import types

import torch

from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner


def _planner() -> ReCogDriveDiffusionPlanner:
    planner = ReCogDriveDiffusionPlanner.__new__(ReCogDriveDiffusionPlanner)
    planner.core_ep_weight = 5.0
    planner.core_ttc_weight = 5.0
    planner.core_comfort_weight = 2.0
    planner.core_normalizer = 12.0
    planner.core_pareto_require_nc = True
    planner.core_pareto_require_dac = True
    planner.core_pareto_use_ddc_guard = True
    planner.core_pareto_ddc_min_absolute = 0.95
    planner.core_pareto_ddc_drop_tolerance = 0.01
    planner.core_pareto_use_ep_floor = True
    planner.core_pareto_ep_floor_tolerance = 0.02
    planner.core_pareto_use_ttc_tradeoff_penalty = True
    planner.core_pareto_tradeoff_tolerance = 0.01
    planner.core_pareto_tradeoff_penalty_weight = 0.2
    planner.core_pareto_use_ttc_floor_penalty = False
    planner.core_pareto_ttc_floor_tolerance = 0.03
    planner.core_pareto_ttc_floor_penalty_weight = 0.1
    planner.core_pareto_score_mode = "pdms_plus_core_margin_minus_slow"
    planner.core_pareto_core_margin_weight = 0.3
    planner.core_pareto_pareto_objectives = (
        "ego_progress",
        "time_to_collision_within_bound",
        "history_comfort",
    )
    planner.core_pareto_use_pareto_front = True
    planner.core_pareto_pareto_front_bonus = 0.2
    planner.core_pareto_min_group_reward_std = 0.005
    planner.core_pareto_use_phenotype_bucket_grpo = False
    planner.core_pareto_all_valid_objective = "core"
    planner.core_pareto_use_reference_margin = True
    planner.core_pareto_reference_margin_scale = 0.05
    planner.core_pareto_reference_margin_clip = 1.0
    planner.core_pareto_reference_margin_weight = 0.3
    planner.core_pareto_use_adaptive_dual = False
    planner.core_pareto_lambda_slow = 0.5
    planner.core_pareto_lambda_safety = 1.0
    planner.core_pareto_slow_rate_ema = 0.0
    planner.core_pareto_unsafe_rate_ema = 0.0
    planner.core_pareto_ddc_drop_rate_ema = 0.0
    planner.core_pareto_all_unsafe_base_offset = 0.7
    planner.core_pareto_all_unsafe_rescue_weight = 0.4
    planner.core_pareto_all_unsafe_adv_min = -1.5
    planner.core_pareto_all_unsafe_adv_max = 0.0
    planner.core_pareto_unsafe_advantage_offset = 1.0
    planner.core_pareto_slow_invalid_advantage = -0.3
    planner.core_pareto_use_all_unsafe_rescue_advantage = True
    planner.core_pareto_dominated_positive_adv_cap = 0.0
    planner.core_pareto_positive_slow_fail_cap = 0.0
    planner.core_pareto_advantage_clip_abs = 5.0
    planner.all_unsafe_group_weight = 0.25
    planner.core_pareto_all_slow_group_weight = 0.25
    planner.core_pareto_all_safe_low_std_group_weight = 0.5
    planner.grpo_use_support_relative = False

    def fail_if_called(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise AssertionError("support-relative branch must not run when disabled")

    planner._compute_support_relative_pareto_advantages = types.MethodType(fail_if_called, planner)
    return planner


def _components() -> dict[str, torch.Tensor]:
    return {
        "pdms": torch.tensor([[0.72, 0.86, 0.91], [0.51, 0.55, 0.62]]),
        "ego_progress": torch.tensor([[0.75, 0.84, 0.88], [0.60, 0.58, 0.64]]),
        "time_to_collision_within_bound": torch.tensor([[0.95, 0.95, 0.93], [0.90, 0.96, 0.97]]),
        "history_comfort": torch.tensor([[0.80, 0.86, 0.90], [0.70, 0.72, 0.76]]),
        "no_at_fault_collisions": torch.ones(2, 3),
        "drivable_area_compliance": torch.ones(2, 3),
        "driving_direction_compliance": torch.full((2, 3), 0.98),
    }


def _reference() -> dict[str, torch.Tensor]:
    ref_pdms = torch.tensor([0.74, 0.50])
    ref_core = torch.tensor([0.82, 0.68])
    return {
        "ref_pdms": ref_pdms,
        "ref_core": ref_core,
        "ref_ep": torch.tensor([0.78, 0.58]),
        "ref_ttc": torch.tensor([0.95, 0.92]),
        "ref_ddc": torch.tensor([0.98, 0.98]),
        "gt_pdms": ref_pdms,
        "il_pdms": ref_pdms,
        "gt_core": ref_core,
        "il_core": ref_core,
    }


def test_core_pareto_v2_disabled_support_relative_ignores_tokens() -> None:
    planner = _planner()
    rewards = _components()["pdms"]
    components = _components()
    trajectories = torch.zeros(2, 3, 8, 3)
    ref = _reference()

    adv_without_tokens, weight_without_tokens, aux_without_tokens = planner._compute_core_pareto_advantages(
        rewards,
        components,
        trajectories,
        ref,
        tokens_list=None,
    )
    adv_with_tokens, weight_with_tokens, aux_with_tokens = planner._compute_core_pareto_advantages(
        rewards,
        components,
        trajectories,
        ref,
        tokens_list=["tok-a", "tok-b"],
    )

    assert torch.allclose(adv_with_tokens, adv_without_tokens)
    assert torch.allclose(weight_with_tokens, weight_without_tokens)
    assert "support_relative_enabled" not in aux_with_tokens
    assert torch.isfinite(adv_with_tokens).all()
    assert torch.isfinite(weight_with_tokens).all()
    assert torch.equal(aux_with_tokens["core_pareto_valid_mask"], aux_without_tokens["core_pareto_valid_mask"])
