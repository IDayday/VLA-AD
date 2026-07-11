from types import SimpleNamespace

import pytest

from navsim.agents.recogdrive.stage3_lfp_grpo import validate_lfp_config_exclusivity


def _grpo(**overrides):
    values = {
        "use_trajectory_level_objective": True,
        "trajectory_logprob_reduce": "discounted_mean",
        "use_gspo_ratio": False,
        "use_core_pareto_grpo": False,
        "use_feasible_pareto_grpo": False,
        "fp_use_pdas": False,
        "core_pareto_use_phenotype_bucket_grpo": False,
        "core_pareto_use_adaptive_dual": False,
        "core_pareto_buffer_bonus_enabled": False,
        "use_dynamic_group_weight": False,
        "use_diversity_reward": False,
        "bc_coeff_start": 0.0,
        "bc_coeff_end": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _offline(**overrides):
    values = {
        "grpo_buffer_guidance_enabled": False,
        "grpo_buffer_reward_bonus_weight": 0.0,
        "grpo_buffer_distill_loss_weight": 0.0,
        "grpo_buffer_distill_loss_weight_start": 0.0,
        "grpo_buffer_distill_loss_weight_end": 0.0,
        "grpo_buffer_preference_dpo_loss_weight": 0.0,
        "grpo_buffer_preference_dpo_loss_weight_start": 0.0,
        "grpo_self_imitation_loss_weight": 0.0,
        "grpo_self_imitation_loss_weight_start": 0.0,
        "grpo_self_imitation_loss_weight_end": 0.0,
        "bc_loss_weight": 0.0,
        "bc_loss_weight_start": 0.0,
        "bc_loss_weight_end": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("scope", "field", "value"),
    [
        ("grpo", "use_gspo_ratio", True),
        ("grpo", "use_feasible_pareto_grpo", True),
        ("grpo", "fp_use_pdas", True),
        ("grpo", "core_pareto_use_adaptive_dual", True),
        ("grpo", "bc_coeff_start", 0.1),
        ("offline", "grpo_buffer_guidance_enabled", True),
        ("offline", "grpo_buffer_distill_loss_weight", 0.1),
        ("offline", "grpo_buffer_preference_dpo_loss_weight", 0.1),
        ("offline", "grpo_self_imitation_loss_weight", 0.1),
        ("offline", "bc_loss_weight", 0.1),
    ],
)
def test_lfp_conflicts_fail_fast(scope: str, field: str, value) -> None:
    grpo = _grpo(**({field: value} if scope == "grpo" else {}))
    offline = _offline(**({field: value} if scope == "offline" else {}))
    with pytest.raises(ValueError, match=field):
        validate_lfp_config_exclusivity("lfp_grpo", grpo, offline)


def test_lfp_replay_conflicts_and_old_algorithm_is_unchanged() -> None:
    with pytest.raises(ValueError, match="grpo_replay"):
        validate_lfp_config_exclusivity(
            "lfp_grpo",
            _grpo(),
            _offline(),
            stage3_objective="grpo_replay",
        )
    # The validator is a no-op for every legacy algorithm.
    validate_lfp_config_exclusivity("none", _grpo(use_gspo_ratio=True), _offline(bc_loss_weight=1.0))
