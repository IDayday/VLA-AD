from __future__ import annotations

import pytest
import torch

from tests.test_last_vla_helpers import make_last_vla_batch, make_last_vla_planner


def _enable_unit_anchor_residual(planner) -> None:
    planner.config.last_vla_use_residual_diffusion = True
    planner.config.last_vla_residual_anchor_source = "vlm_text_traj"
    planner.config.last_vla_require_residual_anchor = True
    planner.config.last_vla_residual_alpha_start = 1.0
    planner.config.last_vla_residual_alpha_end = 1.0
    planner.config.last_vla_residual_alpha_warmup_epochs = 0


def test_last_vla_vlm_text_anchor_diffusion_target_is_gt_minus_fixed_anchor():
    planner = make_last_vla_planner(residual=True)
    _enable_unit_anchor_residual(planner)
    _, action_input = make_last_vla_batch(include_targets=True)
    gt_norm = planner.norm_odo(action_input["action"])
    anchor_norm = planner.norm_odo(action_input["vlm_text_trajectory"])

    target, alpha = planner._last_vla_diffusion_target(
        gt_norm,
        training=True,
        action_input=action_input,
    )

    assert alpha == 1.0
    assert torch.allclose(target, gt_norm - anchor_norm)


def test_last_vla_vlm_text_anchor_get_action_adds_residual_back_to_anchor():
    planner = make_last_vla_planner(residual=True)
    _enable_unit_anchor_residual(planner)
    planner.eval()
    vl_features, action_input = make_last_vla_batch(include_targets=False)
    anchor_norm = torch.full((vl_features.shape[0], 8, 3), 0.2)
    context = type(action_input)(
        data={
            key: value
            for key, value in action_input.items()
            if "target" not in key and key not in {"action", "vlm_text_trajectory"}
        }
    )
    context["vlm_text_trajectory_norm"] = anchor_norm
    context["vlm_text_parse_ok"] = torch.ones(vl_features.shape[0])

    def fake_p_mean_variance(x, *args, **kwargs):
        return torch.zeros_like(x), torch.full_like(x, -100.0), torch.zeros_like(x)

    planner.p_mean_variance = fake_p_mean_variance
    init_actions = torch.zeros(vl_features.shape[0], 8, 3)

    with torch.no_grad():
        pred = planner.get_action(vl_features, context, init_actions=init_actions, deterministic=True)

    expected = planner.denorm_odo(anchor_norm)
    assert torch.allclose(pred["pred_traj"], expected, atol=1e-5)
    assert torch.allclose(pred["pred_vlm_text_anchor_traj"], expected, atol=1e-5)
    assert torch.allclose(pred["pred_residual_norm"], torch.zeros_like(anchor_norm))


def test_last_vla_vlm_text_anchor_missing_anchor_fails_fast():
    planner = make_last_vla_planner(residual=True)
    _enable_unit_anchor_residual(planner)
    planner.eval()
    vl_features, action_input = make_last_vla_batch(include_targets=False)
    context = type(action_input)(
        data={
            key: value
            for key, value in action_input.items()
            if key not in {"action", "vlm_text_trajectory", "vlm_text_trajectory_norm", "vlm_text_parse_ok"}
            and "target" not in key
        }
    )

    with pytest.raises(KeyError, match="vlm_text_trajectory"):
        planner.get_action(
            vl_features,
            context,
            init_actions=torch.zeros(vl_features.shape[0], 8, 3),
            deterministic=True,
        )


def test_last_vla_vlm_text_anchor_sample_chain_runs_and_returns_final_trajectory():
    planner = make_last_vla_planner(residual=True)
    _enable_unit_anchor_residual(planner)
    planner.eval()
    vl_features, action_input = make_last_vla_batch(include_targets=False)
    anchor_norm = torch.full((vl_features.shape[0], 8, 3), -0.1)
    context = type(action_input)(
        data={
            key: value
            for key, value in action_input.items()
            if "target" not in key and key not in {"action", "vlm_text_trajectory"}
        }
    )
    context["vlm_text_trajectory_norm"] = anchor_norm
    context["vlm_text_parse_ok"] = torch.ones(vl_features.shape[0])

    def fake_p_mean_variance(x, *args, **kwargs):
        return torch.zeros_like(x), torch.full_like(x, -100.0), torch.zeros_like(x)

    planner.p_mean_variance = fake_p_mean_variance
    init_actions = torch.zeros(vl_features.shape[0], 8, 3)

    with torch.no_grad():
        chain, final_actions = planner.sample_chain(
            vl_features,
            context["his_traj"],
            context["status_feature"],
            init_actions=init_actions,
            deterministic=True,
            action_input=context,
        )

    assert chain.shape[0] == vl_features.shape[0]
    assert chain.shape[-2:] == (8, 3)
    assert final_actions.shape == (vl_features.shape[0], 8, 3)
    assert torch.allclose(final_actions, planner.denorm_odo(anchor_norm), atol=1e-5)
