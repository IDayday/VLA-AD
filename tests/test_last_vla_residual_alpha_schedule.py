from __future__ import annotations

import torch

from tests.test_last_vla_helpers import make_last_vla_planner


def test_last_vla_diffusion_target_ignores_coarse_prior_settings():
    planner = make_last_vla_planner()
    gt = torch.ones(2, 8, 3)

    planner.config.current_train_epoch = 0
    planner.config.last_vla_residual_alpha_start = 0.0
    planner.config.last_vla_residual_alpha_end = 1.0
    planner.config.last_vla_residual_alpha_warmup_epochs = 10
    target0, alpha0 = planner._last_vla_diffusion_target(gt, training=True)
    assert alpha0 == 0.0
    assert torch.allclose(target0, gt)

    planner.config.current_train_epoch = 10
    target1, alpha1 = planner._last_vla_diffusion_target(gt, training=True)
    assert alpha1 == 0.0
    assert torch.allclose(target1, gt)


def test_last_vla_diffusion_target_inference_matches_gt():
    planner = make_last_vla_planner()
    planner.config.current_train_epoch = 0
    planner.config.last_vla_residual_alpha_start = 0.0
    gt = torch.ones(2, 8, 3)
    target, alpha = planner._last_vla_diffusion_target(gt, training=False)
    assert alpha == 0.0
    assert torch.allclose(target, gt)
