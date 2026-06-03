from __future__ import annotations

import torch

from tests.test_last_vla_helpers import make_last_vla_planner


def test_residual_alpha_zero_target_matches_gt_and_one_matches_residual():
    planner = make_last_vla_planner()
    gt = torch.ones(2, 8, 3)
    coarse = torch.ones(2, 8, 3) * 0.25

    planner.config.current_train_epoch = 0
    planner.config.last_vla_residual_alpha_start = 0.0
    planner.config.last_vla_residual_alpha_end = 1.0
    planner.config.last_vla_residual_alpha_warmup_epochs = 10
    target0, alpha0 = planner._last_vla_residual_diffusion_target(gt, coarse, training=True)
    assert alpha0 == 0.0
    assert torch.allclose(target0, gt)

    planner.config.current_train_epoch = 10
    target1, alpha1 = planner._last_vla_residual_diffusion_target(gt, coarse, training=True)
    assert alpha1 == 1.0
    assert torch.allclose(target1, gt - coarse)


def test_residual_alpha_inference_is_one():
    planner = make_last_vla_planner()
    planner.config.current_train_epoch = 0
    planner.config.last_vla_residual_alpha_start = 0.0
    assert planner._last_vla_residual_alpha(training=False) == 1.0
