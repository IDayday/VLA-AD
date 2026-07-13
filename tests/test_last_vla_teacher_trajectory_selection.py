from __future__ import annotations

import torch

from transformers.feature_extraction_utils import BatchFeature

from tests.test_last_vla_helpers import make_last_vla_planner


def test_teacher_if_better_selects_teacher_or_gt():
    planner = make_last_vla_planner(teacher_mode="teacher_if_better")
    gt = torch.zeros(2, 8, 3)
    teacher = torch.ones(2, 8, 3) * 0.25
    action_input = BatchFeature(
        data={
            "teacher_trajectory_norm": teacher,
            "teacher_score": torch.tensor([0.9, 0.1]),
            "gt_score": torch.tensor([0.2, 0.5]),
        }
    )
    selected, diagnostics = planner._select_last_vla_training_target(action_input, gt)
    assert torch.allclose(selected[0], teacher[0])
    assert torch.allclose(selected[1], gt[1])
    assert torch.allclose(diagnostics["teacher_traj_used_ratio"], torch.tensor(0.5))


def test_mix_mode_interpolates_by_epoch_progress():
    planner = make_last_vla_planner(teacher_mode="mix")
    planner.config.current_train_epoch = 5
    planner.config.total_train_epochs = 10
    planner.config.last_vla_teacher_traj_mix_start = 0.2
    planner.config.last_vla_teacher_traj_mix_end = 0.8
    gt = torch.zeros(1, 8, 3)
    teacher = torch.ones(1, 8, 3)
    selected, diagnostics = planner._select_last_vla_training_target(
        BatchFeature(data={"teacher_trajectory_norm": teacher}),
        gt,
    )
    assert torch.allclose(selected, torch.ones_like(selected) * 0.5)
    assert torch.allclose(diagnostics["teacher_traj_mix"], torch.tensor(0.5))
