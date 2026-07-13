from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

_install_dependency_stubs()

from navsim.agents.recogdrive.fs_norm import FSNormStats, FSNormTransform
from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner


def _identity_transform() -> FSNormTransform:
    stats = FSNormStats(
        mean=torch.zeros(8, 3),
        std=torch.ones(8, 3),
        clip_lower=torch.full((8, 3), -4.0),
        clip_upper=torch.full((8, 3), 4.0),
        clip=5.0,
        version=2,
        scene_balanced=True,
        heading_center_zero=True,
    )
    return FSNormTransform(stats)


def test_fs_target_encoding_never_implicitly_clips() -> None:
    planner = ReCogDriveDiffusionPlanner.__new__(ReCogDriveDiffusionPlanner)
    nn.Module.__init__(planner)
    planner.config = SimpleNamespace(fs_norm_target_clip=5.0, fs_norm_clip=5.0)
    planner.fs_norm_transform = _identity_transform()
    raw = torch.zeros(1, 8, 3)
    raw[..., 0] = torch.arange(1, 9) * 6.0

    encoded = planner._encode_action_target(raw)

    assert encoded[..., 0].max() > 5.0
    assert torch.allclose(encoded[..., 0], torch.full((1, 8), 6.0))


def test_fs_clip_api_is_explicit() -> None:
    transform = _identity_transform()
    delta = torch.full((1, 8, 3), 7.0)
    unclipped = transform.normalize_delta(delta, apply_clip=False)
    scalar_clipped = transform.normalize_delta(delta, apply_clip=True, clip_value=5.0)
    stats_clipped = transform.normalize_delta(delta, apply_clip=True, use_stats_bounds=True)

    assert unclipped.max() == 7.0
    assert scalar_clipped.max() == 5.0
    assert stats_clipped.max() == 4.0


def test_planner_stats_bound_helper_uses_per_step_bounds() -> None:
    planner = ReCogDriveDiffusionPlanner.__new__(ReCogDriveDiffusionPlanner)
    nn.Module.__init__(planner)
    planner.config = SimpleNamespace(
        fs_norm_output_clip_mode="stats_bounds",
        fs_norm_output_clip=12.0,
        fs_norm_clip=5.0,
    )
    planner.fs_norm_transform = _identity_transform()
    representation = torch.full((1, 8, 3), 9.0)
    bounded = planner._bound_output_representation(representation, 1.0)
    assert torch.all(bounded == 4.0)
