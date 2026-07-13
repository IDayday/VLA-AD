from __future__ import annotations

import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

from tests.test_last_vla_helpers import make_last_vla_batch, make_last_vla_planner
from navsim.agents.recogdrive.last_vla_cot_planning import GEOMETRY_MODE_TO_CODE, GeometryTeacherHead, LastVLACoTConfig


def _head(*, allow_patch: bool, require_full: bool = False) -> GeometryTeacherHead:
    return GeometryTeacherHead(
        LastVLACoTConfig(
            planner_dim=384,
            hidden_dim=384,
            vggt_dim=2048,
            geometry_teacher_dim=2048,
            geometry_tokens=4,
            allow_patch_geometry_fallback=allow_patch,
            require_full_geometry=require_full,
        )
    )


def _cot(batch: int = 2) -> torch.Tensor:
    return torch.randn(batch, 4, 384)


def test_full_geometry_mode_allows_geometry_loss():
    head = _head(allow_patch=False, require_full=True)
    action_input = {
        "vggt_geometry_tokens": torch.randn(2, 4, 2048),
        "vggt_geometry_mode_code": torch.full((2,), GEOMETRY_MODE_TO_CODE["full_geometry"]),
    }
    _, target, loss, diagnostics = head(_cot(), action_input, training=True, allow_target_tokens=True)

    assert target is not None
    assert torch.isfinite(loss)
    assert diagnostics["geometry_mode_code"].item() == GEOMETRY_MODE_TO_CODE["full_geometry"]


def test_patch_fallback_mode_raises_when_disabled():
    head = _head(allow_patch=False)
    action_input = {
        "vggt_geometry_tokens": torch.randn(2, 4, 2048),
        "vggt_geometry_mode_code": torch.full((2,), GEOMETRY_MODE_TO_CODE["patch_fallback"]),
    }
    with pytest.raises(KeyError):
        head(_cot(), action_input, training=True, allow_target_tokens=True)


def test_patch_fallback_mode_allowed_when_explicitly_enabled():
    head = _head(allow_patch=True)
    action_input = {
        "vggt_geometry_tokens": torch.randn(2, 4, 2048),
        "vggt_geometry_mode_code": torch.full((2,), GEOMETRY_MODE_TO_CODE["patch_fallback"]),
    }
    _, target, loss, diagnostics = head(_cot(), action_input, training=True, allow_target_tokens=True)

    assert target is not None
    assert torch.isfinite(loss)
    assert diagnostics["geometry_mode_code"].item() == GEOMETRY_MODE_TO_CODE["patch_fallback"]


def test_no_geometry_with_positive_planner_geometry_loss_raises():
    planner = make_last_vla_planner()
    planner.config.last_vla_allow_patch_geometry_fallback = False
    planner.config.last_vla_geometry_loss_weight = 0.1
    planner.last_vla_cot.config.allow_patch_geometry_fallback = False
    planner.train()
    vl_features, action_input = make_last_vla_batch(include_targets=False)
    data = dict(action_input)
    data.pop("vggt_context_tokens", None)
    data["vggt_geometry_mode_code"] = torch.full((vl_features.shape[0],), GEOMETRY_MODE_TO_CODE["missing"])

    with pytest.raises(KeyError):
        planner(vl_features, BatchFeature(data=data))


def test_context_tokens_only_fallback_reports_patch_mode():
    head = _head(allow_patch=True)
    action_input = {"vggt_context_tokens": torch.randn(2, 4, 2048)}
    _, target, loss, diagnostics = head(_cot(), action_input, training=True, allow_target_tokens=True)

    assert target is not None
    assert torch.isfinite(loss)
    assert diagnostics["geometry_mode_code"].item() == GEOMETRY_MODE_TO_CODE["patch_fallback"]
