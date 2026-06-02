from __future__ import annotations

import torch

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

_install_dependency_stubs()

from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.last_vla_cot_planning import LastVLACoTConfig, LastVLACoTTransformer


def _module(action_conditioned: bool) -> LastVLACoTTransformer:
    return LastVLACoTTransformer(
        LastVLACoTConfig(
            planner_dim=32,
            vlm_dim=32,
            jepa_dim=16,
            vggt_dim=24,
            hidden_dim=64,
            cot_num_tokens=6,
            vlm_summary_tokens=2,
            geometry_tokens=3,
            dynamic_tokens=3,
            ego_tokens=2,
            risk_tokens=2,
            use_cot_risk_head=False,
            use_action_conditioned_dynamics=action_conditioned,
            allow_patch_geometry_fallback=True,
            geometry_teacher_dim=24,
        )
    )


def test_action_conditioned_dynamics_changes_future_prediction():
    torch.manual_seed(104)
    model = _module(action_conditioned=True)
    model.eval()
    batch = 2
    vlm_tokens = torch.randn(batch, 5, 32)
    action_input = BatchFeature(
        data={
            "status_feature": torch.randn(batch, 8),
            "high_command_one_hot": torch.eye(3)[torch.tensor([0, 1])].float(),
            "history_trajectory": torch.randn(batch, 4, 3),
            "jepa_context_tokens": torch.randn(batch, 4, 16),
            "vggt_context_tokens": torch.randn(batch, 4, 24),
        }
    )
    noisy_a = torch.zeros(batch, 8, 3)
    noisy_b = torch.ones(batch, 8, 3) * 0.5
    timestep = torch.tensor([2, 2])
    with torch.no_grad():
        out_a = model(vlm_tokens, action_input, training=False, noisy_action_norm=noisy_a, diffusion_timestep=timestep)
        out_b = model(vlm_tokens, action_input, training=False, noisy_action_norm=noisy_b, diffusion_timestep=timestep)
    assert out_a.predicted_future_jepa.shape == (batch, 3, 16)
    assert not torch.allclose(out_a.predicted_future_jepa, out_b.predicted_future_jepa)


def test_unconditioned_dynamics_forward_is_finite():
    torch.manual_seed(105)
    model = _module(action_conditioned=False)
    model.eval()
    batch = 1
    action_input = BatchFeature(data={"status_feature": torch.randn(batch, 8)})
    with torch.no_grad():
        out = model(
            torch.randn(batch, 5, 32),
            action_input,
            training=False,
            noisy_action_norm=torch.randn(batch, 8, 3),
            diffusion_timestep=torch.tensor([1]),
        )
    assert torch.isfinite(out.predicted_future_jepa).all()
