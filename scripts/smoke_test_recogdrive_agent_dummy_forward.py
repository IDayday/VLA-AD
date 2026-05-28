#!/usr/bin/env python3
"""Synthetic ReCogDriveAgent.forward smoke test.

This script verifies the cached-hidden-state ReCogDriveAgent -> action_head data
flow without NAVSIM data, image files, AgentInput, feature builders, metric
caches, or real JEPA/VGGT models. It is a computation-flow test only and does
not measure or imply driving performance.

Run from the repository root:

    python scripts/smoke_test_recogdrive_agent_dummy_forward.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
import sys

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_agent_dependency_stubs() -> None:
    """Install lightweight stubs for imports not needed by this smoke test."""
    from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

    _install_dependency_stubs()

    if "pytorch_lightning" not in sys.modules:
        pl_mod = ModuleType("pytorch_lightning")

        class Callback:
            pass

        pl_mod.Callback = Callback
        sys.modules["pytorch_lightning"] = pl_mod

    if "omegaconf" not in sys.modules:
        omegaconf_mod = ModuleType("omegaconf")

        class DictConfig(dict):
            pass

        class OmegaConf:
            @staticmethod
            def set_struct(*args, **kwargs):
                return None

        omegaconf_mod.DictConfig = DictConfig
        omegaconf_mod.OmegaConf = OmegaConf
        sys.modules["omegaconf"] = omegaconf_mod

    dataclasses_mod = sys.modules["navsim.common.dataclasses"]

    @dataclass
    class AgentInput:
        pass

    class SensorConfig:
        @classmethod
        def build_all_sensors(cls, include=True):
            return cls()

        @classmethod
        def build_no_sensors(cls):
            return cls()

    @dataclass
    class Trajectory:
        poses: object

    dataclasses_mod.AgentInput = AgentInput
    dataclasses_mod.SensorConfig = SensorConfig
    dataclasses_mod.Trajectory = Trajectory

    backbone_mod = ModuleType("navsim.agents.recogdrive.recogdrive_backbone")

    class RecogDriveBackbone(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise RuntimeError("RecogDriveBackbone should not be constructed in cached-hidden-state smoke tests.")

    backbone_mod.RecogDriveBackbone = RecogDriveBackbone
    sys.modules["navsim.agents.recogdrive.recogdrive_backbone"] = backbone_mod

    preprocess_mod = ModuleType("navsim.agents.recogdrive.utils.internvl_preprocess")

    def load_image(*args, **kwargs):
        raise RuntimeError("load_image should not be called in cached-hidden-state smoke tests.")

    preprocess_mod.load_image = load_image
    sys.modules["navsim.agents.recogdrive.utils.internvl_preprocess"] = preprocess_mod


_install_agent_dependency_stubs()

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # noqa: E402

from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent  # noqa: E402


BATCH_SIZE = 2
NUM_VLM_TOKENS = 32
VLM_HIDDEN_DIM = 1536
ACTION_HORIZON = 8
ACTION_DIM = 3
NUM_JEPA_TOKENS = 12
NUM_VGGT_TOKENS = 12
JEPA_DIM = 1024
VGGT_DIM = 2048


def deterministic_tensor(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    scale: float,
    offset: float,
) -> torch.Tensor:
    total = 1
    for dim in shape:
        total *= dim
    values = torch.arange(total, device=device, dtype=torch.float32)
    values = torch.sin(values * 0.017 + offset) * scale
    return values.reshape(shape)


def make_history_trajectory(device: torch.device) -> torch.Tensor:
    steps = torch.linspace(-1.5, 0.0, 4, device=device).view(1, 4)
    batch_offsets = torch.arange(BATCH_SIZE, device=device, dtype=torch.float32).view(BATCH_SIZE, 1)
    command_sign = torch.tensor([-1.0, 1.0], device=device).view(BATCH_SIZE, 1)

    x = 2.0 + steps + 0.2 * batch_offsets
    y = command_sign * 0.2 * (steps + 1.5)
    heading = command_sign * 0.04 * (steps + 1.5)
    return torch.stack([x, y, heading], dim=-1)


def make_future_trajectory(device: torch.device) -> torch.Tensor:
    steps = torch.linspace(0.0, 1.0, ACTION_HORIZON, device=device).view(1, ACTION_HORIZON)
    batch_offsets = torch.arange(BATCH_SIZE, device=device, dtype=torch.float32).view(BATCH_SIZE, 1)
    command_sign = torch.tensor([-1.0, 1.0], device=device).view(BATCH_SIZE, 1)

    x = 1.0 + 17.0 * steps + 0.3 * batch_offsets
    y = command_sign * (0.25 + 1.25 * steps**2)
    heading = command_sign * (0.03 + 0.23 * steps)

    trajectory = torch.stack([x, y, heading], dim=-1)
    trajectory[..., 0].clamp_(0.0, 20.0)
    trajectory[..., 1].clamp_(-3.0, 3.0)
    trajectory[..., 2].clamp_(-0.5, 0.5)
    return trajectory


def make_features(*, device: torch.device, use_expert_features: bool) -> dict[str, torch.Tensor]:
    high_command_one_hot = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )
    velocity = torch.tensor([[3.5, 0.0], [4.0, 0.2]], device=device, dtype=torch.float32)
    acceleration = torch.tensor([[0.1, 0.0, 0.0], [0.0, -0.1, 0.0]], device=device, dtype=torch.float32)

    features = {
        "history_trajectory": make_history_trajectory(device),
        "high_command_one_hot": high_command_one_hot,
        "status_feature": torch.cat([high_command_one_hot, velocity, acceleration], dim=-1),
        "last_hidden_state": deterministic_tensor(
            (BATCH_SIZE, NUM_VLM_TOKENS, VLM_HIDDEN_DIM),
            device=device,
            scale=0.5,
            offset=0.1,
        ),
    }

    if use_expert_features:
        jepa_tokens = deterministic_tensor(
            (BATCH_SIZE, NUM_JEPA_TOKENS, JEPA_DIM),
            device=device,
            scale=0.2,
            offset=0.3,
        )
        vggt_tokens = deterministic_tensor(
            (BATCH_SIZE, NUM_VGGT_TOKENS, VGGT_DIM),
            device=device,
            scale=0.15,
            offset=0.7,
        )
        features.update({
            "jepa_context_tokens": jepa_tokens,
            "vggt_context_tokens": vggt_tokens,
        })

    return features


def make_targets(device: torch.device) -> dict[str, torch.Tensor]:
    return {"trajectory": make_future_trajectory(device)}


def make_agent(*, use_expert_features: bool) -> ReCogDriveAgent:
    return ReCogDriveAgent(
        trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
        cache_hidden_state=True,
        cache_mode=False,
        vlm_path=None,
        checkpoint_path=None,
        dit_type="small",
        vlm_size="small",
        grpo=False,
        use_expert_features=use_expert_features,
        expert_feature_source="dummy" if use_expert_features else "none",
        use_jepa=True,
        use_vggt=True,
        jepa_dim=JEPA_DIM if use_expert_features else 0,
        vggt_dim=VGGT_DIM if use_expert_features else 0,
        expert_alignment_weight=0.0,
        jepa_alignment_weight=0.0,
        vggt_alignment_weight=0.0,
    )


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains non-finite values.")


def assert_has_nonzero_gradient(agent: ReCogDriveAgent) -> None:
    for parameter in agent.action_head.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        if torch.isfinite(grad).all() and grad.abs().sum().item() > 0.0:
            return
    raise AssertionError("No action_head parameter had a finite nonzero gradient.")


def run_case(*, label: str, use_expert_features: bool) -> None:
    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)

    agent = make_agent(use_expert_features=use_expert_features)
    device = next(agent.action_head.parameters()).device

    train_features = make_features(device=device, use_expert_features=False)
    targets = make_targets(device)

    agent.train()
    agent.zero_grad(set_to_none=True)
    predictions = agent.forward(train_features, targets)
    loss = agent.compute_loss(train_features, targets, predictions)
    assert_finite(f"{label} training loss", loss)
    loss.backward()
    assert_has_nonzero_gradient(agent)
    print(f"{label} train forward/backward OK")

    eval_features = make_features(device=device, use_expert_features=False)
    agent.eval()
    with torch.no_grad():
        predictions = agent.forward(eval_features)

    if "pred_traj" not in predictions:
        raise AssertionError(f"{label} inference output does not contain 'pred_traj'.")
    pred_traj = predictions["pred_traj"]
    expected_shape = (BATCH_SIZE, ACTION_HORIZON, ACTION_DIM)
    if pred_traj.shape != expected_shape:
        raise AssertionError(f"{label} pred_traj shape {tuple(pred_traj.shape)} != {expected_shape}.")
    assert_finite(f"{label} pred_traj", pred_traj)
    print(f"{label} inference OK")


def main() -> None:
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running ReCogDriveAgent dummy forward smoke on {device_name}.")
    print("This is a computation-flow smoke test only; it does not evaluate driving performance.")

    run_case(label="baseline", use_expert_features=False)
    run_case(label="expert", use_expert_features=True)

    print("ReCogDriveAgent dummy forward smoke test passed.")


if __name__ == "__main__":
    main()
