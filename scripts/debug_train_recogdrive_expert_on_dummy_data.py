#!/usr/bin/env python3
"""Tiny synthetic optimization loop for ReCogDrive expert-token plumbing.

This script uses no NAVSIM data, images, AgentInput, metric cache, or real
JEPA/VGGT models. It directly trains ReCogDriveDiffusionPlanner for a few steps
on deterministic synthetic data to validate computation flow only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs  # noqa: E402

_install_dependency_stubs()

from transformers.feature_extraction_utils import BatchFeature  # noqa: E402

from navsim.agents.recogdrive.recogdrive_diffusion_planner import (  # noqa: E402
    DDIMConfig,
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)


WARNING = (
    "This uses synthetic data for computation-flow validation only. It is not a training result "
    "and must not be used for performance claims."
)

NUM_VLM_TOKENS = 32
VLM_HIDDEN_DIM = 1536
ACTION_HORIZON = 8
ACTION_DIM = 3
HISTORY_HORIZON = 4
STATUS_DIM = 8
NUM_JEPA_TOKENS = 4
NUM_VGGT_TOKENS = 4
JEPA_DIM = 768
VGGT_DIM = 2048


class SyntheticReCogDriveDataset(Dataset):
    """Deterministic synthetic cached-hidden-state ReCogDrive samples."""

    def __init__(
        self,
        *,
        num_samples: int,
        use_expert_features: bool,
        include_alignment_targets: bool,
        seed: int,
    ) -> None:
        self.num_samples = num_samples
        self.use_expert_features = use_expert_features
        self.include_alignment_targets = include_alignment_targets
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    @staticmethod
    def _deterministic_tensor(shape: Tuple[int, ...], *, idx: int, seed: int, scale: float, offset: float) -> torch.Tensor:
        total = 1
        for dim in shape:
            total *= dim
        values = torch.arange(total, dtype=torch.float32).reshape(shape)
        return torch.sin(values * 0.017 + idx * 0.131 + seed * 0.019 + offset) * scale

    def _history_trajectory(self, idx: int) -> torch.Tensor:
        command_sign = -1.0 if idx % 2 == 0 else 1.0
        steps = torch.linspace(-1.5, 0.0, HISTORY_HORIZON)
        x = 2.0 + steps + 0.03 * idx
        y = command_sign * 0.2 * (steps + 1.5)
        heading = command_sign * 0.04 * (steps + 1.5)
        return torch.stack([x, y, heading], dim=-1)

    def _future_trajectory(self, idx: int) -> torch.Tensor:
        command_sign = -1.0 if idx % 2 == 0 else 1.0
        steps = torch.linspace(0.0, 1.0, ACTION_HORIZON)
        x = 1.0 + 17.0 * steps + 0.05 * idx
        y = command_sign * (0.25 + 1.25 * steps**2)
        heading = command_sign * (0.03 + 0.23 * steps)
        trajectory = torch.stack([x, y, heading], dim=-1)
        trajectory[..., 0].clamp_(0.0, 20.0)
        trajectory[..., 1].clamp_(-3.0, 3.0)
        trajectory[..., 2].clamp_(-0.5, 0.5)
        return trajectory

    def _status_feature(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        command_index = idx % 3
        high_command_one_hot = torch.zeros(3, dtype=torch.float32)
        high_command_one_hot[command_index] = 1.0
        velocity = torch.tensor([3.5 + 0.05 * idx, 0.1 * (command_index - 1)], dtype=torch.float32)
        acceleration = torch.tensor([0.03 * (idx % 2), -0.02 * command_index, 0.0], dtype=torch.float32)
        status_feature = torch.cat([high_command_one_hot, velocity, acceleration], dim=0)
        assert status_feature.shape == (STATUS_DIM,)
        return high_command_one_hot, status_feature

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        high_command_one_hot, status_feature = self._status_feature(idx)
        features: Dict[str, torch.Tensor] = {
            "history_trajectory": self._history_trajectory(idx),
            "high_command_one_hot": high_command_one_hot,
            "status_feature": status_feature,
            "last_hidden_state": self._deterministic_tensor(
                (NUM_VLM_TOKENS, VLM_HIDDEN_DIM),
                idx=idx,
                seed=self.seed,
                scale=0.5,
                offset=0.1,
            ),
        }
        targets = {"trajectory": self._future_trajectory(idx)}

        if self.use_expert_features:
            jepa_tokens = self._deterministic_tensor(
                (NUM_JEPA_TOKENS, JEPA_DIM),
                idx=idx,
                seed=self.seed,
                scale=0.2,
                offset=0.3,
            )
            vggt_tokens = self._deterministic_tensor(
                (NUM_VGGT_TOKENS, VGGT_DIM),
                idx=idx,
                seed=self.seed,
                scale=0.15,
                offset=0.7,
            )
            features["jepa_context_tokens"] = jepa_tokens
            features["vggt_context_tokens"] = vggt_tokens

            if self.include_alignment_targets:
                features["jepa_target_tokens"] = jepa_tokens + self._deterministic_tensor(
                    (NUM_JEPA_TOKENS, JEPA_DIM),
                    idx=idx,
                    seed=self.seed,
                    scale=0.02,
                    offset=1.3,
                )
                features["vggt_target_tokens"] = vggt_tokens + self._deterministic_tensor(
                    (NUM_VGGT_TOKENS, VGGT_DIM),
                    idx=idx,
                    seed=self.seed,
                    scale=0.02,
                    offset=1.7,
                )

        return features, targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="auto")
    parser.add_argument("--use-expert-features", action="store_true")
    parser.add_argument("--use-alignment-loss", action="store_true")
    parser.add_argument("--dit-type", choices=("small",), default="small")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return torch.device(device_arg)


def make_planner_config(*, use_expert_features: bool, use_alignment_loss: bool) -> ReCogDriveDiffusionPlannerConfig:
    return ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg={
            "num_heads": 8,
            "head_dim": 48,
            "num_layers": 16,
            "output_dim": 512,
            "dropout": 0.0,
            "attention_bias": True,
            "norm_eps": 1e-5,
            "interleave_attention": True,
        },
        input_embedding_dim=384,
        hidden_size=1024,
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        max_seq_len=ACTION_HORIZON,
        sampling_method="ddim",
        num_inference_steps=2,
        vlm_size="small",
        ddim_cfg=DDIMConfig(num_train_timesteps=10, ddim_eta=0.0),
        use_expert_features=use_expert_features,
        use_jepa=True,
        use_vggt=True,
        jepa_dim=JEPA_DIM if use_expert_features else 0,
        vggt_dim=VGGT_DIM if use_expert_features else 0,
        expert_dropout=0.0,
        expert_fusion_mode="concat_context",
        use_expert_type_embedding=True,
        use_expert_gates=True,
        expert_alignment_weight=0.0,
        jepa_alignment_weight=0.1 if use_alignment_loss else 0.0,
        vggt_alignment_weight=0.1 if use_alignment_loss else 0.0,
        alignment_loss_type="normalized_mse",
    )


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device=device, dtype=torch.float32) for key, value in batch.items()}


def build_action_input(features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> BatchFeature:
    history_trajectory = features["history_trajectory"]
    action_input = {
        "his_traj": history_trajectory.reshape(history_trajectory.shape[0], -1),
        "status_feature": features["status_feature"],
        "action": targets["trajectory"],
    }
    for key in ("jepa_context_tokens", "vggt_context_tokens", "jepa_target_tokens", "vggt_target_tokens"):
        if key in features:
            action_input[key] = features[key]
    return BatchFeature(data=action_input)


def tensor_value(output: BatchFeature, key: str, reference: torch.Tensor) -> torch.Tensor:
    value = output.get(key)
    if isinstance(value, torch.Tensor):
        return value.detach()
    return reference.detach().new_zeros(())


def expert_gate_summary(planner: ReCogDriveDiffusionPlanner) -> str:
    parts = []
    if hasattr(planner, "jepa_gate"):
        parts.append(f"jepa_gate={torch.sigmoid(planner.jepa_gate.detach()).item():.4f}")
    if hasattr(planner, "vggt_gate"):
        parts.append(f"vggt_gate={torch.sigmoid(planner.vggt_gate.detach()).item():.4f}")
    return " ".join(parts) if parts else "expert_gates=none"


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} is not finite: {tensor}")


def assert_no_nan_parameters(module: nn.Module) -> None:
    for name, parameter in module.named_parameters():
        detached = parameter.detach()
        if torch.isnan(detached).any():
            raise AssertionError(f"Parameter became NaN: {name}")
        if parameter.requires_grad and not torch.isfinite(detached).all():
            raise AssertionError(f"Trainable parameter became non-finite: {name}")


def main() -> None:
    args = parse_args()
    if args.use_alignment_loss and not args.use_expert_features:
        raise ValueError("--use-alignment-loss requires --use-expert-features.")

    print(WARNING)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    print(
        "debug config: "
        f"device={device} dit_type={args.dit_type} batch_size={args.batch_size} steps={args.steps} "
        f"use_expert_features={args.use_expert_features} use_alignment_loss={args.use_alignment_loss}"
    )

    dataset = SyntheticReCogDriveDataset(
        num_samples=args.batch_size * args.steps,
        use_expert_features=args.use_expert_features,
        include_alignment_targets=args.use_alignment_loss,
        seed=args.seed,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    planner = ReCogDriveDiffusionPlanner(
        make_planner_config(
            use_expert_features=args.use_expert_features,
            use_alignment_loss=args.use_alignment_loss,
        )
    ).to(device)
    planner.train()

    optimizer = torch.optim.AdamW(planner.parameters(), lr=1e-4, weight_decay=1e-4)

    for step, (features_cpu, targets_cpu) in enumerate(dataloader, start=1):
        if step > args.steps:
            break

        features = batch_to_device(features_cpu, device)
        targets = batch_to_device(targets_cpu, device)

        output = planner(features["last_hidden_state"], build_action_input(features, targets))
        loss = output["loss"]
        diffusion_loss = tensor_value(output, "diffusion_loss", loss)
        jepa_alignment_loss = tensor_value(output, "jepa_alignment_loss", loss)
        vggt_alignment_loss = tensor_value(output, "vggt_alignment_loss", loss)

        assert_finite("total loss", loss)
        assert_finite("diffusion loss", diffusion_loss)
        assert_finite("jepa alignment loss", jepa_alignment_loss)
        assert_finite("vggt alignment loss", vggt_alignment_loss)

        loss.backward()
        grad_norm = clip_grad_norm_(planner.parameters(), max_norm=1.0)
        assert_finite("grad norm", grad_norm)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        assert_no_nan_parameters(planner)

        print(
            f"step={step:03d} "
            f"loss={loss.detach().float().item():.6f} "
            f"diffusion_loss={diffusion_loss.float().item():.6f} "
            f"jepa_alignment_loss={jepa_alignment_loss.float().item():.6f} "
            f"vggt_alignment_loss={vggt_alignment_loss.float().item():.6f} "
            f"grad_norm={grad_norm.detach().float().item():.6f} "
            f"{expert_gate_summary(planner)}"
        )

    print("Synthetic debug optimization loop completed successfully.")


if __name__ == "__main__":
    main()
