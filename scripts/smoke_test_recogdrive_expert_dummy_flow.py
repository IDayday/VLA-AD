#!/usr/bin/env python3
"""Synthetic ReCogDriveDiffusionPlanner smoke test.

This script does not require NAVSIM data, image files, metric caches, AgentInput,
or real JEPA/VGGT models. It directly instantiates the diffusion planner with
deterministic fake tensors to verify baseline and expert-token forward,
backward, get_action, dtype/device conversion, and no-future-leakage behavior.

Run from the repository root:

    python scripts/smoke_test_recogdrive_expert_dummy_flow.py
    python scripts/smoke_test_recogdrive_expert_dummy_flow.py --test-large
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the lightweight dependency stubs used by the existing planner smoke
# script so this can run before the full NAVSIM dependency stack is installed.
from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs  # noqa: E402

_install_dependency_stubs()

from transformers.feature_extraction_utils import BatchFeature  # noqa: E402

from navsim.agents.recogdrive.recogdrive_diffusion_planner import (  # noqa: E402
    DDIMConfig,
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)


BATCH_SIZE = 2
NUM_VLM_TOKENS = 32
ACTION_HORIZON = 8
ACTION_DIM = 3
HISTORY_DIM = 12
STATUS_DIM = 8
NUM_JEPA_TOKENS = 12
NUM_VGGT_TOKENS = 12
JEPA_DIM = 1024
VGGT_DIM = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-large",
        action="store_true",
        help="Also run the larger planner dimensions: VLM hidden 3584, planner dim 1536.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Execution device. CUDA uses fp16/bf16; CPU always uses fp32.",
    )
    parser.add_argument(
        "--use-expert-features",
        action="store_true",
        help="Accepted for the documented command; the suite always runs both baseline and expert paths.",
    )
    parser.add_argument(
        "--expert-adapter-dim",
        type=int,
        default=768,
        help="Internal expert adapter dimension; default matches ReCogDrive-2B expert-token setup.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return torch.device(device_arg)


def resolve_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def deterministic_tensor(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    scale: float = 1.0,
    offset: float = 0.0,
) -> torch.Tensor:
    total = 1
    for dim in shape:
        total *= dim
    values = torch.arange(total, device=device, dtype=torch.float32)
    values = torch.sin(values * 0.013 + offset) * scale
    return values.reshape(shape).to(dtype=dtype)


def make_plausible_trajectory(
    *,
    batch_size: int,
    horizon: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    steps = torch.linspace(0.0, 1.0, horizon, device=device, dtype=torch.float32).view(1, horizon)
    batch_offsets = torch.arange(batch_size, device=device, dtype=torch.float32).view(batch_size, 1)

    x = 1.0 + 17.5 * steps + 0.25 * batch_offsets
    y_command = torch.tensor([-1.0, 1.0], device=device, dtype=torch.float32).view(batch_size, 1)
    y = y_command * (0.35 + 1.15 * steps**2)
    heading = y_command * (0.04 + 0.24 * steps)

    trajectory = torch.stack([x, y, heading], dim=-1)
    trajectory[..., 0].clamp_(0.0, 20.0)
    trajectory[..., 1].clamp_(-3.0, 3.0)
    trajectory[..., 2].clamp_(-0.5, 0.5)
    return trajectory.to(dtype=dtype)


def make_history(
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    history_horizon = HISTORY_DIM // ACTION_DIM
    steps = torch.linspace(-1.5, 0.0, history_horizon, device=device, dtype=torch.float32).view(1, history_horizon)
    batch_offsets = torch.arange(batch_size, device=device, dtype=torch.float32).view(batch_size, 1)
    y_command = torch.tensor([-1.0, 1.0], device=device, dtype=torch.float32).view(batch_size, 1)

    x = 2.0 + steps + 0.15 * batch_offsets
    y = y_command * 0.2 * (steps + 1.5)
    heading = y_command * 0.03 * (steps + 1.5)
    return torch.stack([x, y, heading], dim=-1).reshape(batch_size, HISTORY_DIM).to(dtype=dtype)


def make_status(
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    status = torch.zeros(batch_size, STATUS_DIM, device=device, dtype=torch.float32)
    status[:, 0] = torch.tensor([3.5, 4.0], device=device)
    status[:, 1] = torch.tensor([0.0, 0.2], device=device)
    status[:, 2] = torch.tensor([0.1, 0.0], device=device)
    status[:, 3] = torch.tensor([0.0, -0.1], device=device)
    status[:, 4:] = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        device=device,
        dtype=torch.float32,
    )
    return status.to(dtype=dtype)


def make_vl_features(
    *,
    batch_size: int,
    num_tokens: int,
    hidden_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return deterministic_tensor(
        (batch_size, num_tokens, hidden_dim),
        device=device,
        dtype=dtype,
        scale=0.5,
        offset=0.1,
    )


def make_action_input(
    *,
    device: torch.device,
    dtype: torch.dtype,
    include_action: bool,
    include_experts: bool,
    include_targets: bool,
    nan_targets: bool = False,
) -> BatchFeature:
    data = {
        "his_traj": make_history(batch_size=BATCH_SIZE, device=device, dtype=dtype),
        "status_feature": make_status(batch_size=BATCH_SIZE, device=device, dtype=dtype),
    }

    if include_action:
        data["action"] = make_plausible_trajectory(
            batch_size=BATCH_SIZE,
            horizon=ACTION_HORIZON,
            device=device,
            dtype=dtype,
        )

    if include_experts:
        jepa_tokens = deterministic_tensor(
            (BATCH_SIZE, NUM_JEPA_TOKENS, JEPA_DIM),
            device=device,
            dtype=dtype,
            scale=0.25,
            offset=0.2,
        )
        vggt_tokens = deterministic_tensor(
            (BATCH_SIZE, NUM_VGGT_TOKENS, VGGT_DIM),
            device=device,
            dtype=dtype,
            scale=0.2,
            offset=0.4,
        )
        data["jepa_context_tokens"] = jepa_tokens
        data["vggt_context_tokens"] = vggt_tokens

        if include_targets:
            if nan_targets:
                data["jepa_target_tokens"] = torch.full_like(jepa_tokens, float("nan"))
                data["vggt_target_tokens"] = torch.full_like(vggt_tokens, float("nan"))
            else:
                data["jepa_target_tokens"] = jepa_tokens + deterministic_tensor(
                    jepa_tokens.shape,
                    device=device,
                    dtype=dtype,
                    scale=0.03,
                    offset=1.3,
                )
                data["vggt_target_tokens"] = vggt_tokens + deterministic_tensor(
                    vggt_tokens.shape,
                    device=device,
                    dtype=dtype,
                    scale=0.025,
                    offset=1.7,
                )

    return BatchFeature(data=data)


def make_config(
    *,
    vlm_hidden_dim: int,
    input_embedding_dim: int,
    use_expert_features: bool,
    dtype: torch.dtype,
) -> ReCogDriveDiffusionPlannerConfig:
    if input_embedding_dim == 384:
        num_heads, head_dim, output_dim, vlm_size = 8, 48, 512, "small"
    elif input_embedding_dim == 1536:
        num_heads, head_dim, output_dim, vlm_size = 32, 48, 1536, "large"
    else:
        raise ValueError(f"Unsupported input_embedding_dim={input_embedding_dim}")

    expected_vlm_dim = 3584 if vlm_size == "large" else 1536
    if vlm_hidden_dim != expected_vlm_dim:
        raise ValueError(f"{vlm_size} planner expects vlm_hidden_dim={expected_vlm_dim}.")

    dtype_name = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }[dtype]

    return ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg={
            "num_heads": num_heads,
            "head_dim": head_dim,
            "num_layers": 2,
            "output_dim": output_dim,
            "dropout": 0.0,
            "attention_bias": True,
            "norm_eps": 1e-5,
            "interleave_attention": True,
        },
        input_embedding_dim=input_embedding_dim,
        hidden_size=input_embedding_dim,
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        max_seq_len=ACTION_HORIZON,
        sampling_method="ddim",
        num_inference_steps=2,
        model_dtype=dtype_name,
        vlm_size=vlm_size,
        ddim_cfg=DDIMConfig(num_train_timesteps=10, ddim_eta=0.0),
        use_expert_features=use_expert_features,
        use_jepa=True,
        use_vggt=True,
        jepa_dim=JEPA_DIM if use_expert_features else 0,
        vggt_dim=VGGT_DIM if use_expert_features else 0,
        expert_adapter_dim=768,
        num_jepa_tokens=NUM_JEPA_TOKENS,
        num_vggt_tokens=NUM_VGGT_TOKENS,
        expert_dropout=0.0,
        expert_fusion_mode="concat_context",
        use_expert_type_embedding=True,
        use_expert_gates=True,
        expert_alignment_weight=0.0,
        jepa_alignment_weight=0.1 if use_expert_features else 0.0,
        vggt_alignment_weight=0.1 if use_expert_features else 0.0,
        alignment_loss_type="normalized_mse",
    )


def move_planner(
    planner: ReCogDriveDiffusionPlanner,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> ReCogDriveDiffusionPlanner:
    planner = planner.to(device=device)
    if dtype != torch.float32:
        planner = planner.to(dtype=dtype)
    return planner


def assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains non-finite values.")


def assert_finite_nonzero_gradient(planner: ReCogDriveDiffusionPlanner) -> None:
    for name, parameter in planner.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        if torch.isfinite(grad).all() and grad.abs().sum().item() > 0.0:
            return
    raise AssertionError("No trainable parameter had a finite nonzero gradient.")


def run_forward_backward(
    *,
    label: str,
    planner: ReCogDriveDiffusionPlanner,
    vl_features: torch.Tensor,
    action_input: BatchFeature,
    expect_alignment: bool,
) -> None:
    planner.train()
    planner.zero_grad(set_to_none=True)
    output = planner.forward(vl_features, action_input)
    if "loss" not in output:
        raise AssertionError(f"{label} forward output does not contain 'loss'.")
    assert_finite_tensor(f"{label} loss", output["loss"])
    if expect_alignment:
        for key in ("jepa_alignment_loss", "vggt_alignment_loss"):
            if key not in output:
                raise AssertionError(f"{label} output does not contain {key!r}.")
            assert_finite_tensor(f"{label} {key}", output[key])

    output["loss"].backward()
    assert_finite_nonzero_gradient(planner)


def run_get_action(
    *,
    label: str,
    planner: ReCogDriveDiffusionPlanner,
    vl_features: torch.Tensor,
    action_input: BatchFeature,
    expect_warning: bool = False,
) -> torch.Tensor:
    planner.eval()
    init_actions = torch.zeros(
        BATCH_SIZE,
        ACTION_HORIZON,
        ACTION_DIM,
        device=vl_features.device,
        dtype=vl_features.dtype,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.no_grad():
            output = planner.get_action(
                vl_features,
                action_input,
                init_actions=init_actions,
                deterministic=True,
            )

    if expect_warning and not any("train-only expert target keys" in str(w.message) for w in caught):
        raise AssertionError(f"{label} did not warn about train-only expert target keys.")
    if "pred_traj" not in output:
        raise AssertionError(f"{label} get_action output does not contain 'pred_traj'.")
    pred_traj = output["pred_traj"]
    if pred_traj.shape != (BATCH_SIZE, ACTION_HORIZON, ACTION_DIM):
        raise AssertionError(f"{label} pred_traj shape {tuple(pred_traj.shape)} != {(BATCH_SIZE, ACTION_HORIZON, ACTION_DIM)}")
    assert_finite_tensor(f"{label} pred_traj", pred_traj)
    return pred_traj


def run_suite(
    *,
    size_name: str,
    vlm_hidden_dim: int,
    input_embedding_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(1234)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1234)

    vl_features = make_vl_features(
        batch_size=BATCH_SIZE,
        num_tokens=NUM_VLM_TOKENS,
        hidden_dim=vlm_hidden_dim,
        device=device,
        dtype=dtype,
    )

    baseline = move_planner(
        ReCogDriveDiffusionPlanner(
            make_config(
                vlm_hidden_dim=vlm_hidden_dim,
                input_embedding_dim=input_embedding_dim,
                use_expert_features=False,
                dtype=dtype,
            )
        ),
        device=device,
        dtype=dtype,
    )
    baseline_input = make_action_input(
        device=device,
        dtype=dtype,
        include_action=True,
        include_experts=False,
        include_targets=False,
    )
    run_forward_backward(
        label=f"{size_name} baseline",
        planner=baseline,
        vl_features=vl_features,
        action_input=baseline_input,
        expect_alignment=False,
    )
    print(f"{size_name} baseline forward OK")
    baseline_inference_input = make_action_input(
        device=device,
        dtype=dtype,
        include_action=False,
        include_experts=False,
        include_targets=False,
    )
    run_get_action(
        label=f"{size_name} baseline",
        planner=baseline,
        vl_features=vl_features,
        action_input=baseline_inference_input,
    )
    print(f"{size_name} baseline get_action OK")

    expert = move_planner(
        ReCogDriveDiffusionPlanner(
            make_config(
                vlm_hidden_dim=vlm_hidden_dim,
                input_embedding_dim=input_embedding_dim,
                use_expert_features=True,
                dtype=dtype,
            )
        ),
        device=device,
        dtype=dtype,
    )
    expert_input = make_action_input(
        device=device,
        dtype=dtype,
        include_action=True,
        include_experts=True,
        include_targets=True,
    )
    run_forward_backward(
        label=f"{size_name} expert",
        planner=expert,
        vl_features=vl_features,
        action_input=expert_input,
        expect_alignment=True,
    )
    print(f"{size_name} expert forward OK")
    print(f"{size_name} backward OK")

    inference_input = make_action_input(
        device=device,
        dtype=dtype,
        include_action=False,
        include_experts=True,
        include_targets=False,
    )
    run_get_action(
        label=f"{size_name} expert",
        planner=expert,
        vl_features=vl_features,
        action_input=inference_input,
    )
    print(f"{size_name} expert get_action OK")

    no_leakage_input = make_action_input(
        device=device,
        dtype=dtype,
        include_action=False,
        include_experts=True,
        include_targets=True,
        nan_targets=True,
    )
    run_get_action(
        label=f"{size_name} no-future-leakage",
        planner=expert,
        vl_features=vl_features,
        action_input=no_leakage_input,
        expect_warning=True,
    )
    print(f"{size_name} no-future-leakage OK")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(device)
    print(f"Running synthetic ReCogDrive expert smoke on device={device}, dtype={dtype}")

    run_suite(
        size_name="small",
        vlm_hidden_dim=1536,
        input_embedding_dim=384,
        device=device,
        dtype=dtype,
    )

    if args.test_large:
        run_suite(
            size_name="large",
            vlm_hidden_dim=3584,
            input_embedding_dim=1536,
            device=device,
            dtype=dtype,
        )

    print("Synthetic ReCogDrive expert dummy flow smoke test passed.")


if __name__ == "__main__":
    main()
