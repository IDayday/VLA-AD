#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
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


def build_config(use_risk_vla: bool) -> ReCogDriveDiffusionPlannerConfig:
    return ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg={
            "num_heads": 4,
            "head_dim": 96,
            "num_layers": 2,
            "output_dim": 384,
            "dropout": 0.0,
            "attention_bias": True,
            "norm_eps": 1e-5,
            "interleave_attention": True,
        },
        input_embedding_dim=384,
        planner_dim=384,
        hidden_size=384,
        action_dim=3,
        action_horizon=8,
        max_seq_len=8,
        sampling_method="ddim",
        num_inference_steps=2,
        vlm_size="small",
        ddim_cfg=DDIMConfig(num_train_timesteps=10),
        use_expert_features=False,
        use_last_rd=False,
        use_bit_drive=False,
        use_risk_vla=use_risk_vla,
        risk_vla_strategy_token_scale=0.25,
        risk_vla_horizon_residual_scale=0.25,
        risk_vla_risk_loss_weight=0.05,
        risk_vla_focal_loss_weight=0.0,
        risk_vla_strategy_entropy_weight=0.0,
        risk_vla_use_bit_summary=True,
    )


def make_batch(batch_size: int, device: torch.device) -> tuple[torch.Tensor, BatchFeature]:
    vl_features = torch.randn(batch_size, 5, 1536, device=device)
    data = {
        "his_traj": torch.randn(batch_size, 12, device=device),
        "history_trajectory": torch.randn(batch_size, 4, 3, device=device),
        "status_feature": torch.randn(batch_size, 8, device=device),
        "high_command_one_hot": torch.eye(3, device=device)[torch.tensor([0, 2], device=device)[:batch_size]],
        "action": torch.randn(batch_size, 8, 3, device=device),
        "risk_labels": torch.randint(0, 2, (batch_size, 8, 6), device=device).float(),
        "terminal_intent": torch.randn(batch_size, 3, device=device),
        "path_intent": torch.randn(batch_size, 8, 3, device=device),
    }
    return vl_features, BatchFeature(data=data)


def assert_finite(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{name} contains non-finite values.")


def run_case(use_risk_vla: bool, device: torch.device) -> None:
    planner = ReCogDriveDiffusionPlanner(build_config(use_risk_vla)).to(device)
    planner.train()
    vl_features, action_input = make_batch(2, device)
    context = planner._prepare_dit_context(vl_features, action_input, training=True)
    expected_tokens = vl_features.shape[1] + (8 if use_risk_vla else 0)
    if context["context_tokens"].shape[1] != expected_tokens:
        raise RuntimeError(f"context token count mismatch: {context['context_tokens'].shape[1]} != {expected_tokens}")
    if use_risk_vla and context["expert_step_condition"] is None:
        raise RuntimeError("RISK-VLA horizon residual was not added.")

    output = planner(vl_features, action_input)
    assert_finite(output["loss"], "loss")
    if use_risk_vla:
        for key in ("risk_vla_risk_loss", "risk_vla_prob_ttc", "risk_vla_weight_path_intent"):
            if key not in output:
                raise RuntimeError(f"missing {key}")
            assert_finite(output[key], key)
    elif "risk_vla_risk_loss" in output:
        raise RuntimeError("use_risk_vla=false unexpectedly produced RISK-VLA output keys.")

    planner.eval()
    eval_input = BatchFeature(
        data={key: value for key, value in action_input.items() if key not in {"action", "risk_labels"}}
    )
    with torch.no_grad():
        pred = planner.get_action(vl_features, eval_input, init_actions=torch.zeros(2, 8, 3, device=device), deterministic=True)
    if pred["pred_traj"].shape != (2, 8, 3):
        raise RuntimeError(f"pred_traj shape mismatch: {tuple(pred['pred_traj'].shape)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU smoke test for config-gated RISK-VLA planner forward.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    device = torch.device(args.device)
    torch.manual_seed(20260603)
    run_case(False, device)
    run_case(True, device)
    print("RISK-VLA planner smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
