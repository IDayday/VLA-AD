#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
import sys

import torch

if hasattr(torch, "compile"):
    torch.compile = lambda fn=None, *args, **kwargs: fn if fn is not None else (lambda f: f)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from transformers.feature_extraction_utils import BatchFeature  # noqa: E402
except ModuleNotFoundError:
    from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs  # noqa: E402

    _install_dependency_stubs()
    from transformers.feature_extraction_utils import BatchFeature  # noqa: E402

from navsim.agents.recogdrive.recogdrive_diffusion_planner import (  # noqa: E402
    DDIMConfig,
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)


def build_config() -> ReCogDriveDiffusionPlannerConfig:
    return ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg={
            "num_heads": 8,
            "head_dim": 48,
            "num_layers": 2,
            "output_dim": 512,
            "dropout": 0.0,
            "attention_bias": True,
            "norm_eps": 1e-5,
            "interleave_attention": True,
        },
        input_embedding_dim=384,
        hidden_size=512,
        action_dim=3,
        action_horizon=8,
        max_seq_len=8,
        sampling_method="ddim",
        num_inference_steps=2,
        vlm_size="small",
        ddim_cfg=DDIMConfig(num_train_timesteps=10),
        use_expert_features=False,
        use_last_rd=False,
        use_bit_drive=True,
        bit_use_terminal_head=True,
        bit_use_path_anchors=True,
        bit_use_reverse_decoder=True,
        bit_condition_mode="tokens_and_action",
        bit_terminal_loss_weight=0.05,
        bit_path_loss_weight=0.05,
        bit_end_consistency_loss_weight=0.03,
        bit_reverse_loss_weight=0.03,
        bit_cycle_loss_weight=0.0,
        bit_use_gt_condition_prob=0.5,
        bit_condition_noise_std=0.05,
        bit_condition_dropout=0.1,
        bit_use_risk_head=True,
        bit_use_risk_tokens=True,
        bit_risk_loss_weight=0.02,
        bit_risk_token_strength=0.05,
    )


def fake_batch(batch_size: int, device: torch.device) -> tuple[torch.Tensor, BatchFeature]:
    vl_features = torch.randn(batch_size, 5, 1536, device=device)
    data = {
        "his_traj": torch.randn(batch_size, 12, device=device),
        "status_feature": torch.randn(batch_size, 8, device=device),
        "action": torch.randn(batch_size, 8, 3, device=device),
        "bit_risk_labels": torch.randint(0, 2, (batch_size, 4), device=device).float(),
    }
    return vl_features, BatchFeature(data=data)


def assert_finite_tensor(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{name} contains non-finite values.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test BiT-Drive planner flow on dummy tensors.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    device = torch.device(args.device)
    torch.manual_seed(20260601)

    planner = ReCogDriveDiffusionPlanner(build_config()).to(device)
    planner.train()
    vl_features, action_input = fake_batch(2, device)
    output = planner(vl_features, action_input)
    for key in (
        "loss",
        "diffusion_loss",
        "bit_terminal_loss",
        "bit_path_loss",
        "bit_end_consistency_loss",
        "bit_reverse_loss",
        "bit_risk_loss",
    ):
        assert_finite_tensor(output[key], key)
    output["loss"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(planner.parameters(), 10.0)
    assert_finite_tensor(torch.as_tensor(grad_norm), "grad_norm")

    planner.eval()
    eval_input = BatchFeature(data={
        "his_traj": action_input["his_traj"],
        "status_feature": action_input["status_feature"],
        "action": torch.full_like(action_input["action"], float("nan")),
    })
    with torch.no_grad():
        pred = planner.get_action(vl_features, eval_input, deterministic=True)
    if tuple(pred["pred_traj"].shape) != (2, 8, 3):
        raise RuntimeError(f"Unexpected pred_traj shape: {tuple(pred['pred_traj'].shape)}")
    assert_finite_tensor(pred["pred_traj"], "pred_traj")
    assert_finite_tensor(pred["bit_terminal_pred"], "bit_terminal_pred")
    assert_finite_tensor(pred["bit_path_anchor_pred"], "bit_path_anchor_pred")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "bit_smoke.ckpt"
        torch.save({"state_dict": planner.state_dict()}, ckpt_path)
        reloaded = ReCogDriveDiffusionPlanner(build_config()).to(device)
        state = torch.load(ckpt_path, map_location=device)
        reloaded.load_state_dict(state["state_dict"], strict=True)
        reloaded.eval()
        with torch.no_grad():
            reloaded_pred = reloaded.get_action(vl_features, eval_input, deterministic=True)
        assert_finite_tensor(reloaded_pred["pred_traj"], "reloaded_pred_traj")

    print("BiT-Drive dummy flow smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
