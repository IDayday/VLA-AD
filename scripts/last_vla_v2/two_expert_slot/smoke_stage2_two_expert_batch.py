#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
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


def make_planner() -> ReCogDriveDiffusionPlanner:
    cfg = ReCogDriveDiffusionPlannerConfig(
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
        use_two_expert_slots=True,
        two_expert_cache_mode=True,
        two_expert_condition_mode="horizon_hmef_lite",
        two_expert_dit_condition_mode="horizon_hmef_lite",
        two_expert_zero_init_deltas=True,
        two_expert_use_raw_vlm_base=True,
        use_expert_features=False,
        use_last_vla=False,
        last_vla_stage="disabled",
        use_last_rd=False,
        last_rd_stage="disabled",
        last_vla_use_residual_diffusion=False,
        last_vla_teacher_traj_mode="none",
    )
    return ReCogDriveDiffusionPlanner(cfg)


def make_batch(batch: int = 2) -> tuple[torch.Tensor, BatchFeature]:
    return torch.randn(batch, 9, 1536), BatchFeature(
        data={
            "his_traj": torch.randn(batch, 12),
            "history_trajectory": torch.randn(batch, 4, 3),
            "status_feature": torch.randn(batch, 8),
            "high_command_one_hot": torch.eye(3)[:batch],
            "action": torch.randn(batch, 8, 3),
            "two_expert_h_dyn": torch.randn(batch, 3, 12, 1536),
            "two_expert_h_geo": torch.randn(batch, 12, 1536),
        }
    )


def run_smoke() -> Dict[str, object]:
    torch.manual_seed(19)
    planner = make_planner()
    planner.train()
    vl, action_input = make_batch()
    target_norm = planner.norm_odo(action_input.action)
    context = planner._prepare_dit_context(vl, action_input, training=True, target_action_norm=target_norm)
    target, alpha = planner._last_vla_diffusion_target(target_norm, training=True)
    out = planner(vl, action_input)

    if context["expert_step_condition"].shape != (2, 8, 384):
        raise AssertionError(f"bad expert_step_condition shape {tuple(context['expert_step_condition'].shape)}")
    if context["context_tokens"].shape != (2, 9, 384):
        raise AssertionError(f"raw VLM context not preserved: {tuple(context['context_tokens'].shape)}")
    if not torch.allclose(context["expert_step_condition"], torch.zeros_like(context["expert_step_condition"])):
        raise AssertionError("zero-init expert deltas should make initial expert_step_condition zero")
    if not torch.allclose(target, target_norm) or alpha != 0.0:
        raise AssertionError("two-expert diffusion target must remain GT normalized trajectory")
    if not torch.isfinite(out["loss"]):
        raise AssertionError("training loss is not finite")
    if "pred_residual_norm" in out:
        raise AssertionError("two-expert training output must not expose residual prediction")

    planner.eval()
    with torch.no_grad():
        eval_input = BatchFeature(data={key: value for key, value in action_input.items() if key != "action"})
        pred = planner.get_action(vl, eval_input, init_actions=torch.zeros(2, 8, 3), deterministic=True)
        if pred["pred_traj"].shape != (2, 8, 3):
            raise AssertionError("get_action did not return pred_traj [B,8,3]")
        for mode in ("raw_vlm_only", "zero_h_dyn", "zero_h_geo"):
            corrupted = BatchFeature(data=dict(eval_input))
            corrupted["two_expert_corruption_mode"] = mode
            corrupted_context = planner._prepare_dit_context(vl, corrupted, training=False)
            if corrupted_context["expert_step_condition"].shape != (2, 8, 384):
                raise AssertionError(f"corruption {mode} broke expert_step_condition shape")

    return {
        "ok": True,
        "loss": float(out["loss"].detach().cpu()),
        "diffusion_loss": float(out["diffusion_loss"].detach().cpu()),
        "expert_step_condition_shape": list(context["expert_step_condition"].shape),
        "context_tokens_shape": list(context["context_tokens"].shape),
        "diffusion_target_is_gt_norm": True,
        "no_residual": True,
    }


def main() -> int:
    result = run_smoke()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
