#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
from transformers.feature_extraction_utils import BatchFeature

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.recogdrive_diffusion_planner import (
    DDIMConfig,
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)
from navsim.agents.recogdrive.fs_norm import load_fs_norm_stats
from scripts.pareto_support.build_fs_norm_stats import _selected_supports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded PTA-FS-DiT smoke on selected real supports.")
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--fs-stats", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--num-scenes", type=int, default=2)
    parser.add_argument("--targets-per-scene", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=53)
    return parser.parse_args()


def load_targets(archive_dir: Path, num_scenes: int, targets_per_scene: int) -> torch.Tensor:
    rows = []
    for path in sorted(archive_dir.glob("*.pkl.xz")):
        supports, _ = _selected_supports(path)
        if len(supports) >= targets_per_scene:
            rows.append(
                torch.stack(
                    [torch.as_tensor(item, dtype=torch.float32) for item in supports[:targets_per_scene]],
                    dim=0,
                )
            )
        if len(rows) >= num_scenes:
            break
    if len(rows) < num_scenes:
        raise RuntimeError(
            f"Only found {len(rows)} scenes with at least {targets_per_scene} selected supports under {archive_dir}."
        )
    targets = torch.stack(rows)
    if targets.shape[2:] != (8, 3):
        raise ValueError(f"Expected selected targets [B, M, 8, 3], got {tuple(targets.shape)}.")
    return targets


def planner_config(fs_stats: Path, *, adapter: bool) -> ReCogDriveDiffusionPlannerConfig:
    stats = load_fs_norm_stats(str(fs_stats), map_location="cpu")
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
        planner_dim=384,
        hidden_size=1024,
        action_dim=3,
        action_horizon=8,
        max_seq_len=8,
        sampling_method="ddim",
        num_inference_steps=5,
        vlm_size="small",
        ddim_cfg=DDIMConfig(num_train_timesteps=100),
        use_fs_norm=True,
        fs_norm_stats_path=str(fs_stats),
        fs_norm_min_version=2,
        fs_norm_use_robust=bool(stats.use_robust),
        fs_norm_target_clip=0.0,
        fs_norm_output_clip=12.0,
        fs_norm_output_clip_mode="stats_bounds",
        use_planning_token_adapter=adapter,
        planning_num_tokens=16,
        planning_num_heads=8,
        planning_condition_layers="cross_attention",
        planning_gate_init=0.05,
        planning_context_gate_init=0.05,
        planning_condition_dropout=0.10,
        trajectory_aux_weight=0.05 if adapter else 0.0,
        feasibility_aux_weight=0.01 if adapter else 0.0,
        aux_alpha_power=1.0,
        aux_warmup_epochs=10,
        tangent_margin_rad=0.08,
        curvature_margin=0.05,
        min_segment_length=0.20,
        x0_aux_weight=0.0,
        delta_aux_weight=0.0,
        geo_aux_weight=0.0,
    )


def make_conditioning(targets: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, BatchFeature]:
    batch = int(targets.shape[0])
    vl_features = torch.randn(batch, 6, 1536, device=device)
    command = torch.eye(3, device=device)[torch.arange(batch, device=device) % 3]
    action_input = BatchFeature(
        data={
            "action": targets[:, 0].to(device),
            "his_traj": torch.randn(batch, 12, device=device),
            "history_trajectory": torch.randn(batch, 4, 3, device=device),
            "status_feature": torch.randn(batch, 8, device=device),
            "high_command_one_hot": command,
        }
    )
    return vl_features, action_input


def scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar diagnostic, got {tuple(value.shape)}.")
        return float(value.detach().cpu().item())
    return float(value)


def main() -> int:
    args = parse_args()
    if not 10 <= args.steps <= 50:
        raise ValueError("--steps must be in [10, 50] for this bounded smoke.")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    targets = load_targets(args.archive_dir, args.num_scenes, args.targets_per_scene).to(device)
    batch, target_count = targets.shape[:2]
    weights = torch.full((batch, target_count), 1.0 / target_count, device=device)
    source_code = torch.full((batch, target_count), 3, device=device, dtype=torch.long)
    source_code[:, 0] = 1
    vl_features, action_input = make_conditioning(targets, device)

    planner = ReCogDriveDiffusionPlanner(planner_config(args.fs_stats, adapter=True)).to(device).train()
    assert planner.planning_adapter is not None
    optimizer = torch.optim.AdamW(planner.parameters(), lr=1e-5, weight_decay=1e-4)
    step_rows = []
    for step in range(args.steps):
        planner._reset_planning_adapter_forward_count()
        optimizer.zero_grad(set_to_none=True)
        diffusion_loss, diagnostics = planner._weighted_diffusion_loss_on_targets(
            vl_features,
            action_input,
            targets,
            weights,
            target_source_code=source_code,
        )
        ramp = diagnostics["aux_warmup_ramp"].to(diffusion_loss)
        total_loss = diffusion_loss + ramp * (
            0.05 * diagnostics["trajectory_aux_loss"].to(diffusion_loss)
            + 0.01 * diagnostics["feasibility_aux_loss"].to(diffusion_loss)
        )
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite total loss at step {step}.")
        total_loss.backward()
        finite_gradients = all(
            torch.isfinite(parameter.grad).all().item()
            for parameter in planner.parameters()
            if parameter.grad is not None
        )
        if not finite_gradients:
            raise RuntimeError(f"Non-finite gradient at step {step}.")
        adapter_grad = sum(
            parameter.grad.detach().float().norm().item()
            for parameter in planner.planning_adapter.parameters()
            if parameter.grad is not None
        )
        if adapter_grad <= 0.0:
            raise RuntimeError(f"Planning adapter received no gradient at step {step}.")
        if planner.planning_adapter.forward_count != 1:
            raise RuntimeError(
                f"Planning adapter forward count at step {step} is {planner.planning_adapter.forward_count}, expected 1."
            )
        optimizer.step()
        step_rows.append(
            {
                "step": step,
                "total_loss": scalar(total_loss),
                "diffusion_loss": scalar(diffusion_loss),
                "trajectory_aux_loss": scalar(diagnostics["trajectory_aux_loss"]),
                "feasibility_aux_loss": scalar(diagnostics["feasibility_aux_loss"]),
                "adapter_grad_norm_sum": adapter_grad,
                "planning_adapter_forward_count": planner.planning_adapter.forward_count,
            }
        )

    planner.eval()
    with torch.no_grad():
        chain, prediction = planner.sample_chain(
            vl_features,
            action_input.his_traj,
            action_input.status_feature,
            init_actions=torch.zeros(batch, 8, 3, device=device),
            deterministic=True,
            action_input=action_input,
        )
    if chain.shape != (batch, 6, 8, 3) or not torch.isfinite(chain).all() or not torch.isfinite(prediction).all():
        raise RuntimeError("Five-step DDIM produced an invalid chain or prediction.")
    adapter_inference_count = planner.planning_adapter.forward_count
    if adapter_inference_count != 1:
        raise RuntimeError(f"Five-step DDIM adapter forward count is {adapter_inference_count}, expected 1.")

    del optimizer, planner
    if device.type == "cuda":
        torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    baseline = ReCogDriveDiffusionPlanner(planner_config(args.fs_stats, adapter=False)).to(device).eval()
    with torch.no_grad():
        baseline_chain, baseline_prediction = baseline.sample_chain(
            vl_features,
            action_input.his_traj,
            action_input.status_feature,
            init_actions=torch.zeros(batch, 8, 3, device=device),
            deterministic=True,
            action_input=action_input,
        )
    flag_off_finite = bool(torch.isfinite(baseline_chain).all() and torch.isfinite(baseline_prediction).all())
    if baseline.planning_adapter is not None or not flag_off_finite:
        raise RuntimeError("Flag-off baseline unexpectedly created an adapter or produced non-finite output.")

    report = {
        "archive_dir": str(args.archive_dir.resolve()),
        "fs_stats": str(args.fs_stats.resolve()),
        "device": str(device),
        "num_layers": 16,
        "planner_dim": 384,
        "num_scenes": batch,
        "targets_per_scene": target_count,
        "train_steps": args.steps,
        "all_losses_finite": True,
        "all_gradients_finite": True,
        "adapter_inference_forward_count": adapter_inference_count,
        "ddim_chain_shape": list(chain.shape),
        "ddim_output_finite": True,
        "flag_off_adapter_absent": baseline.planning_adapter is None,
        "flag_off_output_finite": flag_off_finite,
        "steps": step_rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
