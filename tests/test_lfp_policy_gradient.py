from __future__ import annotations

import lzma
import pickle

import torch

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

_install_dependency_stubs()

from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.recogdrive_diffusion_planner import (
    DDIMConfig,
    GRPOConfig,
    OfflineRLConfig,
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)
from navsim.agents.recogdrive.stage3_lfp_grpo import LFPGRPOConfig
from navsim.agents.recogdrive.stage3_metric_adapter import CanonicalMetricBatch


def _model_config(**kwargs) -> ReCogDriveDiffusionPlannerConfig:
    values = dict(
        diffusion_model_cfg={
            "num_heads": 2,
            "head_dim": 16,
            "num_layers": 2,
            "output_dim": 32,
            "dropout": 0.0,
            "attention_bias": True,
            "norm_eps": 1e-5,
            "interleave_attention": True,
        },
        input_embedding_dim=32,
        planner_dim=32,
        hidden_size=64,
        action_horizon=8,
        sampling_method="ddim",
        num_inference_steps=2,
        vlm_size="small",
        ddim_cfg=DDIMConfig(num_train_timesteps=10),
        use_planning_token_adapter=True,
        planning_num_heads=4,
        planning_condition_dropout=0.5,
    )
    values.update(kwargs)
    return ReCogDriveDiffusionPlannerConfig(**values)


def test_on_policy_loss_reaches_dit_and_planning_adapter_but_not_reference(tmp_path) -> None:
    torch.manual_seed(61)
    initial = ReCogDriveDiffusionPlanner(_model_config())
    reference_checkpoint = tmp_path / "stage2.ckpt"
    torch.save(
        {"state_dict": {f"agent.action_head.{key}": value for key, value in initial.state_dict().items()}},
        reference_checkpoint,
    )

    metric_root = tmp_path / "metric_cache"
    (metric_root / "metadata").mkdir(parents=True)
    token_dir = metric_root / "scene"
    token_dir.mkdir()
    metric_path = token_dir / "metric_cache.pkl"
    with lzma.open(metric_path, "wb") as handle:
        pickle.dump(None, handle)
    (metric_root / "metadata" / "paths.csv").write_text(
        f"path\n{metric_path}\n",
        encoding="utf-8",
    )

    reference_cache = tmp_path / "reference.pt"
    record = {
        "selected": {
            "scalar": 0.6,
            "ep": 0.5,
            "ttc": 1.0,
            "quality": 0.7,
            "nc": 1.0,
            "dac": 1.0,
            "ddc": 1.0,
        },
        "selected_source": "gt",
        "selected_source_code": 1,
        "reference_fallback": False,
        "gt_ddc": 1.0,
    }
    torch.save(
        {"metadata": {"benchmark": "navsim_v1"}, "records": {"scene": record}},
        reference_cache,
    )
    grpo_cfg = GRPOConfig(
        metric_cache_path=str(metric_root),
        reference_policy_checkpoint=str(reference_checkpoint),
        sample_time=4,
        bc_coeff_start=0.0,
        bc_coeff_end=0.0,
        use_gspo_ratio=False,
        behavior_policy_sample=False,
        use_core_pareto_grpo=False,
        use_feasible_pareto_grpo=False,
        fp_use_pdas=False,
        core_pareto_use_phenotype_bucket_grpo=False,
        core_pareto_use_adaptive_dual=False,
        use_dynamic_group_weight=False,
        use_diversity_reward=False,
    )
    offline_cfg = OfflineRLConfig(
        enabled=False,
        bc_loss_weight=0.0,
        bc_loss_weight_start=0.0,
        bc_loss_weight_end=0.0,
    )
    planner = ReCogDriveDiffusionPlanner(
        _model_config(
            grpo=True,
            stage3_algorithm="lfp_grpo",
            grpo_cfg=grpo_cfg,
            offline_rl_cfg=offline_cfg,
            lfp_grpo_cfg=LFPGRPOConfig(
                enabled=True,
                benchmark="navsim_v1",
                reference_cache_path=str(reference_cache),
                curriculum_enabled=False,
            ),
        )
    ).train()
    planner.metric_cache_loader.metric_cache_paths["scene"] = metric_path

    def fixed_metrics(trajectories, tokens_rep, metric_cache, batch_size, group_size):
        scalar = trajectories.new_tensor([[0.5, 0.8, 0.6, 0.7]])
        ep = trajectories.new_tensor([[0.5, 0.8, 0.6, 0.7]])
        ones = torch.ones_like(scalar)
        return CanonicalMetricBatch(
            scalar=scalar,
            ep=ep,
            ttc=ones,
            quality=ep,
            nc=ones,
            dac=ones,
            ddc_guard_value=ones,
            tlc=None,
            diagnostics={},
        )

    planner._evaluate_lfp_rollouts = fixed_metrics
    action_input = BatchFeature(
        data={
            "action": torch.zeros(1, 8, 3),
            "his_traj": torch.randn(1, 12),
            "history_trajectory": torch.randn(1, 4, 3),
            "status_feature": torch.randn(1, 8),
            "high_command_one_hot": torch.tensor([[0.0, 1.0, 0.0]]),
        }
    )
    output = planner.forward_lfp_grpo(torch.randn(1, 5, 1536), action_input, ["scene"], sample_time=4)
    assert torch.isfinite(output["loss"])
    assert output["planning_condition_keep_ratio"].item() == 1.0
    assert output["lfp_exact_kl"].abs().item() < 1e-6
    output["loss"].backward()

    dit_grad = sum(
        parameter.grad.abs().sum().item()
        for parameter in planner.model.parameters()
        if parameter.grad is not None
    )
    adapter_grad = sum(
        parameter.grad.abs().sum().item()
        for parameter in planner.planning_adapter.parameters()
        if parameter.grad is not None
    )
    assert dit_grad > 0.0
    assert adapter_grad > 0.0
    assert all(not parameter.requires_grad for parameter in planner.old_policy.parameters())
    assert all(parameter.grad is None for parameter in planner.old_policy.parameters())
    assert planner.planning_adapter.forward_count == 1
