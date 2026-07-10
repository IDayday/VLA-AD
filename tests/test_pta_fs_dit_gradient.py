from __future__ import annotations

from dataclasses import dataclass
import sys
from types import ModuleType

import torch

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

_install_dependency_stubs()


def _patch_agent_stubs() -> None:
    dataclasses_mod = sys.modules["navsim.common.dataclasses"]

    @dataclass
    class AgentInput:
        pass

    class SensorConfig:
        @staticmethod
        def build_all_sensors(include=None):
            return SensorConfig()

    dataclasses_mod.AgentInput = getattr(dataclasses_mod, "AgentInput", AgentInput)
    dataclasses_mod.SensorConfig = getattr(dataclasses_mod, "SensorConfig", SensorConfig)
    abstract_agent_mod = ModuleType("navsim.agents.abstract_agent")

    class AbstractAgent(torch.nn.Module):
        pass

    abstract_agent_mod.AbstractAgent = AbstractAgent
    sys.modules["navsim.agents.abstract_agent"] = abstract_agent_mod


_patch_agent_stubs()

from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.planning_token_adapter import PlanningTokenAdapter, PlanningTokenAdapterConfig
from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from navsim.agents.recogdrive.recogdrive_dit import LightningDiT
from navsim.agents.recogdrive.recogdrive_diffusion_planner import (
    DDIMConfig,
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)


def test_diffusion_loss_reaches_planning_adapter() -> None:
    torch.manual_seed(37)
    adapter = PlanningTokenAdapter(
        PlanningTokenAdapterConfig(
            planner_dim=32,
            hidden_dim=64,
            num_tokens=8,
            num_heads=4,
            condition_dropout=0.0,
        )
    )
    dit = LightningDiT(
        num_heads=2,
        head_dim=16,
        output_dim=32,
        num_layers=2,
        dropout=0.0,
        attention_bias=True,
        interleave_attention=True,
    )
    dit.configure_planning_adapter_branch(0.05)
    planning_tokens, _ = adapter(
        torch.randn(2, 5, 32),
        torch.randn(2, 8),
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        torch.randn(2, 4, 3),
    )
    output = dit(
        torch.randn(2, 8, 32),
        torch.randn(2, 5, 32),
        torch.randn(2, 32),
        torch.tensor([1, 2]),
        planning_condition_tokens=planning_tokens,
        planning_condition_layers="cross_attention",
        planning_legacy_cot_gradient_path=False,
    )
    loss = torch.nn.functional.mse_loss(output, torch.randn_like(output))
    loss.backward()

    adapter_grad = sum(
        parameter.grad.abs().sum().item()
        for parameter in adapter.parameters()
        if parameter.grad is not None
    )
    planning_projection_grad = sum(
        block.cot_out_proj.weight.grad.abs().sum().item()
        for block in dit.transformer_blocks
        if block.cot_out_proj.weight.grad is not None
    )
    assert adapter_grad > 0.0
    assert planning_projection_grad > 0.0


def _adapter_planner_config() -> ReCogDriveDiffusionPlannerConfig:
    return ReCogDriveDiffusionPlannerConfig(
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
        planning_token_source="adapter",
        planning_num_heads=4,
    )


def _agent_with_adapter() -> ReCogDriveAgent:
    agent = ReCogDriveAgent.__new__(ReCogDriveAgent)
    torch.nn.Module.__init__(agent)
    agent.action_head = ReCogDriveDiffusionPlanner(_adapter_planner_config())
    agent.last_vla_train_vlm_lora = False
    agent.use_last_vla = False
    agent.use_planning_token_adapter = True
    agent.planning_token_source = "adapter"
    agent.freeze_expert = False
    agent.train_expert_only = False
    agent.freeze_base_action_head = False
    agent.allow_random_init = False
    return agent


def test_agent_scope_keeps_adapter_planning_branch_trainable() -> None:
    agent = _agent_with_adapter()

    agent._set_trainable_parameters()

    assert all(block.cot_cross_attn.to_q.weight.requires_grad for block in agent.action_head.model.transformer_blocks)
    assert all(block.cot_out_proj.weight.requires_grad for block in agent.action_head.model.transformer_blocks)
    assert all(block.planning_gate_logit.requires_grad for block in agent.action_head.model.transformer_blocks)


def test_agent_old_checkpoint_preserves_new_planning_projection_init(tmp_path) -> None:
    agent = _agent_with_adapter()
    old_state = {}
    for key, value in agent.state_dict().items():
        if "planning_adapter" in key or "planning_gate_logit" in key:
            continue
        old_state[key] = torch.zeros_like(value) if ".cot_out_proj." in key else value.detach().clone()
    checkpoint = tmp_path / "legacy_stage2.ckpt"
    torch.save({"state_dict": old_state}, checkpoint)
    before = [
        block.cot_out_proj.weight.detach().clone()
        for block in agent.action_head.model.transformer_blocks
    ]

    agent._safe_load_checkpoint(str(checkpoint))

    after = [block.cot_out_proj.weight.detach() for block in agent.action_head.model.transformer_blocks]
    assert all(torch.count_nonzero(weight).item() > 0 for weight in after)
    assert all(torch.equal(expected, actual) for expected, actual in zip(before, after))


def _aux_planner() -> ReCogDriveDiffusionPlanner:
    cfg = ReCogDriveDiffusionPlannerConfig(
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
        trajectory_aux_weight=0.05,
        feasibility_aux_weight=0.01,
        aux_warmup_epochs=1,
    )
    return ReCogDriveDiffusionPlanner(cfg)


def test_auxiliary_gradient_reaches_pred_noise_and_alpha_weights_noise_level() -> None:
    torch.manual_seed(41)
    planner = _aux_planner()
    raw_target = _straight_target = torch.zeros(2, 8, 3)
    _straight_target[..., 0] = torch.arange(1, 9).float()
    selected_repr = planner._encode_action_target(raw_target)
    action_input = BatchFeature(data={"action": raw_target})
    training_target = planner._build_training_target(
        raw_trajectory=raw_target,
        selected_repr=selected_repr,
        action_input=action_input,
    )
    timesteps = torch.tensor([0, 9])
    true_noise = torch.randn_like(selected_repr)
    noisy = (
        planner.extract(planner.ddpm_sqrt_alphas_cumprod, timesteps, selected_repr.shape) * selected_repr
        + planner.extract(planner.ddpm_sqrt_one_minus_alphas_cumprod, timesteps, selected_repr.shape) * true_noise
    )
    pred_noise = torch.zeros_like(selected_repr, requires_grad=True)
    pred_x0 = planner._x0_from_noise(noisy, timesteps, pred_noise)
    trajectory_loss, feasibility_loss, _ = planner._compute_pta_aux_per_sample_losses(
        pred_x0,
        training_target,
        timesteps,
        method="ddpm",
    )
    (trajectory_loss.mean() + feasibility_loss.mean()).backward()

    alpha_weight = planner._aux_alpha_weights(timesteps, selected_repr)
    assert alpha_weight[0] > alpha_weight[1]
    assert pred_noise.grad is not None
    assert torch.isfinite(pred_noise.grad).all()
    assert pred_noise.grad.abs().sum() > 0
