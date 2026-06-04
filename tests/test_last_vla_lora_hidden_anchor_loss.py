from __future__ import annotations

import torch

from tests.test_last_vla_vlm_lora_trainable_scope import ReCogDriveAgent, TrajectorySampling


def _agent(weight: float = 0.01) -> ReCogDriveAgent:
    agent = ReCogDriveAgent(
        trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
        checkpoint_path="",
        allow_random_init=True,
        cache_hidden_state=True,
        use_last_vla=True,
        last_vla_stage="cot_alignment",
        use_last_rd=False,
        last_vla_hidden_anchor_weight=weight,
        last_vla_hidden_anchor_mode="summary_cosine",
    )
    return agent


def test_hidden_anchor_loss_metrics_are_finite():
    agent = _agent()
    lora = torch.randn(2, 5, 16)
    frozen = lora + 0.01 * torch.randn(2, 5, 16)

    loss, cosine, l2 = agent._hidden_anchor_metrics(lora, frozen)

    assert torch.isfinite(loss)
    assert torch.isfinite(cosine)
    assert torch.isfinite(l2)


def test_hidden_anchor_active_only_for_lora_cot_alignment_with_weight():
    agent = _agent(weight=0.01)
    agent.train()
    agent.cache_hidden_state = False
    agent.last_vla_train_vlm_lora = True
    assert agent._hidden_anchor_active()

    agent.last_vla_hidden_anchor_weight = 0.0
    assert not agent._hidden_anchor_active()
    agent.last_vla_hidden_anchor_weight = 0.01
    agent.last_vla_stage = "progressive_sft_bottleneck"
    assert not agent._hidden_anchor_active()
