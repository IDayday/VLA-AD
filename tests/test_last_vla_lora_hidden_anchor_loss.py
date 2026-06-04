from __future__ import annotations

import torch
import pytest

from tests.test_last_vla_vlm_lora_trainable_scope import ReCogDriveAgent, TrajectorySampling


def _agent(weight: float = 0.01, every_n: int = 4) -> ReCogDriveAgent:
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
        last_vla_hidden_anchor_every_n_steps=every_n,
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


def test_hidden_anchor_every_two_steps_computes_every_other_decision():
    agent = _agent(every_n=2)
    agent.train()
    agent.cache_hidden_state = False
    agent.last_vla_train_vlm_lora = True

    decisions = [agent._next_hidden_anchor_decision() for _ in range(4)]

    assert [compute for compute, _ in decisions] == [True, False, True, False]
    assert [index for _, index in decisions] == [0, 1, 2, 3]


def test_hidden_anchor_every_one_step_computes_each_decision():
    agent = _agent(every_n=1)
    agent.train()
    agent.cache_hidden_state = False
    agent.last_vla_train_vlm_lora = True

    assert [agent._next_hidden_anchor_decision()[0] for _ in range(3)] == [True, True, True]


def test_hidden_anchor_every_n_steps_must_be_positive():
    with pytest.raises(ValueError, match="every_n_steps"):
        _agent(every_n=0)


def test_hidden_anchor_skipped_step_does_not_add_weighted_loss():
    agent = _agent(every_n=2)
    predictions = {"loss": torch.tensor(2.0)}

    agent._attach_hidden_anchor_outputs(
        predictions,
        hidden_anchor_loss=None,
        hidden_drift_cosine=None,
        hidden_drift_l2=None,
        hidden_anchor_computed=False,
        hidden_anchor_step_index=1,
    )

    assert predictions["loss"].item() == 2.0
    assert "hidden_anchor_loss" not in predictions
    assert "hidden_anchor_loss_weighted" not in predictions
    assert predictions["hidden_anchor_computed"].item() == 0.0
    assert predictions["hidden_anchor_every_n_steps_tensor"].item() == 2.0
