import pytest
import torch
from torch import nn

from navsim.agents.recogdrive.stage3_frontier_curriculum import (
    DistributedFrontierSampler,
    FrontierStateStore,
)
from navsim.planning.training.checkpoint_utils import (
    _filter_recogdrive_checkpoint_state_dict,
    _load_recogdrive_checkpoint_state_dict,
)


def test_frontier_and_sampler_state_resume() -> None:
    kwargs = dict(
        fast_ema=0.8,
        slow_ema=0.98,
        progress_weight=0.5,
        priority_exponent=0.5,
        uniform_ratio=0.2,
        priority_eps=1e-3,
        priority_quantile_cap=0.95,
    )
    source = FrontierStateStore(**kwargs)
    source.update_epoch({"a": (3.0, 2), "b": (1.0, 1)}, epoch=4)
    weights = source.priorities(["a", "b", "c"])
    sampler = DistributedFrontierSampler(3, weights, warmup_epochs=1, uniform_ratio=0.2)
    sampler.set_epoch(5)
    list(sampler)

    restored = FrontierStateStore(**kwargs)
    restored.load_state_dict(source.state_dict())
    restored_sampler = DistributedFrontierSampler(3, torch.ones(3), warmup_epochs=1, uniform_ratio=0.2)
    restored_sampler.load_state_dict(sampler.state_dict())

    assert restored.states == source.states
    torch.testing.assert_close(restored.priorities(["a", "b", "c"]), weights)
    torch.testing.assert_close(restored_sampler.weights, sampler.weights)
    assert restored_sampler.epoch == 5


class _FilteredCheckpointModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.agent = nn.Module()
        self.agent.trainable = nn.Linear(2, 2)
        self.agent.action_head = nn.Module()
        self.agent.action_head.old_policy = nn.Linear(2, 2)


def test_filtered_checkpoint_strict_resume_allows_only_frozen_reference_keys() -> None:
    source = _FilteredCheckpointModule()
    checkpoint_state = _filter_recogdrive_checkpoint_state_dict(source.state_dict())
    restored = _FilteredCheckpointModule()

    result = _load_recogdrive_checkpoint_state_dict(restored, checkpoint_state, strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    torch.testing.assert_close(restored.agent.trainable.weight, source.agent.trainable.weight)


def test_filtered_checkpoint_strict_resume_rejects_trainable_missing_key() -> None:
    source = _FilteredCheckpointModule()
    checkpoint_state = _filter_recogdrive_checkpoint_state_dict(source.state_dict())
    checkpoint_state.pop("agent.trainable.weight")

    with pytest.raises(RuntimeError, match="agent.trainable.weight"):
        _load_recogdrive_checkpoint_state_dict(
            _FilteredCheckpointModule(),
            checkpoint_state,
            strict=True,
        )
