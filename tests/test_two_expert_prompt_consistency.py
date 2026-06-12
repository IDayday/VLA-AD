from __future__ import annotations

import torch

from scripts.last_vla_v2.two_expert_slot.build_two_expert_hidden_cache import (
    build_two_expert_prompt as hidden_cache_prompt,
)
from scripts.last_vla_v2.two_expert_slot.run_two_expert_vlm_sft import (
    build_two_expert_prompt as stage1_prompt,
)


def _sample():
    return {
        "history_trajectory": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.2, 0.01],
                [2.0, 0.4, 0.02],
                [3.0, 0.6, 0.03],
            ]
        ),
        "high_command_one_hot": torch.tensor([0.0, 0.0, 1.0]),
        "status_feature": torch.randn(8),
    }


def test_stage1_and_hidden_cache_prompt_are_identical():
    sample = _sample()

    assert stage1_prompt(sample) == hidden_cache_prompt(sample)


def test_prompt_contains_history_and_command_not_simplified():
    prompt = stage1_prompt(_sample())

    assert "Historical motion context (last 4 timesteps)" in prompt
    assert "Active navigation command: [TURN RIGHT]" in prompt
    assert "t-3:" in prompt and "t-0:" in prompt
    assert "Predict the ego vehicle trajectory for the next 4 seconds" not in prompt
