from __future__ import annotations

import pytest
import torch

from navsim.agents.recogdrive.trajectory_text_replay import (
    build_replay_labels,
    format_trajectory_answer,
    parse_trajectory_answer,
)


class _TinyTokenizer:
    eos_token_id = 9
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(ch) % 17 + 1 for ch in str(text)]}


def test_format_parse_trajectory_roundtrip():
    traj = torch.arange(24, dtype=torch.float32).view(8, 3) / 10.0
    text = format_trajectory_answer(traj, decimals=2)
    parsed = parse_trajectory_answer(text)
    assert parsed.shape == (8, 3)
    assert torch.allclose(parsed, traj, atol=0.01)


def test_build_replay_labels_masks_prompt_and_keeps_answer():
    labels = build_replay_labels(_TinyTokenizer(), "prompt", "answer")
    prompt_len = len("prompt")
    assert torch.equal(labels["labels"][:prompt_len], torch.full((prompt_len,), -100))
    assert (labels["labels"][prompt_len:] != -100).any()


def test_invalid_answer_fails_fast():
    with pytest.raises(ValueError):
        parse_trajectory_answer("no trajectory here")
