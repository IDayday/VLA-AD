from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_LABELED_TRIPLE_RE = re.compile(
    rf"\(\s*x\s*:\s*({_FLOAT_RE})\s*,\s*y\s*:\s*({_FLOAT_RE})\s*,\s*heading\s*:\s*({_FLOAT_RE})\s*\)",
    flags=re.IGNORECASE,
)
_UNLABELED_TRIPLE_RE = re.compile(rf"\(\s*({_FLOAT_RE})\s*,\s*({_FLOAT_RE})\s*,\s*({_FLOAT_RE})\s*\)")


@dataclass(frozen=True)
class TrajectoryParseResult:
    parse_ok: bool
    trajectory: Optional[torch.Tensor]
    error: Optional[str] = None


def format_trajectory_answer(traj: torch.Tensor, decimals: int = 2) -> str:
    tensor = torch.as_tensor(traj).detach().cpu().float()
    if tuple(tensor.shape) != (8, 3):
        raise ValueError(f"trajectory must have shape [8,3], got {tuple(tensor.shape)}.")
    parts = []
    for x, y, heading in tensor.tolist():
        parts.append(f"({x:+.{decimals}f}, {y:+.{decimals}f}, {heading:+.{decimals}f})")
    return "Here is the planning trajectory [PT, " + ", ".join(parts) + "]."


def _triples_from_patterns(text: str) -> list[tuple[float, float, float]]:
    labeled = [(float(x), float(y), float(h)) for x, y, h in _LABELED_TRIPLE_RE.findall(text)]
    if labeled:
        return labeled
    return [(float(x), float(y), float(h)) for x, y, h in _UNLABELED_TRIPLE_RE.findall(text)]


def try_parse_trajectory_answer(text: str) -> TrajectoryParseResult:
    if not isinstance(text, str) or not text.strip():
        return TrajectoryParseResult(False, None, "empty trajectory answer")
    triples = _triples_from_patterns(text)
    if len(triples) == 8:
        tensor = torch.tensor(triples, dtype=torch.float32)
        if torch.isfinite(tensor).all():
            return TrajectoryParseResult(True, tensor, None)
        return TrajectoryParseResult(False, None, "trajectory contains non-finite values")
    numbers = [float(item) for item in re.findall(_FLOAT_RE, text)]
    if len(numbers) == 24:
        tensor = torch.tensor(numbers, dtype=torch.float32).view(8, 3)
        if torch.isfinite(tensor).all():
            return TrajectoryParseResult(True, tensor, None)
        return TrajectoryParseResult(False, None, "trajectory contains non-finite values")
    return TrajectoryParseResult(False, None, f"expected 8 trajectory triples or 24 numbers, got {len(triples)} triples and {len(numbers)} numbers")


def parse_trajectory_answer(text: str) -> torch.Tensor:
    result = try_parse_trajectory_answer(text)
    if not result.parse_ok or result.trajectory is None:
        raise ValueError(result.error or "failed to parse trajectory answer")
    return result.trajectory


def _tokenize_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else getattr(encoded, "input_ids")
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.detach().cpu().view(-1).tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(item) for item in input_ids]


def build_replay_labels(
    tokenizer: Any,
    prompt: str,
    answer_text: str,
    *,
    max_length: Optional[int] = None,
    pad_to_max_length: bool = False,
) -> Dict[str, torch.Tensor]:
    prompt_ids = _tokenize_ids(tokenizer, prompt)
    answer_ids = _tokenize_ids(tokenizer, answer_text)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        answer_ids = answer_ids + [int(eos_id)]
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    if max_length is not None:
        input_ids = input_ids[: int(max_length)]
        labels = labels[: int(max_length)]
    attention_mask = [1] * len(input_ids)
    if pad_to_max_length:
        if max_length is None:
            raise ValueError("pad_to_max_length=True requires max_length.")
        pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        pad_len = int(max_length) - len(input_ids)
        if pad_len > 0:
            input_ids.extend([pad_id] * pad_len)
            labels.extend([-100] * pad_len)
            attention_mask.extend([0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }
