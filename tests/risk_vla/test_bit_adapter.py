from __future__ import annotations

import pytest
import torch

from navsim.agents.recogdrive.risk_vla.bit_adapter import extract_bit_intents


def test_bit_adapter_handles_missing_and_disabled():
    ref = torch.zeros(2, 4)
    terminal, path, diagnostics = extract_bit_intents(
        {},
        batch_size=2,
        action_horizon=8,
        action_dim=3,
        reference=ref,
        use_bit_summary=False,
    )
    assert terminal is None
    assert path is None
    assert diagnostics["bit_adapter_enabled"].item() == 0.0


def test_bit_adapter_accepts_known_aliases_and_casts():
    ref = torch.zeros(2, 4, dtype=torch.float32)
    action_input = {
        "pred_terminal_intent": torch.ones(2, 3, dtype=torch.float64),
        "pred_path_intent": torch.ones(2, 8, 3, dtype=torch.float64),
    }
    terminal, path, diagnostics = extract_bit_intents(
        action_input,
        batch_size=2,
        action_horizon=8,
        action_dim=3,
        reference=ref,
        use_bit_summary=True,
        strict=True,
    )
    assert terminal is not None and terminal.shape == (2, 3)
    assert path is not None and path.shape == (2, 8, 3)
    assert terminal.dtype == ref.dtype
    assert diagnostics["bit_adapter_has_terminal"].item() == 1.0
    assert diagnostics["bit_adapter_has_path"].item() == 1.0


def test_bit_adapter_strict_shape_error_and_nonstrict_warning():
    ref = torch.zeros(2, 4)
    bad = {"bit_path": torch.zeros(2, 7, 3)}
    with pytest.raises(ValueError):
        extract_bit_intents(
            bad,
            batch_size=2,
            action_horizon=8,
            action_dim=3,
            reference=ref,
            strict=True,
        )
    with pytest.warns(RuntimeWarning):
        terminal, path, _ = extract_bit_intents(
            bad,
            batch_size=2,
            action_horizon=8,
            action_dim=3,
            reference=ref,
            strict=False,
        )
    assert terminal is None
    assert path is None
