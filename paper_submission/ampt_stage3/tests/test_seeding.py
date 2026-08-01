from __future__ import annotations

import pytest

from ampt_stage3.seeding import seed_for_token


def test_scene_seed_is_deterministic_and_token_specific() -> None:
    first = seed_for_token(20260726, "scene-a")
    assert first == seed_for_token(20260726, "scene-a")
    assert first != seed_for_token(20260726, "scene-b")
    with pytest.raises(ValueError):
        seed_for_token(-1, "scene-a")
