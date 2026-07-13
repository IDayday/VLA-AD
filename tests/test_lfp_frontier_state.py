import torch

from navsim.agents.recogdrive.stage3_frontier_curriculum import FrontierStateStore


def _store() -> FrontierStateStore:
    return FrontierStateStore(
        fast_ema=0.8,
        slow_ema=0.98,
        progress_weight=0.5,
        priority_exponent=0.5,
        uniform_ratio=0.2,
        priority_eps=1e-3,
        priority_quantile_cap=0.5,
    )


def test_fast_slow_ema_and_learning_progress() -> None:
    store = _store()
    store.update_epoch({"scene": (4.0, 2)}, epoch=0)
    state = store.states["scene"]
    assert abs(state.fast_ema - 0.4) < 1e-12
    assert abs(state.slow_ema - 0.04) < 1e-12
    store.priorities(["scene"])
    assert abs(store.last_diagnostics["lfp_learning_progress_mean"] - 0.36) < 1e-12


def test_priority_quantile_cap_and_uniform_floor() -> None:
    store = _store()
    store.update_epoch({"a": 0.1, "b": 0.2, "outlier": 1000.0}, epoch=0)
    weights = store.priorities(["a", "b", "outlier", "unseen"])
    assert torch.isclose(weights.sum(), torch.tensor(1.0, dtype=weights.dtype))
    assert torch.all(weights >= 0.2 / 4.0 - 1e-12)
    assert store.last_diagnostics["lfp_frontier_priority_cap"] < store._raw_priority(store.states["outlier"])
    assert store.last_diagnostics["lfp_sampler_unseen_scene_ratio"] == 0.25
