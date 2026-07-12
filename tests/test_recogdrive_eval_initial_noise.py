from types import SimpleNamespace

import torch

from scripts.eval_recogdrive_expert_pdm import make_initial_noise, stable_sample_seed


def test_stable_sample_seed_depends_only_on_base_seed_and_token() -> None:
    assert stable_sample_seed(0, "scene-a") == stable_sample_seed(0, "scene-a")
    assert stable_sample_seed(0, "scene-a") != stable_sample_seed(0, "scene-b")
    assert stable_sample_seed(0, "scene-a") != stable_sample_seed(1, "scene-a")


def test_initial_noise_is_invariant_to_evaluation_order() -> None:
    planner = SimpleNamespace(config=SimpleNamespace(action_horizon=8, action_dim=3))
    features = torch.empty(1, 4)
    forward = {
        token: make_initial_noise(planner, features, token, 17)
        for token in ("scene-a", "scene-b", "scene-c")
    }
    reverse = {
        token: make_initial_noise(planner, features, token, 17)
        for token in ("scene-c", "scene-b", "scene-a")
    }
    for token in forward:
        torch.testing.assert_close(forward[token], reverse[token], rtol=0.0, atol=0.0)

