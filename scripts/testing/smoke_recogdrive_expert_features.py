"""Smoke test for optional ReCogDrive expert feature cache loading.

Run from the repository root:

    python scripts/testing/smoke_recogdrive_expert_features.py
"""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import sys
import warnings

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.recogdrive_features import ReCogDriveFeatureBuilder


def _fake_agent_input(log_name: str, token: str):
    ego_statuses = []
    for i in range(4):
        ego_statuses.append(
            SimpleNamespace(
                ego_pose=np.array([float(i), 0.0, 0.0], dtype=np.float32),
                ego_velocity=np.array([0.0, 0.0], dtype=np.float32),
                ego_acceleration=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                driving_command=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            )
        )

    camera = SimpleNamespace(cam_f0=SimpleNamespace(image="/tmp/fake_front_camera.jpg"))
    return SimpleNamespace(
        ego_statuses=ego_statuses,
        cameras=[camera] * 4,
        log_name=log_name,
        token=token,
        scene_token="scene-token",
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_root = Path(tmp)
        log_name = "log_a"
        token = "token_123"
        sample_dir = cache_root / log_name / token
        sample_dir.mkdir(parents=True)
        torch.save(
            {
                "jepa_tokens": torch.randn(4, 16),
                "vggt_tokens": torch.randn(4, 32),
                "jepa_target_tokens": torch.randn(4, 16),
            },
            sample_dir / "expert_features.pt",
        )

        agent_input = _fake_agent_input(log_name, token)

        baseline_builder = ReCogDriveFeatureBuilder(cache_hidden_state=False)
        baseline_features = baseline_builder.compute_features(agent_input)
        assert "jepa_tokens" not in baseline_features
        assert "vggt_tokens" not in baseline_features

        expert_builder = ReCogDriveFeatureBuilder(
            cache_hidden_state=False,
            use_expert_features=True,
            expert_cache_dir=str(cache_root),
            num_jepa_tokens=4,
            num_vggt_tokens=4,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            features = expert_builder.compute_features(agent_input)

        assert features["jepa_tokens"].shape == (4, 16)
        assert features["vggt_tokens"].shape == (4, 32)
        assert "jepa_target_tokens" not in features
        assert any("Dropping it to prevent future-frame leakage" in str(w.message) for w in caught)
        assert features["jepa_tokens"].dtype == torch.float32

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            augmented = expert_builder.add_expert_features_from_token_path({}, sample_dir)
        assert augmented["jepa_tokens"].shape == (4, 16)
        assert "jepa_target_tokens" not in augmented

        training_expert_builder = ReCogDriveFeatureBuilder(
            cache_hidden_state=False,
            use_expert_features=True,
            expert_cache_dir=str(cache_root),
            num_jepa_tokens=4,
            num_vggt_tokens=4,
            allow_expert_target_features=True,
        )
        training_features = training_expert_builder.compute_features(agent_input)
        assert training_features["jepa_target_tokens"].shape == (4, 16)
        print("ReCogDrive expert feature smoke test passed.")


if __name__ == "__main__":
    main()
