"""Smoke test for dummy expert-cache generation and planner consumption.

Run from the repository root:

    PYTHONDONTWRITEBYTECODE=1 python scripts/testing/smoke_recogdrive_dummy_expert_cache.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs  # noqa: E402

_install_dependency_stubs()

from transformers.feature_extraction_utils import BatchFeature  # noqa: E402

from navsim.agents.recogdrive.recogdrive_diffusion_planner import (  # noqa: E402
    DDIMConfig,
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)
from navsim.agents.recogdrive.recogdrive_features import (  # noqa: E402
    assert_real_expert_cache_for_training,
    is_dummy_expert_cache,
    load_expert_cache_sample,
    load_expert_cache_metadata,
)
from scripts.create_dummy_expert_cache import create_dummy_expert_cache  # noqa: E402


def _planner_config() -> ReCogDriveDiffusionPlannerConfig:
    return ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg={
            "num_heads": 1,
            "head_dim": 8,
            "num_layers": 2,
            "output_dim": 8,
            "dropout": 0.0,
            "attention_bias": True,
            "norm_eps": 1e-5,
            "interleave_attention": True,
        },
        input_embedding_dim=8,
        hidden_size=16,
        action_dim=3,
        action_horizon=8,
        max_seq_len=8,
        sampling_method="ddim",
        num_inference_steps=2,
        vlm_size="small",
        ddim_cfg=DDIMConfig(num_train_timesteps=10),
        use_expert_features=True,
        use_jepa=True,
        use_vggt=True,
        jepa_dim=768,
        vggt_dim=2048,
        expert_dropout=0.0,
        jepa_alignment_weight=0.1,
        vggt_alignment_weight=0.1,
    )


def _action_input(sample: dict[str, torch.Tensor]) -> BatchFeature:
    return BatchFeature(data={
        "his_traj": torch.linspace(-1.0, 1.0, 12, dtype=torch.float32).view(1, 12),
        "status_feature": torch.linspace(0.0, 0.7, 8, dtype=torch.float32).view(1, 8),
        "action": torch.stack([
            torch.linspace(0.0, 8.0, 8),
            torch.linspace(-0.4, 0.4, 8),
            torch.linspace(-0.1, 0.1, 8),
        ], dim=-1).view(1, 8, 3),
        "jepa_tokens": sample["jepa_tokens"].unsqueeze(0),
        "vggt_tokens": sample["vggt_tokens"].unsqueeze(0),
        "jepa_target_tokens": sample["jepa_target_tokens"].unsqueeze(0),
        "vggt_target_tokens": sample["vggt_target_tokens"].unsqueeze(0),
    })


def main() -> None:
    torch.manual_seed(17)

    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "dummy_expert_cache"
        create_dummy_expert_cache(cache_dir, num_samples=3, include_targets=True, seed=11)

        metadata = load_expert_cache_metadata(cache_dir)
        assert metadata is not None
        assert metadata["is_dummy"] is True
        assert is_dummy_expert_cache(cache_dir)

        try:
            assert_real_expert_cache_for_training(
                cache_dir,
                use_expert_features=True,
                allow_dummy_expert_cache=False,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("dummy cache should be rejected for training without override")

        assert_real_expert_cache_for_training(
            cache_dir,
            use_expert_features=True,
            allow_dummy_expert_cache=True,
        )

        sample = load_expert_cache_sample(cache_dir, "sample_000001")
        assert sample["jepa_tokens"].shape == (4, 768)
        assert sample["vggt_tokens"].shape == (4, 2048)
        assert sample["jepa_target_tokens"].shape == (4, 768)
        assert sample["vggt_target_tokens"].shape == (4, 2048)

        planner = ReCogDriveDiffusionPlanner(_planner_config())
        planner.train()
        vl_features = torch.randn(1, 3, 1536)
        output = planner(vl_features, _action_input(sample))

        assert "loss" in output
        assert torch.isfinite(output["loss"])
        assert torch.isfinite(output["jepa_alignment_loss"])
        assert torch.isfinite(output["vggt_alignment_loss"])

    print("Dummy expert-cache smoke test passed.")


if __name__ == "__main__":
    main()
