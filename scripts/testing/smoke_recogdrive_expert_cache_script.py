"""Smoke test for the JEPA/VGGT expert cache generation helpers.

Run from the repository root:

    PYTHONDONTWRITEBYTECODE=1 python scripts/testing/smoke_recogdrive_expert_cache_script.py
"""

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.planning.script.run_recogdrive_expert_feature_caching import (
    DummyExpertFeatureBackend,
    ExpertBatch,
    build_backend,
    cache_file_path,
    pool_dense_tokens,
    save_cache_file,
    write_metadata,
)


def main() -> None:
    args = SimpleNamespace(
        teacher_backend="dummy",
        jepa_checkpoint_or_model_path="/tmp/jepa",
        vggt_checkpoint_or_model_path="/tmp/vggt",
        dummy_jepa_dim=5,
        dummy_vggt_dim=7,
        dummy_dense_jepa_tokens=11,
        dummy_dense_vggt_tokens=13,
    )
    backend = build_backend(args, device=torch.device("cpu"), dtype=torch.float32)
    assert isinstance(backend, DummyExpertFeatureBackend)

    batch = ExpertBatch(
        tokens=["tok_a", "tok_b"],
        log_names=["log_a", "log_b"],
        current_image_paths=[["/images/a0.jpg", "/images/a1.jpg"], ["/images/b0.jpg"]],
        future_image_paths=[["/images/a_future.jpg"], ["/images/b_future.jpg"]],
    )

    dense = backend.encode_current(batch)
    dense_again = backend.encode_current(batch)
    assert torch.equal(dense["jepa"], dense_again["jepa"])
    assert torch.equal(dense["vggt"], dense_again["vggt"])

    jepa_tokens = pool_dense_tokens(dense["jepa"], num_tokens=4)
    vggt_tokens = pool_dense_tokens(dense["vggt"], num_tokens=3)
    assert jepa_tokens.shape == (2, 4, 5)
    assert vggt_tokens.shape == (2, 3, 7)

    future_dense = backend.encode_future_targets(batch)
    jepa_target_tokens = pool_dense_tokens(future_dense["jepa"], num_tokens=4)
    vggt_target_tokens = pool_dense_tokens(future_dense["vggt"], num_tokens=3)

    with tempfile.TemporaryDirectory() as tmp:
        output_root = Path(tmp)
        path = cache_file_path(output_root, "log_a", "tok_a")
        save_cache_file(
            path,
            {
                "jepa_tokens": jepa_tokens[0],
                "vggt_tokens": vggt_tokens[0],
                "jepa_target_tokens": jepa_target_tokens[0],
                "vggt_target_tokens": vggt_target_tokens[0],
            },
        )
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        assert payload["jepa_tokens"].shape == (4, 5)
        assert payload["vggt_tokens"].shape == (3, 7)
        assert payload["jepa_target_tokens"].shape == (4, 5)
        assert payload["vggt_target_tokens"].shape == (3, 7)

        metadata_path = output_root / "metadata.json"
        write_metadata(metadata_path, {"backend": "dummy", "num_samples": 2})
        assert metadata_path.is_file()

    print("ReCogDrive expert cache script smoke test passed.")


if __name__ == "__main__":
    main()
