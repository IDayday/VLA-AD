"""Smoke test safe checkpoint loading without real weights or NAVSIM data.

Run from the repository root:

    PYTHONDONTWRITEBYTECODE=1 python scripts/testing/smoke_recogdrive_checkpoint_loading.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.smoke_test_recogdrive_agent_dummy_forward import (  # noqa: E402
    JEPA_DIM,
    VGGT_DIM,
    _install_agent_dependency_stubs,
)

_install_agent_dependency_stubs()

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # noqa: E402

from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent  # noqa: E402


def _agent(
    *,
    use_expert_features: bool,
    checkpoint_path: str | None = None,
) -> ReCogDriveAgent:
    return ReCogDriveAgent(
        trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
        cache_hidden_state=True,
        cache_mode=False,
        vlm_path=None,
        checkpoint_path=checkpoint_path,
        dit_type="small",
        vlm_size="small",
        grpo=False,
        allow_random_init=True,
        use_expert_features=use_expert_features,
        expert_feature_source="dummy" if use_expert_features else "none",
        use_jepa=True,
        use_vggt=True,
        jepa_dim=JEPA_DIM if use_expert_features else 0,
        vggt_dim=VGGT_DIM if use_expert_features else 0,
    )


def main() -> None:
    torch.manual_seed(19)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        baseline = _agent(use_expert_features=False)
        baseline_ckpt = tmp_path / "baseline_random.ckpt"
        torch.save({"state_dict": baseline.state_dict()}, baseline_ckpt)

        expert_from_baseline = _agent(use_expert_features=True, checkpoint_path=str(baseline_ckpt))
        expert_from_baseline.initialize()
        assert hasattr(expert_from_baseline.action_head, "jepa_projector")
        assert hasattr(expert_from_baseline.action_head, "vggt_projector")

        expert_ckpt = tmp_path / "expert_random.ckpt"
        torch.save({"state_dict": expert_from_baseline.state_dict()}, expert_ckpt)
        expert_from_expert = _agent(use_expert_features=True, checkpoint_path=str(expert_ckpt))
        expert_from_expert.initialize()

        bad_state = baseline.state_dict()
        bad_state["action_head.feature_encoder.weight"] = torch.randn(1)
        bad_ckpt = tmp_path / "bad_shape.ckpt"
        torch.save({"state_dict": bad_state}, bad_ckpt)
        try:
            _agent(use_expert_features=True, checkpoint_path=str(bad_ckpt)).initialize()
        except RuntimeError as exc:
            assert "shape mismatch" in str(exc)
        else:
            raise AssertionError("baseline parameter shape mismatch should fail")

    print("ReCogDrive safe checkpoint loading smoke test passed.")


if __name__ == "__main__":
    main()
