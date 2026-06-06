from __future__ import annotations

import pytest

from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner, ReCogDriveDiffusionPlannerConfig


def test_last_vla_bottleneck_mode_is_removed():
    cfg = ReCogDriveDiffusionPlannerConfig(
        input_embedding_dim=384,
        hidden_size=384,
        vlm_size="small",
        use_last_vla=True,
        last_vla_stage="progressive_sft_decoupled",
        last_vla_condition_mode="cot_bottleneck",
        last_vla_cot_bottleneck_mode=True,
        last_vla_raw_vlm_context_to_dit=False,
    )
    with pytest.raises(ValueError, match="Hard bottleneck"):
        ReCogDriveDiffusionPlanner(cfg)
