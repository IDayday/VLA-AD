from __future__ import annotations

from pathlib import Path

import pytest

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

_install_dependency_stubs()

from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner, ReCogDriveDiffusionPlannerConfig


def test_non_last_vla_a0_style_still_instantiates():
    planner = ReCogDriveDiffusionPlanner(
        ReCogDriveDiffusionPlannerConfig(
            input_embedding_dim=384,
            hidden_size=384,
            vlm_size="small",
            use_last_vla=False,
            last_vla_stage="disabled",
            use_last_rd=False,
            last_rd_stage="disabled",
        )
    )
    assert planner.config.use_last_vla is False


def test_last_vla_legacy_bottleneck_config_is_rejected():
    with pytest.raises(ValueError, match="Hard bottleneck"):
        ReCogDriveDiffusionPlanner(
            ReCogDriveDiffusionPlannerConfig(
                input_embedding_dim=384,
                hidden_size=384,
                vlm_size="small",
                use_last_vla=True,
                last_vla_stage="cot_alignment",
                last_vla_condition_mode="cot_bottleneck",
                last_vla_cot_bottleneck_mode=True,
                last_vla_raw_vlm_context_to_dit=False,
            )
        )


def test_production_launchers_use_decoupled_experiments_only():
    launcher = Path("scripts/last_vla_v2/decoupled_highcap_no_risk/serverA_frozen_vlm_decoupled_highcap_no_risk.sh")
    text = launcher.read_text(encoding="utf-8")
    assert "last_vla_decoupled_cot_alignment_highcap_no_risk" in text
    assert "last_vla_progressive_bottleneck" not in text
    assert "vlm_summary" not in text


def test_hard_bottleneck_entrypoints_are_archived():
    assert Path("configs/last_vla_v2/archive/hard_bottleneck_legacy/last_vla_progressive_bottleneck_highcap_no_risk.yaml").is_file()
    assert Path(
        "navsim/planning/script/config/experiment/archive/hard_bottleneck_legacy/last_vla_progressive_bottleneck_highcap_no_risk.yaml"
    ).is_file()
    assert Path("scripts/last_vla_v2/archive/hard_bottleneck_legacy/serverA_frozen_vlm_highcap_no_risk.sh").is_file()
    assert not Path("navsim/planning/script/config/experiment/last_vla_progressive_bottleneck_highcap_no_risk.yaml").exists()
