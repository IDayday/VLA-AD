from __future__ import annotations

from pathlib import Path

import yaml


def test_two_expert_route_configs_load_directly():
    stage1 = yaml.safe_load(Path("configs/last_vla_v2/two_expert_slot/stage1_vlm_sft.yaml").read_text(encoding="utf-8"))
    stage2 = yaml.safe_load(Path("configs/last_vla_v2/two_expert_slot/stage2_dit_sft.yaml").read_text(encoding="utf-8"))

    assert stage1["use_two_expert_slot_sft"] is True
    assert stage1["num_dyn_groups"] == 3
    assert stage1["num_dyn_tokens_per_group"] == 12
    assert stage1["num_geo_tokens"] == 12
    assert stage2["use_two_expert_slots"] is True
    assert stage2["two_expert_condition_mode"] == "horizon_hmef_lite"
    assert stage2["last_vla_use_residual_diffusion"] is False
    assert stage2["last_vla_teacher_traj_mode"] == "none"
    assert stage2["stage2_target"]["type"] == "gt_normalized_trajectory"


def test_two_expert_hydra_experiment_yaml_values():
    exp_dir = Path("navsim/planning/script/config/experiment")
    stage1 = yaml.safe_load((exp_dir / "two_expert_slot_stage1_vlm_sft.yaml").read_text(encoding="utf-8"))
    stage2 = yaml.safe_load((exp_dir / "two_expert_slot_stage2_dit_sft.yaml").read_text(encoding="utf-8"))

    assert stage1["two_expert_slot_stage"] == "stage1_vlm_sft"
    assert stage1["agent"]["cache_hidden_state"] is False
    assert stage1["agent"]["use_last_vla"] is False
    assert stage1["agent"]["use_last_rd"] is False
    assert stage2["agent"]["use_two_expert_slots"] is True
    assert stage2["agent"]["use_last_vla"] is False
    assert stage2["agent"]["use_last_rd"] is False
    assert stage2["agent"]["use_expert_features"] is False
    assert stage2["agent"]["last_vla_use_residual_diffusion"] is False


def test_two_expert_hydra_compose_if_available():
    try:
        from hydra import compose, initialize_config_module
        from hydra.core.global_hydra import GlobalHydra
    except Exception:
        return
    GlobalHydra.instance().clear()
    with initialize_config_module(config_module="navsim.planning.script.config.training", version_base=None):
        cfg = compose(config_name="default_training", overrides=["train_test_split=trainval", "+experiment=two_expert_slot_stage2_dit_sft"])
    assert cfg.agent.use_two_expert_slots is True
    assert cfg.agent.use_last_vla is False
    assert cfg.agent.use_last_rd is False
    assert cfg.agent.last_vla_use_residual_diffusion is False
