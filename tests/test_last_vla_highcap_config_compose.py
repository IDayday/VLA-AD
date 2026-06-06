from __future__ import annotations

from pathlib import Path

import yaml


def _assert_highcap_agent(agent) -> None:
    assert agent["use_last_vla"] is True
    assert agent["use_last_rd"] is False
    assert agent["last_vla_use_risk_head"] is False
    assert agent["num_risk_tokens"] == 0
    assert agent["last_vla_risk_loss_weight"] == 0.0
    assert agent["last_vla_cot_num_tokens"] == 192
    assert "last_vla_vlm_summary_tokens" not in agent
    assert agent["num_jepa_tokens"] == 128
    assert agent["num_dynamic_tokens"] == 128
    assert agent["num_geometry_tokens"] == 192
    assert agent["last_vla_geometry_teacher_dim"] == 512
    assert agent["last_vla_geometry_grid_rows"] == 12
    assert agent["last_vla_geometry_grid_cols"] == 16
    assert agent["last_vla_require_full_geometry"] is True
    assert agent["last_vla_allow_patch_geometry_fallback"] is False
    assert agent["last_vla_condition_mode"] == "decoupled_cot_residual"
    assert agent["last_vla_cot_bottleneck_mode"] is False
    assert agent["last_vla_raw_vlm_context_to_dit"] is True


def test_highcap_yaml_configs_load_directly():
    config_dir = Path("configs/last_vla_v2/decoupled_highcap_no_risk")
    base = yaml.safe_load((config_dir / "base.yaml").read_text(encoding="utf-8"))
    _assert_highcap_agent(base)
    assert base["policy_kd_mode"] == "none"
    assert base["policy_kd_loss_weight"] == 0.0
    for name in ("cot_alignment.yaml", "progressive_sft.yaml", "progressive_sft_eval.yaml", "vlm_lora_cot_alignment.yaml"):
        data = yaml.safe_load((config_dir / name).read_text(encoding="utf-8"))
        assert "defaults" in data
        assert "last_vla_vlm_summary_tokens" not in data


def test_highcap_hydra_experiment_yaml_loads():
    exp_dir = Path("navsim/planning/script/config/experiment")
    for name in (
        "last_vla_decoupled_cot_alignment_highcap_no_risk.yaml",
        "last_vla_decoupled_progressive_highcap_no_risk.yaml",
    ):
        data = yaml.safe_load((exp_dir / name).read_text(encoding="utf-8"))
        _assert_highcap_agent(data["agent"])
    lora_data = yaml.safe_load((exp_dir / "last_vla_decoupled_vlm_lora_cot_alignment_highcap_no_risk.yaml").read_text(encoding="utf-8"))
    assert lora_data["agent"]["last_vla_train_vlm_lora"] is True
    assert "last_vla_vlm_summary_tokens" not in lora_data["agent"]


def test_highcap_hydra_compose_if_available():
    try:
        from hydra import compose, initialize_config_module
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf
    except Exception:
        return

    for experiment in (
        "last_vla_decoupled_cot_alignment_highcap_no_risk",
        "last_vla_decoupled_progressive_highcap_no_risk",
        "last_vla_decoupled_vlm_lora_cot_alignment_highcap_no_risk",
    ):
        GlobalHydra.instance().clear()
        with initialize_config_module(config_module="navsim.planning.script.config.training", version_base=None):
            cfg = compose(
                config_name="default_training",
                overrides=["train_test_split=trainval", f"+experiment={experiment}"],
            )
        agent = OmegaConf.to_container(cfg.agent, resolve=True)
        _assert_highcap_agent(agent)
