from __future__ import annotations

import os
import subprocess
from pathlib import Path

from hydra.core.override_parser.overrides_parser import OverridesParser


def _env(tmp_path: Path) -> dict[str, str]:
    ckpt = tmp_path / "a0.ckpt"
    cache = tmp_path / "cache"
    vlm = tmp_path / "vlm"
    ckpt.write_bytes(b"dummy")
    vlm.write_text("dummy", encoding="utf-8")
    cache.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "FULL_GEOMETRY_CHUNK_ROOT": str(cache),
            "A0_INIT_CHECKPOINT": str(ckpt),
            "OUT_ROOT": str(tmp_path / "out"),
            "MASTER_PORT": "29990",
            "VLM_PATH": str(vlm),
            "NAVSIM_LOG_PATH": str(tmp_path / "navsim_logs"),
            "SENSOR_BLOBS_PATH": str(tmp_path / "sensor_blobs"),
            "PYTHON_BIN": "python",
            "TORCHRUN_BIN": "/bin/false",
        }
    )
    return env


def test_round2_launchers_dryrun_do_not_launch_training(tmp_path: Path):
    env = _env(tmp_path)
    subprocess.run(["bash", "scripts/last_vla_v2/highcap_no_risk/serverA_frozen_vlm_highcap_no_risk.sh"], env=env, check=True)
    env["RUN_TRAIN"] = "0"
    subprocess.run(["bash", "scripts/last_vla_v2/highcap_no_risk/serverB_vlm_lora_highcap_no_risk.sh"], env=env, check=True)

    logs = [
        tmp_path / "out" / "serverA_frozen_highcap_no_risk" / "commands.log",
        tmp_path / "out" / "serverB_lora_highcap_no_risk" / "commands.log",
    ]
    for path in logs:
        text = path.read_text(encoding="utf-8")
        assert "agent.last_vla_adapter_checkpoint=" in text
        assert "++agent.last_vla_adapter_checkpoint" not in text
        assert "+experiment=last_vla" in text
    server_b_text = logs[1].read_text(encoding="utf-8")
    assert "navsim_log_path=" in server_b_text
    assert "sensor_blobs_path=" in server_b_text
    assert "agent.last_vla_vlm_lora_preset=attention_mlp" in server_b_text
    assert "agent.last_vla_vlm_lora_scope=llm" in server_b_text
    assert "agent.last_vla_vlm_lora_r=32" in server_b_text
    assert "agent.last_vla_vlm_lora_alpha=64" in server_b_text
    assert "agent.last_vla_vlm_lora_dropout=0.05" in server_b_text
    assert "agent.last_vla_vlm_lora_use_rslora=true" in server_b_text


def test_hydra_lora_target_module_quote_is_string_not_sweep():
    override = OverridesParser.create().parse_overrides(
        ["agent.last_vla_vlm_lora_target_modules='q_proj,k_proj,v_proj,o_proj'"]
    )[0]

    assert override.value() == "q_proj,k_proj,v_proj,o_proj"


def test_vlm_lora_alignment_launcher_dryrun(tmp_path: Path):
    env = _env(tmp_path)
    env.update(
        {
            "OUT_ROOT": str(tmp_path / "out" / "lora_sweep"),
            "MASTER_PORT_BASE": "29990",
            "EXPERT_TEACHER_CACHE_ROOT": env["FULL_GEOMETRY_CHUNK_ROOT"],
            "LAST_VLA_DRY_RUN_ALLOW_MISSING_PATHS": "1",
            "LORA_SWEEP_CONFIGS": "last_vla_vlm_lora_attention_mlp_r32",
        }
    )
    subprocess.run(["bash", "scripts/last_vla_v2/lora/run_lora_cot_alignment_sweep.sh"], env=env, check=True)
    text = (Path(env["OUT_ROOT"]) / "last_vla_vlm_lora_attention_mlp_r32" / "commands.log").read_text(encoding="utf-8")
    assert "+experiment=last_vla_vlm_lora_attention_mlp_r32" in text
    assert "agent.cache_hidden_state=false" in text
    assert "agent.last_vla_vlm_lora_preset=attention_mlp" in text
    assert "agent.last_vla_vlm_lora_r=32" in text
    assert "agent.last_vla_vlm_lora_alpha=64" in text
    assert "agent.expert_cache_dir=" in text
    assert "navsim_log_path=" in text
    assert "sensor_blobs_path=" in text
