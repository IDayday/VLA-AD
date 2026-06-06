from __future__ import annotations

import os
import subprocess
from pathlib import Path


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
            "FULL_HIGHCAP_TRAIN_CHUNK_ROOT": str(cache),
            "A0_INIT_CHECKPOINT": str(ckpt),
            "OUT_ROOT": str(tmp_path / "out"),
            "MASTER_PORT": "29991",
            "VLM_PATH": str(vlm),
            "NAVSIM_LOG_PATH": str(tmp_path / "navsim_logs"),
            "SENSOR_BLOBS_PATH": str(tmp_path / "sensor_blobs"),
            "PYTHON_BIN": "python",
            "TORCHRUN_BIN": "/bin/false",
        }
    )
    env.pop("RUN_TRAIN", None)
    return env


def test_highcap_launchers_dryrun_do_not_launch_training_and_include_overrides(tmp_path: Path):
    env = _env(tmp_path)
    env.update(
        {
            "LORA_PRESET": "all_linear",
            "LORA_SCOPE": "llm",
            "LORA_R": "64",
            "LORA_ALPHA": "128",
            "LORA_DROPOUT": "0.05",
            "LORA_USE_RSLORA": "true",
        }
    )
    subprocess.run(["bash", "scripts/last_vla_v2/decoupled_highcap_no_risk/serverA_frozen_vlm_decoupled_highcap_no_risk.sh"], env=env, check=True)
    subprocess.run(["bash", "scripts/last_vla_v2/decoupled_highcap_no_risk/serverB_vlm_lora_decoupled_highcap_no_risk.sh"], env=env, check=True)

    server_a = tmp_path / "out" / "serverA_frozen_vlm_decoupled_highcap_no_risk" / "commands.log"
    server_b = tmp_path / "out" / "serverB_vlm_lora_decoupled_highcap_no_risk" / "commands.log"
    text_a = server_a.read_text(encoding="utf-8")
    text_b = server_b.read_text(encoding="utf-8")

    for text in (text_a, text_b):
        assert "last_vla_decoupled_cot_alignment_highcap_no_risk" in text or "last_vla_decoupled_vlm_lora_cot_alignment_highcap_no_risk" in text
        assert "last_vla_decoupled_progressive_highcap_no_risk" in text
        assert "agent.num_jepa_tokens=128" in text
        assert "agent.num_dynamic_tokens=128" in text
        assert "agent.num_geometry_tokens=192" in text
        assert "agent.last_vla_geometry_teacher_dim=512" in text
        assert "agent.last_vla_use_risk_head=false" in text
        assert "agent.last_vla_allow_patch_geometry_fallback=false" in text
        assert "agent.last_vla_condition_mode=decoupled_cot_residual" in text
        assert "agent.last_vla_cot_bottleneck_mode=false" in text
        assert "agent.last_vla_raw_vlm_context_to_dit=true" in text
        assert "vlm_summary" not in text
    assert "navsim_log_path=" in text_b
    assert "sensor_blobs_path=" in text_b
    assert "--cache-variant decoupled_highcap_no_risk" in text_b
    assert "agent.last_vla_vlm_lora_preset=all_linear" in text_b
    assert "agent.last_vla_vlm_lora_scope=llm" in text_b
    assert "agent.last_vla_vlm_lora_r=64" in text_b
    assert "agent.last_vla_vlm_lora_alpha=128" in text_b
    assert "--preset all_linear" in text_b
    assert "--scope llm" in text_b
    assert "--r 64" in text_b
    assert "--alpha 128" in text_b
    assert "--use-rslora" in text_b
    assert "attention_mlp" not in text_b
