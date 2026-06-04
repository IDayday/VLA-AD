from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_lora_sweep_launcher_dryrun_writes_commands(tmp_path: Path):
    env = os.environ.copy()
    env.update(
        {
            "RUN_TRAIN": "0",
            "LAST_VLA_DRY_RUN_ALLOW_MISSING_PATHS": "1",
            "NAVSIM_LOG_PATH": "/tmp/navsim_logs",
            "SENSOR_BLOBS_PATH": "/tmp/sensor_blobs",
            "EXPERT_TEACHER_CACHE_ROOT": "/tmp/expert_cache",
            "VLM_PATH": "/tmp/vlm",
            "OUT_ROOT": str(tmp_path / "sweep"),
            "MASTER_PORT_BASE": "29600",
            "PYTHON_BIN": "python",
            "TORCHRUN_BIN": "/bin/false",
            "LORA_SWEEP_CONFIGS": "last_vla_vlm_lora_attention_only_r16,last_vla_vlm_lora_attention_mlp_r32",
        }
    )

    subprocess.run(["bash", "scripts/last_vla_v2/lora/run_lora_cot_alignment_sweep.sh"], env=env, check=True)

    text = (tmp_path / "sweep" / "last_vla_vlm_lora_attention_mlp_r32" / "commands.log").read_text(encoding="utf-8")
    assert "+experiment=last_vla_vlm_lora_attention_mlp_r32" in text
    assert "extract_vlm_lora_and_cot_adapters.py" in text
    assert "/bin/false" in text
