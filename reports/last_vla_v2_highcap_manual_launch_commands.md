# Last-VLA v2 High-cap Manual Launch Commands

Baseline: A0-official-aligned full navtest PDMS = `0.864891`.

These commands must not be run until `scripts/last_vla_v2/highcap_no_risk/run_final_readiness_gate.sh` writes a `READY` report.

## Local Server A

```bash
RUN_TRAIN=1 \
FULL_GEOMETRY_CHUNK_ROOT=/path/to/highcap_no_risk/train_full_highcap_chunks \
A0_INIT_CHECKPOINT=/path/to/A0-official-aligned/step_00100000.ckpt \
OUT_ROOT=/path/to/last_vla_v2/highcap_no_risk/train_local \
MASTER_PORT=29531 \
scripts/last_vla_v2/highcap_no_risk/serverA_frozen_vlm_highcap_no_risk.sh
```

## Remote Server B

```bash
ssh training-vla-zt-peer '
cd /mnt/project/VLA-AD &&
git fetch origin &&
git checkout feature/recogdrive-last-vla-v2 &&
git pull --ff-only origin feature/recogdrive-last-vla-v2 &&
RUN_TRAIN=1 \
FULL_GEOMETRY_CHUNK_ROOT=/path/to/highcap_no_risk/train_full_highcap_chunks \
A0_INIT_CHECKPOINT=/path/to/A0-official-aligned/step_00100000.ckpt \
VLM_PATH=/path/to/base_vlm \
NAVSIM_LOG_PATH=/path/to/navsim_logs \
SENSOR_BLOBS_PATH=/path/to/sensor_blobs \
OUT_ROOT=/path/to/last_vla_v2/highcap_no_risk/train_remote \
MASTER_PORT=29541 \
LORA_PRESET=attention_mlp \
LORA_SCOPE=llm \
LORA_R=32 \
LORA_ALPHA=64 \
LORA_DROPOUT=0.05 \
LORA_USE_RSLORA=true \
LORA_USE_DORA=false \
LORA_TARGET_MODULES="" \
LORA_HIDDEN_ANCHOR_EVERY_N_STEPS=4 \
scripts/last_vla_v2/highcap_no_risk/serverB_vlm_lora_highcap_no_risk.sh
'
```

Line B eval must use LoRA-regenerated navtest hidden cache, not the original hidden cache.
