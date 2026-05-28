# AGENT.md — ReCogDrive-2B + JEPA/VGGT Expert-Token Project

## 0. Project objective

This repository lives at:

```bash
/mnt/project/VLA-AD
```

It is a modified `xiaomi-research/recogdrive` project. The goal is to implement and train a ReCogDrive-2B-compatible expert-token conditioning system on NAVSIM, inspired by CoWorld-VLA and LaST-VLA.

The project must not stop at smoke tests. Smoke tests are only gates. The required deliverable is a trainable model, evaluated on NAVSIM with ablations that support a meaningful experimental conclusion.

Core design:

- Keep ReCogDrive-2B small DiT unchanged.
- Keep `planner_dim = 384`.
- Add `expert_adapter_dim = 768` only inside the new expert modules.
- Load official ReCogDrive-2B-IL action_head weights wherever tensor shapes match.
- Randomly initialize only new JEPA/VGGT projectors, adapters, alignment heads, type embeddings, gates, and branch weights.
- Train incrementally from Base-IL, not from scratch.

Default locations:

```bash
PROJECT_ROOT=/mnt/project/VLA-AD
NAVSIM_DATA_ROOT=/mnt/navsim

CHECKPOINT_ROOT=/mnt/project/VLA-AD/checkpoints
HF_HOME=/mnt/project/VLA-AD/hf_home
HF_HUB_CACHE=/mnt/project/VLA-AD/hf_home/hub
HF_XET_CACHE=/mnt/project/VLA-AD/hf_home/xet
MODELSCOPE_CACHE=/mnt/project/VLA-AD/modelscope_cache

EXPERIMENT_ROOT=/mnt/project/VLA-AD/experiments/recogdrive_expert
CHUNK_CACHE_ROOT=/mnt/project/VLA-AD/cache/recogdrive_expert_chunks
```

All code, configs, scripts, tests, and docs are written under `/mnt/project/VLA-AD`. Large artifacts are also under this project directory for operational simplicity, but must be gitignored:

```text
/mnt/project/VLA-AD/checkpoints/
/mnt/project/VLA-AD/cache/
/mnt/project/VLA-AD/experiments/
/mnt/project/VLA-AD/hf_home/
/mnt/project/VLA-AD/modelscope_cache/
```

Update `.gitignore` accordingly. Never commit model weights, generated caches, experiment logs, or evaluation outputs.

---

## 1. Non-negotiable constraints

1. Preserve original behavior when `use_expert_features=False`.
2. Do not change original ReCogDrive small DiT input/output dimensions.
3. Do not send 768-dim tokens into the original DiT. DiT context tokens must remain 384-dim.
4. Do not require a full 1–2 TB global hidden-state cache.
5. Implement chunked cache and chunked training.
6. Do not import V-JEPA2 or VGGT in core model forward files. Teacher models belong only in extraction/cache scripts.
7. Do not use future teacher target tokens at evaluation or inference.
8. Do not hardcode Hugging Face tokens, proxy addresses, or non-default machine paths.
9. Do not stop after dummy smoke tests. Implement true training and evaluation commands.
10. Any real training must fail if it accidentally uses dummy cache unless `--allow-dummy-cache` is explicitly passed.

---

## 2. Download policy

Weights must be downloaded into:

```bash
/mnt/project/VLA-AD/checkpoints
```

Required model repositories:

```text
owl10/ReCogDrive-2B-IL
owl10/ReCogDrive-VLM-2B
facebook/vjepa2-vitl-fpc64-256
facebook/VGGT-1B
```

Optional:

```text
owl10/ReCogDrive-2B-RL
```

Download priority:

1. Prefer domestic mirror sources without proxy.
   - Try ModelScope when a matching model exists.
   - Try Hugging Face mirror endpoint via `HF_ENDPOINT` when requested.
2. If domestic mirror is unavailable or too slow, download from the original Hugging Face Hub with proxy.
3. Support resume and re-run. Never delete partially downloaded model files unless `--force-download` is explicitly set.

Implement:

```text
scripts/download_required_weights.py
scripts/check_required_weights.py
scripts/print_required_weight_paths.py
configs/weights.yaml
```

`download_required_weights.py` must support:

```text
--output-root /mnt/project/VLA-AD/checkpoints
--source auto | modelscope | hf-mirror | hf-original
--hf-endpoint optional, default from HF_ENDPOINT
--modelscope-cache /mnt/project/VLA-AD/modelscope_cache
--hf-cache /mnt/project/VLA-AD/hf_home/hub
--repo-workers 4
--file-workers 8
--dry-run
--force-download
--include-rl
--proxy optional
--proxy-if-slow
--min-speed-mbps 5
--timeout-sec 120
--retries 3
```

Default behavior:

```text
source=auto
proxy disabled
try domestic mirror first
fallback to original Hugging Face only if domestic source fails or is too slow
```

Recommended local paths:

```text
/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-IL
/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B
/mnt/project/VLA-AD/checkpoints/teachers/vjepa2-vitl-fpc64-256
/mnt/project/VLA-AD/checkpoints/teachers/VGGT-1B
/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-RL        # optional
```

`check_required_weights.py` must verify that each directory exists and contains at least config files and weight files. Accept `.safetensors`, `.bin`, `.pt`, `.ckpt`, or repository-specific weight files. Do not assume exact filenames.

---

## 3. Data root policy

NAVSIM data is under:

```bash
/mnt/navsim
```

Implement robust auto-detection. Detect and print:

```text
NAVSIM_DATA_ROOT
OPENSCENE_DATA_ROOT
NUPLAN_MAPS_ROOT
NAVSIM split/index files
sensor blob directory
map directory
```

Default environment:

```bash
export NAVSIM_DATA_ROOT=/mnt/navsim
export OPENSCENE_DATA_ROOT=/mnt/navsim/openscene
export NUPLAN_MAPS_ROOT=/mnt/navsim/maps
```

If `/mnt/navsim/openscene` or `/mnt/navsim/maps` does not exist, search likely subdirectories under `/mnt/navsim` before failing. If still missing, fail with a clear diagnostic and suggested environment variable overrides.

---

## 4. Model architecture to implement

### 4.1 Base model

Target base model:

```text
ReCogDrive-2B
vlm_size = small
DiT size = small
VLM hidden dim = 1536
planner_dim = 384
action_horizon = 8
action_dim = 3
```

Keep original ReCogDrive modules compatible:

```text
feature_encoder: 1536 -> 384
his_traj_encoder: 12 -> 384
ego_status_encoder: 8 -> 384
action_encoder: 3 -> 384
fusion_projector: 3*384 -> 384
DiT encoder_hidden_states dim: 384
```

### 4.2 Expert dimensions

```text
planner_dim = 384
expert_adapter_dim = 768
jepa_dim = 1024
vggt_dim = 2048
num_jepa_tokens = 12
num_vggt_tokens = 12
```

`expert_adapter_dim=768` is internal only. All tokens passed to ReCogDrive DiT must be projected back to 384.

### 4.3 JEPA teacher

Use:

```text
facebook/vjepa2-vitl-fpc64-256
teacher_dim = 1024
```

Feature extraction:

- Use `outputs.last_hidden_state` only.
- Do not use predictor output in v1.
- Context input: NAVSIM `cam_f0` history frames `[t-1.5s, t-1.0s, t-0.5s, t]`.
- Expand 4 frames to 8 frames by repeating each frame twice.
- Target input: future `cam_f0` frames `[t+0.5s, t+1.0s, t+1.5s, t+2.0s]`, train-only.
- Expand target frames to 8 frames by repeating each frame twice.
- Input resolution: 256.
- Pool dense tokens to `[12, 1024]` using `adaptive_avg_pool3d(output_size=(4, 1, 3))`.

### 4.4 VGGT teacher

Use:

```text
facebook/VGGT-1B
teacher_dim = 2048
layer = aggregator final cached layer 23
```

Feature extraction:

- Use `model.aggregator(images)`.
- Do not use depth, point-map, camera, or tracking heads in v1.
- Context input: current `cam_f0` only.
- Target input: same current `cam_f0`, train-only.
- Remove camera/register tokens with `patch_start_idx`.
- Reshape patch features to `37 x 37`.
- Pool to `[12, 2048]` using `adaptive_avg_pool2d(output_size=(3, 4))`.
- Do not use 1x1 global VGGT pooling in the default model. Add 1x1 pooling only as an ablation.

---

## 5. Files to create or modify

Create:

```text
navsim/agents/recogdrive/expert_fusion.py
navsim/agents/recogdrive/expert_cache.py
navsim/agents/recogdrive/expert_feature_provider.py
navsim/agents/recogdrive/expert_extractors/__init__.py
navsim/agents/recogdrive/expert_extractors/pooling.py
navsim/agents/recogdrive/expert_extractors/vjepa2_extractor.py
navsim/agents/recogdrive/expert_extractors/vggt_extractor.py

scripts/download_required_weights.py
scripts/check_required_weights.py
scripts/print_required_weight_paths.py
scripts/create_dummy_expert_cache.py
scripts/build_recogdrive_chunk_cache.py
scripts/check_recogdrive_chunk_cache.py
scripts/train_recogdrive_expert_chunked.py
scripts/eval_recogdrive_expert_pdm.py
scripts/aggregate_recogdrive_expert_results.py
scripts/smoke_test_recogdrive_expert_dummy_flow.py
scripts/smoke_test_recogdrive_agent_dummy_forward.py

configs/weights.yaml
configs/recogdrive2b_expert768_chunk.yaml
configs/recogdrive2b_expert768_warmup.yaml
configs/recogdrive2b_expert768_il.yaml
configs/recogdrive2b_expert768_eval.yaml
configs/ablations/recogdrive2b_baseline_same_subset.yaml
configs/ablations/recogdrive2b_jepa_only.yaml
configs/ablations/recogdrive2b_vggt_only.yaml
configs/ablations/recogdrive2b_jepa_vggt_context.yaml
configs/ablations/recogdrive2b_jepa_vggt_align.yaml
configs/ablations/recogdrive2b_vggt_global_pool.yaml

docs/ExpertTraining.md
```

Modify:

```text
navsim/agents/recogdrive/recogdrive_diffusion_planner.py
navsim/agents/recogdrive/recogdrive_agent.py
navsim/agents/recogdrive/recogdrive_features.py
.gitignore
```

---

## 6. expert_fusion.py design

### 6.1 TeacherTokenProjector

```python
input:  [B, K, teacher_dim]
output: [B, K, 384]

LayerNorm(teacher_dim)
Linear(teacher_dim, 768)
GELU
Linear(768, 384)
LayerNorm(384)
```

Use for:

```text
JEPA: 1024 -> 768 -> 384
VGGT: 2048 -> 768 -> 384
```

### 6.2 ExpertAdapter768

```python
input:  vl_embeds [B, Nv, 384]
output: z_768 [B, 12, 768], z_384 [B, 12, 384]
```

Architecture:

```text
up: Linear(384, 768)
queries: learnable [12, 768]
MultiheadAttention(embed_dim=768, num_heads=12, batch_first=True)
FFN: LayerNorm(768) -> Linear(768, 3072) -> GELU -> Linear(3072, 768)
down: Linear(768, 384)
```

Initialize `down` to zero or near-zero so the new adapter does not strongly perturb Base-IL at the start.

### 6.3 AlignmentHead

```text
JEPA: LayerNorm(768) -> Linear(768, 1024)
VGGT: LayerNorm(768) -> Linear(768, 2048)
```

### 6.4 normalized_mse_loss

```python
pred = F.normalize(pred.float(), dim=-1)
target = F.normalize(target.float().detach(), dim=-1)
loss = F.mse_loss(pred, target)
```

### 6.5 Gates and branch weights

Implement:

```python
init_logit_from_prob(p: float)
```

Defaults:

```text
sigmoid(jepa_gate) = 0.05
sigmoid(vggt_gate) = 0.05
branch weights approx [VLM=0.90, JEPA=0.05, VGGT=0.05]
```

---

## 7. Planner integration

Add config fields to `ReCogDriveDiffusionPlannerConfig`:

```python
use_expert_features: bool = False
expert_adapter_dim: int = 768
use_jepa: bool = True
use_vggt: bool = True
jepa_dim: int = 1024
vggt_dim: int = 2048
num_jepa_tokens: int = 12
num_vggt_tokens: int = 12
use_teacher_context_tokens: bool = True
use_student_latent_adapters: bool = True
use_branch_weighted_mean: bool = True
use_expert_type_embedding: bool = True
expert_dropout: float = 0.10
jepa_gate_init: float = 0.05
vggt_gate_init: float = 0.05
branch_init_vlm: float = 0.90
branch_init_jepa: float = 0.05
branch_init_vggt: float = 0.05
jepa_alignment_weight: float = 0.03
vggt_alignment_weight: float = 0.05
alignment_loss_type: str = "normalized_mse"
allow_future_targets_in_inference: bool = False
```

Add helper methods:

```python
_encode_vlm(vl_features) -> vl_embeds
_build_expert_context(vl_embeds, action_input, training: bool) -> dict
_compute_branch_context_mean(vl_embeds, jepa_all, vggt_all) -> context_mean
_compute_alignment_losses(z_jepa_768, z_vggt_768, action_input) -> dict
_prepare_dit_context(vl_features, action_input, training: bool) -> dict
```

Baseline mode:

```python
vl_embeds = feature_encoder(vl_features)
context_tokens = vl_embeds
context_mean = vl_embeds.mean(dim=1)
```

Expert mode:

```python
vl_embeds = feature_encoder(vl_features)                      # [B, Nv, 384]
jepa_ctx = jepa_projector(action_input.jepa_context_tokens)   # [B, 12, 384]
vggt_ctx = vggt_projector(action_input.vggt_context_tokens)   # [B, 12, 384]

z_jepa_768, z_jepa_384 = jepa_adapter(vl_embeds)
z_vggt_768, z_vggt_384 = vggt_adapter(vl_embeds)

pred_jepa = jepa_alignment_head(z_jepa_768)                   # [B, 12, 1024]
pred_vggt = vggt_alignment_head(z_vggt_768)                   # [B, 12, 2048]

jepa_all = concat([jepa_ctx, z_jepa_384], dim=1)               # [B, 24, 384]
vggt_all = concat([vggt_ctx, z_vggt_384], dim=1)               # [B, 24, 384]

add type embeddings if enabled
apply sigmoid gates
apply expert dropout in training

context_tokens = concat([vl_embeds, jepa_all, vggt_all], dim=1)

vl_mean = vl_embeds.mean(dim=1)
jepa_mean = jepa_all.mean(dim=1)
vggt_mean = vggt_all.mean(dim=1)
branch_w = softmax(branch_logits)
context_mean = branch_w[0] * vl_mean + branch_w[1] * jepa_mean + branch_w[2] * vggt_mean
```

Use `context_mean` where original code used `vl_embeds.mean(1)`. Use `context_tokens` where original code used `vl_embeds` as DiT `encoder_hidden_states`.

Return diagnostics in training:

```text
loss
diffusion_loss
jepa_alignment_loss
vggt_alignment_loss
jepa_gate_value
vggt_gate_value
branch_weight_vlm
branch_weight_jepa
branch_weight_vggt
```

Inference:

- `get_action()` must support expert mode.
- `get_action()` must not require or use `jepa_target_tokens` or `vggt_target_tokens`.
- If target tokens are present during evaluation, ignore them and warn.
- Add a no-future-leakage test with NaN target tokens.

---

## 8. Chunked cache policy

Use chunked cache instead of full global hidden-state cache.

Default chunk cache root:

```text
/mnt/project/VLA-AD/cache/recogdrive_expert_chunks
```

Chunk file layout:

```text
chunk_000000/
  metadata.json
  index.jsonl
  samples/
    <scene_token>.pt
    ...
```

Sample `.pt`:

```python
{
  "scene_token": str,
  "sample_token": str,
  "last_hidden_state": FloatTensor[Nv, 1536],           # optional but preferred
  "jepa_context_tokens": FloatTensor[12, 1024],
  "jepa_target_tokens": FloatTensor[12, 1024],
  "vggt_context_tokens": FloatTensor[12, 2048],
  "vggt_target_tokens": FloatTensor[12, 2048],
}
```

`metadata.json`:

```json
{
  "version": "recogdrive2b_expert768_chunk_v1",
  "is_dummy": false,
  "contains_vlm_hidden": true,
  "contains_jepa": true,
  "contains_vggt": true,
  "target_tokens_are_train_only": true
}
```

`build_recogdrive_chunk_cache.py` must support:

```text
--data-root /mnt/navsim
--project-root /mnt/project/VLA-AD
--split navtrain | navtest
--chunk-index
--chunk-size
--output-dir
--build-vlm-hidden
--build-jepa
--build-vggt
--recogdrive-vlm-path
--jepa-model-path
--vggt-model-path
--precision bf16 | fp16 | fp32
--device cuda
--num-gpus 8
--max-samples
--overwrite
--resume
--log-every
```

Default first real debug chunk:

```text
chunk_size = 1024
```

Do not limit the project to 128-sample smoke tests. Use 128 only for debugging. Real training should use chunk sizes such as 1024, 2048, or 4096 depending on disk.

---

## 9. Training requirements

Training must proceed beyond smoke tests.

### 9.1 Required phases

1. **Smoke gate**: dummy tensors, no NAVSIM, no weights.
2. **Weight check**: verify all required weights are present.
3. **Debug chunk**: build 128 or 1024 samples to validate real data flow.
4. **Tiny overfit**: train on 128 or 1024 samples to verify gradients and save/load.
5. **Warmup training**: train expert modules on multiple chunks.
6. **Main IL training**: train expert modules + action_head from Base-IL.
7. **Evaluation**: evaluate on navtest subset and full navtest if feasible.
8. **Ablation**: run at least baseline same-subset, JEPA-only, VGGT-only, JEPA+VGGT, JEPA+VGGT+alignment.
9. **Conclusion report**: aggregate metrics and write a Markdown result summary.

### 9.2 Parameter groups

Expert modules:

```text
jepa_projector
vggt_projector
jepa_adapter
vggt_adapter
alignment heads
gates
branch logits
type embeddings
```

Default:

```text
lr_expert = 1e-4
```

Original action_head modules:

```text
feature_encoder
his_traj_encoder
ego_status_encoder
action_encoder
fusion_projector
DiT
action_decoder
```

Default main IL:

```text
lr_action_head = 2e-5
```

VLM:

```text
frozen by default
```

### 9.3 Warmup

```text
train expert modules only
lr_expert = 1e-4
lr_action_head = 0
jepa_alignment_weight = 0.10
vggt_alignment_weight = 0.10
steps: at least 1000 unless dataset/chunk is too small
```

### 9.4 Main IL

```text
train expert modules + action_head
lr_expert = 1e-4
lr_action_head = 2e-5
jepa_alignment_weight = 0.03
vggt_alignment_weight = 0.05
expert_dropout = 0.10
jepa_gate_init = 0.05
vggt_gate_init = 0.05
branch weights initial = [0.90, 0.05, 0.05]
```

---

## 10. Evaluation and valid conclusions

The experiment must produce interpretable results, not just a successful run.

Minimum evaluation set:

```text
navtest subset: at least 256 samples
full navtest: run when time permits
```

Minimum ablations:

```text
A0: Base-IL same-subset baseline
A1: JEPA only
A2: VGGT only
A3: JEPA + VGGT context only
A4: JEPA + VGGT + adapters + alignment
A5: VGGT 3x4 pooling vs VGGT global 1x1 pooling
```

All ablations must use the same evaluation subset before claiming improvement.

Metrics to aggregate if available:

```text
PDMS
NC
DAC
TTC
Comfort
EP / progress
trajectory L1/L2 if available
```

Create:

```text
scripts/aggregate_recogdrive_expert_results.py
```

It must output:

```text
results.csv
results.md
```

The final `results.md` must state:

```text
which checkpoint was used
which chunk/cache strategy was used
evaluation split and sample count
whether target teacher tokens were disabled in eval
metrics per ablation
best model
failure cases or regressions
```

---

## 11. Required commands to document

Document these in `docs/ExpertTraining.md`.

### 11.1 Environment

```bash
cd /mnt/project/VLA-AD

export PROJECT_ROOT=/mnt/project/VLA-AD
export NAVSIM_DATA_ROOT=/mnt/navsim
export OPENSCENE_DATA_ROOT=/mnt/navsim/openscene
export NUPLAN_MAPS_ROOT=/mnt/navsim/maps

export CHECKPOINT_ROOT=/mnt/project/VLA-AD/checkpoints
export EXPERIMENT_ROOT=/mnt/project/VLA-AD/experiments/recogdrive_expert
export CHUNK_CACHE_ROOT=/mnt/project/VLA-AD/cache/recogdrive_expert_chunks
export HF_HOME=/mnt/project/VLA-AD/hf_home
export HF_HUB_CACHE=/mnt/project/VLA-AD/hf_home/hub
export HF_XET_CACHE=/mnt/project/VLA-AD/hf_home/xet
export MODELSCOPE_CACHE=/mnt/project/VLA-AD/modelscope_cache
```

### 11.2 Download weights, mirror first, no proxy by default

```bash
cd /mnt/project/VLA-AD
python scripts/download_required_weights.py \
  --output-root /mnt/project/VLA-AD/checkpoints \
  --source auto \
  --repo-workers 4 \
  --file-workers 8
```

If slow, use original source with proxy:

```bash
cd /mnt/project/VLA-AD
python scripts/download_required_weights.py \
  --output-root /mnt/project/VLA-AD/checkpoints \
  --source hf-original \
  --proxy http://127.0.0.1:7890 \
  --repo-workers 4 \
  --file-workers 8
```

### 11.3 Check weights

```bash
cd /mnt/project/VLA-AD
python scripts/check_required_weights.py \
  --root /mnt/project/VLA-AD/checkpoints
```

### 11.4 Smoke gate

```bash
cd /mnt/project/VLA-AD
python scripts/smoke_test_recogdrive_expert_dummy_flow.py \
  --device cuda \
  --use-expert-features \
  --expert-adapter-dim 768
```

### 11.5 Build debug chunk

```bash
cd /mnt/project/VLA-AD
python scripts/build_recogdrive_chunk_cache.py \
  --data-root /mnt/navsim \
  --project-root /mnt/project/VLA-AD \
  --split navtrain \
  --chunk-index 0 \
  --chunk-size 1024 \
  --output-dir /mnt/project/VLA-AD/cache/recogdrive_expert_chunks/chunk_000000 \
  --build-vlm-hidden \
  --build-jepa \
  --build-vggt \
  --recogdrive-vlm-path /mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B \
  --jepa-model-path /mnt/project/VLA-AD/checkpoints/teachers/vjepa2-vitl-fpc64-256 \
  --vggt-model-path /mnt/project/VLA-AD/checkpoints/teachers/VGGT-1B \
  --precision bf16 \
  --num-gpus 8
```

### 11.6 Tiny overfit on real NAVSIM chunk

```bash
cd /mnt/project/VLA-AD
python scripts/train_recogdrive_expert_chunked.py \
  --config configs/recogdrive2b_expert768_chunk.yaml \
  --base-il-checkpoint /mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-IL \
  --chunk-cache-dir /mnt/project/VLA-AD/cache/recogdrive_expert_chunks/chunk_000000 \
  --max-samples 1024 \
  --batch-size 2 \
  --num-steps 500 \
  --train-expert-only \
  --freeze-base-action-head \
  --precision bf16 \
  --debug-overfit \
  --output-dir /mnt/project/VLA-AD/experiments/recogdrive_expert/tiny_overfit
```

### 11.7 Warmup

```bash
cd /mnt/project/VLA-AD
python scripts/train_recogdrive_expert_chunked.py \
  --config configs/recogdrive2b_expert768_warmup.yaml \
  --base-il-checkpoint /mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-IL \
  --chunk-cache-root /mnt/project/VLA-AD/cache/recogdrive_expert_chunks \
  --auto-build-next-chunk \
  --delete-old-chunk \
  --num-chunks 8 \
  --epochs-per-chunk 1 \
  --batch-size 2 \
  --gradient-accumulation-steps 2 \
  --train-expert-only \
  --lr-expert 1e-4 \
  --lr-action-head 0 \
  --jepa-align-weight 0.10 \
  --vggt-align-weight 0.10 \
  --precision bf16 \
  --output-dir /mnt/project/VLA-AD/experiments/recogdrive_expert/warmup
```

### 11.8 Main IL training

```bash
cd /mnt/project/VLA-AD
python scripts/train_recogdrive_expert_chunked.py \
  --config configs/recogdrive2b_expert768_il.yaml \
  --resume-from /mnt/project/VLA-AD/experiments/recogdrive_expert/warmup/latest.ckpt \
  --chunk-cache-root /mnt/project/VLA-AD/cache/recogdrive_expert_chunks \
  --auto-build-next-chunk \
  --delete-old-chunk \
  --num-chunks 32 \
  --epochs-per-chunk 1 \
  --batch-size 2 \
  --gradient-accumulation-steps 2 \
  --lr-expert 1e-4 \
  --lr-action-head 2e-5 \
  --jepa-align-weight 0.03 \
  --vggt-align-weight 0.05 \
  --precision bf16 \
  --output-dir /mnt/project/VLA-AD/experiments/recogdrive_expert/main_il
```

### 11.9 Evaluation subset

```bash
cd /mnt/project/VLA-AD
python scripts/eval_recogdrive_expert_pdm.py \
  --config configs/recogdrive2b_expert768_eval.yaml \
  --checkpoint /mnt/project/VLA-AD/experiments/recogdrive_expert/main_il/best.ckpt \
  --data-root /mnt/navsim \
  --split navtest \
  --feature-source chunk_or_online \
  --max-samples 256 \
  --output-dir /mnt/project/VLA-AD/experiments/recogdrive_expert/eval_navtest_256
```

### 11.10 Aggregate results

```bash
cd /mnt/project/VLA-AD
python scripts/aggregate_recogdrive_expert_results.py \
  --input-root /mnt/project/VLA-AD/experiments/recogdrive_expert \
  --output-csv /mnt/project/VLA-AD/experiments/recogdrive_expert/results.csv \
  --output-md /mnt/project/VLA-AD/experiments/recogdrive_expert/results.md
```

---

## 12. Quality gates

Before declaring the project successful, ensure:

1. `use_expert_features=False` still runs the original code path.
2. Official Base-IL weights load into the modified model with only expected missing expert keys.
3. Dummy smoke tests pass.
4. Real NAVSIM chunk cache builds successfully.
5. Tiny overfit runs with finite losses and gradients.
6. Warmup produces finite alignment losses and saved checkpoints.
7. Main IL training runs beyond a debug subset.
8. Evaluation is performed with target teacher tokens disabled.
9. At least A0–A4 ablations are evaluated on the same subset.
10. `results.md` contains a clear conclusion and failure analysis.

---

## 13. Safety and leakage checks

Never use these in evaluation or inference:

```text
jepa_target_tokens
vggt_target_tokens
future cam_f0 frames
future teacher features
```

Implement a test where target tokens are filled with NaN during `get_action()`. The predicted trajectory must remain finite. If NaN target tokens affect inference, the implementation is invalid.

---

## 14. Expected first conclusion target

The first result does not need to beat ReCogDrive-RL. The first valid conclusion should answer:

```text
Does adding JEPA/VGGT expert-token conditioning to ReCogDrive-2B-IL improve or stabilize NAVSIM IL planning metrics on the same evaluation subset?
Which branch contributes more: JEPA temporal-predictive tokens or VGGT road-aware geometry tokens?
Does adapter alignment outperform context-only fusion?
Does road-aware VGGT 3x4 pooling outperform global VGGT pooling?
```

If the model regresses, still produce a valid conclusion by reporting which component caused regression and which ablation is most stable.
