# ReCogDrive Training and Evaluation

## Stage 1: Vision-Language Models Driving Pretraining

First, you need to download **13 QA datasets** (e.g., *DriveLM*, *LingoQA*, etc.) as mentioned in the paper.  
Due to dataset privacy policies, we are currently unable to release the JSON files. These files may be released later if permission is granted by the dataset authors. Once obtained, you should configure the corresponding JSON files under `./internvl_chat/shell/data_info`.

You can also generate the **ReCogDrive dataset on NAVSIM** following the steps below:

```bash
cd ./scripts
sh generate_dataset/generate_internvl_dataset.sh              # trajectory dataset
sh generate_dataset/generate_internvl_dataset_pipeline.sh     # auto-labeled dataset with pipeline
```
Note: Before running the pipeline script, you need to deploy the corresponding VLM using vllm or Sglang for automatic generation.

Next, download the **InternVL pretrained weights** from HuggingFace:  
👉 [InternVL3-2B Weights](https://huggingface.co/OpenGVLab/InternVL3-2B)
👉 [InternVL3-8B Weights](https://huggingface.co/OpenGVLab/InternVL3-8B)

After downloading, go to `./internvl_chat/shell/internvl3.0/2nd_finetune` and configure the training script.  
You can launch the pretraining process with the following commands:

```bash
cd /path/to/internvl_chat
sh ./shell/internvl3.0/2nd_finetune/internvl3_8b_dynamic_res_2nd_finetune_recogdrive_pretrain.sh
```


## Stage 2: Diffusion Planner Imitation Learning

You can download our pretrained **ReCogDrive VLM** from [ReCogDrive VLM](https://huggingface.co/collections/owl10/recogdrive-68bafa143de172bab8de5752).  

For reproducible local Stage2 work in this repository, use the official-aligned training path documented in [OfficialAlignedBaselineGuardrails.md](OfficialAlignedBaselineGuardrails.md). The verified A0 baseline uses `navsim/planning/script/run_training_recogdrive.py`, PyTorch Lightning mixed precision with fp32 model weights, official train/val log split, AdamW `1e-4`, `WarmupCosLR`, and full navtest fp32 evaluation.

For the diffusion planner training, the first step is to **cache datasets for faster training**.  
Since DiT training converges relatively slowly, training VLM and DiT jointly can be very time-consuming. To accelerate, we cache the hidden states output by the VLM, which enables much faster training.  
> ⚠️ Note: Caching requires approximately **1–2 TB of disk space**. We are also working on faster training methods.  


### Step 1: Cache hidden states
```bash
# cache dataset for training
sh cache_dataset/run_caching_recogdrive_hidden_state.sh
```

Optional expert-token caches for JEPA/VGGT conditioning can be generated before planner training. The initial script includes a deterministic `dummy` backend for smoke tests; replace `--teacher-backend dummy` with a real backend implementation once JEPA/VGGT loaders are wired.

For a tiny standalone cache-loading smoke test before NAVSIM/teacher models are available, use:

```bash
python scripts/create_dummy_expert_cache.py \
  --output-dir /tmp/recogdrive_dummy_expert_cache \
  --num-samples 16 \
  --include-targets
PYTHONDONTWRITEBYTECODE=1 python scripts/testing/smoke_recogdrive_dummy_expert_cache.py
```

Dummy caches are marked with `metadata.json` field `is_dummy=true`. They are only for computation smoke tests: ReCogDrive training fails on dummy caches unless `allow_dummy_expert_cache=true` or `ALLOW_DUMMY_EXPERT_CACHE=true` is set, and evaluation emits a loud warning.

No-data/no-weights smoke mode is available for checking the expert-token code before NAVSIM data, real images, ReCogDrive checkpoints, JEPA checkpoints, or VGGT checkpoints are installed. Use `checkpoint_path=null` or `checkpoint_path=''`, keep `allow_random_init=true`, and set `expert_feature_source="dummy"` to synthesize deterministic current expert tokens. This validates computation flow only; random initialization and dummy features must not be used for performance claims, and no benchmark metrics should be reported from dummy mode.

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/smoke_test_recogdrive_expert_dummy_flow.py --device cpu
PYTHONDONTWRITEBYTECODE=1 python scripts/smoke_test_recogdrive_agent_dummy_forward.py
PYTHONDONTWRITEBYTECODE=1 python scripts/testing/smoke_recogdrive_dummy_expert_cache.py
PYTHONDONTWRITEBYTECODE=1 python scripts/testing/smoke_recogdrive_checkpoint_loading.py
PYTHONDONTWRITEBYTECODE=1 python scripts/debug_train_recogdrive_expert_on_dummy_data.py --device cpu --steps 1 --use-expert-features --use-alignment-loss
```

```bash
python navsim/planning/script/run_recogdrive_expert_feature_caching.py \
  --split trainval \
  --scene-filter navtrain \
  --output-cache-dir ${NAVSIM_EXP_ROOT}/expert_cache_recogdrive \
  --jepa-checkpoint-or-model-path /path/to/jepa \
  --vggt-checkpoint-or-model-path /path/to/vggt \
  --batch-size 8 \
  --num-workers 8 \
  --num-jepa-tokens 4 \
  --num-vggt-tokens 4 \
  --device cuda \
  --precision fp16 \
  --compute-future-targets \
  --max-samples 32 \
  --teacher-backend dummy
```

Expert-token ReCogDrive variants use the same VLM hidden-state cache as the baseline. Run them in this order:

Current expert tokens (`jepa_tokens`, `vggt_tokens`) are allowed at both train and inference time. Future expert target tokens (`jepa_target_tokens`, `vggt_target_tokens`) are train-only auxiliary supervision for alignment losses; feature loading must explicitly enable them, and evaluation/inference ignores them with a warning if they are accidentally present.

1. Build the VLM hidden cache if needed:
```bash
sh scripts/cache_dataset/run_caching_recogdrive_hidden_state.sh
```

2. Build the JEPA/VGGT expert cache:
```bash
export NAVSIM_EXP_ROOT=/path/to/NAVSIM/exp
export OPENSCENE_DATA_ROOT=/path/to/NAVSIM/dataset
export RECOGDRIVE_EXPERT_CACHE_DIR=${NAVSIM_EXP_ROOT}/recogdrive_expert_cache
export RECOGDRIVE_JEPA_MODEL_PATH=/path/to/jepa
export RECOGDRIVE_VGGT_MODEL_PATH=/path/to/vggt
EXPERT_VARIANT=jepa_vggt sh scripts/cache_dataset/run_caching_recogdrive_expert_features.sh
```

3. Train expert IL. Set `EXPERT_VARIANT` to `baseline`, `jepa`, `vggt`, `jepa_vggt`, or `jepa_vggt_alignment`:
```bash
export RECOGDRIVE_VLM_PATH=/path/to/ReCogDrive-VLM-2B
export RECOGDRIVE_HIDDEN_CACHE_DIR=${NAVSIM_EXP_ROOT}/recogdrive_agent_cache_dir_train_2b
EXPERT_VARIANT=jepa_vggt_alignment sh scripts/training/run_recogdrive_train_multi_node_2b_expert.sh
```

A4-V2 should now be trained on the official-aligned baseline instead of the legacy chunked training loop:

```bash
export RECOGDRIVE_VLM_PATH=/path/to/ReCogDrive-VLM-2B
export CACHE_PATH=/path/to/local_chunk_cache_with_train_val_logs
export TRAIN_TEST_SPLIT=navtrain
export OUTPUT_DIR=/path/to/outputs/a4_v2_official_aligned
export MASTER_PORT=29571
bash scripts/run_a4_v2_official_aligned_8gpu.sh
```

For the align-first variant:

```bash
export RECOGDRIVE_VLM_PATH=/path/to/ReCogDrive-VLM-2B
export CACHE_PATH=/path/to/local_chunk_cache_with_train_val_logs
export TRAIN_TEST_SPLIT=navtrain
export OUTPUT_DIR=/path/to/outputs/a4_v2_official_align_first
export MASTER_PORT=29581
bash scripts/run_a4_v2_official_align_first_8gpu.sh
```

Evaluate A4-V2 checkpoints through the same fp32 PDM evaluator:

```bash
export A4_EVAL_CONFIG=configs/ablations/recogdrive2b_A4_v2.yaml
export CHECKPOINT_DIR=/path/to/outputs/a4_v2_official_aligned
export EVAL_CHUNK_CACHE_ROOT=/path/to/navtest_chunk_cache
export EVAL_CHUNK_NAME_PATTERN='navtest_full_chunk_*'
export METRIC_CACHE_DIR=/path/to/navtest_metric_cache
export EVAL_OUTPUT_DIR=/path/to/eval/a4_v2_official_aligned
bash scripts/eval_a4_v2_official_aligned_checkpoints.sh
```

4. Evaluate expert IL:
```bash
export RECOGDRIVE_EVAL_CHECKPOINT=/path/to/best_il.ckpt
EXPERT_VARIANT=jepa_vggt_alignment sh scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_2b_expert.sh
```

5. Train expert RL from the best IL checkpoint:
```bash
export RECOGDRIVE_IL_CHECKPOINT=/path/to/best_il.ckpt
export RECOGDRIVE_METRIC_CACHE_DIR=${NAVSIM_EXP_ROOT}/metric_cache_train
EXPERT_VARIANT=jepa_vggt_alignment sh scripts/training/run_recogdrive_train_multi_node_rl_2b_expert.sh
```

6. Evaluate expert RL:
```bash
export RECOGDRIVE_EVAL_CHECKPOINT=/path/to/best_rl.ckpt
export RECOGDRIVE_METRIC_CACHE_DIR=${NAVSIM_EXP_ROOT}/metric_cache
EXPERT_VARIANT=jepa_vggt_alignment sh scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_2b_expert.sh
```

7. Aggregate ablation results:
```bash
python scripts/evaluation/aggregate_recogdrive_expert_results.py \
  baseline=${NAVSIM_EXP_ROOT}/recogdrive_baseline_expert_eval \
  jepa=${NAVSIM_EXP_ROOT}/recogdrive_jepa_expert_eval \
  vggt=${NAVSIM_EXP_ROOT}/recogdrive_vggt_expert_eval \
  jepa_vggt=${NAVSIM_EXP_ROOT}/recogdrive_jepa_vggt_expert_eval \
  jepa_vggt_alignment=${NAVSIM_EXP_ROOT}/recogdrive_jepa_vggt_alignment_expert_eval \
  --output-csv ${NAVSIM_EXP_ROOT}/recogdrive_expert_ablation_results.csv \
  --output-md ${NAVSIM_EXP_ROOT}/recogdrive_expert_ablation_results.md
```

### Step 2: Configure and run training

Configure the script `training/run_recogdrive_train_multi_node_2b.sh` and then start training:

```bash
sh training/run_recogdrive_train_multi_node_2b.sh
```

You can also enable **EMA (Exponential Moving Average)** during training for faster convergence. Note that this may lead to very slight performance degradation.

```bash
sh training/run_recogdrive_train_multi_node_ema_2b.sh
```

### Step 3: Configure and Run Evaluation

After training is complete, you can configure the evaluation script and launch evaluation:

```bash
sh evaluation/run_recogdrive_agent_pdm_score_evaluation_2b.sh
```

This will evaluate your trained agent using **PDM scores** on the navtest.




## Stage 3: Diffusion Planner Reinforcement Learning Training

In this stage, we perform **reinforcement learning (RL) training** on the Diffusion Planner  to further improve planning performance.

### Step 1: Metric Caching

First, you need to cache metrics for the training and test sets, which will be used for evaluation during RL training.

> ⚠️ **Note:** As mentioned in [Issue #10](https://github.com/xiaomi-research/recogdrive/issues/10#issuecomment-3344730681), you **must use NumPy version 1.26.4 or above** to avoid potential errors during metric caching.

```bash
# cache metrics for navtrain
sh cache_dataset/run_metric_caching_train.sh

# cache metrics for navtest
sh cache_dataset/run_metric_caching.sh
```


### Step 2: Configure and Launch RL Training

After caching metrics, configure the RL training script and launch training:

```bash
# Example path to the RL training script
sh training/run_recogdrive_train_multi_node_rl_2b.sh
```

Before running, modify the script parameters as needed  according to your hardware and training requirements. This command will start RL training immediately after configuration.


### Step 3: Configure and Run Evaluation

After training is complete, you can configure the evaluation script and launch evaluation:

```bash
sh evaluation/run_recogdrive_agent_pdm_score_evaluation_2b.sh
```
This will evaluate your trained agent using **PDM scores** on the navtest.
