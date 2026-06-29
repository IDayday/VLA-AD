# AR Answer Prompt4Hist Retrain - 2026-06-27

## Goal

Retrain OneVL AR Answer as the stage1 VLM for the ReCogDrive-style stage2 DiT path.
This round is scoped to data/prompt alignment, fixed-length hidden cache handling,
and explicit preflight checks. Training hyperparameters and PDMS evaluation code
remain aligned with the previous AR Answer run.

## Stage1 Data

- Source train JSONL: `/mnt/project/onevl_navsim_data/navsim_answer_official_paths.jsonl`
- New train JSONL: `/mnt/project/onevl_navsim_data/navsim_answer_prompt4hist_official_paths.jsonl`
- Train report: `/mnt/project/onevl_navsim_data/navsim_answer_prompt4hist_official_paths.report.json`
- Transform script: `scripts/onevl/prepare_ar_answer_prompt4hist_dataset.py`

Applied changes:

- Keep OneVL command text unchanged: `MOVE FORWARD`, `TURN LEFT`, `TURN RIGHT`.
- Keep images unchanged.
- Keep assistant `<answer>...</answer>` trajectory targets unchanged.
- Keep OneVL prompt body, velocity, acceleration, and `<answer>` output format.
- Change history from 3 points to 4 points by appending current state `[0.00, 0.00, 0.00]`.
- Remove fake `<think></think>` requirement from the user prompt because the data has no CoT target.
- Tighten output instruction to answer-only 8 waypoints without extra text or outer list.

Train validation:

- Rows: 103288.
- Images unchanged: 103288.
- Assistant answers unchanged: 103288.
- Commands unchanged: 103288.
- New prompts with 4 history points: 103288.
- New prompts containing `<think>`: 0.

## Navtest Data For Same Evaluation Pipeline

- Source navtest JSON: `/mnt/project/onevl/test_data/navsim_test.json`
- New navtest JSON: `/mnt/project/onevl/test_data/navsim_test_prompt4hist_onevl.json`
- Navtest report: `/mnt/project/onevl/test_data/navsim_test_prompt4hist_onevl.report.json`
- Input alignment report: `/mnt/project/onevl/test_data/navsim_test_prompt4hist_onevl.input_alignment.report.json`

Only the user prompt is converted. Images, GT fields, assistant placeholder content,
test split, metric cache, inference script, submission conversion, and PDM scoring
logic are unchanged.

Navtest validation:

- Rows: 12146.
- Images unchanged: 12146.
- GT unchanged: 12146.
- Commands unchanged: 12146.
- New prompts with 4 history points: 12146.
- Missing local images: 0.
- Missing NAVSIM log images: 0.
- Matched metric-cache tokens: 12146.
- Missing metric-cache tokens: 0.

## Hidden And Stage2 Cache Rules

Bridge script: `scripts/onevl/bridge_ar_answer_to_recogdrive_dit.py`

New cache-related options:

- `--hidden-padding max_length`
- `--hidden-max-length 2800`
- `--hidden-padding-side left`
- `--hidden-truncation false`

Full stage2 cache generation for this experiment must use these settings unless
explicitly changed in a later experiment note. The script records the padding
policy, max length, padding side, truncation flag, `input_ids_shape`,
`hidden_shape`, and `attention_mask_sum` in sample metadata.

Required stage2 target index:

`/mnt/project/VLA-AD/outputs/psi_drive/support_index/stage2_pareto_support_clean_full_gt_supplemented_20260625T162351Z.pt`

Small smoke cache before full generation:

- Output: `/mnt/project/onevl_navsim_exp/ar_answer_prompt4hist_hidden2800_smoke_20260627_155929`
- Samples: 2.
- Hidden shape: `[2800, 2560]`.
- Planner history shape: `[4, 3]`.
- Target source: `stage2_support_index`.
- DiT: small, DDIM.
- DiT forward: passed.

Before full cache generation:

- Run a small cache first with the final trained AR Answer checkpoint.
- Verify every sampled cache has `last_hidden_state=[2800,2560]`.
- Verify `attention_mask_sum <= 2800` and no truncation was used.
- Verify `history_trajectory=[4,3]`, `status_feature=[8]`, `high_command_one_hot=[3]`, and `trajectory=[8,3]`.
- Verify sample token, log name, scene token, image path, and target source are present in metadata.
- Keep cache manifest and summary under the cache output directory.

## Training Run

- Run root: `/mnt/project/onevl_navsim_exp/answer_bs64_prompt4hist_20260627_160112`
- Swift output: `/mnt/project/onevl_navsim_exp/answer_bs64_prompt4hist_20260627_160112/swift_output/v0-20260627-160133`
- Training script: `/mnt/project/OneVL_training/run_script/train/navsim/sft_distributed_qwen3vl_answer_bs64.sh`
- Model init: `/mnt/project/onevl_models/Qwen3-VL-4B-Instruct`
- Dataset: `/mnt/project/onevl_navsim_data/navsim_answer_prompt4hist_official_paths.jsonl`
- GPUs: 8.
- Per-device train batch: 4.
- Gradient accumulation: 2.
- Effective batch size: 64.
- Epochs: 2.
- LR: `4e-5`.
- Scheduler: cosine.
- Loss type: `latent_cot`.
- Other training settings are inherited from the previous AR Answer bs64 script.

## Evaluation Watcher

- Watcher output: `/mnt/project/onevl_navsim_exp/answer_bs64_prompt4hist_20260627_160112/navtest_eval_prompt4hist_latest`
- Watcher script: `scripts/onevl/watch_answer_training_then_eval_latest.sh`
- Inference/eval script: `scripts/onevl/run_ar_answer_latest_infer_eval_full.sh`
- PDM script path is unchanged through `scripts/onevl/run_ar_answer_infer_eval_full.sh`.
- `TEST_SET_PATH`: `/mnt/project/onevl/test_data/navsim_test_prompt4hist_onevl.json`
- The watcher waits for training to finish, validates the latest checkpoint, runs navtest inference, then runs PDM evaluation.
