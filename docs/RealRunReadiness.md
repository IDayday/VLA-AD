# Real Run Readiness

Date: 2026-05-24
Project root: `/mnt/project/VLA-AD`

## Files Checked

- `navsim/agents/recogdrive/expert_fusion.py`
- `navsim/agents/recogdrive/expert_cache.py`
- `navsim/agents/recogdrive/expert_feature_provider.py`
- `navsim/agents/recogdrive/expert_extractors/pooling.py`
- `navsim/agents/recogdrive/expert_extractors/vjepa2_extractor.py`
- `navsim/agents/recogdrive/expert_extractors/vggt_extractor.py`
- `navsim/agents/recogdrive/recogdrive_diffusion_planner.py`
- `navsim/agents/recogdrive/recogdrive_agent.py`
- `navsim/agents/recogdrive/recogdrive_features.py`
- `scripts/download_required_weights.py`
- `scripts/check_required_weights.py`
- `scripts/resolve_weight_paths.py`
- `scripts/check_real_run_environment.py`
- `scripts/inspect_navsim_real_data.py`
- `scripts/build_recogdrive_chunk_cache.py`
- `scripts/check_recogdrive_chunk_cache.py`
- `scripts/check_real_checkpoint_loading.py`
- `scripts/train_recogdrive_expert_chunked.py`
- `scripts/eval_recogdrive_expert_pdm.py`
- `scripts/run_recogdrive_expert_ablation_plan.py`
- `scripts/aggregate_recogdrive_expert_results.py`
- `scripts/test_no_future_leakage_real.py`
- `configs/recogdrive2b_expert768_chunk.yaml`
- `configs/recogdrive2b_expert768_warmup.yaml`
- `configs/recogdrive2b_expert768_il.yaml`
- `configs/recogdrive2b_expert768_eval.yaml`
- `configs/weights.yaml`
- `configs/ablations/recogdrive2b_A0_base_no_expert.yaml`
- `configs/ablations/recogdrive2b_A1_jepa_only.yaml`
- `configs/ablations/recogdrive2b_A2_vggt_only.yaml`
- `configs/ablations/recogdrive2b_A3_context_only.yaml`
- `configs/ablations/recogdrive2b_A4_full.yaml`
- `configs/ablations/recogdrive2b_A5_vggt_global_pool.yaml`
- `docs/ExpertTraining.md`

## Functions And Classes Checked

- `TeacherTokenProjector`, `ExpertAdapter768`, `AlignmentHead`, `normalized_mse_loss`, `init_logit_from_prob`, `branch_logits_from_probs`
- `ExpertCacheMetadata`, `validate_sample_payload`, `sample_cache_path`, `iter_index`, `load_sample`, `write_index`
- `ChunkExpertFeatureProvider`, `DummyExpertFeatureProvider`, `build_expert_feature_provider`, `move_expert_tensors`
- `pool_vjepa2_tokens`, `pool_vggt_tokens`, `expand_four_frames_to_eight`
- `VJEPA2Extractor.extract_context/extract_target`, `VGGTExtractor.extract_context/extract_target`
- `ReCogDriveDiffusionPlannerConfig`, `_prepare_dit_context`, `_build_expert_context`, `_compute_alignment_losses`, `_compute_branch_context_mean`, `forward`, `get_action`, `sample_chain`, `get_logprobs`
- `ReCogDriveAgent.initialize`, `_safe_load_checkpoint`, `_resolve_checkpoint_path`, `forward`, `get_feature_builders`, `_add_dummy_expert_features_if_needed`
- `ReCogDriveFeatureBuilder.compute_features`, `_load_expert_features`, `_resolve_expert_cache_path`, `assert_real_expert_cache_for_training`, `warn_if_dummy_expert_cache`
- Weight scripts: config loading, source selection, proxy handling, local verification, candidate resolution
- Chunk scripts: NAVSIM path autodetection, split aliasing, cam_f0-only loading, metadata/index writing, shape validation
- Training script: directory checkpoint loading, dummy-cache refusal, chunk pattern selection, finite-loss checks, expert-gradient checks, JSONL/final report/checkpoint writing
- Eval script: checkpoint directory loading, target-token exclusion from `get_action`, finite prediction checks, metrics/prediction writing

## Readiness Answers

1. Real weight paths are supported: yes. `configs/weights.yaml` now defines HF repo, optional ModelScope repo, and local dirs under `/mnt/project/VLA-AD/checkpoints`.
2. `checkpoint_path` can be either a file or a directory: yes. `ReCogDriveAgent` and the standalone train/eval/check scripts recursively resolve checkpoint candidates and prefer IL/model/plausible large files.
3. `/mnt/navsim` is supported as dataset root: yes. Inspection passes for local `navtrain` through `trainval_*` aliases and `navtest` through `test_*` aliases.
4. Chunk cache paths are under `/mnt/project/VLA-AD/cache`: yes. The required root is `/mnt/project/VLA-AD/cache/recogdrive_expert_chunks`.
5. Training/eval output paths are under `/mnt/project/VLA-AD/experiments`: yes. Scripts default/report under `/mnt/project/VLA-AD/experiments/recogdrive_expert`.
6. Dummy cache is blocked in real training: yes. `--debug-overfit` no longer bypasses dummy-cache blocking; only `--allow-dummy-cache` does.
7. Target tokens are blocked during inference: yes. `get_action()` uses context tokens only; eval code drops target tokens and warns when samples contain train-only targets.

## Local Gate Results

- Python compilation passed for touched modules and scripts.
- Environment check passed after installing `pytest`, `transformers`, and `modelscope` into the active env. `vggt` remains missing and is warned as optional until VGGT extraction is requested.
- NAVSIM inspection passed for `navtrain`, `navtest`, and `navmini`; reports are written under `/mnt/project/VLA-AD/experiments/recogdrive_expert/navsim_inspection.md` and `.json`.
- Weight check failed: no complete model checkpoint files are present.
- Weight download with no proxy failed with `Network is unreachable` / DNS errors.
- Weight download with explicit proxy `http://127.0.0.1:7890` started but was too slow to complete in this session and was stopped after partial metadata/incomplete files were written.

## What Still Needs Fixing Or Completing

- Complete all required weight downloads under `/mnt/project/VLA-AD/checkpoints`.
- Install or vendor VGGT code in the existing Python 3.9 environment. Use `bash scripts/install_vggt_py39.sh /path/to/vggt` or set `PYTHONPATH=/path/to/vggt:${PYTHONPATH}`; do not create a Python 3.10 VGGT environment for this project.
- Re-run `scripts/check_required_weights.py` and `scripts/resolve_weight_paths.py` after weights finish.
- Build real `chunk_000000` with VLM hidden, JEPA, and VGGT tokens.
- Run real checkpoint loading, tiny overfit, navtest chunk/eval, pilot training, ablations, leakage check, and final conclusion report.
- Do not claim model quality or improvement until at least one real NAVSIM training run and one real NAVSIM evaluation run complete.
