# Strict LaST-VLA ReCogDrive Readiness Report

Date: 2026-06-11

Baseline: A0 official-aligned `step_00100000` full NAVTEST PDMS = `0.864891`.

## Changed Files

- `navsim/agents/recogdrive/last_vla_latent_slots.py`
- `navsim/agents/recogdrive/last_vla_teacher_adapters.py`
- `navsim/agents/recogdrive/last_vla_strict_reasoner.py`
- `navsim/agents/recogdrive/recogdrive_backbone.py`
- `navsim/agents/recogdrive/recogdrive_diffusion_planner.py`
- `navsim/agents/recogdrive/recogdrive_agent.py`
- `configs/last_vla_v2/strict_last_vla/stage1_latent_alignment.yaml`
- `configs/last_vla_v2/strict_last_vla/stage2_diffusion_sft.yaml`
- `scripts/last_vla_v2/strict_last_vla/*`
- `scripts/audit_strict_last_vla_latent_cache.py`
- `tests/test_strict_last_vla_*.py`
- `docs/Strict_LaST_VLA_ReCogDrive_Design.md`

## LaST-VLA Figure Mapping

- Figure 2 latent reasoning tokens: implemented as VLM hidden soft slots `H_dyn`, `H_geo`, `H_plan`.
- Figure 2 dynamics/geometry alignment: implemented by `DynamicsAdapter` and `GeometryAdapter` over random-masked VLM/image hidden tokens plus latent slots.
- Figure 3 progressive SFT: implemented as strict Stage2 cached diffusion SFT with raw VLM tokens as DiT context and projected latent slots as decoupled condition tokens.

## Strict Assumptions

- No A4-V2 direct expert context path.
- No LoRA.
- No hard bottleneck.
- No VLM summary replacement.
- No residual diffusion target.
- No coarse-plus-residual final output.
- Teacher models are train-time only.
- Inference ignores teacher target keys.
- Stage2 diffusion target remains GT normalized trajectory.

## Implementation Status

- VLM latent slots: implemented through soft `inputs_embeds`; strict structured mask requires explicit 4D mask support from the VLM wrapper.
- Dynamics Adapter: implemented, output `[B,128,1024]`, JEPA dev teacher path active.
- Geometry Adapter: implemented, output `[B,192,512]`, VGGT geometry teacher path active.
- Plan latent head: implemented, diagnostic coarse trajectory only.
- Strict reasoner: implemented and used instead of old action-side `LastVLACoTTransformer` when `use_strict_last_vla=true`.
- Stage1 config: added frozen-VLM latent/adapters alignment config.
- Stage2 config: added A0-official-style cached diffusion SFT config.
- Cache schema audit: implemented.

## Teacher Status

Cosmos teacher is not implemented as a formal path yet. The current strict world-model teacher source is `jepa_dev`. If `last_vla_wm_teacher_source=cosmos` is selected without `cosmos_future_features`, strict mode raises `KeyError`.

## Cache Schema

Required strict latent cache keys:

- `last_hidden_state`
- `last_vla_h_dyn`
- `last_vla_h_geo`
- `last_vla_h_plan`
- `last_vla_slot_metadata`
- `trajectory`
- `history_trajectory`
- `high_command_one_hot`
- `status_feature`

Train cache may include teacher targets. Eval/navtest cache must not include future teacher targets unless explicitly flagged, and inference ignores them.

## Validation

Commands run:

```bash
python -m py_compile \
  navsim/agents/recogdrive/last_vla_latent_slots.py \
  navsim/agents/recogdrive/last_vla_teacher_adapters.py \
  navsim/agents/recogdrive/last_vla_strict_reasoner.py \
  navsim/agents/recogdrive/recogdrive_diffusion_planner.py \
  scripts/last_vla_v2/strict_last_vla/build_strict_last_vla_latent_cache.py \
  scripts/audit_strict_last_vla_latent_cache.py
```

Result: passed.

```bash
pytest -q \
  tests/test_strict_last_vla_*.py \
  tests/test_last_vla_*.py \
  tests/test_no_future_leakage.py
```

Result: `109 passed, 2 skipped, 46 warnings`.

## Remaining Blockers

- Full strict latent cache materialization still needs the project-specific NAVSIM dataloader loop wired into `build_strict_last_vla_latent_cache.py`. The script currently provides the gated entrypoint, preflight, and record schema helpers.
- Strict online VLM latent slots require the InternVL wrapper to support `inputs_embeds` and verified 4D attention masks. The code fails clearly if this support is absent.
- Cosmos teacher path is a strict placeholder until real `cosmos_future_features` are available.

## Execution Status

- No training launched.
- No full NAVTEST eval launched.
- Launchers are gated by `RUN_TRAIN=1`, `RUN_CACHE=1`, or `RUN_EVAL=1`.
