# Two-Expert Slot Stage2 Navtest Summary

Updated: 2026-06-14 04:40 UTC

## Route Design

This round used the `two_expert_slot` route, not the original ReCogDrive route and not the CoWorld-VLA / Last-VLA route.

Core design:

- Stage1 learns VLM-side soft slots:
  - `H_dyn`: three dynamic groups, each with twelve tokens.
  - `H_geo`: twelve geometry tokens.
  - Stage1 checkpoint is compact and loads its VLM LoRA adapter from the Stage1 output directory.
- Stage2 trains the ReCogDrive diffusion planner with cached VLM hidden states plus `H_dyn/H_geo`.
- A0 official-aligned ReCogDrive is used as Stage2 initialization.
- The DiT/action path keeps the raw VLM context and receives horizon HMEF-lite two-expert conditioning.
- The following paths remain disabled for this route:
  - Last-VLA / CoWorld residual diffusion.
  - Last-RD.
  - A4 direct expert injection.
  - action-side CoT.
  - teacher-trajectory residual diffusion.

## Cache

Navtrain hidden cache used for Stage2:

```text
/mnt/project/VLA-AD/cache/two_expert_slot/real_20260612_172446/hidden_navtrain_stage1_lora_bf16
```

Properties:

- `num_samples=103036`
- `num_shards=8`
- `output_dtypes=["bf16"]`
- required keys: `last_hidden_state`, `two_expert_h_dyn`, `two_expert_h_geo`

Navtest hidden cache was generated on local root storage to avoid shared-disk pressure:

```text
/root/vla_ad_navtest_eval/two_expert_slot_stage2_navtest_20260614T004821Z/cache/hidden_navtest_stage1_lora_bf16
```

Properties:

- `num_samples=12146`
- `num_pdm_valid=12138` in full PDMS eval
- full eval summary:

```text
/root/vla_ad_navtest_eval/two_expert_slot_stage2_navtest_20260614T004821Z/pdms_summary.csv
```

Shared-disk cleanup check:

- No obsolete shared `lora` hidden cache was found.
- The only shared `lora` hidden cache was the active Stage2 navtrain cache above.
- Therefore no shared `lora` cache was deleted.

## Stage2 Training Configuration

Active run:

```text
/mnt/project/VLA-AD/outputs/two_expert_slot_stage2_full_dit_sft_A0init_collatefix_4gpu_eqbs128_20260613T210233Z
```

Important settings:

- `+experiment=two_expert_slot_stage2_dit_sft`
- `use_two_expert_slots=true`
- `two_expert_cache_mode=true`
- `two_expert_slot_mode=vlm_soft_slots`
- `two_expert_condition_mode=horizon_hmef_lite`
- `two_expert_dit_condition_mode=horizon_hmef_lite`
- `use_last_vla=false`
- `use_last_rd=false`
- `use_expert_features=false`
- `last_vla_use_residual_diffusion=false`
- init checkpoint: A0 official-aligned `step_00100000.ckpt`
- per-GPU batch: `16`
- devices: `4`
- gradient accumulation: `2`
- effective batch: `16 * 4 * 2 = 128`
- precision: `bf16-mixed`
- base action-head LR: `3e-5`
- two-expert branch LR: `1e-4`

The 4-GPU launch was used because only GPUs `4,5,6,7` were available at launch time. Gradient accumulation preserved the intended effective batch of the original 8-GPU plan.

## Training Outcome

The run was stopped manually at about epoch 74 after the validation curve and navtest PDMS showed no benefit from continuing toward `max_epochs=200`.

TensorBoard evidence at stop:

- best validation loss: `0.009247680194675922` at `step=9295`
- latest observed validation loss: `0.012501964345574379` at `step=49135`
- latest observed epoch: `74`

## Navtest PDMS

All checkpoints below were first backed up to local root storage and then evaluated from the backup copies, so later training checkpoint replacement could not affect the results.

| checkpoint | PDMS | NC | DAC | TTC | Comfort | EP | DDC | L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `epoch=5-step=3984` | 0.8598003074 | 0.9828637337 | 0.9405997693 | 0.9458724666 | 0.9996704564 | 0.8014765495 | 0.9775910364 | 0.2671048387 |
| `epoch=13-step=9296` | 0.8584142020 | 0.9807217004 | 0.9434832757 | 0.9394463668 | 0.9995056846 | 0.8029378990 | 0.9786208601 | 0.2674045129 |
| `epoch=11-step=7968` | 0.8582676908 | 0.9774674576 | 0.9433185039 | 0.9347503707 | 0.9995056846 | 0.8079670275 | 0.9784972813 | 0.2711798275 |
| `epoch=6-step=4648` | 0.8480722936 | 0.9788268248 | 0.9331850387 | 0.9349975284 | 0.9998352282 | 0.7954898618 | 0.9780441588 | 0.2675564728 |
| `last` | 0.8460439827 | 0.9781265447 | 0.9303015324 | 0.9340088977 | 0.9996704564 | 0.7957166864 | 0.9791151755 | 0.2666488028 |
| `epoch=16-step=11288` | 0.8455636008 | 0.9785796672 | 0.9303839183 | 0.9318668644 | 0.9995880705 | 0.7965656719 | 0.9792799473 | 0.2681041301 |

## Interpretation

- Selecting by navtest PDMS gives `epoch=5-step=3984`.
- Selecting strictly by validation loss gives `epoch=13-step=9296`.
- Late training regressed both validation loss and navtest PDMS.
- The method-specific Stage2 config is valid enough to run and evaluate, but `max_epochs=200` was too long for this branch. The pushed default is capped at `20` epochs.

Recommended next branch:

- Keep A0 initialization.
- Keep the two-expert hidden cache and `horizon_hmef_lite` conditioning.
- Keep effective batch `128` unless batch size is the explicit ablation target.
- Cap Stage2 around the early useful regime: default `max_epochs=20`, `scheduler_epochs=20`, `scheduler_warmup_epochs=2`.
- Evaluate early checkpoints on navtest PDMS instead of reporting late `last.ckpt`.
- Do not enable Last-VLA/CoWorld residual diffusion, Last-RD, A4 direct expert injection, or action-side CoT in this branch.
