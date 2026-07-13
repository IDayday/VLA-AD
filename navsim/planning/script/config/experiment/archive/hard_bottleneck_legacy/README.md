# Last-VLA Hard-Bottleneck Legacy Hydra Experiments

These Hydra experiment files are deprecated hard-bottleneck / summary replacement entrypoints.

Do not use them for formal training. The formal SOTA-oriented Hydra experiments are:

- `last_vla_decoupled_cot_alignment_highcap_no_risk`
- `last_vla_decoupled_progressive_highcap_no_risk`
- `last_vla_decoupled_vlm_lora_cot_alignment_highcap_no_risk`

Formal Last-VLA v2 preserves raw VLM tokens and injects latent CoT through the zero-init decoupled residual branch.
