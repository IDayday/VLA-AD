# Route Status And Pruning Plan

## Active Route

- Current active route: `two_expert_slot` Stage1/Stage2.
- Stage1 aligns VLM-internal dynamic and geometry soft slots with strict JEPA dynamic and VGGT Feature(23) teachers.
- Stage2 keeps the ReCogDrive DiT objective as GT normalized trajectory prediction.
- Residual diffusion, action-side CoT, A4 direct expert paths, and coarse+residual output are not part of this active route.

## Stage3 AWAC/IQL

- Stage3 AWAC/IQL is an experimental downstream route.
- It is disabled by default and must not be run before the two-expert Stage1/Stage2 full evaluation is complete.
- Stage3 launchers require explicit `RUN_STAGE3=1` or `RUN_TRAIN=1`.
- Stage3 outputs must not be mixed into two-expert Stage1/Stage2 readiness or PDMS interpretation.

## Interpretation Rule

- Report two-expert Stage1/Stage2 results separately from Stage3 AWAC/IQL results.
- Do not use Stage3 improvements or regressions to justify two-expert slot architecture quality until the two-expert Stage2 baseline has completed full navtest evaluation.
- A0 official-aligned baseline remains: full navtest PDMS `0.864891`.
