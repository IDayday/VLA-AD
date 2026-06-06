# RISK-VLA v3 Round3 10K Results Report

Date: 2026-06-06

## Result Status

No formal >=10000 train/val RISK-VLA v3 experiment has completed in this run.

This is not an experimental evidence report. The only completed real-scale step is input discovery. Downstream 10k label construction, candidate-bank export, critic training, router training, and safealign SFT were blocked by missing explicit train/held-out-val assets.

## Completed

- Full input discovery over project/work roots.
- Manifest generation with strict navtest/test leakage separation.
- v3 code/config/script implementation for utility labels, candidate bank, critic, router, safealign, optional VLM LoRA gate, and disabled anchor-GRPO gate.

## Not Run

- >=10000 train strategy utility labels: blocked.
- >=10000 train candidate bank PDM evaluation: blocked.
- >=10000 critic training/validation: blocked.
- >=10000 router training/validation: blocked.
- >=10000 safealign SFT: blocked.

## Key Discovery Numbers

- Train chunk tokens found: 206072.
- Navtest chunk tokens found: 12146.
- Held-out val chunk tokens found: 1024.
- Formal val >=10000 possible: false.
- Formal train utility labels possible: false.
- Navtest >=10000 analysis possible: true.

## Conclusion

The infrastructure is launchable once explicit matched train PDM/candidate assets and a >=10000 held-out validation set are registered. No PDMS/EPDMS improvement claim is made from this run.
