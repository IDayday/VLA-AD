# RISK-VLA v2 Round2 Dry-Run Report

Method: RISK-VLA v2: Risk-Conditioned Multi-Candidate Strategy Planning for Vision-Language-Action Driving
Max samples: `256`

## Stages
1. discover_inputs
2. build_labels
3. build_candidate_bank
4. train_critic
5. train_router
6. eval_small
7. eval_full_if_ready
8. build_report

## Guardrails
- no navtest/test labels for training
- small pilots before full train/val/navtest
- GRPO disabled until supervised safety gates pass
- shared caches are read-only inputs
