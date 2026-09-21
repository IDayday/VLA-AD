#!/usr/bin/env bash
# Recorded commands, not an auto-retry controller. Campaign is STOPPED.
# Original commands used the same cwd and Python below. Failed invocation logs
# are preserved alongside successful corrected read-only invocations.
set -euo pipefail
cd /mnt/project/VLA-AD-worktrees/grpo-credit-risk-quickcheck-20260921
PROBE_PY=/root/miniconda3/envs/navsim/bin/python
PROBE_DIR=tools/analysis/grpo_credit_risk_quickcheck
PROBE_OUT=outputs/grpo_credit_risk_quickcheck
# Initial isolated worktree creation from root repo:
# git -C /mnt/project/VLA-AD worktree add -b analysis/grpo-credit-risk-quickcheck-20260921 /mnt/project/VLA-AD-worktrees/grpo-credit-risk-quickcheck-20260921 884ea1d0f67c60821d61033d61efb10916d16929
# One-time freeze: refuses to overwrite frozen config.
# "$PROBE_PY" "$PROBE_DIR/prepare.py"
# Smoke performed before formal expansion:
# CUDA_VISIBLE_DEVICES=0 "$PROBE_PY" "$PROBE_DIR/features.py" --smoke
# CUDA_VISIBLE_DEVICES=0 "$PROBE_PY" "$PROBE_DIR/sample.py" --model a5_sft --smoke
# CUDA_VISIBLE_DEVICES=1 "$PROBE_PY" "$PROBE_DIR/sample.py" --model a5_grpo_300 --smoke
# "$PROBE_PY" "$PROBE_DIR/score_nominal.py" --smoke
# "$PROBE_PY" "$PROBE_DIR/verify_smoke.py"
# Formal features: i=0..3, CUDA_VISIBLE_DEVICES=$i features.py --rank $i --world 4
# Formal rollout: i=0..3 on GPU0..3 a5_sft; GPU4..7 a5_grpo_300:
# CUDA_VISIBLE_DEVICES=$gpu "$PROBE_PY" "$PROBE_DIR/sample.py" --model "$model" --rank "$i" --world 4
# "$PROBE_PY" "$PROBE_DIR/score_nominal.py" --workers 32
# "$PROBE_PY" "$PROBE_DIR/credit.py"
# "$PROBE_PY" "$PROBE_DIR/analyze_credit.py"
# "$PROBE_PY" "$PROBE_DIR/matching.py"
# "$PROBE_PY" "$PROBE_DIR/risk_probe.py" --split calibration --workers 24
# "$PROBE_PY" "$PROBE_DIR/risk_probe.py" --split confirmation --workers 24
# "$PROBE_PY" "$PROBE_DIR/analyze_risk.py"
# "$PROBE_PY" "$PROBE_DIR/audit_runtime.py"
# After WITHIN_GROUP_SIGNAL, before updates:
# "$PROBE_PY" "$PROBE_DIR/freeze_tiny.py"
# CUDA_VISIBLE_DEVICES=0 "$PROBE_PY" "$PROBE_DIR/tiny_shared.py" --seed 2026092111
# CUDA_VISIBLE_DEVICES=1 "$PROBE_PY" "$PROBE_DIR/tiny_shared.py" --seed 2026092112
# CUDA_VISIBLE_DEVICES=0 "$PROBE_PY" "$PROBE_DIR/verify_tiny_loss.py"
# Actual eight invalid first-step attempts, one GPU each, no further attempts:
# GPU0 seed2026092111 A; GPU1 seed2026092111 B; GPU2 seed2026092111 C; GPU3 seed2026092111 D
# GPU4 seed2026092112 A; GPU5 seed2026092112 B; GPU6 seed2026092112 C; GPU7 seed2026092112 D
# CUDA_VISIBLE_DEVICES=$gpu "$PROBE_PY" "$PROBE_DIR/tiny_train.py" --seed "$seed" --arm "$arm" --stop-step 1
# NOT_RUN: tiny_train.py --resume --stop-step 8; tiny_norm_control.py; tiny_eval.py
# Read-only failure audit after fixing constructor freeze preservation:
# CUDA_VISIBLE_DEVICES=0 "$PROBE_PY" "$PROBE_DIR/audit_failed_update.py"
# "$PROBE_PY" "$PROBE_DIR/test_probe.py"
# "$PROBE_PY" "$PROBE_DIR/finalize_evidence.py"
# "$PROBE_PY" "$PROBE_DIR/write_report.py"
# Safe default: run only unit tests, never restart a stopped experiment.
"$PROBE_PY" "$PROBE_DIR/test_probe.py"
