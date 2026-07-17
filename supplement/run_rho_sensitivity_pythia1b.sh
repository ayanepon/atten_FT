#!/usr/bin/env bash
# Phase-3 optional experiment: query rho sensitivity (5% / 20%) for pythia-1b.
# Paper fixes rho=10 and defers sensitivity analysis to future work
# (acl_latex.tex line 2074); reuses the EXISTING canonical adapter (no
# retraining -- rho only changes query-position selection at extraction time)
# and re-runs fixed-20 attention extraction with TOPK_LOSS_PERCENT=5 and =20.
set -euo pipefail
cd /remote/homes/user/anonymous_experiments
VENV=/remote/homes/user/implementation/.venv_hosta/bin/python3
RUN_DIR=mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2
SEED=42

for RHO in 5 20; do
  FEATURES_ROOT="attention_features_mimir_hardsplit_rho${RHO}"
  echo "[$(TZ=Asia/Tokyo date +%Y-%m-%d\ %H:%M:%S)] Extract rho=${RHO}"
  TOPK_LOSS_PERCENT=$RHO $VENV orchestrate.py extract \
    --model pythia-1b \
    --run-dir "$RUN_DIR" \
    --features-root "$FEATURES_ROOT" \
    --fixed-steps 20 \
    --n-per-group 500 \
    --seed "$SEED" \
    --gpus 3 \
    --wait

  echo "[$(TZ=Asia/Tokyo date +%Y-%m-%d\ %H:%M:%S)] Eval rho=${RHO}"
  $VENV run_strict_fixed20_comparison_10runs.py \
    --models pythia1b \
    --methods proposed_all proposed_en \
    --pythia1b-proposed-root "$FEATURES_ROOT" \
    --output-dir "results/rho${RHO}_pythia1b"
done

echo "[$(TZ=Asia/Tokyo date +%Y-%m-%d\ %H:%M:%S)] RHO-SENSITIVITY PIPELINE FINISHED"
