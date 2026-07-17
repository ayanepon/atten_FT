#!/usr/bin/env bash
# Phase-3 optional experiment: second LoRA FT seed for pythia-1b (stability check).
# Full new training run (seed=123, distinct from canonical seed=42) + fixed-20
# re-extraction with the new adapter + eval, kept minimal (no baselines/exp1/exp3)
# since the purpose is only to check proposed_all/proposed_en AUC stability.
set -euo pipefail
cd /remote/homes/user/anonymous_experiments
VENV=/remote/homes/user/implementation/.venv_hosta/bin/python3
RUN_DIR=mimir_lora_pythia1b_seed123
FEATURES_ROOT=attention_features_pythia1b_seed123
SEED=123

export CUDA_VISIBLE_DEVICES=1

echo "[$(TZ=Asia/Tokyo date +%Y-%m-%d\ %H:%M:%S)] Stage 1/3: train (seed=$SEED)"
$VENV orchestrate.py train \
  --model pythia-1b \
  --run-dir "$RUN_DIR" \
  --seed "$SEED" \
  --from-csv-dir mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2/data

echo "[$(TZ=Asia/Tokyo date +%Y-%m-%d\ %H:%M:%S)] Stage 2/3: extract fixed-20 (seed=$SEED)"
$VENV orchestrate.py extract \
  --model pythia-1b \
  --run-dir "$RUN_DIR" \
  --features-root "$FEATURES_ROOT" \
  --fixed-steps 20 \
  --n-per-group 500 \
  --seed "$SEED" \
  --gpus 1 \
  --wait

echo "[$(TZ=Asia/Tokyo date +%Y-%m-%d\ %H:%M:%S)] Stage 3/3: eval"
$VENV run_strict_fixed20_comparison_10runs.py \
  --models pythia1b \
  --methods proposed_all proposed_en \
  --pythia1b-proposed-root "$FEATURES_ROOT" \
  --output-dir results/second_seed123_pythia1b

echo "[$(TZ=Asia/Tokyo date +%Y-%m-%d\ %H:%M:%S)] SECOND-SEED PIPELINE FINISHED"
