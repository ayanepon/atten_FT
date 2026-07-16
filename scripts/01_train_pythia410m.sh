#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="src/train:${PYTHONPATH:-}"

python src/train/train_mimir_wikipedia_hardsplit_lora_pythia410m.py \
  --output-dir models/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_pythia410m
