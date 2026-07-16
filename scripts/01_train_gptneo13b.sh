#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="src/train:${PYTHONPATH:-}"

python src/train/train_mimir_wikipedia_hardsplit_lora_gptneo13b.py \
  --pythia-split-data-dir data/mimir_hardsplit \
  --output-dir models/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_gptneo13b \
  --learning-rate 1e-4 \
  --num-train-epochs 5
