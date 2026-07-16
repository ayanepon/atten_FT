#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="src/baselines:${PYTHONPATH:-}"

# AttenMIA-style attention baseline.
python src/baselines/run_attenmia_official_mimir_hardsplit.py \
  --run-dir models/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2 \
  --output-dir results/attenmia_official_mimir_hardsplit

# LoRA-Leak / Min-k%++ scores.
python src/baselines/run_lora_leak_official_mimir_hardsplit.py \
  --run-dir models/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2 \
  --output-dir results/lora_leak_official_mimir_hardsplit

# Plain Min-k% repeated-fold evaluation from saved LoRA-Leak scores.
python src/baselines/compare_mink_strict_fixedstep_10runs.py \
  --output-dir results/min_k_strict_fixedstep_10runs
