#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="src/baselines:${PYTHONPATH:-}"

# Pythia-1B AttenMIA-style attention baseline.
python src/baselines/run_attenmia_official_mimir_hardsplit.py \
  --run-dir models/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2 \
  --output-dir results/attenmia_official_mimir_hardsplit

# Pythia-1B LoRA-Leak / Min-k%++ scores.
python src/baselines/run_lora_leak_official_mimir_hardsplit.py \
  --run-dir models/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2 \
  --output-dir results/lora_leak_official_mimir_hardsplit

# Pythia-410M baselines.
python src/baselines/run_attenmia_official_mimir_hardsplit_pythia410m.py
python src/baselines/run_lora_leak_official_mimir_hardsplit_pythia410m.py

# GPT-Neo-2.7B baselines.
python src/baselines/run_attenmia_official_mimir_hardsplit_gptneo27b.py
python src/baselines/run_lora_leak_official_mimir_hardsplit_gptneo27b.py

# Plain Min-k% repeated-fold evaluation from saved LoRA-Leak scores.
python src/baselines/compare_mink_strict_fixedstep_10runs.py \
  --output-dir results/min_k_strict_fixedstep_10runs
