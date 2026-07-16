#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="src/analysis:${PYTHONPATH:-}"

python src/analysis/analyze_gptneo27b_fixed20_ft_vs_pt_unseen.py \
  --input-root results/mimir_wikipedia_hardsplit_fixed20_gptneo27b \
  --output-dir results/mimir_wikipedia_hardsplit_fixed20_gptneo27b/ft_vs_pt_unseen_auc_analysis
