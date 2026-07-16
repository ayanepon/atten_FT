#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="src/proposed:${PYTHONPATH:-}"

# Run these three commands on separate GPUs if available.
python src/proposed/experiment4_gptneo27b_fixed20_ft.py
python src/proposed/experiment4_gptneo27b_fixed20_pt.py
python src/proposed/experiment4_gptneo27b_fixed20_unseen.py
