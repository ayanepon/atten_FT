# GitHub Experiment Code Summary

This file summarizes the repository contents for reproducing the experiments.
All paths are relative to the repository root.

## Recommended Upload Directory

Upload the whole repository directory:

```text
anonymous_github_experiment_code/
  README.md
  README_JA.md
  BASELINE_FIDELITY.md
  requirements.txt
  configs/
  data/
  models/
  results/
  scripts/
  supplement/
  src/
```

Do not upload large model checkpoints or generated result CSVs unless the
repository is intended to include artifacts via Git LFS.

## Core Proposed Method

The proposed method code is under:

```text
src/proposed/
```

Important files:

```text
src/proposed/mimir_hardsplit_attention_common.py
src/proposed/experiment4_mimir_hardsplit_stopping_condition.py
src/proposed/experiment4_gptneo27b_fixed20_common.py
src/proposed/experiment4_gptneo27b_fixed20_ft.py
src/proposed/experiment4_gptneo27b_fixed20_pt.py
src/proposed/experiment4_gptneo27b_fixed20_unseen.py
src/proposed/experiment4_pythia410m_fixed20_common.py
src/proposed/experiment4_pythia410m_fixed20_ft.py
src/proposed/experiment4_pythia410m_fixed20_pt.py
src/proposed/experiment4_pythia410m_fixed20_unseen.py
src/proposed/run_pythia1b_stopping_conditions.py
src/proposed/run_pythia410m_stopping_conditions.py
src/proposed/run_gptneo27b_stopping_conditions.py
```

## Supplementary Original Pipeline

Additional code used for the paper experiments is included under:

```text
supplement/
```

This directory contains the original paper pipeline, orchestration utilities,
robustness checks, reviewer-follow-up experiments, and CPU-only tests. See:

```text
supplement/README.md
supplement/PAPER_ALIGNMENT.md
supplement/STRUCTURE.md
```

## Training Code

Training code is under:

```text
src/train/
```

Important files:

```text
src/train/train_mimir_wikipedia_hardsplit_lora.py
src/train/train_mimir_wikipedia_hardsplit_lora_pythia410m.py
src/train/train_mimir_wikipedia_hardsplit_lora_gptneo27b.py
```

All model variants should use the same MIMIR hard-split CSV files.

## Analysis Code

Analysis code is under:

```text
src/analysis/
```

Important files:

```text
src/analysis/analyze_mimir_fixed_steps_repeated_auc.py
src/analysis/compare_fixedstep_proposed_baselines_strict.py
src/analysis/compare_proposed_attenmia_loraleak_10runs.py
src/analysis/run_strict_fixed20_3model_comparison_10runs.py
src/analysis/evaluate_loss_direction_selected_3model.py
```

These scripts compute AUC, AUPRC, TPR@FPR, repeated-run summaries, and
comparison tables.

## Baseline Code

Baseline code is under:

```text
src/baselines/
```

Included baselines:

```text
src/baselines/run_attenmia_official_mimir_hardsplit.py
src/baselines/run_attenmia_official_mimir_hardsplit_pythia410m.py
src/baselines/run_attenmia_official_mimir_hardsplit_gptneo27b.py
src/baselines/run_lora_leak_official_mimir_hardsplit.py
src/baselines/run_lora_leak_official_mimir_hardsplit_pythia410m.py
src/baselines/run_lora_leak_official_mimir_hardsplit_gptneo27b.py
src/baselines/compare_mink_strict_fixedstep_10runs.py
```

## Canonical Data

All experiments should use the same split files:

```text
data/mimir_hardsplit/
  mimir_wikipedia_pt_member.csv
  mimir_wikipedia_ft_nonmember.csv
  mimir_wikipedia_unseen_nonmember.csv
```

These files may be omitted from GitHub if redistribution is not allowed. In
that case, keep `.gitkeep` and document how to place the files locally.

## Main Reproduction Order

Run from inside the repository root.

```bash
pip install -r requirements.txt
bash scripts/00_prepare_splits.sh
bash scripts/01_train_pythia410m.sh
bash scripts/01_train_gptneo27b.sh
bash scripts/02_extract_fixed20_gptneo27b.sh
bash scripts/03_analyze_gptneo27b.sh
bash scripts/04_run_baselines.sh
```

For Pythia-1B and Pythia-410M, use the corresponding model/checkpoint paths in
the Python wrappers or analysis scripts.

## Do Not Upload

Avoid uploading:

```text
__pycache__/
.DS_Store
*.pyc
large model checkpoint directories
large raw result CSVs
temporary plot folders
personal local paths outside this repository
```

If large files are required, use Git LFS or provide download instructions.
