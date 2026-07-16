# GitHub Experiment Code Summary

This file summarizes the code that should be uploaded for reproducing the
experiments.

## Recommended Upload Directory

Upload the `submission_artifact/` directory as the clean reproduction package.
It already contains the organized source code, scripts, configuration files,
requirements, and README.

```text
submission_artifact/
  README.md
  BASELINE_FIDELITY.md
  requirements.txt
  configs/
  data/
  models/
  results/
  scripts/
  scripts_abs/
  src/
```

Do not upload large model checkpoints or generated result CSVs unless the
repository is intended to include artifacts via Git LFS.

## Core Proposed Method

The proposed method code is under:

```text
submission_artifact/src/proposed/
```

Important files:

```text
mimir_hardsplit_attention_common.py
experiment4_mimir_hardsplit_stopping_condition.py
experiment4_gptneo27b_fixed20_common.py
experiment4_gptneo27b_fixed20_ft.py
experiment4_gptneo27b_fixed20_pt.py
experiment4_gptneo27b_fixed20_unseen.py
experiment4_pythia410m_fixed20_common.py
experiment4_pythia410m_fixed20_ft.py
experiment4_pythia410m_fixed20_pt.py
experiment4_pythia410m_fixed20_unseen.py
```

For Pythia-1B, use `experiment4_mimir_hardsplit_stopping_condition.py`
directly with the Pythia-1B LoRA checkpoint path.

## Training Code

Training code is under:

```text
submission_artifact/src/train/
```

Important files:

```text
train_mimir_wikipedia_hardsplit_lora.py
train_mimir_wikipedia_hardsplit_lora_gptneo27b.py
```

All model variants should use the same MIMIR hard-split CSV files.

## Analysis Code

Analysis code is under:

```text
submission_artifact/src/analysis/
```

Important files:

```text
analyze_mimir_fixed_steps_repeated_auc.py
compare_fixedstep_proposed_baselines_strict.py
compare_proposed_attenmia_loraleak_10runs.py
```

These scripts compute AUC, AUPRC, TPR@FPR, repeated-run summaries, and
comparison tables.

## Baseline Code

Baseline code is under:

```text
submission_artifact/src/baselines/
```

Existing baselines:

```text
run_attenmia_official_mimir_hardsplit.py
run_attenmia_official_mimir_hardsplit_pythia410m.py
run_lora_leak_official_mimir_hardsplit.py
run_lora_leak_official_mimir_hardsplit_pythia410m.py
compare_mink_strict_fixedstep_10runs.py
```

Additional baselines added for the current experiments:

```text
run_g_driftmia_mimir_hardsplit.py
run_g_driftmia_pythia1b_mimir_hardsplit.py
run_g_driftmia_pythia410m_mimir_hardsplit.py
run_g_driftmia_gptneo27b_mimir_hardsplit.py
run_gds_mimir_hardsplit.py
run_gds_pythia1b_mimir_hardsplit.py
run_gds_pythia410m_mimir_hardsplit.py
run_gds_gptneo27b_mimir_hardsplit.py
```

## Canonical Data

All experiments should use the same split files:

```text
submission_artifact/data/mimir_hardsplit/
  mimir_wikipedia_pt_member.csv
  mimir_wikipedia_ft_nonmember.csv
  mimir_wikipedia_unseen_nonmember.csv
```

These files may be omitted from GitHub if redistribution is not allowed. In
that case, keep `.gitkeep` and document how to place the files locally.

## Main Reproduction Order

Run from inside `submission_artifact/`.

```bash
pip install -r requirements.txt
bash scripts/00_prepare_splits.sh
bash scripts/01_train_gptneo27b.sh
bash scripts/02_extract_fixed20_gptneo27b.sh
bash scripts/03_analyze_gptneo27b.sh
bash scripts/04_run_baselines.sh
```

For Pythia-1B and Pythia-410M, use the corresponding model/checkpoint paths in
the Python wrappers or analysis scripts.

## Additional Baseline Runs

G-DriftMIA:

```bash
PYTHONPATH=src/baselines python src/baselines/run_g_driftmia_pythia1b_mimir_hardsplit.py
PYTHONPATH=src/baselines python src/baselines/run_g_driftmia_pythia410m_mimir_hardsplit.py
PYTHONPATH=src/baselines python src/baselines/run_g_driftmia_gptneo27b_mimir_hardsplit.py
```

GDS:

```bash
PYTHONPATH=src/baselines python src/baselines/run_gds_pythia1b_mimir_hardsplit.py
PYTHONPATH=src/baselines python src/baselines/run_gds_pythia410m_mimir_hardsplit.py
PYTHONPATH=src/baselines python src/baselines/run_gds_gptneo27b_mimir_hardsplit.py
```

## Do Not Upload

Avoid uploading:

```text
__pycache__/
.DS_Store
*.pyc
large model checkpoint directories
large raw result CSVs
temporary plot folders
personal local paths outside submission_artifact/
```

If large files are required, use Git LFS or provide download instructions.

