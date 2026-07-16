# Professor Experiment Code Set

This is the compact code set to upload to GitHub. It corresponds to the
experiment code shown to the professor, plus the additional baseline scripts
implemented afterward.

## Upload This Directory

```text
submission_artifact/
```

This directory is the clean reproduction package. It contains the organized
training, proposed-method, baseline, and analysis code.

## Main Proposed Method Code

```text
submission_artifact/src/proposed/mimir_hardsplit_attention_common.py
submission_artifact/src/proposed/experiment4_mimir_hardsplit_stopping_condition.py
```

Model-specific fixed-20 wrappers:

```text
submission_artifact/src/proposed/experiment4_gptneo27b_fixed20_common.py
submission_artifact/src/proposed/experiment4_gptneo27b_fixed20_ft.py
submission_artifact/src/proposed/experiment4_gptneo27b_fixed20_pt.py
submission_artifact/src/proposed/experiment4_gptneo27b_fixed20_unseen.py

submission_artifact/src/proposed/experiment4_pythia410m_fixed20_common.py
submission_artifact/src/proposed/experiment4_pythia410m_fixed20_ft.py
submission_artifact/src/proposed/experiment4_pythia410m_fixed20_pt.py
submission_artifact/src/proposed/experiment4_pythia410m_fixed20_unseen.py
```

For Pythia-1B, use:

```text
submission_artifact/src/proposed/experiment4_mimir_hardsplit_stopping_condition.py
```

with the Pythia-1B checkpoint path.

## Training Code

```text
submission_artifact/src/train/train_mimir_wikipedia_hardsplit_lora.py
submission_artifact/src/train/train_mimir_wikipedia_hardsplit_lora_gptneo27b.py
```

## Analysis Code

```text
submission_artifact/src/analysis/analyze_mimir_fixed_steps_repeated_auc.py
submission_artifact/src/analysis/compare_fixedstep_proposed_baselines_strict.py
submission_artifact/src/analysis/compare_proposed_attenmia_loraleak_10runs.py
```

## Baseline Code

Original comparison baselines:

```text
submission_artifact/src/baselines/run_attenmia_official_mimir_hardsplit.py
submission_artifact/src/baselines/run_attenmia_official_mimir_hardsplit_pythia410m.py
submission_artifact/src/baselines/run_lora_leak_official_mimir_hardsplit.py
submission_artifact/src/baselines/run_lora_leak_official_mimir_hardsplit_pythia410m.py
submission_artifact/src/baselines/compare_mink_strict_fixedstep_10runs.py
```

## Data Assumption

All methods use the same MIMIR hard split:

```text
submission_artifact/data/mimir_hardsplit/mimir_wikipedia_pt_member.csv
submission_artifact/data/mimir_hardsplit/mimir_wikipedia_ft_nonmember.csv
submission_artifact/data/mimir_hardsplit/mimir_wikipedia_unseen_nonmember.csv
```

If the data cannot be redistributed, leave `.gitkeep` and write in the README
that users must place these files manually.

## Do Not Include

Do not commit:

```text
__pycache__/
*.pyc
.DS_Store
large model checkpoint directories
large raw result CSVs
temporary plots
```

Use Git LFS only if large checkpoints or result files must be shared.
