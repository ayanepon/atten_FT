# Professor Experiment Code Set

This repository is the compact code set prepared for GitHub. Paths below are
relative to the repository root.

## Upload Directory

Upload the whole repository directory:

```text
anonymous_github_experiment_code/
```

It contains the organized training, proposed-method, baseline, and analysis
code. Large checkpoints and raw result CSVs are intentionally excluded.

## Main Proposed Method Code

```text
src/proposed/mimir_hardsplit_attention_common.py
src/proposed/experiment4_mimir_hardsplit_stopping_condition.py
```

Model-specific fixed-20 wrappers:

```text
src/proposed/experiment4_gptneo27b_fixed20_common.py
src/proposed/experiment4_gptneo27b_fixed20_ft.py
src/proposed/experiment4_gptneo27b_fixed20_pt.py
src/proposed/experiment4_gptneo27b_fixed20_unseen.py

src/proposed/experiment4_pythia410m_fixed20_common.py
src/proposed/experiment4_pythia410m_fixed20_ft.py
src/proposed/experiment4_pythia410m_fixed20_pt.py
src/proposed/experiment4_pythia410m_fixed20_unseen.py
```

Stopping-condition ablation wrappers:

```text
src/proposed/run_pythia1b_stopping_conditions.py
src/proposed/run_pythia410m_stopping_conditions.py
src/proposed/run_gptneo27b_stopping_conditions.py
```

## Training Code

```text
src/train/train_mimir_wikipedia_hardsplit_lora.py
src/train/train_mimir_wikipedia_hardsplit_lora_gptneo27b.py
```

## Analysis Code

```text
src/analysis/analyze_mimir_fixed_steps_repeated_auc.py
src/analysis/compare_fixedstep_proposed_baselines_strict.py
src/analysis/compare_proposed_attenmia_loraleak_10runs.py
src/analysis/run_strict_fixed20_3model_comparison_10runs.py
src/analysis/evaluate_loss_direction_selected_pythia1b.py
```

## Baseline Code

```text
src/baselines/run_attenmia_official_mimir_hardsplit.py
src/baselines/run_attenmia_official_mimir_hardsplit_pythia410m.py
src/baselines/run_lora_leak_official_mimir_hardsplit.py
src/baselines/run_lora_leak_official_mimir_hardsplit_pythia410m.py
src/baselines/compare_mink_strict_fixedstep_10runs.py
```

## Data Assumption

All methods use the same MIMIR hard split:

```text
data/mimir_hardsplit/mimir_wikipedia_pt_member.csv
data/mimir_hardsplit/mimir_wikipedia_ft_nonmember.csv
data/mimir_hardsplit/mimir_wikipedia_unseen_nonmember.csv
```

The CSV files may be omitted from GitHub if redistribution is not allowed. In
that case, keep `.gitkeep` and place the files manually before running.

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
