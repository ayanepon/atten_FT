# GitHub Experiment Code Package

This directory is the compact GitHub-ready code package for the experiments.

## What Is Included

- Proposed attention-update method code
- LoRA fine-tuning code
- Repeated-CV analysis code
- Baseline code:
  - AttenMIA-style
  - LoRA-Leak-style
  - Min-K / Min-K++
- Stopping-condition ablation entry points for fixed 20/50/100 and early stopping
- Loss-baseline analysis entry points
- Reproduction scripts
- Data placement instructions

## What Is Not Included

Large artifacts are intentionally excluded:

- FT model checkpoints
- raw result CSVs
- generated plots
- cached Python files

Place model checkpoints under:

```text
models/
```

Place generated results under:

```text
results/
```

## Main Files To Read

```text
README.md
PROFESSOR_EXPERIMENT_CODE_SET.md
BASELINE_FIDELITY.md
```

## Data

All experiments assume the same MIMIR hard split:

```text
data/mimir_hardsplit/mimir_wikipedia_pt_member.csv
data/mimir_hardsplit/mimir_wikipedia_ft_nonmember.csv
data/mimir_hardsplit/mimir_wikipedia_unseen_nonmember.csv
```

If these data files cannot be redistributed, keep the directory structure and
place the files manually before running the experiments.
