# Experiment Code

This repository contains the code used for the paper experiments in a compact
GitHub-ready form.

It includes:

- proposed attention-update feature extraction code
- LoRA fine-tuning code
- analysis code for AUC, AUPRC, TPR@FPR, repeated runs, and significance tests
- baseline implementations
  - AttenMIA-style baseline
  - LoRA-Leak-style baseline
  - Min-K / Min-K++ baselines

Large model checkpoints and generated result CSVs are not included.

## Directory Structure

```text
anonymous_github_experiment_code/
  README.md
  README_JA.md
  requirements.txt
  configs/
  data/
  models/
  results/
  scripts/
  src/
    train/
    proposed/
    baselines/
    analysis/
```

## Data

All experiments use the same MIMIR hard split.

Place the following CSV files under:

```text
data/mimir_hardsplit/
  mimir_wikipedia_pt_member.csv
  mimir_wikipedia_ft_nonmember.csv
  mimir_wikipedia_unseen_nonmember.csv
```

The files correspond to:

```text
mimir_wikipedia_pt_member.csv
  PT data assumed to be included in pre-training.

mimir_wikipedia_ft_nonmember.csv
  FT data used for LoRA fine-tuning.

mimir_wikipedia_unseen_nonmember.csv
  Unseen data used neither for pre-training membership nor fine-tuning.
```

If the data cannot be redistributed, keep the directory structure and place the
CSV files manually before running the experiments.

## Fine-Tuned Models

Fine-tuned model checkpoints are not included in this repository.

Place checkpoints under:

```text
models/
```

Alternatively, create them using the training scripts under `src/train/`.

Main training scripts:

```text
src/train/train_mimir_wikipedia_hardsplit_lora.py
src/train/train_mimir_wikipedia_hardsplit_lora_pythia410m.py
src/train/train_mimir_wikipedia_hardsplit_lora_gptneo27b.py
```

## Proposed Method

The main proposed-method code is:

```text
src/proposed/mimir_hardsplit_attention_common.py
src/proposed/experiment4_mimir_hardsplit_stopping_condition.py
```

Model-specific fixed-20 entry points:

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

Stopping-condition ablation entry points for fixed 20/50/100 steps and early
stopping:

```text
src/proposed/run_pythia1b_stopping_conditions.py
src/proposed/run_pythia410m_stopping_conditions.py
src/proposed/run_gptneo27b_stopping_conditions.py
```

These scripts call the shared implementation:

```text
src/proposed/experiment4_mimir_hardsplit_stopping_condition.py
```

## Analysis Code

The following scripts compute AUC, AUPRC, TPR@FPR, repeated-run summaries, and
baseline comparisons:

```text
src/analysis/analyze_mimir_fixed_steps_repeated_auc.py
src/analysis/compare_fixedstep_proposed_baselines_strict.py
src/analysis/compare_proposed_attenmia_loraleak_10runs.py
src/analysis/run_strict_fixed20_3model_comparison_10runs.py
src/analysis/evaluate_loss_direction_selected_3model.py
```

`run_strict_fixed20_3model_comparison_10runs.py` compares the proposed method,
AttenMIA, LoRA-Leak, Initial loss, and Loss decrease using the same repeated
cross-validation splits.

`evaluate_loss_direction_selected_3model.py` evaluates loss-only baselines
for Pythia-1B, Pythia-410M, and GPT-Neo-2.7B with score direction selected
inside each training fold.

## Baselines

Baseline scripts are placed under `src/baselines/`.

```text
src/baselines/run_attenmia_official_mimir_hardsplit.py
src/baselines/run_attenmia_official_mimir_hardsplit_gptneo27b.py
src/baselines/run_lora_leak_official_mimir_hardsplit.py
src/baselines/run_lora_leak_official_mimir_hardsplit_gptneo27b.py
src/baselines/compare_mink_strict_fixedstep_10runs.py
```

## Reproduction Order

The basic execution order is:

```bash
pip install -r requirements.txt
bash scripts/00_prepare_splits.sh
bash scripts/01_train_pythia410m.sh
bash scripts/01_train_gptneo27b.sh
bash scripts/02_extract_fixed20_gptneo27b.sh
bash scripts/03_analyze_gptneo27b.sh
bash scripts/04_run_baselines.sh
```

To run the stopping-condition ablation:

```bash
PYTHONPATH=src/proposed python src/proposed/run_pythia1b_stopping_conditions.py
PYTHONPATH=src/proposed python src/proposed/run_pythia410m_stopping_conditions.py
PYTHONPATH=src/proposed python src/proposed/run_gptneo27b_stopping_conditions.py
```

To run the comparison including loss baselines:

```bash
python src/analysis/run_strict_fixed20_3model_comparison_10runs.py
python src/analysis/evaluate_loss_direction_selected_3model.py
```

For a different local environment, override model and output paths with
environment variables.

## Files Not Included

The following files are intentionally excluded from GitHub:

```text
__pycache__/
*.pyc
.DS_Store
large checkpoints under models/
large result CSVs under results/
temporary plot outputs
```

Therefore, `models/` and `results/` contain only `.gitkeep` files by default.

## Notes

- FT is always treated as the positive class.
- AUC is not flipped after observing the result.
- Elastic Net feature selection for the proposed method is performed only
  inside each training fold.
- All models and baselines use the same MIMIR hard split.

Japanese documentation is available in `README_JA.md`.
