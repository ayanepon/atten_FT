# MIMIR Hard-Split Attention Update Experiments

This artifact contains the code needed to reproduce the main experiments:

- LoRA fine-tuning on the MIMIR Wikipedia hard split
- Proposed fixed-step Attention update extraction
- FT vs PT and FT vs Unseen classification
- AttenMIA-style, LoRA-Leak-style, Min-k%, and Min-k%++ baselines

All paths are relative to this artifact directory. The shell scripts under
`scripts/` are the recommended entry points for reproduction. The Python files
under `src/` implement the underlying training, extraction, baseline, and
analysis routines.

## Directory layout

```text
configs/
  paths.json
data/
  mimir_hardsplit/
models/
results/
scripts/
src/
  train/
  proposed/
  baselines/
  analysis/
```

## Before Running

Run all commands from the artifact root:

```bash
cd submission_artifact
```

Install the required Python packages in your environment:

```bash
pip install -r requirements.txt
```

The experiments download Hugging Face models and datasets when they are not
already cached. A CUDA GPU is recommended for LoRA fine-tuning and attention
extraction.

## Data Placement

The MIMIR Wikipedia hard-split CSV files are placed here:

```text
data/mimir_hardsplit/
  mimir_wikipedia_pt_member.csv
  mimir_wikipedia_ft_nonmember.csv
  mimir_wikipedia_unseen_nonmember.csv
  mimir_wikipedia_pt_ft_unseen_targets.csv
```

The same split CSVs should be reused across Pythia and GPT-Neo experiments.

For fair cross-model and cross-method comparison, use these CSV files as the
single shared data split for all experiments. Do not regenerate separate
FT/PT/Unseen splits for each model.

## Reproduction Steps

The following commands reproduce the main GPT-Neo 1.3B experiment. Run them in
order. The GPT-Neo 2.7B experiment uses the same data splits and the same
workflow; see the GPT-Neo 2.7B commands below.

### Step 0: Prepare the Data Splits

```bash
bash scripts/00_prepare_splits.sh
```

This creates the MIMIR hard-split CSV files and copies them to:

```text
data/mimir_hardsplit/
```

These files are the canonical split files for this artifact. Later training,
attention-extraction, and baseline runs should use this same split.

Expected files:

```text
data/mimir_hardsplit/mimir_wikipedia_pt_member.csv
data/mimir_hardsplit/mimir_wikipedia_ft_nonmember.csv
data/mimir_hardsplit/mimir_wikipedia_unseen_nonmember.csv
data/mimir_hardsplit/mimir_wikipedia_pt_ft_unseen_targets.csv
```

### Step 1: Fine-tune the Target Model

```bash
bash scripts/01_train_gptneo13b.sh
```

This fine-tunes GPT-Neo 1.3B with LoRA on the FT split.
The script explicitly reads the shared split files from:

```text
data/mimir_hardsplit/
```

Expected output:

```text
models/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_gptneo13b/
```

### Step 2: Extract Fixed-20 Attention Updates

```bash
bash scripts/02_extract_fixed20_gptneo13b.sh
```

This extracts the proposed fixed-20-step attention-update features for FT, PT,
and Unseen samples.

Expected outputs:

```text
results/mimir_wikipedia_hardsplit_fixed20_gptneo13b/fixed_attention_20_ft/
results/mimir_wikipedia_hardsplit_fixed20_gptneo13b/fixed_attention_20_pt/
results/mimir_wikipedia_hardsplit_fixed20_gptneo13b/fixed_attention_20_unseen/
```

This step is the slowest part of the proposed method. If multiple GPUs are
available, the three groups can be extracted separately:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src/proposed python src/proposed/experiment4_gptneo13b_fixed20_ft.py
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src/proposed python src/proposed/experiment4_gptneo13b_fixed20_pt.py
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src/proposed python src/proposed/experiment4_gptneo13b_fixed20_unseen.py
```

### Step 3: Analyze Proposed-Method Results

```bash
bash scripts/03_analyze_gptneo13b.sh
```

This computes FT vs PT and FT vs Unseen classification results from the
fixed-20 attention-update features.

Main output:

```text
results/mimir_wikipedia_hardsplit_fixed20_gptneo13b/ft_vs_pt_unseen_auc_analysis/repeated_auc_summary.csv
```

### Step 4: Run Baselines

```bash
bash scripts/04_run_baselines.sh
```

This runs the AttenMIA-style, LoRA-Leak-style, Min-k%, and Min-k%++ baselines.
The default baseline script uses the Pythia-1B run directory:

```text
models/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2/
```

If this model is not present, train or copy the corresponding Pythia-1B LoRA
model before running the baseline script, or edit the script to point to the
model you want to evaluate.

When using a copied or pre-existing model directory, confirm that its `data/`
subdirectory contains the same split CSVs as `data/mimir_hardsplit/`.

## GPT-Neo 2.7B Experiment

GPT-Neo 2.7B uses the same canonical split files in `data/mimir_hardsplit/`.
Run Step 0 first if the split files do not already exist.

```bash
bash scripts/00_prepare_splits.sh
bash scripts/01_train_gptneo27b.sh
bash scripts/02_extract_fixed20_gptneo27b.sh
bash scripts/03_analyze_gptneo27b.sh
```

Expected model output:

```text
models/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_gptneo27b/
```

Expected proposed-method outputs:

```text
results/mimir_wikipedia_hardsplit_fixed20_gptneo27b/fixed_attention_20_ft/
results/mimir_wikipedia_hardsplit_fixed20_gptneo27b/fixed_attention_20_pt/
results/mimir_wikipedia_hardsplit_fixed20_gptneo27b/fixed_attention_20_unseen/
results/mimir_wikipedia_hardsplit_fixed20_gptneo27b/ft_vs_pt_unseen_auc_analysis/
```

As with GPT-Neo 1.3B, fixed-20 extraction can be split across GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src/proposed python src/proposed/experiment4_gptneo27b_fixed20_ft.py
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src/proposed python src/proposed/experiment4_gptneo27b_fixed20_pt.py
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src/proposed python src/proposed/experiment4_gptneo27b_fixed20_unseen.py
```

## Main outputs

```text
results/mimir_wikipedia_hardsplit_fixed20_gptneo13b/
  fixed_attention_20_ft/
  fixed_attention_20_pt/
  fixed_attention_20_unseen/
  ft_vs_pt_unseen_auc_analysis/
```

The main analysis output is:

```text
results/mimir_wikipedia_hardsplit_fixed20_gptneo13b/ft_vs_pt_unseen_auc_analysis/repeated_auc_summary.csv
```

## Baseline fidelity notes

See also:

```text
BASELINE_FIDELITY.md
```

The baseline implementations are intended to be faithful enough for controlled comparison, but their status differs by method.

- Min-k%: implemented directly from the standard Min-k% probability idea using the lowest-probability token subset.
- Min-k%++: implemented using standardized token log-probabilities, matching the key idea of Min-k%++.
- LoRA-Leak-style: includes target loss, zlib-normalized loss, Min-k%, Min-k%++, GradNormx, and pretrained-reference variants. This follows the LoRA-Leak paper's evaluation idea that the pretrained model can act as a reference for LoRA fine-tuning leakage.
- AttenMIA-style: uses attention-derived features, perturbation features, and an MLP classifier. It follows the AttenMIA paper's high-level design, but should be described as an AttenMIA-style reimplementation unless directly validated against the authors' official code.

For the paper, avoid saying "official implementation" for AttenMIA unless the exact authors' code is used. Recommended wording:

```text
We implement an AttenMIA-style baseline based on attention transition and perturbation features, and evaluate it under the same folds as the proposed method.
```

## Notes

- FT is treated as the positive class in FT vs PT and FT vs Unseen.
- AUC is not flipped after observing results.
- Elastic Net feature selection is performed inside each training fold only.
- The fixed-20 condition means 20 optimizer steps for each target sample during the per-sample additional training phase.
