# Anonymous experiment code supplement

This directory contains the experiment programs used for the paper, including
the original experiments, baselines, robustness checks, and reviewer-follow-up
experiments E7--E14. All documentation, module docstrings, and source comments
are in English.

The code is provided for reviewer inspection and reproducibility. Model
weights, raw datasets, raw attention tensors, and per-sample scores are not
included.

## Quick start

Run all commands from this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# CPU-only integrity and protocol tests
PYTHONPATH=. python -m unittest -q \
  test_reviewer_revision test_reviewer_followup \
  test_additional_experiments test_multi_model

# Inspect the original experiment plan without launching jobs
./run_paper_experiments.sh plan

# Materialize and inspect the E7--E12 command plan
PYTHONPATH=. python -m reviewer_followup.controller plan \
  --output-root results/reviewer_followup
PYTHONPATH=. python -m reviewer_followup.controller status \
  --output-root results/reviewer_followup

# Materialize the isolated E13/E14 plan
PYTHONPATH=. python -m reviewer_followup.revision_controller plan \
  --source-root results/reviewer_followup \
  --output-root results/reviewer_revision
```

GPU stages are never launched by the reviewer-follow-up controller without
`--yes-really-run-gpu`. Before using those stages, set `GPU_STATUS_URL` to a
site-local endpoint returning the GPU-status JSON expected by
`reviewer_followup.controller`:

```bash
export GPU_STATUS_URL=https://example.edu/api/gpu/status
```

## Directory map

- `run_paper_experiments.py` and `run_paper_experiments.sh`: full original
  paper pipeline across models and Experiments 1--3.
- `orchestrate.py` and `orchestrate.sh`: single-model train, extract, baseline,
  analysis, and evaluation driver.
- `train_mimir_wikipedia_hardsplit_lora.py`: LoRA fine-tuning on the MIMIR
  Wikipedia hard split.
- `extract_attention_hardsplit.py` and
  `mimir_hardsplit_attention_common.py`: target-specific additional training
  and attention-update feature extraction.
- `experiment4_*`: model- and group-specific fixed-step or early-stopping
  extraction entry points.
- `run_strict_fixed20_comparison_10runs.py`: repeated stratified evaluation of
  the proposed features and baselines.
- `run_attenmia_official_mimir_hardsplit*.py`: the AttenMIA baseline.
- `run_lora_leak_official_mimir_hardsplit*.py`: score extraction for the
  LoRA-Leak method family. The paper reports only the frozen
  `target_mink++_0.2` scalar as **Min-K%++ (LoRA-FT)**, not the complete
  LoRA-Leak attack suite.
- `run_crossfit_fusion_en_lora_leak.py`: leakage-safe, cross-fitted score
  fusion.
- `run_nested_step_selection.py`, `run_paired_robustness.py`,
  `run_data_confound_diagnostics.py`, and the `run_*sensitivity*.sh` scripts:
  step selection, multi-run robustness, and confound controls.
- `analyze_exp1_layer_head_significance.py`: FDR-corrected layer--head
  localization and effect-size analysis.
- `reviewer_followup/`: self-contained E7--E14 package for crossed designs,
  update-feature baselines, checkpoint stability, a controlled model-family
  study, full nested protocol selection, shard merging, and final audits.
- `baseline_fidelity_manifest.json`: implementation hashes, adaptation
  boundaries, and official/adapted status for manuscript baselines.
- `hardsplit/`: shared model, AMP, progress, sharding, and CLI utilities.
- `test_reviewer_followup.py`, `test_additional_experiments.py`, and
  `test_multi_model.py`: CPU-only tests.
- `PAPER_ALIGNMENT.md` and `STRUCTURE.md`: detailed script-to-paper mapping and
  performance-oriented code layout.

## Reviewer-follow-up workflow (E7--E14)

The controller first freezes every argv vector, seed, expected output, and GPU
flag in `experiment_plan.json`:

```bash
PYTHONPATH=. python -m reviewer_followup.controller prepare \
  --output-root results/reviewer_followup
PYTHONPATH=. python -m reviewer_followup.controller run-stage \
  --stage e7 --output-root results/reviewer_followup \
  --yes-really-run-gpu
```

Dependencies are E7 -> E8, E9 -> E10, and E12 extraction shards -> validated
merge -> two nested evaluations. E11 is independent. E12 evaluates 80 raw
attention candidates per sample (20 query protocols times four update
schedules); its sample-level training summary contains four update schedules
per sample.

Completion is intentionally strict. Each command must produce all declared,
non-empty outputs and a matching controller marker. JSON must parse, CSV files
must contain a header and data, extraction runs must record a completed status,
and the E12 merge must cover all 1,500 target IDs. The final audit is:

```bash
PYTHONPATH=. python -m reviewer_followup.audit_results \
  --output-root results/reviewer_followup
```

E13/E14 use a separate frozen result root through
`reviewer_followup.revision_controller`. E13 performs hierarchical
checkpoint-then-target inference; E14 uses the exact controlled E11 data with
Pythia-160M. Both retain explicit GPU opt-in and fresh GPU-status snapshots.

## Data and model inputs

The scripts expect the MIMIR Wikipedia hard-split CSVs and compatible
Hugging Face base models/LoRA adapters. Default model identifiers and LoRA
module mappings are defined in `model_registry.py`. Paths can be overridden by
the documented command-line arguments; inspect `--help` before a full run.

## Anonymization and exclusions

Hostnames, usernames, institutional paths, and the original GPU-status URL are
replaced by generic placeholders. The supplement contains no credentials,
weights, raw feature dumps, canonical result archive, or per-sample membership
decisions. Aggregate paper values therefore remain inspectable without
releasing sensitive sample-level artifacts.
