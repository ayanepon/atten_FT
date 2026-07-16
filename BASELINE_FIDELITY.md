# Baseline fidelity check

This note records how the baseline implementations correspond to the cited methods.

## AttenMIA

Reference: AttenMIA: LLM Membership Inference Attack through Attention Signals  
URL: https://arxiv.org/abs/2601.18110

The paper describes an MIA framework that uses self-attention patterns, information from attention heads across layers, perturbation-based divergence metrics, and a learned MIA classifier.

Included implementation:

```text
src/baselines/run_attenmia_official_mimir_hardsplit.py
```

Implemented components:

- attention transition features between adjacent layers
- per-layer/head attention concentration features
- token drop / token replacement / prefix insertion perturbation features
- MLP classifier with stratified cross-validation

Important caveat:

This is an AttenMIA-style reimplementation based on the paper description. Unless the authors' original code is used and matched line-by-line, the paper should not call this the "official AttenMIA implementation." Recommended wording:

```text
AttenMIA-style baseline
```

## LoRA-Leak

Reference: LoRA-Leak: Membership Inference Attacks Against LoRA Fine-tuned Language Models  
URL: https://arxiv.org/abs/2507.18302

The paper frames LoRA-Leak as an evaluation framework for LoRA fine-tuning MIAs and emphasizes using the pretrained model as a reference signal.

Included implementation:

```text
src/baselines/run_lora_leak_official_mimir_hardsplit.py
```

Implemented components:

- target-model loss score
- zlib-normalized loss
- Min-k%
- Min-k%++
- GradNormx
- pretrained-reference variants using the base pretrained model

Recommended wording:

```text
LoRA-Leak-style baseline with pretrained-reference scores
```

## Min-k%

Reference: Detecting Pretraining Data from Large Language Models  
URL: https://arxiv.org/abs/2310.16789

The method uses the average probability/log-probability of the lowest-probability token subset as a reference-free pretraining-data detection score.

Included implementation:

```text
src/baselines/compare_mink_strict_fixedstep_10runs.py
```

This script evaluates plain Min-k% only. It intentionally excludes Min-k%++ columns.

## Min-k%++

Reference: Min-K%++: Improved Baseline for Detecting Pre-Training Data from Large Language Models  
URL: https://arxiv.org/abs/2404.02936

The method standardizes token likelihood using the model's predicted token distribution and then applies the Min-k idea to the standardized scores.

Included implementation:

```text
src/baselines/run_lora_leak_official_mimir_hardsplit.py
```

The Min-k%++ scores are generated as `mink++_*` / `target_mink++_*` columns.

## Final recommendation for the paper

Use precise wording:

- "Min-k%" for the plain Min-k% implementation.
- "Min-k%++" for the standardized variant.
- "LoRA-Leak-style" unless exact official code is used.
- "AttenMIA-style" unless exact official code is used.

This avoids overclaiming fidelity while still making the comparison reproducible.
