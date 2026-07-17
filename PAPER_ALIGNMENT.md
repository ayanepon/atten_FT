# Paper alignment notes

## Directory layout

| Path | Role |
|------|------|
| `model_registry.py` | Multi-model presets (HF id, LoRA modules, default dirs) |
| `extract_attention_hardsplit.py` | Feature extraction entry (was `experiment4_...py`) |
| `attention_features_mimir_hardsplit/` | **Current** fixed-step features (Pythia-1B) |
| `attention_features_pythia410m/` | Pythia-410M features |
| `attention_features_gptneo27b/` | GPT-Neo-2.7B features |
| `attention_features_mimir_hardsplit_legacy/` | Old results (pre mask/lr cleanup) |
| `orchestrate.py` / `orchestrate.sh` | Multi-GPU train/extract/baselines/exp1/eval (single model) |
| `run_paper_experiments.py` / `.sh` | **Full paper pipeline**: all models + Exp.1–3 |
| `run_unified_fixed20_pipeline.py` | Simpler single-process extract → Exp.1 → eval |
| `run_strict_fixed20_comparison_10runs.py` | Unified classifier comparison (multi-model) |
| `run_strict_fixed20_pythia1b_comparison_10runs.py` | Alias → comparison script |
| `analyze_exp1_layer_head_significance.py` | Exp.1 FDR / Cliff's δ |
| `train_mimir_wikipedia_hardsplit_lora.py` | LoRA FT trainer (`--model` presets) |
| `run_lora_leak_official_mimir_hardsplit_2.py` | LoRA-Leak baseline (core) |
| `run_lora_leak_official_mimir_hardsplit_{pythia410m,gptneo27b}.py` | Thin model-specific wrappers |
| `run_attenmia_official_mimir_hardsplit.py` | AttenMIA baseline (core) |
| `run_attenmia_official_mimir_hardsplit_{pythia410m,gptneo27b}.py` | Thin model-specific wrappers |
| `experiment4_fixed20_common.py` | Shared fixed-20 extract factory (`make_runner`) |
| `experiment4_gptneo27b_fixed20_{ft,pt,unseen}.py` | GPT-Neo fixed-20 extract per group |
| `experiment4_gptneo27b_fixed20_common.py` | GPT-Neo extract runner |
| `experiment4_pythia410m_fixed20_{ft,pt,unseen}.py` | Pythia-410M fixed-20 extract per group |
| `experiment4_pythia410m_fixed20_common.py` | Pythia-410M extract runner |

## Hyperparameters

| Setting | Value |
|---------|-------|
| LoRA FT lr | `1e-4` |
| Additional training lr | **`1e-5`** |
| LoRA FT micro-batch | 1 + grad accum 16 (effective 16) |
| Additional training batch | 1 |
| Query ρ | 10% |
| Fixed main setting | 20 steps |

## Multi-model presets (`model_registry.py`)

**Single source of truth** for HF ids, LoRA modules, default paths, adapter resolution,
and CLI namespace filling (`apply_model_namespace`). Train / extract / baselines /
orchestrate / strict-eval all consume this module (no duplicated preset tables).

| `--model` | HF id | LoRA target modules | Default run dir | Features root |
|-----------|-------|---------------------|-----------------|---------------|
| `pythia-1b` | `EleutherAI/pythia-1b` | NeoX: qkv/dense/h_to_4h/4h_to_h | `mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2` | `attention_features_mimir_hardsplit` |
| `pythia-410m` | `EleutherAI/pythia-410m` | same as 1B | `mimir_lora_pythia410m` | `attention_features_pythia410m` |
| `gpt-neo-2.7b` | `EleutherAI/gpt-neo-2.7B` | q/k/v/out/c_fc/c_proj | `mimir_lora_gptneo27b` | `attention_features_gptneo27b` |

Base model is also inferred from `adapter/adapter_config.json` when `--model-name` is empty.

Shared helpers:
- `apply_model_namespace(args, profile=...)` — profiles: `train` / `pipeline` / `lora_leak` / `attenmia`
- `resolve_adapter_dir` / `resolve_model_name` / `resolve_from_args`
- `strict_eval_model_configs()` — eval `MODEL_CONFIGS`
- `add_model_arguments(parser)` — standard `--model` / `--model-name`

## Recommended workflow

### Full paper (all experiments)

```bash
# Print planned stages
./run_paper_experiments.sh plan

# Everything: train missing adapters → fixed-20 extract → baselines →
# Exp.1 → Exp.2 strict eval (per model + joint) → Exp.3 (50/100/early)
./run_paper_experiments.sh full --gpus auto \
  --from-csv-dir mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2/data

# Main tables only (skip Exp.3 long extract)
./run_paper_experiments.sh full --skip-exp3 --gpus auto

# Dry-run / status / stop
./run_paper_experiments.sh full --dry-run
./run_paper_experiments.sh status
./run_paper_experiments.sh stop
```

Pipeline state: `paper_pipeline/state.json`

| Phase | Outputs |
|-------|---------|
| A per model | adapter, `attention_features_*` fixed-20, LoRA-Leak / AttenMIA, Exp.1 stats, strict eval |
| B joint | `results/strict_fixed20_paper_all_models/` |
| C Exp.3 | `attention_features_pythia1b_steps{50,100}/`, `_dynamic/`, `results/exp3_step_ablation_pythia1b/` |

### Single-model (via orchestrate)

```bash
# 1) Train LoRA FT (reuse hard-split CSVs from the 1B run when available)
python train_mimir_wikipedia_hardsplit_lora.py \
  --model pythia-410m \
  --from-csv-dir mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2/data

# 2) Multi-GPU extract → Exp.1 → strict eval
./orchestrate.sh all --model pythia-410m --gpus auto --fresh

# Or train + extract + eval in one shot:
./orchestrate.sh all --model gpt-neo-2.7b --do-train \
  --from-csv-dir mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2/data \
  --gpus auto --min-free-gib 20 --fresh

# 3) Optional baselines then re-eval
./orchestrate.sh baselines --model pythia-410m
./orchestrate.sh eval --model pythia-410m \
  --lora-root results/lora_leak_pythia410m \
  --attenmia-root results/attenmia_pythia410m \
  --methods proposed_all proposed_en initial_loss loss_decrease lora_leak attenmia

# Progress / stop
./orchestrate.sh status --model pythia-410m
./orchestrate.sh stop --model pythia-410m
```

Pythia-1B (existing defaults):

```bash
./orchestrate.sh all \
  --run-dir mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2 \
  --features-root attention_features_mimir_hardsplit \
  --fixed-steps 20 --lr 1e-5 --gpus auto --fresh
```

Job state:
`attention_features_*/orchestrator/jobs.json`

Logs:
`attention_features_*/orchestrator/logs/`

## Eval model keys

Strict eval uses short names:

| Orchestrator `--model` | Eval `--models` |
|------------------------|-----------------|
| `pythia-1b` | `pythia1b` |
| `pythia-410m` | `pythia410m` |
| `gpt-neo-2.7b` | `gptneo27b` |

## Performance knobs (paper-safe defaults)

| Stage | Flag / behavior | Notes |
|-------|-----------------|-------|
| extract | `--flush-every 25` | Progress CSV rewrite interval (default 25) |
| extract | (default) no per-step accuracy on fixed-N overfit | Use `--record-train-curve` if needed |
| extract | fused before loss + token loss (1 forward) | Same numbers |
| extract | LoRA snapshot on GPU | `SNAPSHOT_ON_CPU=1` to save VRAM |
| extract | attention metrics stay on GPU | Same features |
| AttenMIA | shared non-prefix feature cache across comparisons | FT computed once |
| LoRA-Leak | `--fast` | Min-K=0.2 only, no GradNormx |
| LoRA-Leak | `--no-gradnormx` | Large speedup when GradNormx not needed |
| strict eval | `proposed_features_fixed20_cache.parquet` | Auto-built; `--refresh-feature-cache` to rebuild |

The strict evaluator also shares each comparison's train-only standardized fold
arrays between Proposed (all) and Proposed+EN.  Its default Elastic-Net
selection settings are `max_iter=1000`, `tol=5e-4`; these are solver convergence
settings, not experimental variables.  The full settings remain configurable
with `--elasticnet-max-iter` and `--elasticnet-tol`.  Proposed+EN repeats use up
to four threads by default (`EVAL_N_JOBS` or `--n-jobs` overrides this).

The extraction entry point combines the pre-update loss, top-loss query
selection, and pre-update attention snapshot into one forward pass.  After
additional training, the loss and post-update attention snapshot are likewise
computed in one forward pass.  The causal + padding mask and all paper metrics
are unchanged; the attention metric aggregation is vectorized over queries and
heads.

Paper protocol (lr=1e-5, fixed-20, masked attention features, FT positive, no test flip) is unchanged.

## E6 reproducibility freeze (2026-07-16)

Protocol, score columns, Elastic Net settings, and artifact roots for paper tables and additional experiments are frozen in:

- `results/additional_20260715/E6_REPRODUCIBILITY_MANIFEST.json`
- `results/additional_20260715/E6_REPRODUCIBILITY_MANIFEST.md`

Key freezes: LoRA-Leak paper score = `target_mink++_0.2`; EN `max_iter=1000`, `tol=5e-4`; query offset main = 1; no test-label AUC flip.
