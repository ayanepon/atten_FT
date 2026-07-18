# Code layout

```
data/
  hardsplit/                 # shared package (prefer importing from here)
    models.py                # multi-model presets / CLI namespace
    cli_utils.py             # thin baseline entry helpers
    progress.py              # incremental extract CSV writers + shard merge
    parallel.py              # sample-shard planning, eval map_repeats
    amp_utils.py             # bf16/fp16 autocast for additional training

  model_registry.py          # re-export of hardsplit.models (compat)
  extract_attention_hardsplit.py
  mimir_hardsplit_attention_common.py
  train_mimir_wikipedia_hardsplit_lora.py
  run_strict_fixed20_comparison_10runs.py
  test_reviewer_revision.py
  reviewer_followup/          # E7--E14 controls, uncertainty, and controllers
  run_attenmia_official_mimir_hardsplit.py
  run_lora_leak_official_mimir_hardsplit.py   # alias → _2
  run_lora_leak_official_mimir_hardsplit_2.py
  orchestrate.py / orchestrate.sh          # single-model multi-GPU
  run_paper_experiments.py / .sh           # full paper (all models + Exp.1–3)
  _run_dynamic_extract.py                  # Exp.3 early-stopping extract helper

  # thin model-specific launchers (delegate to cores + hardsplit.cli_utils)
  run_*_{pythia410m,gptneo27b}.py
  experiment4_{pythia410m,gptneo27b}_fixed20_*.py
```

## Speed flags (quick reference)

```bash
# Extract: AMP + incremental CSV + sample shards (multi-GPU)
python extract_attention_hardsplit.py ... --amp --flush-every 25 --shard 0/4

./orchestrate.sh extract --model pythia-1b --gpus auto --sample-shards 2 --wait

# Eval: reuse feature cache + parallel EN repeats
python run_strict_fixed20_comparison_10runs.py ... --n-jobs 4

# Min-K%++ / LoRA-Leak-family score-extraction fast path
python run_lora_leak_official_mimir_hardsplit.py --fast
```
