# -*- coding: utf-8 -*-
"""Backward-compatible re-export of ``hardsplit.models``.

Prefer:
  from hardsplit.models import resolve_model_spec, apply_model_namespace
"""

from __future__ import annotations

from hardsplit.models import *  # noqa: F401,F403
from hardsplit.models import (  # noqa: F401
    DEFAULT_ATTENMIA_DIR,
    DEFAULT_EVAL_DIR,
    DEFAULT_EXP1_DIR,
    DEFAULT_HF_ID,
    DEFAULT_LORA_LEAK_DIR,
    DEFAULT_PYTHIA_LORA_CSV,
    GPT_NEO_LORA_MODULES,
    MODEL_CLI_HELP,
    MODEL_PRESETS,
    PYTHIA1B_FEATURES_ROOT,
    PYTHIA1B_RUN_DIR,
    PYTHIA1B_RUN_DIR_RESULTS,
    PYTHIA_LORA_MODULES,
    ModelSpec,
    add_model_arguments,
    apply_model_namespace,
    eval_key_from_model,
    list_eval_keys,
    list_model_keys,
    lora_modules_for_hf_id,
    model_spec_or_custom,
    normalize_model_key,
    read_base_model_from_adapter,
    resolve_adapter_dir,
    resolve_from_args,
    resolve_model_name,
    resolve_model_spec,
    strict_eval_model_configs,
    try_resolve_model_spec,
)
