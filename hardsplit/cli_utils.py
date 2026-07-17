# -*- coding: utf-8 -*-
"""Shared CLI helpers for thin model-specific entrypoints."""

from __future__ import annotations

import os
import sys
from typing import Callable, List, Optional, Sequence


def add_default_arg(argv: List[str], flag: str, values: Sequence[str]) -> None:
    """Append ``flag values...`` only when flag is not already present."""
    if flag not in argv:
        argv.extend([flag, *values])


def run_with_model_defaults(
    *,
    core_main: Callable[[], None],
    model_key: str,
    env_prefix: str,
    kind: str,
) -> None:
    """Inject ``--model`` / paths for baseline wrappers then call core.main().

    kind: 'attenmia' | 'lora_leak'
    """
    from hardsplit.models import resolve_model_spec

    spec = resolve_model_spec(model_key)
    run_dir = os.environ.get(f"{env_prefix}_MIMIR_MODEL_ROOT", spec.default_run_dir)
    if kind == "attenmia":
        out_default = spec.default_attenmia_root
        out_env = f"{env_prefix}_ATTENMIA_OUTPUT_DIR"
    elif kind == "lora_leak":
        out_default = spec.default_lora_root
        out_env = f"{env_prefix}_LORA_LEAK_OUTPUT_DIR"
    else:
        raise ValueError(f"unknown kind: {kind}")
    output_dir = os.environ.get(out_env, out_default)
    data_dir = os.environ.get(
        f"{env_prefix}_MIMIR_DATA_DIR",
        "mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2/data",
    )

    argv = sys.argv[1:]
    add_default_arg(argv, "--model", [spec.key])
    add_default_arg(argv, "--model-name", [spec.hf_id])
    add_default_arg(argv, "--run-dir", [run_dir])
    add_default_arg(argv, "--adapter-dir", [run_dir])
    add_default_arg(argv, "--data-dir", [data_dir])
    add_default_arg(argv, "--output-dir", [output_dir])
    add_default_arg(argv, "--experiments", ["ft_vs_pt", "ft_vs_unseen"])
    sys.argv = [sys.argv[0], *argv]
    core_main()
