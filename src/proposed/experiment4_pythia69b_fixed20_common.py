# -*- coding: utf-8 -*-
"""Shared configuration for the Pythia-6.9B MIMIR fixed-20 rerun."""

import os
import sys
from pathlib import Path
from typing import Sequence


MODEL_ROOT = Path(
    "models/"
    "mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_pythia69b"
)
OUTPUT_ROOT = Path(
    "results/"
    "mimir_wikipedia_hardsplit_fixed20_pythia69b_rerun"
)


def resolve_adapter_dir() -> Path:
    candidates = [MODEL_ROOT / "adapter", MODEL_ROOT]
    for candidate in candidates:
        if (candidate / "adapter_config.json").exists():
            return candidate
    tried = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Pythia-6.9B LoRA adapter was not found. Tried:\n" + tried
    )


def run_group(group: str) -> None:
    if group not in {"ft", "pt", "unseen"}:
        raise ValueError(f"Unsupported group: {group}")

    adapter_dir = resolve_adapter_dir()
    os.environ["BASE_MODEL_NAME"] = "EleutherAI/pythia-6.9b"
    os.environ["MIMIR_HARDSPLIT_RUN_DIR"] = str(MODEL_ROOT)
    os.environ["MIMIR_HARDSPLIT_ADAPTER_DIR"] = str(adapter_dir)
    os.environ["OUTPUT_DIR"] = str(OUTPUT_ROOT / f"fixed_attention_20_{group}")

    # Import after setting BASE_MODEL_NAME because the attention common module
    # reads the model name at import time.
    from experiment4_mimir_hardsplit_stopping_condition import main

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            old_argv[0],
            "--run-dir",
            str(MODEL_ROOT),
            "--adapter-dir",
            str(adapter_dir),
            "--output-dir",
            str(OUTPUT_ROOT / f"fixed_attention_20_{group}"),
            "--no-run-dynamic",
            "--fixed-steps",
            "20",
            "--groups",
            group,
        ]
        main()
    finally:
        sys.argv = old_argv


def run_groups(groups: Sequence[str]) -> None:
    for group in groups:
        run_group(group)
