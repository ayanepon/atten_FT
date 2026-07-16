# -*- coding: utf-8 -*-
"""Shared configuration for the GPT-Neo-1.3B MIMIR fixed-20 experiment."""

import os
import sys
from pathlib import Path
from typing import Sequence


MODEL_ROOT = Path(
    os.environ.get(
        "GPTNEO13B_MIMIR_MODEL_ROOT",
        "models/"
        "mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_gptneo13b",
    )
)
OUTPUT_ROOT = Path(
    os.environ.get(
        "GPTNEO13B_FIXED20_OUTPUT_ROOT",
        "results/"
        "mimir_wikipedia_hardsplit_fixed20_gptneo13b",
    )
)


def resolve_adapter_dir() -> Path:
    candidates = [
        Path(os.environ["GPTNEO13B_ADAPTER_DIR"]) if os.environ.get("GPTNEO13B_ADAPTER_DIR") else None,
        MODEL_ROOT / "adapter",
        MODEL_ROOT,
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "adapter_config.json").exists():
            return candidate
    tried = "\n".join(f"  - {path}" for path in candidates if path is not None)
    raise FileNotFoundError(
        "GPT-Neo-1.3B LoRA adapter was not found. Tried:\n" + tried
    )


def run_group(group: str) -> None:
    if group not in {"ft", "pt", "unseen"}:
        raise ValueError(f"Unsupported group: {group}")

    adapter_dir = resolve_adapter_dir()
    output_dir = OUTPUT_ROOT / f"fixed_attention_20_{group}"

    os.environ["BASE_MODEL_NAME"] = "EleutherAI/gpt-neo-1.3B"
    os.environ["MIMIR_HARDSPLIT_RUN_DIR"] = str(MODEL_ROOT)
    os.environ["MIMIR_HARDSPLIT_BASE_DIR"] = str(MODEL_ROOT)
    os.environ["MIMIR_HARDSPLIT_ADAPTER_DIR"] = str(adapter_dir)
    os.environ["OUTPUT_DIR"] = str(output_dir)

    # Import after setting BASE_MODEL_NAME because mimir_hardsplit_attention_common
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
            str(output_dir),
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
