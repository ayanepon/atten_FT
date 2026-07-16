# -*- coding: utf-8 -*-
"""Shared configuration for the Pythia-410m MIMIR fixed-20 rerun."""

import os
from pathlib import Path
from typing import Sequence

from experiment4_mimir_hardsplit_fixed20_attention import run_fixed20


MODEL_ROOT = Path(
    "models/"
    "mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_pythia410m"
)
OUTPUT_ROOT = Path(
    "results/"
    "mimir_wikipedia_hardsplit_fixed20_pythia410m_rerun"
)


def resolve_adapter_dir() -> Path:
    candidates = [MODEL_ROOT / "adapter", MODEL_ROOT]
    for candidate in candidates:
        if (candidate / "adapter_config.json").exists():
            return candidate
    tried = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Pythia-410m LoRA adapter was not found. Tried:\n" + tried
    )


def run_group(group: str) -> None:
    if group not in {"ft", "pt", "unseen"}:
        raise ValueError(f"Unsupported group: {group}")

    adapter_dir = resolve_adapter_dir()
    os.environ["BASE_MODEL_NAME"] = "EleutherAI/pythia-410m"
    os.environ["RUN_DIR"] = str(MODEL_ROOT)
    os.environ["ADAPTER_DIR"] = str(adapter_dir)
    os.environ["OUTPUT_DIR"] = str(OUTPUT_ROOT / f"fixed_attention_20_{group}")
    run_fixed20([group])


def run_groups(groups: Sequence[str]) -> None:
    for group in groups:
        run_group(group)
