# -*- coding: utf-8 -*-
"""Run Pythia-1B attention-update extraction for 20/50/100 and early stopping.

This is the reproduction entry point for the stopping-condition ablation.
It uses the same MIMIR hard-split data and the same fine-tuned checkpoint as
the main Pythia-1B experiment.
"""

import os
import sys
from pathlib import Path


MODEL_ROOT = Path(
    os.environ.get(
        "PYTHIA1B_MIMIR_MODEL_ROOT",
        "results/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2",
    )
)
OUTPUT_ROOT = Path(
    os.environ.get(
        "PYTHIA1B_STOPPING_OUTPUT_ROOT",
        "results/experiment4_mimir_hardsplit_stopping_condition",
    )
)


def resolve_adapter_dir() -> Path:
    candidates = [
        Path(os.environ["PYTHIA1B_ADAPTER_DIR"]) if os.environ.get("PYTHIA1B_ADAPTER_DIR") else None,
        MODEL_ROOT / "adapter",
        MODEL_ROOT,
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "adapter_config.json").exists():
            return candidate
    tried = "\n".join(f"  - {path}" for path in candidates if path is not None)
    raise FileNotFoundError("Pythia-1B LoRA adapter was not found. Tried:\n" + tried)


def main() -> None:
    adapter_dir = resolve_adapter_dir()
    os.environ["BASE_MODEL_NAME"] = "EleutherAI/pythia-1b"
    os.environ["MIMIR_HARDSPLIT_RUN_DIR"] = str(MODEL_ROOT)
    os.environ["MIMIR_HARDSPLIT_BASE_DIR"] = str(MODEL_ROOT)
    os.environ["MIMIR_HARDSPLIT_ADAPTER_DIR"] = str(adapter_dir)
    os.environ["OUTPUT_DIR"] = str(OUTPUT_ROOT)

    from experiment4_mimir_hardsplit_stopping_condition import main as run_main

    sys.argv = [
        sys.argv[0],
        "--run-dir",
        str(MODEL_ROOT),
        "--adapter-dir",
        str(adapter_dir),
        "--output-dir",
        str(OUTPUT_ROOT),
        "--fixed-steps",
        "20",
        "50",
        "100",
        "--run-dynamic",
        "--groups",
        "ft",
        "pt",
        "unseen",
    ]
    run_main()


if __name__ == "__main__":
    main()
