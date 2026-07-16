# -*- coding: utf-8 -*-
"""LoRA fine-tuning on the MIMIR Wikipedia hard split for Pythia-410M.

This wrapper reuses the Pythia-1B training implementation but changes the base
model and output directory to EleutherAI/pythia-410m.
"""

import sys

import train_mimir_wikipedia_hardsplit_lora as base


BASE_MODEL_NAME = "EleutherAI/pythia-410m"
DEFAULT_OUTPUT_DIR = "models/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_pythia410m"


def main() -> None:
    base.BASE_MODEL_NAME = BASE_MODEL_NAME
    base.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR

    argv = sys.argv[1:]
    if "--model-name" not in argv:
        argv = argv + ["--model-name", BASE_MODEL_NAME]
    if "--output-dir" not in argv:
        argv = argv + ["--output-dir", DEFAULT_OUTPUT_DIR]

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + argv
        base.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
