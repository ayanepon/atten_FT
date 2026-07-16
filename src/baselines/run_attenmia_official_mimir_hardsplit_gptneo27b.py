# -*- coding: utf-8 -*-
"""Run the AttenMIA-style baseline for GPT-Neo-2.7B on the MIMIR hard split."""

import run_attenmia_official_mimir_hardsplit as baseline


baseline.BASE_MODEL_NAME = "EleutherAI/gpt-neo-2.7B"
baseline.DEFAULT_RUN_DIR = "models/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_gptneo27b"
baseline.DEFAULT_OUTPUT_DIR = "results/attenmia_official_mimir_hardsplit_gptneo27b"


if __name__ == "__main__":
    baseline.main()
