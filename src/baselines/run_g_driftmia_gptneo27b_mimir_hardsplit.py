# -*- coding: utf-8 -*-
"""Run G-DriftMIA for GPT-Neo-2.7B on the shared MIMIR hard split."""

from __future__ import annotations

import sys

from run_g_driftmia_mimir_hardsplit import main


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--model-name",
        "EleutherAI/gpt-neo-2.7B",
        "--run-dir",
        "/workplace/FT/BlackNLP_2/models/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_gptneo27b",
        "--adapter-dir",
        "/workplace/FT/BlackNLP_2/models/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_gptneo27b/adapter",
        "--data-dir",
        "/workplace/FT/BlackNLP_2/results/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2/data",
        "--output-dir",
        "/workplace/FT/BlackNLP_2/results/g_driftmia_mimir_hardsplit_gptneo27b",
        "--trainable-scope",
        "lora",
        "--max-length",
        "256",
        "--n-per-group",
        "500",
        "--repeats",
        "10",
        "--cv-splits",
        "5",
        "--comparisons",
        "ft_vs_pt",
        "ft_vs_unseen",
    ]
    main()
