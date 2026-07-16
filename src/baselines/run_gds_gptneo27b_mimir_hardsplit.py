# -*- coding: utf-8 -*-
"""Run GDS for GPT-Neo-2.7B on the shared MIMIR hard split."""

from __future__ import annotations

import sys

from run_gds_mimir_hardsplit import main


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--model-name",
        "EleutherAI/gpt-neo-2.7B",
        "--data-dir",
        "/workplace/FT/BlackNLP_2/results/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2/data",
        "--output-dir",
        "/workplace/FT/BlackNLP_2/results/gds_mimir_hardsplit_gptneo27b",
        "--max-length",
        "256",
        "--n-per-group",
        "500",
        "--comparisons",
        "ft_vs_pt",
        "ft_vs_unseen",
    ]
    main()
