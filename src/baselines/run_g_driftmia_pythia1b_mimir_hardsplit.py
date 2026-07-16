# -*- coding: utf-8 -*-
"""Run G-DriftMIA for Pythia-1B on the shared MIMIR hard split."""

from __future__ import annotations

import sys

from run_g_driftmia_mimir_hardsplit import main


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--model-name",
        "EleutherAI/pythia-1b",
        "--run-dir",
        "/workplace/FT/BlackNLP_2/results/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2",
        "--adapter-dir",
        "/workplace/FT/BlackNLP_2/results/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2/adapter",
        "--data-dir",
        "/workplace/FT/BlackNLP_2/results/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2/data",
        "--output-dir",
        "/workplace/FT/BlackNLP_2/results/g_driftmia_mimir_hardsplit_pythia1b",
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
