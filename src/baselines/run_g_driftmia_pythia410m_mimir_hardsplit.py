# -*- coding: utf-8 -*-
"""Run G-DriftMIA for Pythia-410M on the shared MIMIR hard split."""

from __future__ import annotations

import sys

from run_g_driftmia_mimir_hardsplit import main


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--model-name",
        "EleutherAI/pythia-410m",
        "--run-dir",
        "/workplace/FT/BlackNLP_2/models/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_pythia410m",
        "--adapter-dir",
        "/workplace/FT/BlackNLP_2/models/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_pythia410m/adapter",
        "--data-dir",
        "/workplace/FT/BlackNLP_2/results/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2/data",
        "--output-dir",
        "/workplace/FT/BlackNLP_2/results/g_driftmia_mimir_hardsplit_pythia410m",
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
