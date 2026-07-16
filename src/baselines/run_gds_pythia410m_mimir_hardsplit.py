# -*- coding: utf-8 -*-
"""Run GDS for Pythia-410M on the shared MIMIR hard split."""

from __future__ import annotations

import sys

from run_gds_mimir_hardsplit import main


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--model-name",
        "EleutherAI/pythia-410m",
        "--data-dir",
        "/workplace/FT/BlackNLP_2/results/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2/data",
        "--output-dir",
        "/workplace/FT/BlackNLP_2/results/gds_mimir_hardsplit_pythia410m",
        "--max-length",
        "256",
        "--n-per-group",
        "500",
        "--comparisons",
        "ft_vs_pt",
        "ft_vs_unseen",
    ]
    main()
