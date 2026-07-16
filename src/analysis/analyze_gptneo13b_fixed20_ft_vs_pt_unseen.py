# -*- coding: utf-8 -*-
"""
Analyze GPT-Neo-1.3B fixed-20 attention-update results.

This script reads the already extracted result folders:

  fixed_attention_20_ft/
  fixed_attention_20_pt/
  fixed_attention_20_unseen/

and evaluates:

  - FT vs PT
  - FT vs Unseen

using the same repeated 5-fold OOF AUC analysis as the previous fixed-step
MIMIR experiments. FT is the positive class, and AUC is not flipped.

Default input:
  results/mimir_wikipedia_hardsplit_fixed20_gptneo13b

Default output:
  results/mimir_wikipedia_hardsplit_fixed20_gptneo13b/ft_vs_pt_unseen_auc_analysis

Run:
  python3 analyze_gptneo13b_fixed20_ft_vs_pt_unseen.py

Optional:
  python3 analyze_gptneo13b_fixed20_ft_vs_pt_unseen.py \
    --input-root /path/to/mimir_wikipedia_hardsplit_fixed20_gptneo13b \
    --output-dir /path/to/output
"""

import argparse
import sys
from pathlib import Path

import analyze_mimir_fixed_steps_repeated_auc as core


DEFAULT_INPUT_ROOT = "results/mimir_wikipedia_hardsplit_fixed20_gptneo13b"
DEFAULT_OUTPUT_DIR = (
    "results/"
    "mimir_wikipedia_hardsplit_fixed20_gptneo13b/"
    "ft_vs_pt_unseen_auc_analysis"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--methods",
        default="elasticnet,all",
        help="elasticnet: fold-internal feature selection + L2 logistic; all: all layer/head features + L2 logistic",
    )
    parser.add_argument("--selection-c", type=float, default=0.1)
    parser.add_argument("--classifier-c", type=float, default=1.0)
    parser.add_argument("--elasticnet-l1-ratio", type=float, default=0.7)
    parser.add_argument("--elasticnet-max-iter", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            old_argv[0],
            "--input-root",
            args.input_root,
            "--output-dir",
            args.output_dir,
            "--steps",
            "20",
            "--repeats",
            str(args.repeats),
            "--cv-splits",
            str(args.cv_splits),
            "--seed",
            str(args.seed),
            "--methods",
            args.methods,
            "--selection-c",
            str(args.selection_c),
            "--classifier-c",
            str(args.classifier_c),
            "--elasticnet-l1-ratio",
            str(args.elasticnet_l1_ratio),
            "--elasticnet-max-iter",
            str(args.elasticnet_max_iter),
        ]
        core.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
