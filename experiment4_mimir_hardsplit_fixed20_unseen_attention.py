# -*- coding: utf-8 -*-
"""Run Experiment 4 fixed-20-step attention extraction for Unseen only."""

import os
import sys

from extract_attention_hardsplit import DEFAULT_OUTPUT_DIR, main


if __name__ == "__main__":
    out = os.environ.get("OUTPUT_DIR", f"{DEFAULT_OUTPUT_DIR}/fixed_attention_20_unseen")
    sys.argv.extend([
        "--output-dir", out,
        "--no-run-dynamic",
        "--fixed-steps", "20",
        "--groups", "unseen",
        "--skip-analyze",
    ])
    main()
