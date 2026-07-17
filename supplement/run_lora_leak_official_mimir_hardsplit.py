# -*- coding: utf-8 -*-
"""Canonical entrypoint for LoRA-Leak on MIMIR hard split.

Implementation lives in ``run_lora_leak_official_mimir_hardsplit_2.py``
(historical name kept for remote/workspace compatibility).

Usage:
  python run_lora_leak_official_mimir_hardsplit.py --model pythia-1b
  python run_lora_leak_official_mimir_hardsplit.py --model pythia-410m
  python run_lora_leak_official_mimir_hardsplit.py --model gpt-neo-2.7b
"""

from run_lora_leak_official_mimir_hardsplit_2 import *  # noqa: F401,F403
from run_lora_leak_official_mimir_hardsplit_2 import main

if __name__ == "__main__":
    main()
