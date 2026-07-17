# -*- coding: utf-8 -*-
"""Pythia-410M fixed-20 extract runner (via shared factory)."""
from __future__ import annotations

import sys

from experiment4_fixed20_common import make_runner

run_group, run_groups, SPEC = make_runner("pythia-410m", env_prefix="PYTHIA410M")

if __name__ == "__main__":
    run_groups(sys.argv[1:] or ["ft", "pt", "unseen"])
