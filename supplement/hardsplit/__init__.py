# -*- coding: utf-8 -*-
"""BlackboxNLP MIMIR hard-split experiment package.

Layout:
  hardsplit.models      — multi-model presets / CLI namespace
  hardsplit.cli_utils   — thin model-specific entry helpers
  hardsplit.progress    — incremental CSV progress writers
  hardsplit.parallel    — sample sharding helpers
  hardsplit.amp_utils   — mixed-precision helpers for overfit

Top-level scripts (extract_attention_hardsplit.py, orchestrate.py, …)
remain the user-facing entrypoints and re-export / call into this package.
"""

from __future__ import annotations

__version__ = "0.2.0"
