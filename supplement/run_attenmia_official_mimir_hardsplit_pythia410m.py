# -*- coding: utf-8 -*-
"""AttenMIA entry for Pythia-410M (thin wrapper)."""
from __future__ import annotations

import run_attenmia_official_mimir_hardsplit as core
from hardsplit.cli_utils import run_with_model_defaults


def main() -> None:
    run_with_model_defaults(
        core_main=core.main,
        model_key="pythia-410m",
        env_prefix="PYTHIA410M",
        kind="attenmia",
    )


if __name__ == "__main__":
    main()
