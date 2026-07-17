# -*- coding: utf-8 -*-
"""LoRA-Leak entry for GPT-Neo-2.7B (thin wrapper)."""
from __future__ import annotations

try:
    import run_lora_leak_official_mimir_hardsplit as core
except ImportError:  # pragma: no cover
    import run_lora_leak_official_mimir_hardsplit_2 as core
from hardsplit.cli_utils import run_with_model_defaults


def main() -> None:
    run_with_model_defaults(
        core_main=core.main,
        model_key="gpt-neo-2.7b",
        env_prefix="GPTNEO27B",
        kind="lora_leak",
    )


if __name__ == "__main__":
    main()
