# -*- coding: utf-8 -*-
"""Generic fixed-20 attention extraction runner for any model_registry preset.

Used by model-specific entrypoints (pythia-410m / gpt-neo-2.7b / …).
Wraps ``extract_attention_hardsplit.py`` with paper-aligned defaults:
  fixed steps = 20, additional-training lr = 1e-5, skip post-hoc analyze.

Example:
  from experiment4_fixed20_common import make_runner
  run_group = make_runner("pythia-410m", env_prefix="PYTHIA410M")
  run_group("ft")
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Sequence

from model_registry import ModelSpec, resolve_adapter_dir, resolve_model_spec


def make_runner(
    model_key: str,
    *,
    env_prefix: str | None = None,
    model_root_env: str | None = None,
    adapter_env: str | None = None,
    output_root_env: str | None = None,
) -> tuple[Callable[[str], None], Callable[[Sequence[str]], None], ModelSpec]:
    """Return (run_group, run_groups, spec) for a model preset."""
    spec = resolve_model_spec(model_key)
    prefix = env_prefix or spec.short_name.upper()
    root_env = model_root_env or f"{prefix}_MIMIR_MODEL_ROOT"
    ad_env = adapter_env or f"{prefix}_ADAPTER_DIR"
    out_env = output_root_env or f"{prefix}_FIXED20_OUTPUT_ROOT"

    model_root = Path(os.environ.get(root_env, spec.default_run_dir))
    output_root = Path(os.environ.get(out_env, spec.default_features_root))

    def resolve_run_adapter() -> Path:
        if os.environ.get(ad_env):
            return resolve_adapter_dir(os.environ[ad_env])
        return resolve_adapter_dir(model_root)

    def run_group(group: str) -> None:
        if group not in {"ft", "pt", "unseen"}:
            raise ValueError(f"Unsupported group: {group}")

        adapter_dir = resolve_run_adapter()
        output_dir = output_root / f"fixed_attention_20_{group}"

        os.environ["BASE_MODEL_NAME"] = spec.hf_id
        os.environ["MIMIR_HARDSPLIT_RUN_DIR"] = str(model_root)
        os.environ["MIMIR_HARDSPLIT_BASE_DIR"] = str(model_root)
        os.environ["MIMIR_HARDSPLIT_ADAPTER_DIR"] = str(adapter_dir)
        os.environ["OUTPUT_DIR"] = str(output_dir)

        from extract_attention_hardsplit import main

        old_argv = sys.argv[:]
        try:
            sys.argv = [
                old_argv[0],
                "--run-dir",
                str(model_root),
                "--adapter-dir",
                str(adapter_dir),
                "--model-name",
                spec.hf_id,
                "--output-dir",
                str(output_dir),
                "--no-run-dynamic",
                "--fixed-steps",
                "20",
                "--groups",
                group,
                "--lr",
                "1e-5",
                "--skip-analyze",
                "--flush-every",
                "25",
            ]
            main()
        finally:
            sys.argv = old_argv

    def run_groups(groups: Sequence[str]) -> None:
        for group in groups:
            run_group(group)

    return run_group, run_groups, spec
