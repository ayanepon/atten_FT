# -*- coding: utf-8 -*-
"""Run MIMIR hard-split attention extraction for exactly 20 optimizer steps.

Thin multi-model launcher around ``extract_attention_hardsplit.py``.

Defaults come from ``model_registry`` (or env overrides).  Before loading, the
LoRA adapter's ``base_model_name_or_path`` is validated against the requested
model to avoid opaque PEFT tensor-size mismatches.

Environment overrides:
  MODEL_PRESET                 e.g. pythia-410m | gpt-neo-2.7b | pythia-1b
  BASE_MODEL_NAME              HF id override
  MIMIR_HARDSPLIT_RUN_DIR      (legacy alias: RUN_DIR)
  MIMIR_HARDSPLIT_ADAPTER_DIR  (legacy alias: ADAPTER_DIR)
  OUTPUT_DIR / OUTPUT_ROOT

Workplace example (Pythia-410M):
  export MODEL_PRESET=pythia-410m
  export MIMIR_HARDSPLIT_RUN_DIR=/workplace/FT/BlackNLP_2/models/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_pythia410m
  python experiment4_mimir_hardsplit_fixed20_attention.py ft
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from model_registry import resolve_adapter_dir, resolve_model_name, resolve_model_spec


def resolve_defaults() -> tuple[str, Path, Path, Path]:
    """Return (hf_id, run_dir, adapter_hint, default_output_root)."""
    preset = os.environ.get("MODEL_PRESET") or os.environ.get("BASE_MODEL_PRESET") or "pythia-410m"
    try:
        spec = resolve_model_spec(preset)
        hf_default = spec.hf_id
        run_default = spec.default_run_dir
        out_default = spec.default_features_root
    except KeyError:
        hf_default = "EleutherAI/pythia-410m"
        run_default = "mimir_lora_pythia410m"
        out_default = "attention_features_pythia410m"

    hf = os.environ.get("BASE_MODEL_NAME", hf_default)
    run_dir = Path(
        os.environ.get(
            "MIMIR_HARDSPLIT_RUN_DIR",
            os.environ.get("RUN_DIR", run_default),
        )
    )
    adapter_hint = Path(
        os.environ.get(
            "MIMIR_HARDSPLIT_ADAPTER_DIR",
            os.environ.get("ADAPTER_DIR", str(run_dir / "adapter")),
        )
    )
    out_root = Path(os.environ.get("OUTPUT_ROOT", out_default))
    # model-specific convenience env vars
    if os.environ.get("PYTHIA410M_FIXED20_OUTPUT_ROOT") and "410" in hf.lower():
        out_root = Path(os.environ["PYTHIA410M_FIXED20_OUTPUT_ROOT"])
    if os.environ.get("GPTNEO27B_FIXED20_OUTPUT_ROOT") and "gpt-neo" in hf.lower():
        out_root = Path(os.environ["GPTNEO27B_FIXED20_OUTPUT_ROOT"])
    if os.environ.get("OUTPUT_ROOT"):
        out_root = Path(os.environ["OUTPUT_ROOT"])
    return hf, run_dir, adapter_hint, out_root


def validate_adapter(adapter_or_run: Path, expected_model: str) -> Path:
    """Resolve adapter dir and ensure base model matches expected HF id."""
    resolved = resolve_adapter_dir(adapter_or_run, run_dir=adapter_or_run)
    config_path = resolved / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    adapter_model = config.get("base_model_name_or_path")
    expected = resolve_model_name(explicit=expected_model, default=expected_model)
    actual = resolve_model_name(explicit=adapter_model, default=adapter_model or "")
    if actual and expected and actual != expected:
        raise RuntimeError(
            "LoRA adapter/base-model mismatch.\n"
            f"  adapter was trained for: {adapter_model!r}\n"
            f"  fixed20 will load:       {expected_model!r}\n"
            "Use an adapter trained with the same base model."
        )
    return resolved


def run_fixed20(groups: Optional[Sequence[str]] = None) -> None:
    model_name, run_dir, adapter_hint, output_root = resolve_defaults()
    model_name = os.environ.get("BASE_MODEL_NAME", model_name)

    group_list: List[str] = list(groups) if groups else ["ft", "pt", "unseen"]
    if len(group_list) == 1:
        default_out = output_root / f"fixed_attention_20_{group_list[0]}"
    else:
        default_out = output_root / f"fixed_attention_20_{'_'.join(group_list)}"
    output_dir = Path(os.environ.get("OUTPUT_DIR", str(default_out)))

    try:
        adapter_dir = validate_adapter(adapter_hint, model_name)
    except FileNotFoundError:
        adapter_dir = validate_adapter(run_dir, model_name)

    os.environ["BASE_MODEL_NAME"] = model_name
    os.environ["MIMIR_HARDSPLIT_RUN_DIR"] = str(run_dir)
    os.environ["MIMIR_HARDSPLIT_BASE_DIR"] = str(run_dir)
    os.environ["MIMIR_HARDSPLIT_ADAPTER_DIR"] = str(adapter_dir)
    os.environ["OUTPUT_DIR"] = str(output_dir)

    from extract_attention_hardsplit import main as experiment_main

    arguments = [
        "--run-dir",
        str(run_dir),
        "--adapter-dir",
        str(adapter_dir),
        "--model-name",
        model_name,
        "--output-dir",
        str(output_dir),
        "--no-run-dynamic",
        "--fixed-steps",
        "20",
        "--lr",
        "1e-5",
        "--skip-analyze",
        "--groups",
        *group_list,
    ]
    old_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], *arguments]
        print(f"Base model: {model_name}")
        print(f"LoRA adapter: {adapter_dir}")
        print(f"Run dir: {run_dir}")
        print(f"Output: {output_dir}")
        print(f"Groups: {group_list}")
        experiment_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    cli_groups = [g for g in sys.argv[1:] if not g.startswith("-")]
    run_fixed20(cli_groups or None)
