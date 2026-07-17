# -*- coding: utf-8 -*-
"""Unified fixed-20 pipeline: extract (optional) → Exp.1 stats → strict eval.

Paper-aligned end-to-end orchestration for multi-model MIMIR hard split.
Prefer ``orchestrate.py`` for multi-GPU jobs; this script is a simpler single-process path.

Examples:
  python run_unified_fixed20_pipeline.py --skip-extract \\
    --features-root attention_features_mimir_hardsplit

  python run_unified_fixed20_pipeline.py --model pythia-410m --skip-extract

  python run_unified_fixed20_pipeline.py --model pythia-1b --groups ft pt unseen
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from model_registry import (
    DEFAULT_EVAL_DIR,
    DEFAULT_EXP1_DIR,
    PYTHIA1B_FEATURES_ROOT,
    PYTHIA1B_RUN_DIR,
    add_model_arguments,
    apply_model_namespace,
)


HERE = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(HERE))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified fixed-20 extract + evaluate pipeline")
    p.add_argument("--python", default=sys.executable)
    add_model_arguments(p)
    p.add_argument("--run-dir", default=PYTHIA1B_RUN_DIR)
    p.add_argument("--features-root", default=PYTHIA1B_FEATURES_ROOT)
    p.add_argument("--groups", nargs="+", default=["ft", "pt", "unseen"], choices=["ft", "pt", "unseen"])
    p.add_argument("--n-per-group", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-5, help="Additional-training lr (paper-correct: 1e-5)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-exp1", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument(
        "--methods",
        nargs="+",
        default=["proposed_all", "proposed_en", "initial_loss", "loss_decrease", "lora_leak", "attenmia"],
    )
    p.add_argument("--lora-root", default="")
    p.add_argument("--attenmia-root", default="")
    p.add_argument("--eval-output-dir", default=DEFAULT_EVAL_DIR)
    p.add_argument("--exp1-output-dir", default=DEFAULT_EXP1_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args = apply_model_namespace(args, profile="pipeline", log=print)
    py = args.python
    root = Path(args.features_root)
    model_key = getattr(args, "model_key", None) or "pythia1b"

    if not args.skip_extract:
        for group in args.groups:
            out = root / f"fixed_attention_20_{group}"
            cmd = [
                py,
                "extract_attention_hardsplit.py",
                "--run-dir",
                args.run_dir,
                "--output-dir",
                str(out),
                "--no-run-dynamic",
                "--fixed-steps",
                "20",
                "--groups",
                group,
                "--n-per-group",
                str(args.n_per_group),
                "--lr",
                str(args.lr),
                "--seed",
                str(args.seed),
                "--skip-analyze",
            ]
            if args.model_name:
                cmd.extend(["--model-name", args.model_name])
            cmd.append("--no-resume" if args.no_resume else "--resume")
            run(cmd)

    if not args.skip_exp1:
        run(
            [
                py,
                "analyze_exp1_layer_head_significance.py",
                "--root",
                str(root),
                "--output-dir",
                args.exp1_output_dir,
            ]
        )

    if not args.skip_eval:
        cmd = [
            py,
            "run_strict_fixed20_comparison_10runs.py",
            "--models",
            model_key,
            f"--{model_key}-proposed-root",
            str(root),
            "--output-dir",
            args.eval_output_dir,
            "--repeats",
            str(args.repeats),
            "--seed",
            str(args.seed),
            "--methods",
            *args.methods,
        ]
        if args.lora_root:
            cmd.extend([f"--{model_key}-lora-root", args.lora_root])
        if args.attenmia_root:
            cmd.extend([f"--{model_key}-attenmia-root", args.attenmia_root])
        run(cmd)

    print("\nUnified pipeline finished.")
    print(f"  model:    {model_key}")
    print(f"  features: {root.resolve()}")
    print(f"  exp1:     {(HERE / args.exp1_output_dir).resolve()}")
    print(f"  eval:     {(HERE / args.eval_output_dir).resolve()}")


if __name__ == "__main__":
    main()
