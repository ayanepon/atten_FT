# -*- coding: utf-8 -*-
"""One-shot dynamic-only extract (early stopping) for paper Exp.3.

Paper settings (acl_latex.tex Additional Training Conditions):
  - lr = 1e-5
  - patience = 50 steps (loss and accuracy; tol = 1e-6)
  - min_steps = 1
  - max_steps = 5000
  - early_stop_on_accuracy = True
"""
from __future__ import annotations

import argparse
import sys

import extract_attention_hardsplit as ex
from hardsplit.progress import ExtractOwnershipError

# Distinguishable exit code so the orchestrator can tell "another host already
# owns this output_dir" apart from a real failure and skip instead of crashing
# the whole run.
EXIT_OWNERSHIP_CONFLICT = 75


def main() -> None:
    p = argparse.ArgumentParser(description="Dynamic (early-stopping) attention extract")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--model-name", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--groups", nargs="+", default=["ft"])
    p.add_argument("--n-per-group", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early-stopping-patience", type=int, default=50)
    p.add_argument("--early-stopping-tol", type=float, default=1e-6)
    p.add_argument("--early-stopping-min-steps", type=int, default=1)
    p.add_argument("--max-overfit-steps", type=int, default=5000)
    p.add_argument("--flush-every", type=int, default=25)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--shard", default="", help="Sample shard K/N (0-based), e.g. 0/2")
    args = p.parse_args()

    ns = argparse.Namespace(
        run_dir=args.run_dir,
        adapter_dir=args.adapter_dir,
        model_name=args.model_name,
        output_dir=args.output_dir,
        n_per_group=args.n_per_group,
        fixed_steps=[],  # dynamic only
        run_dynamic=True,
        groups=args.groups,
        lr=args.lr,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_tol=args.early_stopping_tol,
        early_stopping_min_steps=args.early_stopping_min_steps,
        max_overfit_steps=args.max_overfit_steps,
        seed=args.seed,
        analyze_only=False,
        skip_analyze=True,
        resume=True,
        flush_every=args.flush_every,
        record_train_curve=False,
        verbose_samples=False,
        early_stop_on_accuracy=True,
        shard=args.shard,
        amp=not args.no_amp,
    )
    try:
        ex.run_extraction(ns)
    except ExtractOwnershipError as e:
        print(f"[skip] {e}", file=sys.stderr)
        sys.exit(EXIT_OWNERSHIP_CONFLICT)


if __name__ == "__main__":
    main()
