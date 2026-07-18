#!/usr/bin/env python3
"""Aggregate crossed-design OOF predictions over independent FT checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from reviewer_followup.analyze_factorial_uncertainty import average_target_scores
from reviewer_followup.common import atomic_write_csv, atomic_write_json, base_manifest, sha256_file, t_interval


def parse_seed_path(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise ValueError(f"Expected SEED=CSV, got {value}")
    seed, path = value.split("=", 1)
    return int(seed), Path(path)


def aggregate(inputs: dict[int, Path], *, n_bootstrap: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    averaged_by_seed: dict[int, pd.DataFrame] = {}
    for checkpoint_seed, path in sorted(inputs.items()):
        averaged = average_target_scores(pd.read_csv(path))
        averaged_by_seed[checkpoint_seed] = averaged
        for (comparison, method), part in averaged.groupby(["comparison", "method"]):
            rows.append(
                {
                    "ft_seed": checkpoint_seed, "comparison": comparison, "method": method,
                    "auc": float(roc_auc_score(part["y_true"], part["score"])),
                    "n_targets": int(len(part)), "source": str(path),
                }
            )
    per_checkpoint = pd.DataFrame(rows)
    summary_rows = []
    rng = np.random.default_rng(seed)
    for (comparison, method), part in per_checkpoint.groupby(["comparison", "method"]):
        aucs = part.sort_values("ft_seed")["auc"].to_numpy(float)
        low_t, high_t = t_interval(aucs)
        boot = np.empty(n_bootstrap, dtype=float)
        seeds = part["ft_seed"].to_numpy(int)
        for draw in range(n_bootstrap):
            sampled_seeds = rng.choice(seeds, len(seeds), replace=True)
            seed_aucs = []
            for checkpoint_seed in sampled_seeds:
                target = averaged_by_seed[int(checkpoint_seed)]
                target = target[(target["comparison"] == comparison) & (target["method"] == method)]
                y = target["y_true"].to_numpy(int)
                score = target["score"].to_numpy(float)
                pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
                index = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
                seed_aucs.append(roc_auc_score(y[index], score[index]))
            boot[draw] = float(np.mean(seed_aucs))
        low, high = np.quantile(boot, [0.025, 0.975])
        summary_rows.append(
            {
                "comparison": comparison, "method": method, "n_ft_checkpoints": int(len(part)),
                "auc_checkpoint_mean": float(aucs.mean()), "auc_checkpoint_sd": float(aucs.std(ddof=1)),
                "auc_checkpoint_t_ci_low": low_t, "auc_checkpoint_t_ci_high": high_t,
                "auc_hierarchical_bootstrap_ci_low": float(low), "auc_hierarchical_bootstrap_ci_high": float(high),
                "n_bootstrap": n_bootstrap, "bootstrap_units": "ft_checkpoint_then_target",
            }
        )
    return per_checkpoint, pd.DataFrame(summary_rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", required=True, help="SEED=factorial_outer_predictions.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    inputs = dict(parse_seed_path(value) for value in args.prediction)
    if len(inputs) < 2:
        raise ValueError("At least two independent FT checkpoints are required")
    per_checkpoint, summary = aggregate(inputs, n_bootstrap=args.n_bootstrap, seed=args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output / "crossed_checkpoint_results.csv", per_checkpoint)
    atomic_write_csv(output / "crossed_checkpoint_summary.csv", summary)
    manifest = base_manifest(experiment="e13_crossed_multicheckpoint", command=sys.argv)
    manifest.update(
        {
            "status": "completed", "ft_seeds": sorted(inputs), "n_bootstrap": args.n_bootstrap,
            "bootstrap_units": ["ft_checkpoint", "target"],
            "inputs": {str(seed): {"path": str(path), "sha256": sha256_file(path)} for seed, path in inputs.items()},
        }
    )
    atomic_write_json(output / "crossed_checkpoint_manifest.json", manifest)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
