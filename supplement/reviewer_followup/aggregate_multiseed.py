#!/usr/bin/env python3
"""Aggregate classifier results at the checkpoint level, not CV-repeat level."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from reviewer_followup.common import atomic_write_csv, atomic_write_json, base_manifest, t_interval


def parse_seed_paths(values: list[str]) -> dict[int, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected SEED=CSV, got {value}")
        seed, path = value.split("=", 1)
        result[int(seed)] = Path(path)
    if len(result) < 2:
        raise ValueError("At least two checkpoint seeds are required")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True, help="SEED=summary_auc.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", default="proposed_en")
    parser.add_argument("--seed-axis", choices=["ft", "sample"], default="ft")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    inputs = parse_seed_paths(args.result)
    rows = []
    for seed, path in sorted(inputs.items()):
        frame = pd.read_csv(path)
        if "method" in frame.columns:
            frame = frame[frame["method"] == args.method]
        for row in frame.itertuples(index=False):
            auc = row.auc_mean if hasattr(row, "auc_mean") else row.auc
            tpr = (
                row.tpr_at_10_fpr_mean
                if hasattr(row, "tpr_at_10_fpr_mean")
                else getattr(row, "tpr_at_10_fpr", float("nan"))
            )
            rows.append(
                {
                    f"{args.seed_axis}_seed": seed,
                    "comparison": row.comparison,
                    "method": getattr(row, "method", args.method),
                    "auc": float(auc),
                    "split_sd": float(getattr(row, "auc_std", float("nan"))),
                    "tpr_at_10_fpr": float(tpr),
                    "source": str(path),
                }
            )
    per_checkpoint = pd.DataFrame(rows)
    summary_rows = []
    rng = np.random.default_rng(args.seed)
    for comparison, part in per_checkpoint.groupby("comparison"):
        values = part["auc"].to_numpy(float)
        low, high = t_interval(values.tolist())
        boot = np.asarray(
            [rng.choice(values, size=len(values), replace=True).mean() for _ in range(args.n_bootstrap)], dtype=float
        )
        prefix = "checkpoint" if args.seed_axis == "ft" else "sample_seed"
        summary_rows.append(
            {
                "comparison": comparison,
                "method": args.method,
                f"n_{prefix}s": int(len(part)),
                f"auc_{prefix}_mean": float(part["auc"].mean()),
                f"auc_{prefix}_sd": float(part["auc"].std(ddof=1)),
                f"auc_{prefix}_t_ci_low": low,
                f"auc_{prefix}_t_ci_high": high,
                f"auc_{prefix}_bootstrap_ci_low": float(np.quantile(boot, 0.025)),
                f"auc_{prefix}_bootstrap_ci_high": float(np.quantile(boot, 0.975)),
                f"n_{prefix}s_auc_gt_0_5": int((part["auc"] > 0.5).sum()),
                "split_sd_mean_descriptive": float(part["split_sd"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = "checkpoint" if args.seed_axis == "ft" else "sample_seed"
    atomic_write_csv(output / f"{stem}_results.csv", per_checkpoint)
    atomic_write_csv(output / f"{stem}_summary.csv", summary)
    manifest = base_manifest(experiment="e9_multiseed_aggregation", command=sys.argv)
    manifest.update(
        {
            "status": "completed",
            "method": args.method,
            "seed_axis": args.seed_axis,
            "seeds": sorted(inputs),
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_seed": args.seed,
        }
    )
    atomic_write_json(output / "checkpoint_manifest.json", manifest)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
