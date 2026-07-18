#!/usr/bin/env python3
"""Target-bootstrap AUC intervals and paired deltas from strict OOF scores."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from reviewer_followup.common import atomic_write_csv, atomic_write_json, base_manifest, sha256_file


def infer_oof_uncertainty(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    reference_method: str = "proposed_en",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"model", "comparison", "method", "repeat", "uid", "y_true", "score"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"OOF prediction CSV missing {sorted(missing)}")
    if predictions.duplicated(["model", "comparison", "method", "repeat", "uid"]).any():
        raise ValueError("Duplicate OOF predictions")
    averaged = (
        predictions.groupby(["model", "comparison", "method", "uid", "y_true"], as_index=False)["score"].mean()
    )
    method_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    for group_index, ((model, comparison), part) in enumerate(
        averaged.groupby(["model", "comparison"], sort=True)
    ):
        wide = part.pivot(index=["uid", "y_true"], columns="method", values="score").reset_index()
        y = wide["y_true"].to_numpy(int)
        positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        rng = np.random.default_rng(seed + group_index * 100_000)
        samples = [
            np.concatenate(
                [rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)]
            )
            for _ in range(n_bootstrap)
        ]
        methods = [column for column in wide.columns if column not in {"uid", "y_true"}]
        for method in methods:
            valid = wide[method].notna().to_numpy()
            if not valid.all():
                continue
            scores = wide[method].to_numpy(float)
            point = float(roc_auc_score(y, scores))
            boot = np.asarray([roc_auc_score(y[index], scores[index]) for index in samples])
            low, high = np.quantile(boot, [0.025, 0.975])
            method_rows.append(
                {
                    "model": model, "comparison": comparison, "method": method,
                    "auc": point, "ci_low": float(low), "ci_high": float(high),
                    "n_targets": int(len(wide)), "n_bootstrap": n_bootstrap,
                    "resampling_unit": "target_after_averaging_repeated_cv_scores",
                }
            )
        if reference_method not in wide.columns:
            continue
        reference = wide[reference_method].to_numpy(float)
        for method in methods:
            if method == reference_method or wide[method].isna().any():
                continue
            baseline = wide[method].to_numpy(float)
            point = float(roc_auc_score(y, reference) - roc_auc_score(y, baseline))
            boot = np.asarray(
                [roc_auc_score(y[index], reference[index]) - roc_auc_score(y[index], baseline[index]) for index in samples]
            )
            low, high = np.quantile(boot, [0.025, 0.975])
            paired_rows.append(
                {
                    "model": model, "comparison": comparison,
                    "reference_method": reference_method, "baseline_method": method,
                    "delta_auc": point, "ci_low": float(low), "ci_high": float(high),
                    "excludes_zero": bool(low > 0 or high < 0),
                    "n_targets": int(len(wide)), "n_bootstrap": n_bootstrap,
                }
            )
    return pd.DataFrame(method_rows), pd.DataFrame(paired_rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-method", default="proposed_en")
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source = Path(args.predictions_csv)
    methods, deltas = infer_oof_uncertainty(
        pd.read_csv(source), n_bootstrap=args.n_bootstrap, seed=args.seed,
        reference_method=args.reference_method,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output / "method_target_bootstrap_auc.csv", methods)
    atomic_write_csv(output / "paired_target_bootstrap_auc_deltas.csv", deltas)
    manifest = base_manifest(experiment="strict_fixed20_target_uncertainty", command=sys.argv)
    manifest.update(
        {
            "status": "completed", "input": str(source), "input_sha256": sha256_file(source),
            "reference_method": args.reference_method, "n_bootstrap": args.n_bootstrap, "seed": args.seed,
        }
    )
    atomic_write_json(output / "strict_uncertainty_manifest.json", manifest)


if __name__ == "__main__":
    main(sys.argv[1:])
