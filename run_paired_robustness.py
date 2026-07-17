# -*- coding: utf-8 -*-
"""Paired uncertainty and permutation checks for strict evaluation outputs.

The canonical evaluator reports one AUC per repeated CV run.  This utility
does not pretend that those repetitions are independent datasets: it reports
repeat-level paired bootstrap intervals and sign-permutation p-values as a
robustness diagnostic, with Holm correction across the requested comparisons.
For sample-level out-of-fold predictions, use the same functions through the
Python API after exporting one row per prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def paired_permutation_pvalue(differences: np.ndarray) -> float:
    """Two-sided sign permutation p-value, exact for <=15 paired units."""
    diff = np.asarray(differences, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return 1.0
    observed = abs(float(diff.mean()))
    n = int(diff.size)
    if n <= 15:
        signs = np.array(
            [1 if (mask >> i) & 1 else -1 for mask in range(1 << n) for i in range(n)],
            dtype=float,
        ).reshape(1 << n, n)
        null = np.abs((signs * diff.reshape(1, -1)).mean(axis=1))
        return float(np.mean(null >= observed - 1e-12))
    rng = np.random.default_rng(42)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(10000, n))
    null = np.abs((signs * diff.reshape(1, -1)).mean(axis=1))
    return float((1 + np.sum(null >= observed - 1e-12)) / (len(null) + 1))


def paired_bootstrap_ci(
    differences: np.ndarray,
    *,
    n_bootstrap: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    diff = np.asarray(differences, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return {"mean_diff": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, diff.size, size=(n_bootstrap, diff.size))
    means = diff[indices].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "std_diff": float(diff.std(ddof=1)) if diff.size > 1 else 0.0,
        "n_pairs": int(diff.size),
    }


def holm_adjust(pvalues: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvalues), dtype=float)
    if p.size == 0:
        return p
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def analyze_auc_table(
    auc_df: pd.DataFrame,
    proposed_method: str = "proposed_en",
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    required = {"model", "comparison", "method", "repeat", "auc"}
    missing = required - set(auc_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    rows: List[Dict[str, object]] = []
    for (model, comparison), sub in auc_df.groupby(["model", "comparison"], sort=True):
        proposed = sub[sub["method"] == proposed_method][["repeat", "auc"]].rename(
            columns={"auc": "proposed_auc"}
        )
        for baseline in sorted(set(sub["method"]) - {proposed_method}):
            other = sub[sub["method"] == baseline][["repeat", "auc"]].rename(
                columns={"auc": "baseline_auc"}
            )
            merged = proposed.merge(other, on="repeat", how="inner").sort_values("repeat")
            if merged.empty:
                continue
            diff = merged["proposed_auc"].to_numpy(dtype=float) - merged["baseline_auc"].to_numpy(dtype=float)
            ci = paired_bootstrap_ci(diff, n_bootstrap=n_bootstrap, seed=seed)
            rows.append(
                {
                    "model": model,
                    "comparison": comparison,
                    "proposed_method": proposed_method,
                    "baseline_method": baseline,
                    **ci,
                    "permutation_p": paired_permutation_pvalue(diff),
                    "proposed_auc_mean": float(merged["proposed_auc"].mean()),
                    "baseline_auc_mean": float(merged["baseline_auc"].mean()),
                    "bootstrap_unit": "repeated_cv_run",
                    "interpretation": "robustness diagnostic; repeats are correlated fold resamples",
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["holm_p"] = holm_adjust(result["permutation_p"].to_numpy())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to auc_10runs.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--proposed-method", default="proposed_en")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input).expanduser()
    auc_df = pd.read_csv(input_path)
    result = analyze_auc_table(
        auc_df,
        proposed_method=args.proposed_method,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    result.to_csv(output_dir / "paired_bootstrap_permutation.csv", index=False)
    manifest = {
        "script": Path(__file__).name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "input": str(input_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "proposed_method": args.proposed_method,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "unit": "repeated_cv_run",
        "warning": "Repeated CV runs are correlated resamples, not independent datasets.",
    }
    (output_dir / "robustness_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
