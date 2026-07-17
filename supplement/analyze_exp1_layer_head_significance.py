# -*- coding: utf-8 -*-
"""Experiment 1: layer--head significance (paper-aligned).

For each binary comparison (FT--PT, FT--Unseen):
  - 8 attention features × 128 layer--head pairs
  - Mann--Whitney U (two-sided)
  - Benjamini--Hochberg FDR across all feature--layer--head tests
    within that comparison (paper: global FDR within each comparison)
  - Cliff's delta (FT minus comparison group)

Also reports:
  - per-feature significant counts (Table fullmatrix)
  - pairs significant in both comparisons with consistent Cliff's delta sign
  - Spearman correlation between each feature and loss decrease (Table exp2_corr)

Usage:
  python analyze_exp1_layer_head_significance.py \\
    --root attention_features_mimir_hardsplit \\
    --output-dir results/exp1_layer_head_stats
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, spearmanr

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:  # pragma: no cover
    multipletests = None


GROUP_FT = "mimir_wikipedia_nonmember_ft"
GROUP_PT = "mimir_wikipedia_member_pt"
GROUP_UNSEEN = "mimir_wikipedia_nonmember_unseen"

METRICS = [
    "entropy_delta",
    "l1_mean",
    "l2_rms",
    "js_div",
    "max_shift",
    "top1_shift_mean",
    "top5_shift_mean",
    "top10_shift_mean",
]

PAPER_NAMES = {
    "entropy_delta": "Entropy diff.",
    "l1_mean": "Mean diff.",
    "l2_rms": "RMSE",
    "js_div": "JSD",
    "max_shift": "Max shift",
    "top1_shift_mean": "Top-1% mean",
    "top5_shift_mean": "Top-5% mean",
    "top10_shift_mean": "Top-10% mean",
}


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    if multipletests is not None:
        return multipletests(p, method="fdr_bh")[1]
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(q, 0, 1)
    return out


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta for x vs y: positive means x tends larger than y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    ranks = rankdata(np.concatenate([x, y]))
    rx = ranks[:nx].sum()
    u = rx - nx * (nx + 1) / 2.0
    return float((2 * u) / (nx * ny) - 1.0)


def load_raw(root: Path) -> Dict[str, pd.DataFrame]:
    parts = {}
    for key, group in [("ft", GROUP_FT), ("pt", GROUP_PT), ("unseen", GROUP_UNSEEN)]:
        candidates = list(root.glob(f"**/fixed_attention_20_{key}/raw_experiment4_attention_shift.csv"))
        if not candidates:
            raise FileNotFoundError(f"raw CSV not found for {key} under {root}")
        path = max(candidates, key=lambda p: p.stat().st_size)
        df = pd.read_csv(path)
        if "group" not in df.columns:
            df["group"] = group
        parts[key] = df
    return parts


def load_loss(root: Path) -> pd.DataFrame:
    frames = []
    for key in ["ft", "pt", "unseen"]:
        candidates = list(root.glob(f"**/fixed_attention_20_{key}/sample_level_experiment4.csv"))
        if not candidates:
            continue
        path = max(candidates, key=lambda p: p.stat().st_size)
        df = pd.read_csv(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def test_comparison(parts: Dict[str, pd.DataFrame], pos_key: str, neg_key: str) -> pd.DataFrame:
    rows = []
    pos = parts[pos_key]
    neg = parts[neg_key]
    metrics = [m for m in METRICS if m in pos.columns and m in neg.columns]
    layers = sorted(pos["layer"].unique())
    heads = sorted(pos["head"].unique())
    for metric in metrics:
        for layer in layers:
            for head in heads:
                a = pos[(pos["layer"] == layer) & (pos["head"] == head)][metric].to_numpy(float)
                b = neg[(neg["layer"] == layer) & (neg["head"] == head)][metric].to_numpy(float)
                a = a[np.isfinite(a)]
                b = b[np.isfinite(b)]
                if len(a) < 2 or len(b) < 2:
                    continue
                _, p = mannwhitneyu(a, b, alternative="two-sided")
                rows.append(
                    {
                        "positive_group": pos[pos["group"].notna()]["group"].iloc[0] if "group" in pos else pos_key,
                        "negative_group": neg[neg["group"].notna()]["group"].iloc[0] if "group" in neg else neg_key,
                        "metric": metric,
                        "paper_name": PAPER_NAMES.get(metric, metric),
                        "layer": int(layer),
                        "head": int(head),
                        "mean_positive": float(np.mean(a)),
                        "mean_negative": float(np.mean(b)),
                        "mean_difference_pos_minus_neg": float(np.mean(a) - np.mean(b)),
                        "mannwhitney_p_raw": float(p),
                        "cliffs_delta_pos_minus_neg": cliffs_delta(a, b),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["mannwhitney_p_fdr_global"] = bh_fdr(out["mannwhitney_p_raw"].to_numpy())
    out["significant_fdr_0.05"] = out["mannwhitney_p_fdr_global"] < 0.05
    return out


def direction_summary(ftpt: pd.DataFrame, ftun: pd.DataFrame) -> pd.DataFrame:
    a = ftpt[ftpt["significant_fdr_0.05"]][
        ["metric", "layer", "head", "cliffs_delta_pos_minus_neg"]
    ].rename(columns={"cliffs_delta_pos_minus_neg": "delta_pt"})
    b = ftun[ftun["significant_fdr_0.05"]][
        ["metric", "layer", "head", "cliffs_delta_pos_minus_neg"]
    ].rename(columns={"cliffs_delta_pos_minus_neg": "delta_un"})
    both = a.merge(b, on=["metric", "layer", "head"], how="inner")
    rows = []
    for metric, g in both.groupby("metric"):
        rows.append(
            {
                "metric": metric,
                "paper_name": PAPER_NAMES.get(metric, metric),
                "both_sig_pairs": int(len(g)),
                "FT_gt_both": int(((g["delta_pt"] > 0) & (g["delta_un"] > 0)).sum()),
                "FT_lt_both": int(((g["delta_pt"] < 0) & (g["delta_un"] < 0)).sum()),
            }
        )
    cols = ["metric", "paper_name", "both_sig_pairs", "FT_gt_both", "FT_lt_both"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).sort_values("metric")


def spearman_with_loss(parts: Dict[str, pd.DataFrame], loss_df: pd.DataFrame) -> pd.DataFrame:
    if loss_df.empty or "delta_loss_before_minus_after" not in loss_df.columns:
        return pd.DataFrame()
    raw = pd.concat(parts.values(), ignore_index=True)
    # Drop loss columns from raw if present to avoid merge suffixes
    drop_cols = [c for c in ["delta_loss_before_minus_after", "before_loss", "after_loss"] if c in raw.columns]
    raw = raw.drop(columns=drop_cols)
    loss = (
        loss_df[["sample_id", "group", "delta_loss_before_minus_after"]]
        .drop_duplicates(subset=["sample_id", "group"])
        .rename(columns={"delta_loss_before_minus_after": "loss_decrease"})
    )
    merged = raw.merge(loss, on=["sample_id", "group"], how="inner")
    if "loss_decrease" not in merged.columns:
        return pd.DataFrame()
    rows = []
    for metric in [m for m in METRICS if m in merged.columns]:
        rhos = []
        for (_, _), g in merged.groupby(["layer", "head"]):
            if g[metric].nunique() < 2:
                continue
            rho, _ = spearmanr(g[metric], g["loss_decrease"])
            if np.isfinite(rho):
                rhos.append(rho)
        if not rhos:
            continue
        rows.append(
            {
                "metric": metric,
                "paper_name": PAPER_NAMES.get(metric, metric),
                "mean_rho": float(np.mean(rhos)),
                "mean_abs_rho": float(np.mean(np.abs(rhos))),
                "n_layer_head": int(len(rhos)),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="attention_features_mimir_hardsplit")
    p.add_argument("--output-dir", default="results/exp1_layer_head_stats")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        # local fallback next to this script
        cand = Path(__file__).resolve().parent / args.root
        root = cand if cand.exists() else root
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    parts = load_raw(root)
    ftpt = test_comparison(parts, "ft", "pt")
    ftun = test_comparison(parts, "ft", "unseen")
    ftpt.to_csv(out / "ft_vs_pt_layer_head_tests.csv", index=False)
    ftun.to_csv(out / "ft_vs_unseen_layer_head_tests.csv", index=False)

    sig_summary = pd.DataFrame(
        [
            {
                "comparison": "FT--PT",
                "metric": m,
                "paper_name": PAPER_NAMES.get(m, m),
                "n_sig": int(ftpt.loc[ftpt["metric"] == m, "significant_fdr_0.05"].sum()),
            }
            for m in METRICS
            if m in set(ftpt["metric"])
        ]
        + [
            {
                "comparison": "FT--Unseen",
                "metric": m,
                "paper_name": PAPER_NAMES.get(m, m),
                "n_sig": int(ftun.loc[ftun["metric"] == m, "significant_fdr_0.05"].sum()),
            }
            for m in METRICS
            if m in set(ftun["metric"])
        ]
    )
    sig_summary.to_csv(out / "significant_counts_by_feature.csv", index=False)

    direction = direction_summary(ftpt, ftun)
    direction.to_csv(out / "direction_summary_both_comparisons.csv", index=False)

    loss_df = load_loss(root)
    corr = spearman_with_loss(parts, loss_df)
    corr.to_csv(out / "spearman_with_loss_decrease.csv", index=False)

    # Heatmap-ready matrix for entropy_delta (Cliff's delta, FT vs PT)
    ent = ftpt[ftpt["metric"] == "entropy_delta"].copy()
    if not ent.empty:
        piv = ent.pivot(index="layer", columns="head", values="cliffs_delta_pos_minus_neg")
        piv.to_csv(out / "entropy_delta_cliffs_ft_vs_pt.csv")
        sig = ent.pivot(index="layer", columns="head", values="significant_fdr_0.05")
        sig.to_csv(out / "entropy_delta_sig_ft_vs_pt.csv")

    print("Significant counts:")
    print(sig_summary.to_string(index=False))
    print("\nDirection summary (both comparisons significant):")
    print(direction.to_string(index=False))
    if not corr.empty:
        print("\nSpearman with loss decrease:")
        print(corr.to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
