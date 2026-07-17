#!/usr/bin/env python3
"""Evaluate the crossed PT-membership x FT-exposure attention experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize

from reviewer_followup.common import atomic_write_csv, atomic_write_json, base_manifest, bh_fdr, validate_factorial_targets
from reviewer_followup.evaluation import aggregate_repeats, evaluate_feature_sets, wide_attention


CONTRASTS = {
    "ft_effect_pt1": ("p1f1", "p1f0"),
    "ft_effect_pt0": ("p0f1", "p0f0"),
    "pt_effect_ft1": ("p1f1", "p0f1"),
    "pt_effect_ft0": ("p1f0", "p0f0"),
}


def factorial_ols(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    pt = pd.to_numeric(frame["pt_member"]).to_numpy(float)
    ft = pd.to_numeric(frame["ft_exposed"]).to_numpy(float)
    design = np.column_stack([np.ones(len(frame)), pt, ft, pt * ft])
    names = ["intercept", "pt_main_at_f0", "ft_main_at_p0", "interaction"]
    rows = []
    for feature in feature_columns:
        y = pd.to_numeric(frame[feature], errors="coerce").to_numpy(float)
        valid = np.isfinite(y)
        x = design[valid]
        outcome = y[valid]
        if len(outcome) <= x.shape[1] + 2:
            continue
        inv = np.linalg.pinv(x.T @ x)
        beta = inv @ x.T @ outcome
        residual = outcome - x @ beta
        h = np.sum((x @ inv) * x, axis=1)
        meat = x.T @ np.diag((residual / np.clip(1.0 - h, 1e-8, None)) ** 2) @ x
        covariance = inv @ meat @ inv
        se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
        dof = len(outcome) - x.shape[1]
        for index, term in enumerate(names):
            statistic = beta[index] / max(se[index], 1e-30)
            p_value = 2.0 * student_t.sf(abs(statistic), dof)
            rows.append(
                {
                    "feature": feature,
                    "term": term,
                    "estimate": float(beta[index]),
                    "se_hc3": float(se[index]),
                    "ci_low": float(beta[index] - student_t.ppf(0.975, dof) * se[index]),
                    "ci_high": float(beta[index] + student_t.ppf(0.975, dof) * se[index]),
                    "p": float(p_value),
                    "n": int(len(outcome)),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_adj_bh_within_term"] = result.groupby("term")["p"].transform(lambda values: bh_fdr(values))
    return result


def multiclass_cv(frame: pd.DataFrame, columns: list[str], *, repeats: int, seed: int) -> pd.DataFrame:
    labels = sorted(frame["group"].unique())
    y = frame["group"].map({label: index for index, label in enumerate(labels)}).to_numpy(int)
    x = frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median()).fillna(0.0).to_numpy(float)
    rows = []
    for repeat in range(1, repeats + 1):
        splitter = StratifiedKFold(5, shuffle=True, random_state=seed + repeat - 1)
        scores = np.zeros((len(y), len(labels)), dtype=float)
        for train, test in splitter.split(x, y):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x[train])
            x_test = scaler.transform(x[test])
            classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
            classifier.fit(x_train, y[train])
            scores[test] = classifier.predict_proba(x_test)
        binary = label_binarize(y, classes=np.arange(len(labels)))
        rows.append({"repeat": repeat, "macro_ovr_auc": float(roc_auc_score(binary, scores, average="macro", multi_class="ovr"))})
    return pd.DataFrame(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-csv", required=True)
    parser.add_argument("--targets-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--condition", default="fixed_attention_20")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raw = pd.read_csv(args.attention_csv)
    if "condition" in raw.columns:
        raw = raw[raw["condition"] == args.condition].copy()
    targets = pd.read_csv(args.targets_csv, dtype=str, keep_default_na=False)
    validate_factorial_targets(targets)
    wide = wide_attention(raw)
    factors = targets[["group", "pt_member", "ft_exposed"]].drop_duplicates("group")
    wide = wide.merge(factors, on="group", how="left")
    feature_columns = [column for column in wide.columns if column.startswith("attn_")]
    ols = factorial_ols(wide, feature_columns)
    repeat_rows, predictions, selections = evaluate_feature_sets(
        wide,
        {"attention_en": feature_columns},
        CONTRASTS,
        repeats=args.repeats,
        seed=args.seed,
    )
    multiclass = multiclass_cv(wide, feature_columns, repeats=args.repeats, seed=args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output / "factorial_feature_effects.csv", ols)
    atomic_write_csv(output / "factorial_contrast_repeats.csv", repeat_rows)
    atomic_write_csv(output / "factorial_contrast_summary.csv", aggregate_repeats(repeat_rows))
    atomic_write_csv(output / "factorial_outer_predictions.csv", predictions)
    atomic_write_csv(output / "factorial_selected_features.csv", selections)
    atomic_write_csv(output / "factorial_multiclass_auc.csv", multiclass)
    manifest = base_manifest(experiment="e7_crossed_2x2_evaluation", command=sys.argv)
    manifest.update({"condition": args.condition, "n_features": len(feature_columns), "contrasts": CONTRASTS, "status": "completed"})
    atomic_write_json(output / "factorial_evaluation_manifest.json", manifest)
    print(aggregate_repeats(repeat_rows).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
