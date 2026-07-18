#!/usr/bin/env python3
"""Compare attention features with matched gradient and parameter updates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from reviewer_followup.common import atomic_write_csv, atomic_write_json, base_manifest, sha256_file
from reviewer_followup.evaluation import (
    aggregate_repeats,
    bootstrap_auc_delta,
    evaluate_feature_sets,
    wide_attention,
    wide_updates,
)


def parse_comparisons(values: list[str]) -> dict[str, tuple[str, str]]:
    result = {}
    for value in values:
        try:
            name, pair = value.split("=", 1)
            positive, negative = pair.split(",", 1)
        except ValueError as exc:
            raise ValueError(f"comparison must be NAME=POSITIVE,NEGATIVE: {value}") from exc
        if not name or not positive or not negative:
            raise ValueError(f"comparison must be NAME=POSITIVE,NEGATIVE: {value}")
        result[name] = (positive, negative)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-csv", required=True)
    parser.add_argument("--update-csv", required=True)
    parser.add_argument("--sample-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--comparison", action="append", required=True, help="NAME=POSITIVE,NEGATIVE")
    parser.add_argument("--condition", default="fixed_attention_20")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=1000)
    return parser.parse_args(argv)


def _is_global(column: str) -> bool:
    return "_global_" in column


def _is_cosine(column: str) -> bool:
    return column.endswith(("grad_weight_cosine", "grad_delta_cosine"))


def target_score_inference(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Infer on target-averaged OOF scores, never on CV-repeat metrics."""
    bootstrap_rows = []
    permutation_rows = []
    for index, ((comparison, method), part) in enumerate(
        predictions.groupby(["comparison", "method"], sort=True)
    ):
        averaged = part.groupby(["sample_id", "y_true"], as_index=False)["score"].mean()
        y = averaged["y_true"].to_numpy(int)
        score = averaged["score"].to_numpy(float)
        positive = np.flatnonzero(y == 1)
        negative = np.flatnonzero(y == 0)
        rng = np.random.default_rng(seed + index)
        observed = float(roc_auc_score(y, score))
        boot = np.empty(n_bootstrap, dtype=float)
        for draw in range(n_bootstrap):
            sampled = np.concatenate(
                [rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)]
            )
            boot[draw] = roc_auc_score(y[sampled], score[sampled])
        low, high = np.quantile(boot, [0.025, 0.975])
        bootstrap_rows.append(
            {
                "comparison": comparison,
                "method": method,
                "auc": observed,
                "ci_low": float(low),
                "ci_high": float(high),
                "n_targets": int(len(averaged)),
                "n_bootstrap": n_bootstrap,
                "resampling_unit": "target_after_averaging_repeated_cv_scores",
            }
        )
        permuted = np.asarray([roc_auc_score(rng.permutation(y), score) for _ in range(n_permutations)])
        p_value = float((1 + np.count_nonzero(np.abs(permuted - 0.5) >= abs(observed - 0.5))) / (n_permutations + 1))
        permutation_rows.append(
            {
                "comparison": comparison,
                "method": method,
                "observed_auc": observed,
                "permutation_auc_mean": float(permuted.mean()),
                "two_sided_p": p_value,
                "n_permutations": n_permutations,
                "test": "fixed_oof_score_label_permutation",
            }
        )
    return pd.DataFrame(bootstrap_rows), pd.DataFrame(permutation_rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    attention_raw = pd.read_csv(args.attention_csv)
    update_raw = pd.read_csv(args.update_csv)
    sample = pd.read_csv(args.sample_csv)
    for frame in (attention_raw, update_raw, sample):
        if "condition" in frame.columns:
            frame.drop(frame.index[frame["condition"] != args.condition], inplace=True)
    attention = wide_attention(attention_raw)
    updates = wide_updates(update_raw)
    keys = ["condition", "sample_id", "group"] if "condition" in attention.columns else ["sample_id", "group"]
    frame = attention.merge(updates, on=keys, how="inner")
    trajectory_columns = [column for column in sample.columns if column.startswith(("train_loss_curve_", "train_gradient_curve_"))]
    if trajectory_columns:
        frame = frame.merge(sample[keys + trajectory_columns].drop_duplicates(keys), on=keys, how="left")
    attention_columns = [column for column in frame.columns if column.startswith("attn_")]
    gradient_columns = [column for column in frame.columns if column.startswith("upd_gradient_")]
    delta_columns = [column for column in frame.columns if column.startswith("upd_parameter_delta_")]
    update_columns = gradient_columns + delta_columns + trajectory_columns
    feature_sets = {
        "attention": attention_columns,
        "gradient_global_magnitude": [
            column for column in gradient_columns if _is_global(column) and column.endswith(("grad_l1", "grad_l2", "grad_max"))
        ],
        "update_cosines": [column for column in gradient_columns + delta_columns if _is_cosine(column)],
        "gradient_layer_only": [column for column in gradient_columns if not _is_global(column)],
        "gradient": gradient_columns,
        "parameter_delta": delta_columns,
        "gradient_plus_delta": gradient_columns + delta_columns,
        "loss_gradient_trajectory": trajectory_columns,
        "all_update_baselines": update_columns,
        "attention_plus_all_updates": attention_columns + update_columns,
    }
    repeats, predictions, selections = evaluate_feature_sets(
        frame,
        feature_sets,
        parse_comparisons(args.comparison),
        repeats=args.repeats,
        seed=args.seed,
    )
    deltas = bootstrap_auc_delta(
        predictions,
        augmented_method="attention_plus_all_updates",
        baseline_method="all_update_baselines",
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    inference, permutations = target_score_inference(
        predictions,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    dictionary_rows = [
        {"method": method, "feature": feature, "feature_family": feature.split("_", 2)[1] if feature.startswith("upd_") else "attention_or_trajectory"}
        for method, columns in feature_sets.items()
        for feature in columns
    ]
    selection_frequency = (
        selections.groupby(["comparison", "method", "feature"], as_index=False)
        .size()
        .rename(columns={"size": "selected_fold_count"})
    )
    denominators = selections.groupby(["comparison", "method"])[["repeat", "fold"]].apply(
        lambda part: len(part.drop_duplicates())
    )
    if not selection_frequency.empty:
        selection_frequency["eligible_fold_count"] = [
            int(denominators.loc[(row.comparison, row.method)]) for row in selection_frequency.itertuples(index=False)
        ]
        selection_frequency["selection_frequency"] = (
            selection_frequency["selected_fold_count"] / selection_frequency["eligible_fold_count"]
        )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output / "update_baseline_repeats.csv", repeats)
    atomic_write_csv(output / "update_baseline_summary.csv", aggregate_repeats(repeats))
    atomic_write_csv(output / "update_baseline_outer_predictions.csv", predictions)
    atomic_write_csv(output / "update_baseline_selected_features.csv", selections)
    atomic_write_csv(output / "attention_incremental_bootstrap.csv", deltas)
    atomic_write_csv(output / "update_baseline_target_bootstrap.csv", inference)
    atomic_write_csv(output / "update_baseline_label_permutation.csv", permutations)
    atomic_write_csv(output / "update_baseline_feature_dictionary.csv", pd.DataFrame(dictionary_rows))
    atomic_write_csv(output / "update_baseline_selection_frequency.csv", selection_frequency)
    manifest = base_manifest(experiment="e8_update_baselines", command=sys.argv)
    manifest.update(
        {
            "status": "completed",
            "feature_counts": {key: len(value) for key, value in feature_sets.items()},
            "n_bootstrap": args.n_bootstrap,
            "n_permutations": args.n_permutations,
            "resampling_unit": "target_after_averaging_repeated_cv_scores",
            "permutation_test": "fixed_oof_score_label_permutation",
            "input_sha256": {
                "attention_csv": sha256_file(Path(args.attention_csv)),
                "update_csv": sha256_file(Path(args.update_csv)),
                "sample_csv": sha256_file(Path(args.sample_csv)),
            },
        }
    )
    atomic_write_json(output / "update_baseline_manifest.json", manifest)
    print(aggregate_repeats(repeats).to_string(index=False))
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
