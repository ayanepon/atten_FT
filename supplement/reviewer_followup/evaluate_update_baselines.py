#!/usr/bin/env python3
"""Compare attention features with matched gradient and parameter updates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from reviewer_followup.common import atomic_write_csv, atomic_write_json, base_manifest
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
    return parser.parse_args(argv)


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
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output / "update_baseline_repeats.csv", repeats)
    atomic_write_csv(output / "update_baseline_summary.csv", aggregate_repeats(repeats))
    atomic_write_csv(output / "update_baseline_outer_predictions.csv", predictions)
    atomic_write_csv(output / "update_baseline_selected_features.csv", selections)
    atomic_write_csv(output / "attention_incremental_bootstrap.csv", deltas)
    manifest = base_manifest(experiment="e8_update_baselines", command=sys.argv)
    manifest.update({"status": "completed", "feature_counts": {key: len(value) for key, value in feature_sets.items()}})
    atomic_write_json(output / "update_baseline_manifest.json", manifest)
    print(aggregate_repeats(repeats).to_string(index=False))
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
