#!/usr/bin/env python3
"""Leakage-safe repeated-CV evaluation for an arbitrary attention extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from reviewer_followup.common import atomic_write_csv, atomic_write_json, base_manifest
from reviewer_followup.evaluation import aggregate_repeats, evaluate_feature_sets, wide_attention


def parse_comparisons(values: list[str]) -> dict[str, tuple[str, str]]:
    comparisons: dict[str, tuple[str, str]] = {}
    for value in values:
        try:
            name, pair = value.split("=", 1)
            positive, negative = pair.split(",", 1)
        except ValueError as exc:
            raise ValueError(f"comparison must be NAME=POSITIVE,NEGATIVE: {value}") from exc
        if not name or not positive or not negative:
            raise ValueError(f"comparison must be NAME=POSITIVE,NEGATIVE: {value}")
        comparisons[name] = (positive, negative)
    return comparisons


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--comparison", action="append", required=True, help="NAME=POSITIVE,NEGATIVE")
    parser.add_argument("--condition", default="fixed_attention_20")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raw = pd.read_csv(args.attention_csv)
    if "condition" in raw.columns:
        raw = raw[raw["condition"] == args.condition].copy()
    wide = wide_attention(raw)
    features = [column for column in wide.columns if column.startswith("attn_")]
    if not features:
        raise ValueError("No attention features were found")
    repeats, predictions, selections = evaluate_feature_sets(
        wide,
        {"proposed_en": features},
        parse_comparisons(args.comparison),
        repeats=args.repeats,
        cv_splits=args.cv_splits,
        seed=args.seed,
    )
    summary = aggregate_repeats(repeats)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output / "attention_repeats.csv", repeats)
    atomic_write_csv(output / "attention_summary.csv", summary)
    atomic_write_csv(output / "attention_outer_predictions.csv", predictions)
    atomic_write_csv(output / "attention_selected_features.csv", selections)
    manifest = base_manifest(experiment="reviewer_followup_attention_evaluation", command=sys.argv)
    manifest.update(
        {
            "status": "completed",
            "condition": args.condition,
            "n_features": len(features),
            "comparisons": parse_comparisons(args.comparison),
        }
    )
    atomic_write_json(output / "attention_evaluation_manifest.json", manifest)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
