#!/usr/bin/env python3
"""Measure layer/head effect and feature-selection stability across checkpoints."""

from __future__ import annotations

import argparse
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from reviewer_followup.common import atomic_write_csv, atomic_write_json, base_manifest


FEATURE_PATTERNS = (
    re.compile(r"attn_l(?P<layer>\d+)_h(?P<head>\d+)_(?P<metric>.+)"),
    re.compile(r"attn_(?P<metric>.+)_L(?P<layer>\d+)_H(?P<head>\d+)"),
)


def parse_seed_paths(values: list[str]) -> dict[int, Path]:
    result = {}
    for value in values:
        seed, path = value.split("=", 1)
        result[int(seed)] = Path(path)
    if len(result) < 2:
        raise ValueError("At least two seeds are required")
    return result


def normalize_effects(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    required = {"metric", "layer", "head"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Effect file missing {sorted(required - set(frame.columns))}")
    effect_column = "cliffs_delta_pos_minus_neg" if "cliffs_delta_pos_minus_neg" in frame.columns else "effect"
    significant_column = "significant_fdr_0.05" if "significant_fdr_0.05" in frame.columns else "significant"
    out = frame[["metric", "layer", "head", effect_column, significant_column]].copy()
    out.columns = ["metric", "layer", "head", "effect", "significant"]
    out["seed"] = seed
    # Use item access here: ``Series.head`` is a method, so attribute access
    # does not retrieve the column named ``head``.
    out["key"] = out.apply(
        lambda row: f"{row['metric']}|L{int(row['layer'])}|H{int(row['head'])}",
        axis=1,
    )
    out["significant"] = out["significant"].astype(str).str.lower().isin({"1", "true", "yes"})
    return out


def selection_frequency(seed_paths: dict[int, Path], comparison: str = "") -> pd.DataFrame:
    rows = []
    for seed, path in seed_paths.items():
        frame = pd.read_csv(path)
        if "feature" not in frame.columns:
            raise ValueError(f"{path} needs feature column")
        if comparison:
            if "comparison" not in frame.columns:
                raise ValueError(f"{path} needs comparison column when --comparison is used")
            frame = frame[frame["comparison"] == comparison].copy()
            if frame.empty:
                raise ValueError(f"{path} has no selected features for comparison={comparison}")
        total_folds = frame[[column for column in ("repeat", "fold") if column in frame.columns]].drop_duplicates().shape[0]
        total_folds = max(total_folds, 1)
        for feature, count in frame["feature"].value_counts().items():
            parsed = None
            for pattern in FEATURE_PATTERNS:
                match = pattern.fullmatch(str(feature))
                if match:
                    parsed = match.groupdict()
                    break
            rows.append(
                {
                    "seed": seed,
                    "feature": feature,
                    "metric": parsed["metric"] if parsed else "unknown",
                    "layer": int(parsed["layer"]) if parsed else -1,
                    "head": int(parsed["head"]) if parsed else -1,
                    "selected_count": int(count),
                    "selected_frequency": float(count / total_folds),
                }
            )
    return pd.DataFrame(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effect", action="append", required=True, help="SEED=layer_head_tests.csv")
    parser.add_argument("--selection", action="append", default=[], help="SEED=selected_features.csv")
    parser.add_argument("--comparison", default="", help="Optional comparison filter for selection files")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    effect_paths = parse_seed_paths(args.effect)
    effects = pd.concat([normalize_effects(pd.read_csv(path), seed) for seed, path in effect_paths.items()], ignore_index=True)
    wide = effects.pivot(index="key", columns="seed", values="effect")
    significant_sets = {seed: set(part.loc[part["significant"], "key"]) for seed, part in effects.groupby("seed")}
    pair_rows = []
    for first, second in combinations(sorted(effect_paths), 2):
        common = wide[[first, second]].dropna()
        rho = spearmanr(common[first], common[second]).statistic if len(common) > 1 else float("nan")
        union = significant_sets[first] | significant_sets[second]
        intersection = significant_sets[first] & significant_sets[second]
        pair_rows.append(
            {
                "seed_a": first,
                "seed_b": second,
                "effect_spearman": float(rho),
                "n_common_features": int(len(common)),
                "significant_jaccard": float(len(intersection) / len(union)) if union else 1.0,
                "n_significant_intersection": int(len(intersection)),
                "n_significant_union": int(len(union)),
            }
        )
    direction = (
        effects.assign(direction=np.sign(effects["effect"]))
        .groupby(["key", "metric", "layer", "head"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_effect=("effect", "mean"),
            direction_consistency=("direction", lambda values: float(max((values > 0).mean(), (values < 0).mean()))),
            significant_frequency=("significant", "mean"),
        )
    )
    layer_summary = (
        direction.groupby(["metric", "layer"], as_index=False)
        .agg(
            mean_direction_consistency=("direction_consistency", "mean"),
            mean_significant_frequency=("significant_frequency", "mean"),
            n_heads=("head", "nunique"),
        )
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output / "head_pairwise_stability.csv", pd.DataFrame(pair_rows))
    atomic_write_csv(output / "head_direction_stability.csv", direction)
    atomic_write_csv(output / "layer_stability_summary.csv", layer_summary)
    if args.selection:
        selections = selection_frequency(parse_seed_paths(args.selection), args.comparison)
        atomic_write_csv(output / "selected_feature_frequency_by_seed.csv", selections)
        overall = selections.groupby(["feature", "metric", "layer", "head"], as_index=False)["selected_frequency"].mean()
        atomic_write_csv(output / "selected_feature_frequency_across_seeds.csv", overall)
    manifest = base_manifest(experiment="e10_head_stability", command=sys.argv)
    manifest.update({"status": "completed", "seeds": sorted(effect_paths), "comparison": args.comparison})
    atomic_write_json(output / "head_stability_manifest.json", manifest)
    print(pd.DataFrame(pair_rows).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
