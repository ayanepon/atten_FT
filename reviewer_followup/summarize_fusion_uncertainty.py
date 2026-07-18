#!/usr/bin/env python3
"""Validate and render the paired target-bootstrap fusion comparisons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from reviewer_followup.common import atomic_write_csv, atomic_write_json, base_manifest, sha256_file


REQUIRED_BASELINES = {"lora_leak", "proposed_en"}
REQUIRED_COMPARISONS = {"ft_vs_pt", "ft_vs_unseen"}


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(
        columns={
            "fused_method": "augmented_method",
            "mean_auc_delta": "delta_auc",
            "ci95_low": "ci_low",
            "ci95_high": "ci_high",
        }
    ).copy()
    if "augmented_method" in frame.columns:
        frame["augmented_method"] = frame["augmented_method"].replace(
            {"fusion_2d_crossfit": "fusion_2d", "fusion_alpha_crossfit": "fusion_alpha"}
        )
    required = {"comparison", "augmented_method", "baseline_method", "delta_auc", "ci_low", "ci_high"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Fusion bootstrap CSV missing {sorted(missing)}")
    selected = frame[
        (frame["augmented_method"] == "fusion_2d")
        & frame["baseline_method"].isin(REQUIRED_BASELINES)
        & frame["comparison"].isin(REQUIRED_COMPARISONS)
    ].copy()
    pairs = set(zip(selected["comparison"], selected["baseline_method"]))
    expected = {(comparison, baseline) for comparison in REQUIRED_COMPARISONS for baseline in REQUIRED_BASELINES}
    if pairs != expected or len(selected) != len(expected):
        raise ValueError(f"Expected exactly four Fusion2D comparisons; missing={sorted(expected - pairs)}")
    selected["excludes_zero"] = (selected["ci_low"] > 0) | (selected["ci_high"] < 0)
    return selected.sort_values(["comparison", "baseline_method"]).reset_index(drop=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source = Path(args.input_csv)
    rows = validate(pd.read_csv(source))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output / "fusion_target_bootstrap_required_rows.csv", rows)
    tex = ["% Generated target-level paired-bootstrap fusion comparisons."]
    for row in rows.itertuples(index=False):
        comparison = row.comparison.replace("_", r"\_")
        baseline = row.baseline_method.replace("_", r"\_")
        tex.append(
            f"{comparison} vs. {baseline} & "
            f"{row.delta_auc:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}] \\\\"
        )
    (output / "fusion_target_bootstrap_rows.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    manifest = base_manifest(experiment="reviewer_revision_fusion_uncertainty", command=sys.argv)
    manifest.update({"status": "completed", "input_csv": str(source), "input_sha256": sha256_file(source)})
    atomic_write_json(output / "fusion_uncertainty_manifest.json", manifest)
    print(rows.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
