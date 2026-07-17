# -*- coding: utf-8 -*-
"""Build multi-step trajectory (slope) features for the strict eval harness
(Phase-3 optional experiment).

For each sample, computes the OLS slope of every (layer, head, metric) value
across the three already-extracted fixed-step conditions (20/50/100 steps),
producing one derived feature set with the same column count as a single-step
cache. No new GPU extraction: this only recombines existing caches, and
requires the three roots to share identical (sample_id, group) sets and
column sets (verified explicitly, not assumed).

Output is written as its own proposed_root directory: a
proposed_features_fixed20_cache.csv holding the slope features, plus symlinks
to the step-20 root's group directories (so read_group_files /
load_loss_sample_scores in run_strict_fixed20_comparison_10runs.py still find
the raw/sample_level CSVs they require).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = ["ft", "pt", "unseen"]
STEPS = [20, 50, 100]


def find_cache(root: Path, condition_prefix: str) -> Path:
    stem = (
        "proposed_features_fixed20_cache"
        if condition_prefix == "fixed_attention_20"
        else f"proposed_features_{condition_prefix}_cache"
    )
    for ext in ("parquet", "csv"):
        p = root / f"{stem}.{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No proposed feature cache found under {root} for {condition_prefix}")


def load_wide(root: Path, condition_prefix: str) -> pd.DataFrame:
    path = find_cache(root, condition_prefix)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return df


def symlink_groups(source_root: Path, output_root: Path, condition_prefix: str) -> None:
    for group in GROUPS:
        name = f"{condition_prefix}_{group}"
        src = (source_root / name).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Expected source group dir missing: {src}")
        dst = output_root / name
        if dst.exists() or dst.is_symlink():
            continue
        dst.symlink_to(src)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step20-root", required=True)
    ap.add_argument("--step50-root", required=True)
    ap.add_argument("--step100-root", required=True)
    ap.add_argument("--step20-condition-prefix", default="fixed_attention_20")
    ap.add_argument("--step50-condition-prefix", default="fixed_attention_50")
    ap.add_argument("--step100-condition-prefix", default="fixed_attention_100")
    ap.add_argument("--output-root", required=True)
    args = ap.parse_args()

    root20 = Path(args.step20_root).resolve()
    root50 = Path(args.step50_root).resolve()
    root100 = Path(args.step100_root).resolve()

    df20 = load_wide(root20, args.step20_condition_prefix)
    df50 = load_wide(root50, args.step50_condition_prefix)
    df100 = load_wide(root100, args.step100_condition_prefix)

    id_cols = ["sample_id", "group"]
    feat20 = [c for c in df20.columns if c.startswith("attn_l")]
    feat50 = [c for c in df50.columns if c.startswith("attn_l")]
    feat100 = [c for c in df100.columns if c.startswith("attn_l")]

    if not (set(feat20) == set(feat50) == set(feat100)):
        raise ValueError(
            "Feature column sets differ across step roots; cannot build trajectory "
            f"features. |20|={len(feat20)} |50|={len(feat50)} |100|={len(feat100)}"
        )

    df20 = df20.sort_values(id_cols).reset_index(drop=True)
    df50 = df50.sort_values(id_cols).reset_index(drop=True)
    df100 = df100.sort_values(id_cols).reset_index(drop=True)

    key20 = list(zip(df20["sample_id"], df20["group"]))
    key50 = list(zip(df50["sample_id"], df50["group"]))
    key100 = list(zip(df100["sample_id"], df100["group"]))
    if not (key20 == key50 == key100):
        raise ValueError(
            "(sample_id, group) rows are not identical/aligned across the three step "
            "roots after sorting; refusing to build trajectory features on misaligned data."
        )

    x = np.array(STEPS, dtype=float)
    xbar = x.mean()
    denom = float(((x - xbar) ** 2).sum())
    w = (x - xbar) / denom  # w20, w50, w100 s.t. slope = w20*y20 + w50*y50 + w100*y100

    cols = feat20
    v20 = df20[cols].to_numpy(dtype=float)
    v50 = df50[cols].to_numpy(dtype=float)
    v100 = df100[cols].to_numpy(dtype=float)
    slope = w[0] * v20 + w[1] * v50 + w[2] * v100

    slope_cols = [f"{c}_slope" for c in cols]
    out = pd.DataFrame(slope, columns=slope_cols)
    out.insert(0, "group", df20["group"].to_numpy())
    out.insert(0, "sample_id", df20["sample_id"].to_numpy())

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    symlink_groups(root20, out_root, args.step20_condition_prefix)
    if args.step20_condition_prefix != "fixed_attention_20":
        # Eval harness default condition-prefix is fixed_attention_20; alias the
        # symlinked group dirs under that name too so no extra CLI flag is needed.
        symlink_groups(root20, out_root, "fixed_attention_20")

    out_csv = out_root / "proposed_features_fixed20_cache.csv"
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(slope_cols)} slope feature cols, {len(out)} rows)")


if __name__ == "__main__":
    main()
