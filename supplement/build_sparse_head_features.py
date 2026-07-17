# -*- coding: utf-8 -*-
"""Build sparse feature caches for the strict eval harness (Phase-3 optional experiment).

Two variants, both derived from an already-extracted fixed-20 proposed feature
cache (no new GPU extraction):
  - significant_heads: keep only (layer, head, metric) columns that were FDR<0.05
    significant in Exp.1 (union of FT-PT and FT-Unseen layer/head tests).
  - entropy_only: keep only entropy_delta columns for all layers/heads.

Each variant is written as its own proposed_root directory: a filtered
proposed_features_fixed20_cache.csv plus symlinks to the original group
directories (so read_group_files/load_loss_sample_scores in
run_strict_fixed20_comparison_10runs.py still find the raw/sample_level CSVs
they require, without duplicating multi-GB files).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

GROUPS = ["ft", "pt", "unseen"]


def load_significant_triples(exp1_root: Path) -> set:
    triples = set()
    for fname in ["ft_vs_pt_layer_head_tests.csv", "ft_vs_unseen_layer_head_tests.csv"]:
        df = pd.read_csv(exp1_root / fname)
        sig = df[df["significant_fdr_0.05"] == True]  # noqa: E712
        for _, row in sig.iterrows():
            triples.add((int(row["layer"]), int(row["head"]), row["metric"]))
    return triples


def col_to_triple(col: str):
    assert col.startswith("attn_l")
    rest = col[len("attn_l"):]
    layer_str, rest2 = rest.split("_h", 1)
    head_str, metric = rest2.split("_", 1)
    return (int(layer_str), int(head_str), metric)


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
    ap.add_argument("--source-root", required=True, help="Existing proposed_root with a fixed20 cache")
    ap.add_argument("--exp1-root", required=True, help="results/exp1_layer_head_stats_<model> dir")
    ap.add_argument("--output-root-significant", required=True)
    ap.add_argument("--output-root-entropy", required=True)
    ap.add_argument("--condition-prefix", default="fixed_attention_20")
    args = ap.parse_args()

    source_root = Path(args.source_root).resolve()
    exp1_root = Path(args.exp1_root).resolve()

    cache_path = find_cache(source_root, args.condition_prefix)
    wide = pd.read_parquet(cache_path) if cache_path.suffix == ".parquet" else pd.read_csv(cache_path)

    id_cols = ["sample_id", "group"]
    feature_cols = [c for c in wide.columns if c.startswith("attn_l")]

    triples = load_significant_triples(exp1_root)
    print(f"Significant (layer,head,metric) triples (union FT-PT / FT-Unseen, FDR<0.05): {len(triples)}")

    sig_cols = [c for c in feature_cols if col_to_triple(c) in triples]
    entropy_cols = [c for c in feature_cols if c.endswith("_entropy_delta")]

    print(f"Significant-head feature columns: {len(sig_cols)} / {len(feature_cols)}")
    print(f"Entropy-only feature columns: {len(entropy_cols)} / {len(feature_cols)}")

    stem = (
        "proposed_features_fixed20_cache"
        if args.condition_prefix == "fixed_attention_20"
        else f"proposed_features_{args.condition_prefix}_cache"
    )

    for out_root_arg, cols, label in [
        (args.output_root_significant, sig_cols, "significant_heads"),
        (args.output_root_entropy, entropy_cols, "entropy_only"),
    ]:
        if not cols:
            print(f"[{label}] no columns selected, skipping")
            continue
        out_root = Path(out_root_arg).resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        symlink_groups(source_root, out_root, args.condition_prefix)
        sub = wide[id_cols + cols].copy()
        out_csv = out_root / f"{stem}.csv"
        sub.to_csv(out_csv, index=False)
        print(f"[{label}] wrote {out_csv} ({len(cols)} feature cols, {len(sub)} rows)")


if __name__ == "__main__":
    main()
