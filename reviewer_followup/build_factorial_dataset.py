#!/usr/bin/env python3
"""Build balanced 2x2 PT-membership x FT-exposure datasets.

Two modes are supported:

``mimir-membership``
    P=1 comes from the MIMIR member pool and P=0 from the two non-member
    pools.  This fills the missing P1F1 cell in the current paper protocol.

``controlled-exposure``
    All four cells come from documents treated as original-model non-members.
    P marks exposure in a researcher-controlled continued-pretraining stage;
    this is used for the cross-family ground-truth experiment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from reviewer_followup.common import (
    GROUP_FACTORS,
    atomic_write_csv,
    atomic_write_json,
    base_manifest,
    load_text_csv,
    sha256_file,
    validate_factorial_targets,
)


def _take(pool: pd.DataFrame, indices: np.ndarray, group: str) -> pd.DataFrame:
    pt_member, ft_exposed = GROUP_FACTORS[group]
    out = pool.iloc[indices].copy().reset_index(drop=True)
    out["group"] = group
    out["pt_member"] = pt_member
    out["ft_exposed"] = ft_exposed
    out["cell_index"] = np.arange(len(out), dtype=int)
    return out


def _deduplicate(pools: List[pd.DataFrame]) -> List[pd.DataFrame]:
    seen: set[str] = set()
    cleaned: List[pd.DataFrame] = []
    for pool in pools:
        keep = ~pool["text_sha256"].isin(seen)
        current = pool.loc[keep].drop_duplicates("text_sha256").reset_index(drop=True)
        seen.update(current["text_sha256"].tolist())
        cleaned.append(current)
    return cleaned


def build_mimir_membership(args: argparse.Namespace) -> pd.DataFrame:
    member = load_text_csv(Path(args.member_csv))
    nonmember_a = load_text_csv(Path(args.nonmember_csv))
    nonmember_b = load_text_csv(Path(args.nonmember_extra_csv))
    member, nonmember_a, nonmember_b = _deduplicate([member, nonmember_a, nonmember_b])
    nonmember = pd.concat([nonmember_a, nonmember_b], ignore_index=True).drop_duplicates("text_sha256")
    rng = np.random.default_rng(args.seed)
    if len(member) < 2 * args.n_per_cell:
        raise ValueError(f"member pool has {len(member)} rows; need {2 * args.n_per_cell}")
    if len(nonmember) < 2 * args.n_per_cell:
        raise ValueError(f"nonmember pool has {len(nonmember)} rows; need {2 * args.n_per_cell}")
    member_idx = rng.permutation(len(member))[: 2 * args.n_per_cell]
    nonmember_idx = rng.permutation(len(nonmember))[: 2 * args.n_per_cell]
    return pd.concat(
        [
            _take(member, member_idx[: args.n_per_cell], "p1f1"),
            _take(member, member_idx[args.n_per_cell :], "p1f0"),
            _take(nonmember, nonmember_idx[: args.n_per_cell], "p0f1"),
            _take(nonmember, nonmember_idx[args.n_per_cell :], "p0f0"),
        ],
        ignore_index=True,
    )


def build_controlled_exposure(args: argparse.Namespace) -> pd.DataFrame:
    pools = [load_text_csv(Path(args.nonmember_csv))]
    if args.nonmember_extra_csv:
        pools.append(load_text_csv(Path(args.nonmember_extra_csv)))
    pools = _deduplicate(pools)
    pool = pd.concat(pools, ignore_index=True).drop_duplicates("text_sha256").reset_index(drop=True)
    needed = 4 * args.n_per_cell
    if len(pool) < needed:
        raise ValueError(f"controlled pool has {len(pool)} rows; need {needed}")
    rng = np.random.default_rng(args.seed)
    selected = rng.permutation(len(pool))[:needed]
    parts = []
    for offset, group in enumerate(("p1f1", "p1f0", "p0f1", "p0f0")):
        start = offset * args.n_per_cell
        parts.append(_take(pool, selected[start : start + args.n_per_cell], group))
    return pd.concat(parts, ignore_index=True)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["mimir-membership", "controlled-exposure"], required=True)
    parser.add_argument("--member-csv", default="")
    parser.add_argument("--nonmember-csv", required=True)
    parser.add_argument("--nonmember-extra-csv", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-per-cell", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "mimir-membership":
        if not args.member_csv or not args.nonmember_extra_csv:
            raise ValueError("mimir-membership requires --member-csv and --nonmember-extra-csv")
        targets = build_mimir_membership(args)
    else:
        targets = build_controlled_exposure(args)
    targets = targets.sort_values(["group", "cell_index"]).reset_index(drop=True)
    targets["sample_id"] = [f"{g}::{i}" for g, i in zip(targets["group"], targets["cell_index"])]
    targets["label"] = targets["ft_exposed"].astype(int)
    validation = validate_factorial_targets(targets)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets_path = output_dir / "factorial_targets.csv"
    ft_path = output_dir / "factorial_ft_train.csv"
    pt_path = output_dir / "factorial_controlled_pt_train.csv"
    atomic_write_csv(targets_path, targets)
    atomic_write_csv(ft_path, targets[targets["ft_exposed"] == 1].reset_index(drop=True))
    atomic_write_csv(pt_path, targets[targets["pt_member"] == 1].reset_index(drop=True))

    input_paths = [Path(args.nonmember_csv)]
    if args.member_csv:
        input_paths.append(Path(args.member_csv))
    if args.nonmember_extra_csv:
        input_paths.append(Path(args.nonmember_extra_csv))
    manifest = base_manifest(experiment="e7_crossed_2x2" if args.mode == "mimir-membership" else "e11_controlled_family")
    manifest.update(
        {
            "status": "completed",
            "mode": args.mode,
            "seed": args.seed,
            "n_per_cell": args.n_per_cell,
            "inputs": {str(p): sha256_file(p) for p in input_paths},
            "outputs": {
                "targets": str(targets_path),
                "ft_train": str(ft_path),
                "controlled_pt_train": str(pt_path),
            },
            "validation": validation,
        }
    )
    atomic_write_json(output_dir / "factorial_manifest.json", manifest)
    print(pd.Series(validation).to_string())
    print(f"Wrote {targets_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
