# -*- coding: utf-8 -*-
"""Direction-selected loss baselines for the three fixed-20 model settings.

This script is designed for the stricter loss-baseline setting:

  1. Build the same sample-level repeated 5-fold splits.
  2. For each fold, choose the scalar score direction using only the train split.
  3. Apply the chosen direction to the held-out test split.
  4. Average fold metrics within each repeat.
  5. Save AUC, AUPRC, and TPR@10%FPR summaries for the loss-only baselines.

Loss baselines:
  - initial_loss: raw score is before_loss.
  - loss_diff: raw score is before_loss - after_loss.

The reported score direction is not selected from test labels.  FT is always the
positive class.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


GROUP_FT = "mimir_wikipedia_nonmember_ft"
GROUP_PT = "mimir_wikipedia_member_pt"
GROUP_UNSEEN = "mimir_wikipedia_nonmember_unseen"

COMPARISONS = {
    "ft_vs_pt": (GROUP_FT, GROUP_PT),
    "ft_vs_unseen": (GROUP_FT, GROUP_UNSEEN),
}

MODEL_ROOTS = {
    "pythia1b": (
        "/workplace/FT/BlackNLP_2/results/"
        "experiment4_mimir_hardsplit_stopping_condition"
    ),
    "pythia410m": (
        "/workplace/FT/BlackNLP_2/results/"
        "mimir_wikipedia_hardsplit_fixed20_pythia410m_rerun"
    ),
    "gptneo27b": (
        "/workplace/FT/BlackNLP_2/results/"
        "mimir_wikipedia_hardsplit_fixed20_gptneo27b"
    ),
}
DEFAULT_OUTPUT_DIR = (
    "/workplace/FT/BlackNLP_2/results/"
    "loss_direction_selected_3model"
)


def here() -> Path:
    return Path(__file__).resolve().parent


def has_fixed20_groups(path: Path) -> bool:
    return all(
        (path / f"fixed_attention_20_{g}" / "sample_level_experiment4.csv").exists()
        for g in ["ft", "pt", "unseen"]
    )


def resolve_path(path_like: str, *, fixed20: bool = False, required_file: str | None = None) -> Path:
    raw = Path(path_like)
    candidates = [raw, here() / raw.name]
    path_str = str(path_like)
    for prefix in [
        "/workplace/FT/BlackboxNLP_2/results/",
        "/workplace/FT/BlackNLP_2/results/",
        "/workplace/FT/BlackNLP/results/",
        "/workplace/FT/",
    ]:
        if path_str.startswith(prefix):
            candidates.append(here() / path_str.replace(prefix, ""))
    if fixed20:
        candidates.append(here())
    seen = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        if fixed20 and has_fixed20_groups(candidate):
            return candidate
        if required_file and (candidate / required_file).exists():
            return candidate / required_file
        if not fixed20 and required_file is None and candidate.exists():
            return candidate
    raise FileNotFoundError(f"Path not found: {path_like}")


def load_group_csv(root: Path, group_key: str) -> pd.DataFrame:
    path = root / f"fixed_attention_20_{group_key}" / "sample_level_experiment4.csv"
    if not path.exists():
        matches = list(root.glob(f"**/fixed_attention_20_{group_key}/sample_level_experiment4.csv"))
        if not matches:
            raise FileNotFoundError(f"sample_level_experiment4.csv not found for {group_key} under {root}")
        path = matches[0]
    df = pd.read_csv(path)
    df["source_file"] = str(path)
    return df


def ensure_uid(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sample_id" not in out.columns:
        out["sample_id"] = out.groupby("group").cumcount()
    out["sample_id"] = out["sample_id"].astype(int)
    out["_local_sample_id"] = 0
    for group, idx in out.groupby("group").groups.items():
        ids = sorted(out.loc[idx, "sample_id"].drop_duplicates().tolist())
        mapper = {sid: i for i, sid in enumerate(ids)}
        out.loc[idx, "_local_sample_id"] = out.loc[idx, "sample_id"].map(mapper).astype(int)
    out["uid"] = out["group"].astype(str) + "::" + out["_local_sample_id"].astype(str)
    return out


def load_sample_scores(root: Path) -> pd.DataFrame:
    df = pd.concat(
        [
            load_group_csv(root, "ft"),
            load_group_csv(root, "pt"),
            load_group_csv(root, "unseen"),
        ],
        ignore_index=True,
    )
    required = {"group", "sample_id", "before_loss", "delta_loss_before_minus_after"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required sample-level columns: {sorted(missing)}")
    df = ensure_uid(df)
    df["initial_loss_raw"] = pd.to_numeric(df["before_loss"], errors="coerce")
    df["loss_diff_raw"] = pd.to_numeric(df["delta_loss_before_minus_after"], errors="coerce")
    return df


def make_splits(
    base: pd.DataFrame,
    positive: str,
    negative: str,
    repeats: int,
    cv_splits: int,
    seed: int,
) -> Tuple[List[Dict], pd.DataFrame]:
    sub = (
        base[base["group"].isin([positive, negative])][["uid", "group"]]
        .drop_duplicates("uid")
        .sort_values(["group", "uid"])
        .reset_index(drop=True)
    )
    y = (sub["group"].to_numpy() == positive).astype(int)
    if len(np.bincount(y)) != 2 or np.bincount(y).min() < cv_splits:
        raise ValueError(f"Not enough samples: {np.bincount(y).tolist()}")

    splits = []
    split_rows = []
    for repeat in range(1, repeats + 1):
        rng = np.random.default_rng(seed + repeat - 1)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        pos_folds = np.array_split(pos_idx, cv_splits)
        neg_folds = np.array_split(neg_idx, cv_splits)
        all_idx = np.arange(len(y))
        for fold in range(1, cv_splits + 1):
            test_idx = np.concatenate([pos_folds[fold - 1], neg_folds[fold - 1]])
            train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=False)
            train_uids = set(sub.iloc[train_idx]["uid"].tolist())
            test_uids = set(sub.iloc[test_idx]["uid"].tolist())
            splits.append({"repeat": repeat, "fold": fold, "train_uids": train_uids, "test_uids": test_uids})
            for split_name, idxs in [("train", train_idx), ("test", test_idx)]:
                for row in sub.iloc[idxs].itertuples(index=False):
                    split_rows.append(
                        {
                            "repeat": repeat,
                            "fold": fold,
                            "split": split_name,
                            "uid": row.uid,
                            "group": row.group,
                            "label_positive": int(row.group == positive),
                        }
                    )
    return splits, pd.DataFrame(split_rows)


def tpr_at_fpr(y: np.ndarray, score: np.ndarray, max_fpr: float = 0.10) -> float:
    fpr, tpr, _ = roc_curve(y, score)
    valid = fpr <= max_fpr
    return float(np.max(tpr[valid])) if np.any(valid) else 0.0


def choose_direction_on_train(y_train: np.ndarray, raw_train: np.ndarray) -> Tuple[int, float]:
    auc_pos = float(roc_auc_score(y_train, raw_train))
    auc_neg = float(roc_auc_score(y_train, -raw_train))
    if auc_pos >= auc_neg:
        return 1, auc_pos
    return -1, auc_neg


def evaluate_direction_selected(
    df: pd.DataFrame,
    raw_col: str,
    method: str,
    positive: str,
    negative: str,
    splits: Sequence[Dict],
    repeats: int,
) -> Tuple[List[Dict], pd.DataFrame]:
    sub = (
        df[df["group"].isin([positive, negative])]
        .dropna(subset=[raw_col])
        .drop_duplicates("uid")
        .reset_index(drop=True)
    )
    y = (sub["group"].to_numpy() == positive).astype(int)
    raw = sub[raw_col].to_numpy(float)
    uid_to_idx = {uid: i for i, uid in enumerate(sub["uid"].tolist())}

    repeat_rows = []
    fold_rows = []
    for repeat in range(1, repeats + 1):
        fold_auc = []
        fold_auprc = []
        fold_tpr10 = []
        signs = []
        train_aucs = []
        for split in [s for s in splits if s["repeat"] == repeat]:
            tr = np.array([uid_to_idx[u] for u in split["train_uids"] if u in uid_to_idx], dtype=int)
            te = np.array([uid_to_idx[u] for u in split["test_uids"] if u in uid_to_idx], dtype=int)
            sign, train_auc = choose_direction_on_train(y[tr], raw[tr])
            test_score = sign * raw[te]
            test_auc = float(roc_auc_score(y[te], test_score))
            test_auprc = float(average_precision_score(y[te], test_score))
            test_tpr10 = tpr_at_fpr(y[te], test_score, 0.10)
            fold_auc.append(test_auc)
            fold_auprc.append(test_auprc)
            fold_tpr10.append(test_tpr10)
            signs.append(sign)
            train_aucs.append(train_auc)
            fold_rows.append(
                {
                    "repeat": repeat,
                    "fold": split["fold"],
                    "method": method,
                    "raw_col": raw_col,
                    "selected_sign": sign,
                    "train_selected_auc": train_auc,
                    "test_auc": test_auc,
                    "test_auprc": test_auprc,
                    "test_tpr_at_10_fpr": test_tpr10,
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                }
            )
        repeat_rows.append(
            {
                "method": method,
                "raw_col": raw_col,
                "repeat": repeat,
                "auc": float(np.mean(fold_auc)),
                "auprc": float(np.mean(fold_auprc)),
                "tpr_at_10_fpr": float(np.mean(fold_tpr10)),
                "n_positive": int(y.sum()),
                "n_negative": int((1 - y).sum()),
                "n_features": 1,
                "selected_sign_mean": float(np.mean(signs)),
                "selected_positive_direction_folds": int(np.sum(np.array(signs) == 1)),
                "selected_negative_direction_folds": int(np.sum(np.array(signs) == -1)),
                "train_selected_auc_mean": float(np.mean(train_aucs)),
            }
        )
    return repeat_rows, pd.DataFrame(fold_rows)


def summarize_auc(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["model", "comparison", "method"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
            tpr_at_10_fpr_mean=("tpr_at_10_fpr", "mean"),
            tpr_at_10_fpr_std=("tpr_at_10_fpr", "std"),
            n_repeats=("repeat", "nunique"),
            n_positive=("n_positive", "first"),
            n_negative=("n_negative", "first"),
            selected_positive_direction_folds=("selected_positive_direction_folds", "sum"),
            selected_negative_direction_folds=("selected_negative_direction_folds", "sum"),
        )
        .sort_values(["model", "comparison", "method"])
    )


def run_model(
    model: str,
    root: Path,
    args: argparse.Namespace,
    out: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sample = load_sample_scores(root)
    model_dir = out / model
    model_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(model_dir / "loss_direction_selected_sample_scores.csv", index=False)
    sample.groupby("group").size().rename("n").reset_index().to_csv(model_dir / "group_counts.csv", index=False)

    auc_rows = []
    fold_tables = []
    for comparison in args.comparisons:
        positive, negative = COMPARISONS[comparison]
        splits, split_df = make_splits(sample, positive, negative, args.repeats, args.cv_splits, args.seed)
        split_df.insert(0, "model", model)
        split_df.insert(1, "comparison", comparison)
        split_df.to_csv(model_dir / f"common_folds_{comparison}.csv", index=False)
        for raw_col, method in [
            ("initial_loss_raw", "initial_loss"),
            ("loss_diff_raw", "loss_diff"),
        ]:
            rows, folds = evaluate_direction_selected(
                sample,
                raw_col,
                method,
                positive,
                negative,
                splits,
                args.repeats,
            )
            auc_rows.extend({"model": model, "comparison": comparison, **row} for row in rows)
            folds.insert(0, "model", model)
            folds.insert(1, "comparison", comparison)
            fold_tables.append(folds)

    auc = pd.DataFrame(auc_rows)
    folds = pd.concat(fold_tables, ignore_index=True)
    auc.to_csv(model_dir / "loss_direction_selected_auc_10runs.csv", index=False)
    folds.to_csv(model_dir / "loss_direction_selected_fold_details.csv", index=False)
    summary = summarize_auc(auc)
    summary.to_csv(model_dir / "loss_direction_selected_summary.csv", index=False)
    return auc, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="+", default=["pythia1b", "pythia410m", "gptneo27b"], choices=list(MODEL_ROOTS.keys()))
    for model, default_root in MODEL_ROOTS.items():
        parser.add_argument(f"--{model}-root", default=default_root)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--comparisons",
        nargs="+",
        default=["ft_vs_pt", "ft_vs_unseen"],
        choices=list(COMPARISONS.keys()),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    roots = {
        model: resolve_path(getattr(args, f"{model}_root"), fixed20=True)
        for model in args.models
    }

    all_auc = []
    all_summary = []
    for model in args.models:
        auc, summary = run_model(model, roots[model], args, out)
        all_auc.append(auc)
        all_summary.append(summary)

    auc_df = pd.concat(all_auc, ignore_index=True)
    summary_df = pd.concat(all_summary, ignore_index=True)

    auc_df.to_csv(out / "loss_direction_selected_auc_10runs.csv", index=False)
    summary_df.to_csv(out / "loss_direction_selected_summary.csv", index=False)

    with open(out / "loss_direction_selected_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                "resolved_roots": {k: str(v) for k, v in roots.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nDirection-selected loss baseline summary:")
    print(summary_df.round(6).to_string(index=False))
    print(f"\nOutput directory: {out}")


if __name__ == "__main__":
    main()
