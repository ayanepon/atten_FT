#!/usr/bin/env python3
"""Leak-free cross-fitted fusion of Proposed+EN and LoRA-Leak.

For each outer test fold, base-model scores for the outer training rows are
created by an inner CV.  The fusion model and alpha weight are fitted only on
these inner out-of-fold scores, then evaluated once on the outer test fold.
This avoids fitting the fusion layer on in-sample base predictions.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from run_fusion_en_lora_leak import (
    _fit_1d_lr_scores,
    _fit_en_scores,
    _pick_alpha,
    align_proposed_with_lora,
    choose_lora_col,
    load_lora_scores_csv,
    load_target_map,
    log,
    resolve_local_path,
)
from run_strict_fixed20_comparison_10runs import (
    COMPARISONS,
    compute_metrics,
    ensure_uid,
    fit_transform_train_only,
    load_or_build_proposed_features,
    make_common_splits,
    read_group_files,
    summarize,
)


def crossfit_base_scores(
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    lora_train_raw: np.ndarray,
    *,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate one inner-OOF EN and LoRA score for every outer-train row."""
    n = len(y_train)
    if n < args.inner_splits:
        raise ValueError("Not enough outer-train rows for the requested inner splits")
    en_oof = np.full(n, np.nan, dtype=float)
    lora_oof = np.full(n, np.nan, dtype=float)
    inner = StratifiedKFold(
        n_splits=args.inner_splits,
        shuffle=True,
        random_state=seed,
    )
    for inner_fold, (inner_train, inner_test) in enumerate(
        inner.split(np.zeros(n), y_train), start=1
    ):
        x_inner_train, x_inner_test = fit_transform_train_only(
            x_train_raw[inner_train], x_train_raw[inner_test]
        )
        _, en_pred, _ = _fit_en_scores(
            x_inner_train,
            y_train[inner_train],
            x_inner_test,
            args,
            seed + inner_fold,
        )
        _, lora_pred = _fit_1d_lr_scores(
            lora_train_raw[inner_train],
            y_train[inner_train],
            lora_train_raw[inner_test],
            seed + inner_fold,
        )
        en_oof[inner_test] = en_pred
        lora_oof[inner_test] = lora_pred
    if not np.isfinite(en_oof).all() or not np.isfinite(lora_oof).all():
        raise RuntimeError("Inner cross-fitting did not cover every outer-train row")
    return en_oof, lora_oof


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, max_fpr: float = 0.10) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = fpr <= max_fpr
    return float(np.max(tpr[valid])) if np.any(valid) else 0.0


def run_crossfit_repeat(
    repeat: int,
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    positive_group: str,
    negative_group: str,
    common_splits: List[Dict],
    args: argparse.Namespace,
) -> Tuple[List[Dict], List[Dict]]:
    sub = df[df["group"].isin([positive_group, negative_group])].drop_duplicates("uid").reset_index(drop=True)
    x_all = sub[list(feature_cols)].to_numpy(dtype=float)
    y_all = (sub["group"].to_numpy() == positive_group).astype(int)
    lora_all = sub["lora_raw"].to_numpy(dtype=float)
    uid_to_idx = {uid: i for i, uid in enumerate(sub["uid"].tolist())}
    fold_metrics = {name: [] for name in ("proposed_en", "lora_leak", "fusion_alpha_crossfit", "fusion_2d_crossfit")}
    rows: List[Dict] = []

    for split in [item for item in common_splits if item["repeat"] == repeat]:
        train_idx = np.asarray([uid_to_idx[uid] for uid in sorted(split["train_uids"])], dtype=int)
        test_idx = np.asarray([uid_to_idx[uid] for uid in sorted(split["test_uids"])], dtype=int)
        rng = args.seed + repeat * 1000 + split["fold"] * 10
        x_train_raw, x_test_raw = x_all[train_idx], x_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        lora_train, lora_test = lora_all[train_idx], lora_all[test_idx]

        en_inner_oof, lora_inner_oof = crossfit_base_scores(
            x_train_raw, y_train, lora_train, args=args, seed=rng
        )
        fusion = LogisticRegression(
            solver="lbfgs", C=1.0, max_iter=2000, class_weight="balanced", random_state=rng
        )
        fusion.fit(np.column_stack([en_inner_oof, lora_inner_oof]), y_train)
        alpha = _pick_alpha(en_inner_oof, lora_inner_oof, y_train, [i / 10.0 for i in range(11)])

        x_train, x_test = fit_transform_train_only(x_train_raw, x_test_raw)
        _, en_test, n_selected = _fit_en_scores(x_train, y_train, x_test, args, rng + 1)
        _, lora_test_score = _fit_1d_lr_scores(lora_train, y_train, lora_test, rng + 1)
        fusion_2d = fusion.predict_proba(np.column_stack([en_test, lora_test_score]))[:, 1]
        fusion_alpha = alpha * en_test + (1.0 - alpha) * lora_test_score
        scores = {
            "proposed_en": en_test,
            "lora_leak": lora_test_score,
            "fusion_alpha_crossfit": fusion_alpha,
            "fusion_2d_crossfit": fusion_2d,
        }
        for method, prediction in scores.items():
            fold_metrics[method].append(compute_metrics(y_test, prediction))
        for local_index, sample_index in enumerate(test_idx):
            rows.append(
                {
                    "repeat": repeat,
                    "fold": split["fold"],
                    "uid": sub.iloc[sample_index]["uid"],
                    "group": sub.iloc[sample_index]["group"],
                    "y": int(y_test[local_index]),
                    "s_en": float(en_test[local_index]),
                    "s_lora": float(lora_test_score[local_index]),
                    "s_fusion_alpha_crossfit": float(fusion_alpha[local_index]),
                    "s_fusion_2d_crossfit": float(fusion_2d[local_index]),
                    "alpha": float(alpha),
                    "n_selected_features": int(n_selected),
                }
            )

    summary_rows: List[Dict] = []
    # Mean selected features across folds (EN only); other methods leave NaN.
    n_selected_mean = float("nan")
    if rows:
        by_fold = {int(r["fold"]): int(r["n_selected_features"]) for r in rows}
        if by_fold:
            n_selected_mean = float(np.mean(list(by_fold.values())))
    for method, metrics in fold_metrics.items():
        summary_rows.append(
            {
                "method": method,
                "repeat": repeat,
                "auc": float(np.mean([item["auc"] for item in metrics])),
                "auprc": float(np.mean([item["auprc"] for item in metrics])),
                "tpr_at_10_fpr": float(np.mean([item["tpr_at_10_fpr"] for item in metrics])),
                "n_pos": int(y_all.sum()),
                "n_neg": int((1 - y_all).sum()),
                "n_features": len(feature_cols) if method == "proposed_en" else 2 if "fusion" in method else 1,
                "n_selected_mean": n_selected_mean if method == "proposed_en" else float("nan"),
            }
        )
    return summary_rows, rows


def descriptive_oof_bootstrap(oof: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    """Bootstrap AUC deltas after averaging each target's outer-test scores over repeats."""
    methods = {
        "fusion_alpha_crossfit": "s_fusion_alpha_crossfit",
        "fusion_2d_crossfit": "s_fusion_2d_crossfit",
    }
    rows: List[Dict] = []
    for comparison, part in oof.groupby("comparison", sort=True):
        values = part.groupby(["uid", "group", "y"], as_index=False).mean(numeric_only=True)
        y = values["y"].to_numpy(dtype=int)
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        rng = np.random.default_rng(seed)
        for fused_name, fused_col in methods.items():
            for base_name, base_col in (("proposed_en", "s_en"), ("lora_leak", "s_lora")):
                deltas = np.empty(n_bootstrap, dtype=float)
                for draw in range(n_bootstrap):
                    indices = np.concatenate([
                        rng.choice(pos, size=len(pos), replace=True),
                        rng.choice(neg, size=len(neg), replace=True),
                    ])
                    deltas[draw] = roc_auc_score(y[indices], values[fused_col].to_numpy()[indices]) - roc_auc_score(
                        y[indices], values[base_col].to_numpy()[indices]
                    )
                rows.append(
                    {
                        "comparison": comparison,
                        "fused_method": fused_name,
                        "baseline_method": base_name,
                        "mean_auc_delta": float(deltas.mean()),
                        "ci95_low": float(np.quantile(deltas, 0.025)),
                        "ci95_high": float(np.quantile(deltas, 0.975)),
                        "n_targets": int(len(values)),
                        "bootstrap_unit": "target after repeat-averaged outer-test score",
                        "interpretation": "descriptive within-constructed-dataset uncertainty; not independent-run significance",
                    }
                )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposed-root", default="attention_features_mimir_hardsplit_legacy")
    parser.add_argument("--lora-root", default="results/lora_leak_pythia1b")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--condition-prefix", default="fixed_attention_20")
    parser.add_argument("--model-label", default="pythia1b")
    parser.add_argument("--comparisons", nargs="+", default=["ft_vs_pt", "ft_vs_unseen"], choices=list(COMPARISONS))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-c", type=float, default=0.1)
    parser.add_argument("--classifier-c", type=float, default=1.0)
    parser.add_argument("--elasticnet-l1-ratio", type=float, default=0.7)
    parser.add_argument("--elasticnet-max-iter", type=int, default=1000)
    parser.add_argument("--elasticnet-tol", type=float, default=5e-4)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    proposed_root = resolve_local_path(args.proposed_root)
    lora_root = resolve_local_path(args.lora_root)
    raw = read_group_files(proposed_root, "raw_experiment4_attention_shift.csv", condition_prefix=args.condition_prefix)
    proposed = load_or_build_proposed_features(proposed_root, raw, condition_prefix=args.condition_prefix)
    lora = load_lora_scores_csv(lora_root)
    lora_col = choose_lora_col(lora, "auto")
    aligned = align_proposed_with_lora(proposed, lora, load_target_map(proposed_root, args.condition_prefix), lora_col)
    feature_cols = [column for column in aligned.columns if column.startswith("attn_l")]
    all_summary: List[Dict] = []
    all_oof: List[pd.DataFrame] = []

    for comparison in args.comparisons:
        positive_group, negative_group = COMPARISONS[comparison]
        common_splits, split_frame = make_common_splits(
            aligned, positive_group, negative_group, repeats=args.repeats, cv_splits=args.cv_splits, seed=args.seed
        )
        split_frame.insert(0, "comparison", comparison)
        split_frame.to_csv(out / f"common_folds_{comparison}.csv", index=False)
        for repeat in range(1, args.repeats + 1):
            log(f"cross-fitted fusion {comparison}: repeat {repeat}/{args.repeats}")
            rows, oof_rows = run_crossfit_repeat(
                repeat, aligned, feature_cols, positive_group, negative_group, common_splits, args
            )
            all_summary.extend({"model": args.model_label, "comparison": comparison, **row} for row in rows)
            all_oof.extend({"comparison": comparison, **row} for row in oof_rows)

    auc = pd.DataFrame(all_summary)
    if "n_selected_mean" not in auc.columns:
        auc["n_selected_mean"] = float("nan")
    auc.to_csv(out / "auc_10runs.csv", index=False)
    oof = pd.DataFrame(all_oof)
    oof.to_csv(out / "oof_scores.csv", index=False)
    # summarize after oof write so a summary KeyError never drops outer-test scores
    summary = summarize(auc)
    summary.to_csv(out / "summary_auc.csv", index=False)
    descriptive_oof_bootstrap(oof, args.n_bootstrap, args.seed).to_csv(
        out / "oof_bootstrap_auc_deltas.csv", index=False
    )
    manifest = vars(args).copy()
    manifest.update(
        {
            "script": Path(__file__).name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "proposed_root": str(proposed_root),
            "lora_root": str(lora_root),
            "lora_score_col_resolved": lora_col,
            "n_aligned": int(len(aligned)),
            "fusion_protocol": "inner-OOF base scores train fusion; outer test evaluated once",
        }
    )
    (out / "crossfit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
