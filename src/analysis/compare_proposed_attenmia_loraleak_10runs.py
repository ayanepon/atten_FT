# -*- coding: utf-8 -*-
"""
Compare Proposed / AttenMIA / LoRA-Leak with 10 repeated AUC runs.

Default comparisons:
  - FT vs PT
  - FT vs Unseen

The proposed method uses fixed-20 attention-update outputs and keeps every
layer/head/metric as a feature.  It reports both:
  - proposed_layer_head_all
  - proposed_l1_selected_layer_head

AttenMIA is re-evaluated from saved AttenMIA feature CSVs using 10 repeated
5-fold CV.  LoRA-Leak is evaluated from saved score CSVs.  For LoRA-Leak,
the score column is selected from the best strict AUC in
lora_leak_pairwise_results.csv unless explicitly overridden.

Outputs:
  - auc_10runs.csv
  - paired_auc_tests.csv
  - summary_auc.csv
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


GROUP_FT = "mimir_wikipedia_nonmember_ft"
GROUP_PT = "mimir_wikipedia_member_pt"
GROUP_UNSEEN = "mimir_wikipedia_nonmember_unseen"

COMPARISONS = {
    "ft_vs_pt": (GROUP_FT, GROUP_PT),
    "ft_vs_unseen": (GROUP_FT, GROUP_UNSEEN),
}

ATTENTION_METRICS = [
    "l1_mean",
    "l2_rms",
    "mse",
    "js_div",
    "entropy_delta",
    "max_shift",
    "top1_shift_mean",
    "top5_shift_mean",
    "top10_shift_mean",
]

DEFAULT_1B_PROPOSED_ROOT = "results/experiment4_mimir_hardsplit_stopping_condition"
DEFAULT_1B_ATTENMIA_DIR = "results/attenmia_official_mimir_hardsplit"
DEFAULT_1B_LORA_LEAK_DIR = "results/lora_leak_official_mimir_hardsplit"

DEFAULT_410M_PROPOSED_ROOT = "results/mimir_wikipedia_hardsplit_fixed20_pythia410m_rerun"
DEFAULT_410M_ATTENMIA_DIR = "results/attenmia_official_mimir_hardsplit_pythia410m"
DEFAULT_410M_LORA_LEAK_DIR = "results/lora_leak_official_mimir_hardsplit_pythia410m"


def local_root() -> Path:
    return Path(__file__).resolve().parent


def log(message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def resolve_path(path_like: str, required_files: Sequence[str] = ()) -> Path:
    """Resolve server paths or downloaded local mirrors."""
    original = Path(path_like)
    candidates = [original]

    root = local_root()
    path_str = str(path_like)
    replacements = [
        "results/",
        "models/",
        "results/",
        "",
    ]
    for prefix in replacements:
        if path_str.startswith(prefix):
            candidates.append(root / path_str.replace(prefix, ""))
    candidates.append(root / original.name)

    # Some previous runs were nested under another result directory.
    for nested in root.glob(f"**/{original.name}"):
        candidates.append(nested)

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve() if candidate.exists() else candidate
        if str(candidate) in seen:
            continue
        seen.add(str(candidate))
        if not candidate.exists():
            continue
        if required_files and not all((candidate / f).exists() for f in required_files):
            continue
        return candidate

    # Compatibility for downloaded proposed-method outputs where
    # fixed_attention_20_{ft,pt,unseen} were placed directly next to this script.
    proposed_dirs = [
        root / "fixed_attention_20_ft",
        root / "fixed_attention_20_pt",
        root / "fixed_attention_20_unseen",
    ]
    if not required_files and all(d.exists() for d in proposed_dirs):
        if "experiment4_mimir_hardsplit_stopping_condition" in path_str:
            return root

    missing = ", ".join(required_files) if required_files else "path"
    raise FileNotFoundError(f"Could not resolve {path_like} with required files: {missing}")


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.10) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    ok = np.where(fpr <= target_fpr)[0]
    return float(np.max(tpr[ok])) if len(ok) else 0.0


def metric_row(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "tpr_at_10_fpr": tpr_at_fpr(y_true, scores, 0.10),
    }


def ensure_uid(df: pd.DataFrame) -> pd.DataFrame:
    """Attach stable per-sample uid = group::group-local-sample-id.

    Most generated files already contain sample_id.  If a baseline file lacks it,
    fall back to the within-group row order, which matches the saved target order
    used by these scripts.  If sample_id is global, remap sorted unique sample_id
    values to 0..N-1 within each group.
    """
    out = df.copy()
    if "sample_id" not in out.columns:
        out["sample_id"] = out.groupby("group").cumcount()
    out["sample_id"] = out["sample_id"].astype(int)
    out["_local_sample_id"] = 0
    for group, idx in out.groupby("group").groups.items():
        unique_ids = sorted(out.loc[idx, "sample_id"].drop_duplicates().tolist())
        mapper = {sid: i for i, sid in enumerate(unique_ids)}
        out.loc[idx, "_local_sample_id"] = out.loc[idx, "sample_id"].map(mapper).astype(int)
    out["uid"] = out["group"].astype(str) + "::" + out["_local_sample_id"].astype(str)
    return out


def make_common_splits(
    base_df: pd.DataFrame,
    positive_group: str,
    negative_group: str,
    repeats: int,
    n_splits: int,
    seed: int,
) -> Tuple[List[Dict], pd.DataFrame]:
    base = ensure_uid(base_df)
    sub = (
        base[base["group"].isin([positive_group, negative_group])][["uid", "group"]]
        .drop_duplicates("uid")
        .sort_values(["group", "uid"])
        .reset_index(drop=True)
    )
    y = (sub["group"].to_numpy() == positive_group).astype(int)
    splits: List[Dict] = []
    rows = []
    for rep in range(repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y), start=1):
            train_uids = set(sub.iloc[tr]["uid"].tolist())
            test_uids = set(sub.iloc[te]["uid"].tolist())
            splits.append({
                "repeat": rep + 1,
                "fold": fold,
                "train_uids": train_uids,
                "test_uids": test_uids,
            })
            for uid in test_uids:
                rows.append({"repeat": rep + 1, "fold": fold, "uid": uid, "split": "test"})
            for uid in train_uids:
                rows.append({"repeat": rep + 1, "fold": fold, "uid": uid, "split": "train"})
    return splits, pd.DataFrame(rows)


def load_proposed_raw(proposed_root: Path) -> pd.DataFrame:
    dirs = [
        proposed_root / "fixed_attention_20_ft",
        proposed_root / "fixed_attention_20_pt",
        proposed_root / "fixed_attention_20_unseen",
    ]
    parts = []
    for d in dirs:
        path = d / "raw_experiment4_attention_shift.csv"
        if not path.exists():
            raise FileNotFoundError(f"Proposed raw CSV not found: {path}")
        parts.append(pd.read_csv(path))
    raw = pd.concat(parts, ignore_index=True)
    return raw


def load_proposed_sample_level(proposed_root: Path) -> pd.DataFrame:
    dirs = [
        proposed_root / "fixed_attention_20_ft",
        proposed_root / "fixed_attention_20_pt",
        proposed_root / "fixed_attention_20_unseen",
    ]
    parts = []
    for d in dirs:
        path = d / "sample_level_experiment4.csv"
        if not path.exists():
            raise FileNotFoundError(f"Proposed sample-level CSV not found: {path}")
        parts.append(pd.read_csv(path))
    return pd.concat(parts, ignore_index=True)


def proposed_wide_features(raw: pd.DataFrame) -> pd.DataFrame:
    wide = None
    for metric in ATTENTION_METRICS:
        if metric not in raw.columns:
            continue
        piv = raw.pivot_table(
            index=["sample_id", "group"],
            columns=["layer", "head"],
            values=metric,
            aggfunc="mean",
        )
        piv.columns = [f"{metric}_L{int(layer):02d}_H{int(head):02d}" for layer, head in piv.columns]
        piv = piv.reset_index()
        wide = piv if wide is None else wide.merge(piv, on=["sample_id", "group"], how="inner")
    if wide is None:
        raise ValueError("No attention metrics were found in proposed raw CSVs.")
    return wide


def repeated_cv_logistic(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    positive_group: str,
    negative_group: str,
    repeats: int,
    n_splits: int,
    seed: int,
    method: str,
    common_splits: List[Dict],
    l1_select: bool = False,
    l1_c: float = 0.02,
) -> List[Dict]:
    sub = ensure_uid(df)
    sub = sub[sub["group"].isin([positive_group, negative_group])].dropna(subset=list(feature_cols)).copy()
    sub = sub.drop_duplicates("uid").reset_index(drop=True)
    y = (sub["group"].to_numpy() == positive_group).astype(int)
    X = sub[list(feature_cols)].to_numpy(dtype=float)
    uid_to_idx = {uid: i for i, uid in enumerate(sub["uid"].tolist())}
    rows = []

    for rep in range(repeats):
        log(f"{method}: repeat {rep + 1}/{repeats}")
        scores = np.zeros(len(y), dtype=float)
        selected_counts = []
        fold_metrics = []
        rep_splits = [s for s in common_splits if s["repeat"] == rep + 1]
        for split in rep_splits:
            tr = np.array([uid_to_idx[u] for u in split["train_uids"] if u in uid_to_idx], dtype=int)
            te = np.array([uid_to_idx[u] for u in split["test_uids"] if u in uid_to_idx], dtype=int)
            if len(tr) == 0 or len(te) == 0:
                continue
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[tr])
            Xte = scaler.transform(X[te])
            if l1_select:
                selector = LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=l1_c,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=seed + rep,
                )
                selector.fit(Xtr, y[tr])
                mask = np.abs(selector.coef_[0]) > 1e-12
                if not mask.any():
                    mask[np.argmax(np.abs(selector.coef_[0]))] = True
                selected_counts.append(int(mask.sum()))
            else:
                mask = np.ones(X.shape[1], dtype=bool)
                selected_counts.append(int(mask.sum()))

            clf = LogisticRegression(
                solver="lbfgs",
                class_weight="balanced",
                max_iter=3000,
                random_state=seed + rep,
            )
            clf.fit(Xtr[:, mask], y[tr])
            fold_score = clf.decision_function(Xte[:, mask])
            scores[te] = fold_score
            fm = metric_row(y[te], fold_score)
            fm.update({"fold": split["fold"]})
            fold_metrics.append(fm)

        row = metric_row(y, scores)
        row.update({
            "method": method,
            "repeat": rep + 1,
            "n_pos": int(y.sum()),
            "n_neg": int((1 - y).sum()),
            "n_features": int(X.shape[1]),
            "n_selected_mean": float(np.mean(selected_counts)),
            "fold_auc_mean": float(np.mean([m["auc"] for m in fold_metrics])),
        })
        rows.append(row)
    return rows


def load_attenmia_features(attenmia_dir: Path, comparison: str) -> pd.DataFrame:
    candidates = [
        attenmia_dir / comparison / "attenmia_official_base_features.csv",
        attenmia_dir / attenmia_dir.name / comparison / "attenmia_official_base_features.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(candidate)
    # Last-resort nested search.
    matches = list(attenmia_dir.glob(f"**/{comparison}/attenmia_official_base_features.csv"))
    if matches:
        return pd.read_csv(matches[0])
    raise FileNotFoundError(f"AttenMIA feature CSV not found for {comparison} under {attenmia_dir}")


def repeated_cv_attenmia_mlp(
    attenmia_dir: Path,
    comparison: str,
    positive_group: str,
    negative_group: str,
    repeats: int,
    n_splits: int,
    seed: int,
    common_splits: List[Dict],
) -> List[Dict]:
    df = load_attenmia_features(attenmia_dir, comparison)
    df = ensure_uid(df)
    feature_cols = [c for c in df.columns if c.startswith(("trans_", "base_", "pert_"))]
    if not feature_cols:
        raise ValueError(f"No AttenMIA feature columns found for {comparison}")
    sub = df[df["group"].isin([positive_group, negative_group])].copy()
    sub = sub.drop_duplicates("uid").reset_index(drop=True)
    y = (sub["group"].to_numpy() == positive_group).astype(int)
    X = sub[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    uid_to_idx = {uid: i for i, uid in enumerate(sub["uid"].tolist())}
    rows = []
    for rep in range(repeats):
        log(f"attenmia_mlp {comparison}: repeat {rep + 1}/{repeats}")
        scores = np.zeros(len(y), dtype=float)
        fold_aucs = []
        rep_splits = [s for s in common_splits if s["repeat"] == rep + 1]
        for split in rep_splits:
            tr = np.array([uid_to_idx[u] for u in split["train_uids"] if u in uid_to_idx], dtype=int)
            te = np.array([uid_to_idx[u] for u in split["test_uids"] if u in uid_to_idx], dtype=int)
            if len(tr) == 0 or len(te) == 0:
                continue
            clf = Pipeline([
                ("scale", StandardScaler()),
                ("mlp", MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=500,
                    early_stopping=True,
                    random_state=seed + rep,
                )),
            ])
            clf.fit(X[tr], y[tr])
            fold_score = clf.predict_proba(X[te])[:, 1]
            scores[te] = fold_score
            fold_aucs.append(float(roc_auc_score(y[te], fold_score)))
        row = metric_row(y, scores)
        row.update({
            "method": "attenmia_mlp",
            "repeat": rep + 1,
            "n_pos": int(y.sum()),
            "n_neg": int((1 - y).sum()),
            "n_features": int(len(feature_cols)),
            "n_selected_mean": math.nan,
            "fold_auc_mean": float(np.mean(fold_aucs)),
        })
        rows.append(row)
    return rows


def best_lora_score_col(lora_dir: Path, comparison: str, mode: str) -> str:
    pairwise_path = lora_dir / "lora_leak_pairwise_results.csv"
    if not pairwise_path.exists():
        matches = list(lora_dir.glob("**/lora_leak_pairwise_results.csv"))
        if not matches:
            raise FileNotFoundError(f"lora_leak_pairwise_results.csv not found under {lora_dir}")
        pairwise_path = matches[0]
    df = pd.read_csv(pairwise_path)
    sub = df[df["comparison"] == comparison].copy()
    if sub.empty:
        raise ValueError(f"No LoRA-Leak rows for comparison={comparison}")
    if mode == "target_mink_family_best":
        sub = sub[sub["score_col"].str.startswith(("target_mink_", "target_mink++_"))].copy()
    elif mode == "target_mink_best":
        sub = sub[sub["score_col"].str.startswith("target_mink_")].copy()
    elif mode == "target_minkpp_best":
        sub = sub[sub["score_col"].str.startswith("target_mink++_")].copy()
    elif mode.startswith("score:"):
        score_col = mode.split(":", 1)[1]
        if score_col not in set(sub["score_col"]):
            raise ValueError(f"Requested LoRA-Leak score not found for {comparison}: {score_col}")
        return score_col
    elif mode != "best_strict":
        raise ValueError(f"Unknown --lora-score-mode: {mode}")
    if sub.empty:
        raise ValueError(f"No LoRA-Leak rows remain after applying mode={mode} for comparison={comparison}")
    return str(sub.sort_values("auroc", ascending=False).iloc[0]["score_col"])


def load_lora_scores(lora_dir: Path) -> pd.DataFrame:
    path = lora_dir / "lora_leak_scores.csv"
    if not path.exists():
        matches = list(lora_dir.glob("**/lora_leak_scores.csv"))
        if not matches:
            raise FileNotFoundError(f"lora_leak_scores.csv not found under {lora_dir}")
        path = matches[0]
    return pd.read_csv(path)


def repeated_fold_auc_from_fixed_scores(
    scores_df: pd.DataFrame,
    score_col: str,
    positive_group: str,
    negative_group: str,
    repeats: int,
    n_splits: int,
    seed: int,
    common_splits: List[Dict],
) -> List[Dict]:
    sub = ensure_uid(scores_df)
    sub = sub[sub["group"].isin([positive_group, negative_group])].dropna(subset=[score_col]).copy()
    sub = sub.drop_duplicates("uid").reset_index(drop=True)
    y = (sub["group"].to_numpy() == positive_group).astype(int)
    score = sub[score_col].to_numpy(float)
    uid_to_idx = {uid: i for i, uid in enumerate(sub["uid"].tolist())}
    rows = []
    for rep in range(repeats):
        log(f"lora_leak:{score_col}: repeat {rep + 1}/{repeats}")
        fold_auc = []
        fold_auprc = []
        fold_tpr10 = []
        rep_splits = [s for s in common_splits if s["repeat"] == rep + 1]
        for split in rep_splits:
            te = np.array([uid_to_idx[u] for u in split["test_uids"] if u in uid_to_idx], dtype=int)
            if len(te) == 0:
                continue
            fold_auc.append(float(roc_auc_score(y[te], score[te])))
            fold_auprc.append(float(average_precision_score(y[te], score[te])))
            fold_tpr10.append(tpr_at_fpr(y[te], score[te], 0.10))
        rows.append({
            "method": f"lora_leak:{score_col}",
            "repeat": rep + 1,
            "auc": float(np.mean(fold_auc)),
            "auprc": float(np.mean(fold_auprc)),
            "tpr_at_10_fpr": float(np.mean(fold_tpr10)),
            "n_pos": int(y.sum()),
            "n_neg": int((1 - y).sum()),
            "n_features": 1,
            "n_selected_mean": math.nan,
            "fold_auc_mean": float(np.mean(fold_auc)),
        })
    return rows


def repeated_fold_auc_from_sample_column(
    sample_df: pd.DataFrame,
    score_col: str,
    method: str,
    positive_group: str,
    negative_group: str,
    repeats: int,
    n_splits: int,
    seed: int,
    common_splits: List[Dict],
) -> List[Dict]:
    sub = ensure_uid(sample_df)
    sub = sub[sub["group"].isin([positive_group, negative_group])].dropna(subset=[score_col]).copy()
    sub = sub.drop_duplicates("uid").reset_index(drop=True)
    y = (sub["group"].to_numpy() == positive_group).astype(int)
    score = sub[score_col].to_numpy(float)
    uid_to_idx = {uid: i for i, uid in enumerate(sub["uid"].tolist())}
    rows = []
    for rep in range(repeats):
        log(f"{method}: repeat {rep + 1}/{repeats}")
        fold_auc = []
        fold_auprc = []
        fold_tpr10 = []
        rep_splits = [s for s in common_splits if s["repeat"] == rep + 1]
        for split in rep_splits:
            te = np.array([uid_to_idx[u] for u in split["test_uids"] if u in uid_to_idx], dtype=int)
            if len(te) == 0:
                continue
            fold_auc.append(float(roc_auc_score(y[te], score[te])))
            fold_auprc.append(float(average_precision_score(y[te], score[te])))
            fold_tpr10.append(tpr_at_fpr(y[te], score[te], 0.10))
        rows.append({
            "method": method,
            "repeat": rep + 1,
            "auc": float(np.mean(fold_auc)),
            "auprc": float(np.mean(fold_auprc)),
            "tpr_at_10_fpr": float(np.mean(fold_tpr10)),
            "n_pos": int(y.sum()),
            "n_neg": int((1 - y).sum()),
            "n_features": 1,
            "n_selected_mean": math.nan,
            "fold_auc_mean": float(np.mean(fold_auc)),
        })
    return rows


def paired_tests(auc_df: pd.DataFrame, proposed_method: str) -> pd.DataFrame:
    rows = []
    for (model, comparison), sub in auc_df.groupby(["model", "comparison"]):
        methods = sorted(sub["method"].unique())
        if proposed_method not in methods:
            continue
        p = sub[sub["method"] == proposed_method].sort_values("repeat")
        for method in methods:
            if method == proposed_method:
                continue
            b = sub[sub["method"] == method].sort_values("repeat")
            merged = p[["repeat", "auc"]].merge(b[["repeat", "auc"]], on="repeat", suffixes=("_proposed", "_baseline"))
            if len(merged) < 2:
                continue
            diff = merged["auc_proposed"].to_numpy(float) - merged["auc_baseline"].to_numpy(float)
            try:
                w = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
                wilcoxon_p = float(w.pvalue)
            except ValueError:
                wilcoxon_p = math.nan
            t = ttest_rel(merged["auc_proposed"], merged["auc_baseline"])
            rows.append({
                "model": model,
                "comparison": comparison,
                "proposed_method": proposed_method,
                "baseline_method": method,
                "n_repeats": int(len(merged)),
                "proposed_auc_mean": float(merged["auc_proposed"].mean()),
                "baseline_auc_mean": float(merged["auc_baseline"].mean()),
                "mean_auc_diff": float(diff.mean()),
                "std_auc_diff": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
                "wilcoxon_p": wilcoxon_p,
                "paired_t_p": float(t.pvalue),
            })
    return pd.DataFrame(rows)


def summarize_auc(auc_df: pd.DataFrame) -> pd.DataFrame:
    return (
        auc_df.groupby(["model", "comparison", "method"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            auprc_mean=("auprc", "mean"),
            tpr_at_10_fpr_mean=("tpr_at_10_fpr", "mean"),
            n_repeats=("repeat", "count"),
            n_features=("n_features", "first"),
            n_selected_mean=("n_selected_mean", "mean"),
        )
    )


def append_rows(rows: List[Dict], new_rows: List[Dict], output_dir: Path) -> List[Dict]:
    rows.extend(new_rows)
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "auc_10runs.partial.csv", index=False)
    return rows


def run_for_model(model_name: str, proposed_root: str, attenmia_dir: str, lora_dir: str, args, output_dir: Path) -> List[Dict]:
    log(f"Start model={model_name}")
    proposed_root_p = resolve_path(proposed_root)
    attenmia_dir_p = resolve_path(attenmia_dir)
    lora_dir_p = resolve_path(lora_dir, required_files=["lora_leak_scores.csv", "lora_leak_pairwise_results.csv"])

    log(f"{model_name}: loading proposed raw")
    raw = load_proposed_raw(proposed_root_p)
    sample_level = load_proposed_sample_level(proposed_root_p)
    log(f"{model_name}: building layer/head feature table")
    wide = proposed_wide_features(raw)
    feature_cols = [c for c in wide.columns if c not in {"sample_id", "group"}]
    lora_scores = load_lora_scores(lora_dir_p)

    rows: List[Dict] = []
    selected_comparisons = {name: COMPARISONS[name] for name in args.comparisons}
    for comparison, (positive, negative) in selected_comparisons.items():
        log(f"{model_name} {comparison}: start")
        common_splits, split_df = make_common_splits(
            wide,
            positive,
            negative,
            args.repeats,
            args.cv_splits,
            args.seed,
        )
        split_df.insert(0, "comparison", comparison)
        split_df.insert(0, "model", model_name)
        split_df.to_csv(output_dir / f"common_folds_{model_name}_{comparison}.csv", index=False)

        if "proposed_all" in args.methods:
            append_rows(rows, [
            {"model": model_name, "comparison": comparison, **r}
            for r in repeated_cv_logistic(
                wide,
                feature_cols,
                positive,
                negative,
                args.repeats,
                args.cv_splits,
                args.seed,
                "proposed_layer_head_all",
                common_splits,
                l1_select=False,
            )
            ], output_dir)

        if "proposed_l1" in args.methods:
            append_rows(rows, [
            {"model": model_name, "comparison": comparison, **r}
            for r in repeated_cv_logistic(
                wide,
                feature_cols,
                positive,
                negative,
                args.repeats,
                args.cv_splits,
                args.seed,
                "proposed_l1_selected_layer_head",
                common_splits,
                l1_select=True,
                l1_c=args.l1_c,
            )
            ], output_dir)

        if "attenmia" in args.methods:
            try:
                append_rows(rows, [
                    {"model": model_name, "comparison": comparison, **r}
                    for r in repeated_cv_attenmia_mlp(
                        attenmia_dir_p,
                        comparison,
                        positive,
                        negative,
                        args.repeats,
                        args.cv_splits,
                        args.seed,
                        common_splits,
                    )
                ], output_dir)
            except FileNotFoundError as e:
                log(f"[WARN] Skipping AttenMIA {model_name} {comparison}: {e}")

        if "lora_leak" in args.methods:
            score_col = best_lora_score_col(lora_dir_p, comparison, args.lora_score_mode)
            append_rows(rows, [
                {"model": model_name, "comparison": comparison, **r}
                for r in repeated_fold_auc_from_fixed_scores(
                    lora_scores,
                    score_col,
                    positive,
                    negative,
                    args.repeats,
                    args.cv_splits,
                    args.seed,
                    common_splits,
                )
            ], output_dir)

        if "initial_loss" in args.methods:
            append_rows(rows, [
                {"model": model_name, "comparison": comparison, **r}
                for r in repeated_fold_auc_from_sample_column(
                    sample_level,
                    "before_loss",
                    "initial_loss",
                    positive,
                    negative,
                    args.repeats,
                    args.cv_splits,
                    args.seed,
                    common_splits,
                )
            ], output_dir)

        if "loss_delta" in args.methods:
            append_rows(rows, [
                {"model": model_name, "comparison": comparison, **r}
                for r in repeated_fold_auc_from_sample_column(
                    sample_level,
                    "delta_loss_before_minus_after",
                    "loss_delta",
                    positive,
                    negative,
                    args.repeats,
                    args.cv_splits,
                    args.seed,
                    common_splits,
                )
            ], output_dir)
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/method_comparison_10runs")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--l1-c", type=float, default=0.02)
    parser.add_argument("--proposed-test-method", default="proposed_l1_selected_layer_head")
    parser.add_argument(
        "--comparisons",
        nargs="+",
        default=["ft_vs_pt", "ft_vs_unseen"],
        choices=list(COMPARISONS.keys()),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["proposed_l1", "attenmia", "lora_leak", "initial_loss", "loss_delta"],
        choices=["proposed_l1", "proposed_all", "attenmia", "lora_leak", "initial_loss", "loss_delta"],
        help="Methods to run. proposed_all is skipped by default because it is slow and not the main test target.",
    )
    parser.add_argument(
        "--lora-score-mode",
        default="target_mink_family_best",
        help=(
            "LoRA-Leak score selection. Default uses only Min-K family scores "
            "(target_mink_* and target_mink++_*) and picks the best strict AUC."
            " Use target_mink_best for pure Min-K, target_minkpp_best for Min-K++, "
            "best_strict for any LoRA-Leak score, or score:<column> for a fixed score."
        ),
    )

    parser.add_argument("--run-1b", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-410m", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--proposed-1b-root", default=DEFAULT_1B_PROPOSED_ROOT)
    parser.add_argument("--attenmia-1b-dir", default=DEFAULT_1B_ATTENMIA_DIR)
    parser.add_argument("--lora-leak-1b-dir", default=DEFAULT_1B_LORA_LEAK_DIR)

    parser.add_argument("--proposed-410m-root", default=DEFAULT_410M_PROPOSED_ROOT)
    parser.add_argument("--attenmia-410m-dir", default=DEFAULT_410M_ATTENMIA_DIR)
    parser.add_argument("--lora-leak-410m-dir", default=DEFAULT_410M_LORA_LEAK_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = resolve_path(args.output_dir) if Path(args.output_dir).exists() else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    if args.run_1b:
        all_rows += run_for_model(
            "pythia1b",
            args.proposed_1b_root,
            args.attenmia_1b_dir,
            args.lora_leak_1b_dir,
            args,
            out_dir,
        )
    if args.run_410m:
        all_rows += run_for_model(
            "pythia410m",
            args.proposed_410m_root,
            args.attenmia_410m_dir,
            args.lora_leak_410m_dir,
            args,
            out_dir,
        )

    auc_df = pd.DataFrame(all_rows)
    auc_df.to_csv(out_dir / "auc_10runs.csv", index=False)
    summary = summarize_auc(auc_df)
    summary.to_csv(out_dir / "summary_auc.csv", index=False)
    tests = paired_tests(auc_df, args.proposed_test_method)
    tests.to_csv(out_dir / "paired_auc_tests.csv", index=False)

    with open(out_dir / "comparison_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(f"Saved: {out_dir}")
    print("\nSummary:")
    print(summary.round(6).to_string(index=False))
    print("\nPaired tests:")
    print(tests.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
