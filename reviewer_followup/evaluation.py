"""Leakage-safe feature construction and repeated-CV helpers."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


META_COLUMNS = {"condition", "sample_id", "group", "source", "label", "feature_family", "scope"}


def elastic_net_penalty_kwargs() -> Dict[str, str]:
    """Avoid sklearn 1.8's deprecated explicit penalty while supporting older releases."""
    return {} if LogisticRegression().get_params().get("penalty") == "deprecated" else {"penalty": "elasticnet"}


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target: float = 0.10) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = np.where(fpr <= target)[0]
    return float(np.max(tpr[valid])) if len(valid) else 0.0


def wide_attention(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_id", "group", "layer", "head"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Attention CSV missing {sorted(required - set(raw.columns))}")
    metrics = [
        name
        for name in (
            "l1_mean",
            "l2_rms",
            "js_div",
            "entropy_delta",
            "max_shift",
            "top1_shift_mean",
            "top5_shift_mean",
            "top10_shift_mean",
        )
        if name in raw.columns
    ]
    index = ["sample_id", "group"]
    if "condition" in raw.columns:
        index.insert(0, "condition")
    result = raw[index].drop_duplicates().copy()
    for metric in metrics:
        pivot = raw.pivot_table(index=index, columns=["layer", "head"], values=metric, aggfunc="mean")
        pivot.columns = [f"attn_{metric}_L{int(layer):02d}_H{int(head):02d}" for layer, head in pivot.columns]
        result = result.merge(pivot.reset_index(), on=index, how="inner")
    return result


def wide_updates(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_id", "group", "feature_family", "scope"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Update CSV missing {sorted(required - set(raw.columns))}")
    numeric = []
    for column in raw.columns:
        if column in META_COLUMNS:
            continue
        converted = pd.to_numeric(raw[column], errors="coerce")
        if converted.notna().any():
            numeric.append(column)
    index = ["sample_id", "group"]
    if "condition" in raw.columns:
        index.insert(0, "condition")
    result = raw[index].drop_duplicates().copy()
    for metric in numeric:
        work = raw[index + ["feature_family", "scope", metric]].copy()
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
        pivot = work.pivot_table(index=index, columns=["feature_family", "scope"], values=metric, aggfunc="mean")
        pivot.columns = [f"upd_{family}_{scope}_{metric}" for family, scope in pivot.columns]
        result = result.merge(pivot.reset_index(), on=index, how="inner")
    return result


def _train_only_transform(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.asarray(x_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    finite = np.isfinite(x_train)
    keep = np.flatnonzero((finite.sum(axis=0) >= 2) & (np.nanstd(x_train, axis=0) > 1e-12))
    if len(keep) == 0:
        keep = np.arange(x_train.shape[1])
    x_train = x_train[:, keep]
    x_test = x_test[:, keep]
    median = np.nanmedian(x_train, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    for matrix in (x_train, x_test):
        missing = np.where(~np.isfinite(matrix))
        if len(missing[0]):
            matrix[missing] = np.take(median, missing[1])
    scaler = StandardScaler()
    return np.clip(scaler.fit_transform(x_train), -20, 20), np.clip(scaler.transform(x_test), -20, 20), keep


def evaluate_feature_sets(
    frame: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    comparisons: Mapping[str, Tuple[str, str]],
    *,
    repeats: int = 10,
    cv_splits: int = 5,
    seed: int = 42,
    use_elastic_net: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: List[Dict[str, object]] = []
    predictions: List[Dict[str, object]] = []
    selections: List[Dict[str, object]] = []
    for comparison, (positive, negative) in comparisons.items():
        subset = frame[frame["group"].isin([positive, negative])].drop_duplicates("sample_id").reset_index(drop=True)
        y = (subset["group"].to_numpy() == positive).astype(int)
        if len(np.unique(y)) != 2 or np.min(np.bincount(y)) < cv_splits:
            raise ValueError(f"Insufficient rows for {comparison}: {np.bincount(y).tolist()}")
        for repeat in range(1, repeats + 1):
            folds = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed + repeat - 1)
            for method, columns in feature_sets.items():
                columns = [column for column in columns if column in subset.columns]
                if not columns:
                    continue
                x = subset[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
                oof = np.full(len(subset), np.nan)
                selected_counts: List[int] = []
                for fold, (train_index, test_index) in enumerate(folds.split(x, y), start=1):
                    x_train, x_test, keep = _train_only_transform(x[train_index], x[test_index])
                    selected = np.arange(x_train.shape[1])
                    if use_elastic_net:
                        selector = LogisticRegression(
                            **elastic_net_penalty_kwargs(),
                            solver="saga",
                            l1_ratio=0.7,
                            C=0.1,
                            tol=5e-4,
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=seed + repeat * 100 + fold,
                        )
                        selector.fit(x_train, y[train_index])
                        selected = np.flatnonzero(np.abs(selector.coef_[0]) > 1e-10)
                        if len(selected) == 0:
                            selected = np.arange(x_train.shape[1])
                    classifier = LogisticRegression(
                        solver="lbfgs",
                        C=1.0,
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=seed + repeat * 100 + fold,
                    )
                    classifier.fit(x_train[:, selected], y[train_index])
                    oof[test_index] = classifier.predict_proba(x_test[:, selected])[:, 1]
                    selected_counts.append(int(len(selected)))
                    original_selected = keep[selected]
                    for feature_index in original_selected:
                        selections.append(
                            {
                                "comparison": comparison,
                                "method": method,
                                "repeat": repeat,
                                "fold": fold,
                                "feature": columns[int(feature_index)],
                            }
                        )
                summaries.append(
                    {
                        "comparison": comparison,
                        "method": method,
                        "repeat": repeat,
                        "auc": float(roc_auc_score(y, oof)),
                        "auprc": float(average_precision_score(y, oof)),
                        "tpr_at_10_fpr": tpr_at_fpr(y, oof),
                        "n_positive": int(y.sum()),
                        "n_negative": int((1 - y).sum()),
                        "n_features": int(len(columns)),
                        "n_selected_mean": float(np.mean(selected_counts)),
                    }
                )
                for index, score in enumerate(oof):
                    predictions.append(
                        {
                            "comparison": comparison,
                            "method": method,
                            "repeat": repeat,
                            "sample_id": subset.loc[index, "sample_id"],
                            "group": subset.loc[index, "group"],
                            "y_true": int(y[index]),
                            "score": float(score),
                        }
                    )
    return pd.DataFrame(summaries), pd.DataFrame(predictions), pd.DataFrame(selections)


def aggregate_repeats(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["comparison", "method"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
            tpr_at_10_fpr_mean=("tpr_at_10_fpr", "mean"),
            tpr_at_10_fpr_std=("tpr_at_10_fpr", "std"),
            n_repeats=("repeat", "nunique"),
            n_features=("n_features", "max"),
            n_selected_mean=("n_selected_mean", "mean"),
        )
        .sort_values(["comparison", "auc_mean"], ascending=[True, False])
    )


def bootstrap_auc_delta(
    predictions: pd.DataFrame,
    *,
    augmented_method: str,
    baseline_method: str,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for comparison, part in predictions.groupby("comparison"):
        averaged = (
            part.groupby(["method", "sample_id", "y_true"], as_index=False)["score"].mean()
            .pivot(index=["sample_id", "y_true"], columns="method", values="score")
            .dropna(subset=[augmented_method, baseline_method])
            .reset_index()
        )
        y = averaged["y_true"].to_numpy(int)
        positive = np.where(y == 1)[0]
        negative = np.where(y == 0)[0]
        deltas = []
        for _ in range(n_bootstrap):
            index = np.concatenate(
                [rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)]
            )
            deltas.append(
                roc_auc_score(y[index], averaged.loc[index, augmented_method])
                - roc_auc_score(y[index], averaged.loc[index, baseline_method])
            )
        point = roc_auc_score(y, averaged[augmented_method]) - roc_auc_score(y, averaged[baseline_method])
        rows.append(
            {
                "comparison": comparison,
                "augmented_method": augmented_method,
                "baseline_method": baseline_method,
                "delta_auc": float(point),
                "ci_low": float(np.quantile(deltas, 0.025)),
                "ci_high": float(np.quantile(deltas, 0.975)),
                "n_bootstrap": n_bootstrap,
            }
        )
    return pd.DataFrame(rows)
