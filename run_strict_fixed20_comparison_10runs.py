# -*- coding: utf-8 -*-
"""Strict fixed-20 multi-model comparison (paper-aligned evaluation).

Recomputes 10 × 5-fold evaluations from already-extracted attention features.
Does not re-run model inference.

Supported models (``--models``):
  pythia1b | pythia410m | gptneo27b

Paper protocol (acl_latex.tex Evaluation Method + Appendix classifier details):
  - FT is always the positive class
  - 5-fold CV repeated 10 times with seeds 42--51
  - No test-set post-hoc score flipping
  - Proposed+EN: train-fold StandardScaler → Elastic Net (saga, l1_ratio=0.7,
    C=0.1) → L2 logistic regression on selected features
  - Proposed (all): train-fold StandardScaler → L2 logistic regression
  - Scalar baselines (loss / LoRA-Leak scores): 1D logistic regression with
    direction learned only from the training fold
  - Metrics: mean AUC and TPR@10%FPR over repeats
  - Wilcoxon signed-rank tests of Proposed+EN vs baselines (paired over repeats)

Performance/reproducibility:
  - train-only standardized folds are prepared once and shared by attention
    classifiers;
  - Elastic-Net solver defaults are max_iter=1000, tol=5e-4, configurable for
    sensitivity checks;
  - repeats use up to four threads by default, while fold construction and
    sample ordering remain deterministic.

Usage:
  python run_strict_fixed20_comparison_10runs.py \\
    --models pythia1b \\
    --pythia1b-proposed-root attention_features_mimir_hardsplit \\
    --output-dir results/strict_fixed20_eval

  python run_strict_fixed20_comparison_10runs.py \\
    --models pythia410m gptneo27b \\
    --pythia410m-proposed-root attention_features_pythia410m \\
    --gptneo27b-proposed-root attention_features_gptneo27b
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scipy.stats import wilcoxon
except ImportError:  # pragma: no cover
    wilcoxon = None


GROUP_FT = "mimir_wikipedia_nonmember_ft"
GROUP_PT = "mimir_wikipedia_member_pt"
GROUP_UNSEEN = "mimir_wikipedia_nonmember_unseen"

COMPARISONS = {
    "ft_vs_pt": (GROUP_FT, GROUP_PT),
    "ft_vs_unseen": (GROUP_FT, GROUP_UNSEEN),
    "pt_vs_unseen": (GROUP_PT, GROUP_UNSEEN),
}

COMPARISON_LABELS = {
    "ft_vs_pt": "FT--PT",
    "ft_vs_unseen": "FT--Unseen",
    "pt_vs_unseen": "PT--Unseen",
}

# Paper: eight attention-update features (no separate MSE feature)
ATTENTION_METRICS = [
    "l1_mean",
    "l2_rms",
    "js_div",
    "entropy_delta",
    "max_shift",
    "top1_shift_mean",
    "top5_shift_mean",
    "top10_shift_mean",
]

try:
    from model_registry import list_eval_keys, strict_eval_model_configs

    MODEL_CONFIGS = strict_eval_model_configs()
except ImportError:  # pragma: no cover
    list_eval_keys = lambda: ["pythia1b", "pythia410m", "gptneo27b"]  # noqa: E731
    MODEL_CONFIGS = {
        "pythia1b": {
            "label": "Pythia-1B",
            "proposed_root": "attention_features_mimir_hardsplit",
            "lora_root": "",
            "attenmia_root": "",
        },
        "pythia410m": {
            "label": "Pythia-410M",
            "proposed_root": "attention_features_pythia410m",
            "lora_root": "",
            "attenmia_root": "",
        },
        "gptneo27b": {
            "label": "GPT-Neo-2.7B",
            "proposed_root": "attention_features_gptneo27b",
            "lora_root": "",
            "attenmia_root": "",
        },
    }


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def elastic_net_selector_kwargs() -> Dict[str, Any]:
    """Return version-compatible Elastic-Net arguments.

    scikit-learn 1.8 deprecates explicitly passing ``penalty`` while older
    supported versions still require ``penalty='elasticnet'`` when using
    ``l1_ratio``.  Detect the default value instead of emitting warnings or
    silently falling back to L2 on older installations.
    """
    kwargs: Dict[str, Any] = {}
    default_penalty = LogisticRegression().get_params().get("penalty")
    if default_penalty != "deprecated":
        kwargs["penalty"] = "elasticnet"
    return kwargs


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def resolve_path(path_like: str, required_files: Sequence[str] = ()) -> Path:
    path = Path(path_like).expanduser()
    root = script_dir()
    candidates = [
        path,
        Path.cwd() / path,
        root / path.name,
        root / path,
        root / "results" / path.name,
        root.parent / path.name,
    ]
    path_str = str(path_like)
    for prefix in [
        "/workplace/FT/BlackNLP_2/results/",
        "/workplace/FT/BlackNLP_2/",
        "/workplace/FT/BlackNLP/results/",
        "results/",
    ]:
        if path_str.startswith(prefix):
            suffix = path_str[len(prefix) :]
            candidates.extend([root / suffix, root / "results" / suffix])

    def _ok(candidate: Path) -> bool:
        if not candidate.exists():
            return False
        if required_files and not all((candidate / item).exists() for item in required_files):
            return False
        return True

    seen = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        if _ok(candidate):
            return candidate

    # Expensive recursive search only as a last resort
    name = Path(path_like).name
    for nested in root.glob(f"**/{name}"):
        key = str(nested.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        if _ok(nested):
            return nested
    raise FileNotFoundError(f"Could not resolve path: {path_like}")


def ensure_uid(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sample_id" not in out.columns:
        out["sample_id"] = out.groupby("group").cumcount()
    out["sample_id"] = out["sample_id"].astype(int)
    out["_local_sample_id"] = 0
    for _, idx in out.groupby("group").groups.items():
        ids = sorted(out.loc[idx, "sample_id"].drop_duplicates().tolist())
        mapper = {sample_id: i for i, sample_id in enumerate(ids)}
        out.loc[idx, "_local_sample_id"] = out.loc[idx, "sample_id"].map(mapper)
    out["_local_sample_id"] = out["_local_sample_id"].astype(int)
    out["uid"] = out["group"].astype(str) + "::" + out["_local_sample_id"].astype(str)
    return out


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, max_fpr: float = 0.10) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = fpr <= max_fpr
    return float(np.max(tpr[valid])) if np.any(valid) else 0.0


def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "tpr_at_10_fpr": tpr_at_fpr(y_true, scores),
    }


def make_common_splits(
    base_df: pd.DataFrame,
    positive_group: str,
    negative_group: str,
    repeats: int,
    cv_splits: int,
    seed: int,
) -> Tuple[List[Dict], pd.DataFrame]:
    """Paper: 5-fold CV × 10 repeats, seeds 42--51 (seed + repeat - 1)."""
    base = ensure_uid(base_df)
    sub = (
        base[base["group"].isin([positive_group, negative_group])][["uid", "group"]]
        .drop_duplicates("uid")
        .sort_values(["group", "uid"])
        .reset_index(drop=True)
    )
    y = (sub["group"].to_numpy() == positive_group).astype(int)
    splits = []
    rows = []
    for repeat in range(1, repeats + 1):
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed + repeat - 1)
        for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(y)), y), start=1):
            train_uids = set(sub.iloc[train_idx]["uid"].tolist())
            test_uids = set(sub.iloc[test_idx]["uid"].tolist())
            splits.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "train_uids": train_uids,
                    "test_uids": test_uids,
                    # Keep deterministic row order for every method.  The old
                    # set-only representation made the order depend on hash
                    # randomization and also forced each method to rebuild it.
                    "train_idx": train_idx.astype(int),
                    "test_idx": test_idx.astype(int),
                }
            )
            for split_name, idxs in [("train", train_idx), ("test", test_idx)]:
                for row in sub.iloc[idxs].itertuples(index=False):
                    rows.append(
                        {
                            "repeat": repeat,
                            "fold": fold,
                            "split": split_name,
                            "uid": row.uid,
                            "group": row.group,
                        }
                    )
    return splits, pd.DataFrame(rows)


def read_group_files(
    root: Path,
    filename: str,
    *,
    condition_prefix: str = "fixed_attention_20",
    groups: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load per-group CSVs under ``{condition_prefix}_{ft,pt,unseen}/``.

    ``condition_prefix`` defaults to the paper main setting (fixed-20).
    Exp.3 uses ``fixed_attention_50``, ``fixed_attention_100``, or ``dynamic_attention``.
    ``groups`` defaults to all three; pass a subset when only some comparisons are run
    (e.g. FT--PT ablations that do not yet have an Unseen extraction).
    """
    all_candidates = list(root.glob(f"**/{condition_prefix}_all/{filename}"))
    if all_candidates:
        path = max(all_candidates, key=lambda item: item.stat().st_size)
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        return frame

    group_keys = list(groups) if groups is not None else ["ft", "pt", "unseen"]
    if not group_keys:
        raise ValueError("groups must be non-empty")
    parts = []
    for group_key in group_keys:
        candidates = list(root.glob(f"**/{condition_prefix}_{group_key}/{filename}"))
        if not candidates:
            raise FileNotFoundError(
                f"{filename} for {condition_prefix}_{group_key} not found under {root}"
            )
        path = max(candidates, key=lambda item: item.stat().st_size)
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def make_proposed_features(raw: pd.DataFrame) -> pd.DataFrame:
    metrics_found = [m for m in ATTENTION_METRICS if m in raw.columns]
    if not metrics_found:
        raise ValueError(f"No paper attention metrics found. Need {ATTENTION_METRICS}")
    # Faster path: build feature names without melt when possible
    cols = ["sample_id", "group", "layer", "head"] + metrics_found
    sub = raw[cols].copy()
    sub["layer"] = sub["layer"].astype(int)
    sub["head"] = sub["head"].astype(int)
    pieces = []
    for metric in metrics_found:
        piv = sub.pivot_table(
            index=["sample_id", "group"],
            columns=["layer", "head"],
            values=metric,
            aggfunc="mean",
        )
        piv.columns = [f"attn_l{int(layer)}_h{int(head)}_{metric}" for layer, head in piv.columns]
        pieces.append(piv)
    wide = pieces[0]
    for piece in pieces[1:]:
        wide = wide.join(piece, how="outer")
    return wide.reset_index()


def _feature_cache_stem(condition_prefix: str) -> str:
    """Stable cache filename stem from condition prefix."""
    safe = condition_prefix.replace("/", "_").replace(" ", "_")
    if safe == "fixed_attention_20":
        return "proposed_features_fixed20_cache"
    return f"proposed_features_{safe}_cache"


def load_or_build_proposed_features(
    proposed_root: Path,
    raw: pd.DataFrame,
    *,
    refresh: bool = False,
    condition_prefix: str = "fixed_attention_20",
) -> pd.DataFrame:
    """Cache wide proposed features next to the feature root (parquet preferred)."""
    stem = _feature_cache_stem(condition_prefix)
    cache_parquet = proposed_root / f"{stem}.parquet"
    cache_csv = proposed_root / f"{stem}.csv"
    if not refresh:
        if cache_parquet.exists():
            try:
                log(f"Loading proposed feature cache: {cache_parquet}")
                return pd.read_parquet(cache_parquet)
            except Exception as exc:
                log(f"Parquet cache failed ({exc}); trying CSV / rebuild")
        if cache_csv.exists():
            log(f"Loading proposed feature cache: {cache_csv}")
            return pd.read_csv(cache_csv)

    log("Building proposed features from raw attention CSVs (one-time cost)...")
    wide = make_proposed_features(raw)
    try:
        wide.to_parquet(cache_parquet, index=False)
        log(f"Wrote cache {cache_parquet}")
    except Exception:
        wide.to_csv(cache_csv, index=False)
        log(f"Wrote cache {cache_csv}")
    return wide


def fit_transform_train_only(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Standardize using training fold only (paper: no test leakage)."""
    x_train = np.asarray(x_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    x_train = np.where(np.isfinite(x_train), x_train, np.nan)
    x_test = np.where(np.isfinite(x_test), x_test, np.nan)
    # Drop near-constant columns using train only
    finite = np.isfinite(x_train)
    keep = np.flatnonzero(
        (finite.sum(axis=0) >= 2)
        & (np.nanstd(x_train, axis=0) > 1e-12)
    )
    if keep.size == 0:
        keep = np.arange(x_train.shape[1], dtype=int)
    x_train = x_train[:, keep]
    x_test = x_test[:, keep]
    # Median impute from train
    med = np.nanmedian(x_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    inds = np.where(~np.isfinite(x_train))
    if inds[0].size:
        x_train[inds] = np.take(med, inds[1])
    inds = np.where(~np.isfinite(x_test))
    if inds[0].size:
        x_test[inds] = np.take(med, inds[1])
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    # Numerical stability for high-dimensional LR (avoid overflow warnings)
    x_train_s = np.clip(x_train_s, -20.0, 20.0)
    x_test_s = np.clip(x_test_s, -20.0, 20.0)
    return x_train_s, x_test_s


@dataclass
class PreparedFeatureEvaluation:
    """Fold-indexed feature arrays shared by all attention classifiers.

    Standardization and train-only imputation are identical for Proposed (all)
    and Proposed+EN.  Preparing them once per comparison avoids doing the same
    six transformations twice and keeps the paper's train-only boundary intact.
    """

    feature_cols: List[str]
    df: pd.DataFrame
    x_all: np.ndarray
    y: np.ndarray
    uid_to_idx: Dict[str, int]
    folds: List[Dict[str, Any]]


def prepare_feature_evaluation(
    features: pd.DataFrame,
    positive_group: str,
    negative_group: str,
    common_splits: List[Dict],
) -> PreparedFeatureEvaluation:
    feature_cols = [c for c in features.columns if c.startswith("attn_l")]
    if not feature_cols:
        raise ValueError("No proposed attention feature columns found")
    df = ensure_uid(features)
    df = df[df["group"].isin([positive_group, negative_group])].drop_duplicates("uid").reset_index(drop=True)
    x_all = df[feature_cols].to_numpy(dtype=float)
    y = (df["group"].to_numpy() == positive_group).astype(int)
    uid_to_idx = {uid: i for i, uid in enumerate(df["uid"].tolist())}

    folds = []
    for split in common_splits:
        train_idx = np.asarray(
            split.get("train_idx", [uid_to_idx[u] for u in sorted(split["train_uids"])]),
            dtype=int,
        )
        test_idx = np.asarray(
            split.get("test_idx", [uid_to_idx[u] for u in sorted(split["test_uids"])]),
            dtype=int,
        )
        x_train, x_test = fit_transform_train_only(x_all[train_idx], x_all[test_idx])
        folds.append({**split, "train_idx": train_idx, "test_idx": test_idx, "x_train": x_train, "x_test": x_test})
    return PreparedFeatureEvaluation(feature_cols, df, x_all, y, uid_to_idx, folds)


def run_proposed_en(
    features: pd.DataFrame,
    positive_group: str,
    negative_group: str,
    common_splits: List[Dict],
    args: argparse.Namespace,
    prepared: Optional[PreparedFeatureEvaluation] = None,
    prediction_sink: Optional[List[Dict]] = None,
) -> List[Dict]:
    prepared = prepared or prepare_feature_evaluation(features, positive_group, negative_group, common_splits)
    feature_cols, df, x_all, y, uid_to_idx = (
        prepared.feature_cols,
        prepared.df,
        prepared.x_all,
        prepared.y,
        prepared.uid_to_idx,
    )
    folds = prepared.folds

    def _one_repeat(repeat: int) -> Dict:
        fold_metrics = []
        selected_counts = []
        for split in [s for s in folds if s["repeat"] == repeat]:
            train_idx = split["train_idx"]
            test_idx = split["test_idx"]
            x_train, x_test = split["x_train"], split["x_test"]

            # Paper Appendix: Elastic-Net-regularized LR for selection
            selector = LogisticRegression(
                solver="saga",
                l1_ratio=args.elasticnet_l1_ratio,
                C=args.selection_c,
                tol=args.elasticnet_tol,
                max_iter=args.elasticnet_max_iter,
                class_weight="balanced",
                random_state=args.seed + repeat * 100 + split["fold"],
                **elastic_net_selector_kwargs(),
            )
            selector.fit(x_train, y[train_idx])
            selected = np.flatnonzero(np.abs(selector.coef_[0]) > 1e-10)
            if len(selected) == 0:
                selected = np.arange(x_train.shape[1])
            selected_counts.append(len(selected))

            # Paper: separate L2 LR on selected features
            clf = LogisticRegression(
                solver="lbfgs",
                C=args.classifier_c,
                max_iter=2000,
                class_weight="balanced",
                random_state=args.seed + repeat * 100 + split["fold"],
            )
            clf.fit(x_train[:, selected], y[train_idx])
            scores = clf.predict_proba(x_test[:, selected])[:, 1]
            fold_metrics.append(compute_metrics(y[test_idx], scores))
            if prediction_sink is not None:
                prediction_sink.extend(
                    {
                        "method": "proposed_en", "repeat": repeat, "fold": split["fold"],
                        "uid": df.iloc[int(i)]["uid"], "group": df.iloc[int(i)]["group"],
                        "y_true": int(y[int(i)]), "score": float(score),
                    }
                    for i, score in zip(test_idx, scores)
                )

        return {
            "method": "proposed_en",
            "repeat": repeat,
            "auc": float(np.mean([m["auc"] for m in fold_metrics])),
            "auprc": float(np.mean([m["auprc"] for m in fold_metrics])),
            "tpr_at_10_fpr": float(np.mean([m["tpr_at_10_fpr"] for m in fold_metrics])),
            "n_pos": int(y.sum()),
            "n_neg": int((1 - y).sum()),
            "n_features": len(feature_cols),
            "n_selected_mean": float(np.mean(selected_counts)),
        }

    n_jobs = int(getattr(args, "n_jobs", 1) or 1)
    if n_jobs != 1:
        log(f"Proposed+EN parallel n_jobs={n_jobs}")
        from hardsplit.parallel import map_repeats

        return map_repeats(_one_repeat, args.repeats, n_jobs=n_jobs, prefer="threads")

    rows = []
    for repeat in range(1, args.repeats + 1):
        log(f"Proposed+EN repeat {repeat}/{args.repeats}")
        rows.append(_one_repeat(repeat))
    return rows


def run_proposed_all(
    features: pd.DataFrame,
    positive_group: str,
    negative_group: str,
    common_splits: List[Dict],
    args: argparse.Namespace,
    prepared: Optional[PreparedFeatureEvaluation] = None,
    prediction_sink: Optional[List[Dict]] = None,
) -> List[Dict]:
    prepared = prepared or prepare_feature_evaluation(features, positive_group, negative_group, common_splits)
    feature_cols, df, x_all, y, uid_to_idx = (
        prepared.feature_cols,
        prepared.df,
        prepared.x_all,
        prepared.y,
        prepared.uid_to_idx,
    )
    folds = prepared.folds
    rows = []

    for repeat in range(1, args.repeats + 1):
        log(f"Proposed all-features repeat {repeat}/{args.repeats}")
        fold_metrics = []
        for split in [s for s in folds if s["repeat"] == repeat]:
            train_idx = split["train_idx"]
            test_idx = split["test_idx"]
            x_train, x_test = split["x_train"], split["x_test"]
            clf = LogisticRegression(
                solver="lbfgs",
                C=args.classifier_c,
                max_iter=2000,
                class_weight="balanced",
                random_state=args.seed + repeat * 100 + split["fold"],
            )
            clf.fit(x_train, y[train_idx])
            scores = clf.predict_proba(x_test)[:, 1]
            fold_metrics.append(compute_metrics(y[test_idx], scores))
            if prediction_sink is not None:
                prediction_sink.extend(
                    {
                        "method": "proposed_all", "repeat": repeat, "fold": split["fold"],
                        "uid": df.iloc[int(i)]["uid"], "group": df.iloc[int(i)]["group"],
                        "y_true": int(y[int(i)]), "score": float(score),
                    }
                    for i, score in zip(test_idx, scores)
                )
        rows.append(
            {
                "method": "proposed_all",
                "repeat": repeat,
                "auc": float(np.mean([m["auc"] for m in fold_metrics])),
                "auprc": float(np.mean([m["auprc"] for m in fold_metrics])),
                "tpr_at_10_fpr": float(np.mean([m["tpr_at_10_fpr"] for m in fold_metrics])),
                "n_pos": int(y.sum()),
                "n_neg": int((1 - y).sum()),
                "n_features": len(feature_cols),
                "n_selected_mean": float(len(feature_cols)),
            }
        )
    return rows


def run_scalar_lr(
    df: pd.DataFrame,
    score_col: str,
    method: str,
    positive_group: str,
    negative_group: str,
    common_splits: List[Dict],
    args: argparse.Namespace,
    prediction_sink: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Paper: scalar baseline as 1D input to logistic regression; direction from train only."""
    df = ensure_uid(df)
    df = df[df["group"].isin([positive_group, negative_group])].dropna(subset=[score_col])
    df = df.drop_duplicates("uid").reset_index(drop=True)
    y = (df["group"].to_numpy() == positive_group).astype(int)
    raw = df[score_col].to_numpy(dtype=float)
    uid_to_idx = {uid: i for i, uid in enumerate(df["uid"].tolist())}
    rows = []

    for repeat in range(1, args.repeats + 1):
        log(f"{method} repeat {repeat}/{args.repeats}")
        fold_metrics = []
        for split in [s for s in common_splits if s["repeat"] == repeat]:
            train_idx = np.asarray(
                split.get("train_idx", [uid_to_idx[u] for u in sorted(split["train_uids"]) if u in uid_to_idx]),
                dtype=int,
            )
            test_idx = np.asarray(
                split.get("test_idx", [uid_to_idx[u] for u in sorted(split["test_uids"]) if u in uid_to_idx]),
                dtype=int,
            )
            x_train = raw[train_idx].reshape(-1, 1)
            x_test = raw[test_idx].reshape(-1, 1)
            # Learn direction only from train via LR coefficients
            clf = LogisticRegression(
                solver="lbfgs",
                C=1.0,
                max_iter=2000,
                class_weight="balanced",
                random_state=args.seed + repeat * 100 + split["fold"],
            )
            clf.fit(x_train, y[train_idx])
            scores = clf.predict_proba(x_test)[:, 1]
            fold_metrics.append(compute_metrics(y[test_idx], scores))
            if prediction_sink is not None:
                prediction_sink.extend(
                    {
                        "method": method, "repeat": repeat, "fold": split["fold"],
                        "uid": df.iloc[int(i)]["uid"], "group": df.iloc[int(i)]["group"],
                        "y_true": int(y[int(i)]), "score": float(score),
                    }
                    for i, score in zip(test_idx, scores)
                )
        rows.append(
            {
                "method": method,
                "repeat": repeat,
                "auc": float(np.mean([m["auc"] for m in fold_metrics])),
                "auprc": float(np.mean([m["auprc"] for m in fold_metrics])),
                "tpr_at_10_fpr": float(np.mean([m["tpr_at_10_fpr"] for m in fold_metrics])),
                "n_pos": int(y.sum()),
                "n_neg": int((1 - y).sum()),
                "n_features": 1,
                "n_selected_mean": math.nan,
            }
        )
    return rows


def load_loss_sample_scores(
    proposed_root: Path,
    *,
    condition_prefix: str = "fixed_attention_20",
    groups: Sequence[str] | None = None,
) -> pd.DataFrame:
    sample = read_group_files(
        proposed_root,
        "sample_level_experiment4.csv",
        condition_prefix=condition_prefix,
        groups=groups,
    )
    sample = ensure_uid(sample)
    sample["initial_loss"] = pd.to_numeric(sample["before_loss"], errors="coerce")
    sample["loss_decrease"] = pd.to_numeric(sample["delta_loss_before_minus_after"], errors="coerce")
    return sample


def load_lora_scores(root: Path) -> Optional[pd.DataFrame]:
    try:
        path = resolve_path(str(root))
    except FileNotFoundError:
        return None
    matches = list(path.glob("**/lora_leak_scores.csv"))
    if not matches and (path / "lora_leak_scores.csv").exists():
        matches = [path / "lora_leak_scores.csv"]
    if not matches:
        return None
    return pd.read_csv(matches[0])


def load_attenmia_features(root: Path, comparison: str) -> Optional[pd.DataFrame]:
    try:
        path = resolve_path(str(root))
    except FileNotFoundError:
        return None
    matches = list(path.glob(f"**/{comparison}/attenmia_official_base_features.csv"))
    if not matches:
        matches = list(path.glob(f"**/*{comparison}*/attenmia_official_base_features.csv"))
    if not matches:
        return None
    return pd.read_csv(matches[0])


def run_attenmia_mlp(
    features: pd.DataFrame,
    positive_group: str,
    negative_group: str,
    common_splits: List[Dict],
    args: argparse.Namespace,
    prediction_sink: Optional[List[Dict]] = None,
) -> List[Dict]:
    """AttenMIA features with the same common folds as the proposed method."""
    df = ensure_uid(features)
    feature_cols = [c for c in df.columns if c.startswith(("trans_", "base_", "pert_"))]
    if not feature_cols:
        raise ValueError("No AttenMIA feature columns found")
    df = df[df["group"].isin([positive_group, negative_group])].drop_duplicates("uid").reset_index(drop=True)
    x_all = df[feature_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    y = (df["group"].to_numpy() == positive_group).astype(int)
    uid_to_idx = {uid: i for i, uid in enumerate(df["uid"].tolist())}
    rows = []
    for repeat in range(1, args.repeats + 1):
        log(f"AttenMIA repeat {repeat}/{args.repeats}")
        fold_metrics = []
        for split in [s for s in common_splits if s["repeat"] == repeat]:
            train_idx = np.asarray(
                split.get("train_idx", [uid_to_idx[u] for u in sorted(split["train_uids"]) if u in uid_to_idx]),
                dtype=int,
            )
            test_idx = np.asarray(
                split.get("test_idx", [uid_to_idx[u] for u in sorted(split["test_uids"]) if u in uid_to_idx]),
                dtype=int,
            )
            x_train, x_test = fit_transform_train_only(x_all[train_idx], x_all[test_idx])
            clf = Pipeline(
                [
                    (
                        "mlp",
                        MLPClassifier(
                            hidden_layer_sizes=(128, 64),
                            activation="relu",
                            solver="adam",
                            alpha=1e-4,
                            learning_rate_init=1e-3,
                            max_iter=args.attenmia_max_iter,
                            early_stopping=True,
                            random_state=args.seed + repeat * 100 + split["fold"],
                        ),
                    )
                ]
            )
            # Already standardized in fit_transform_train_only
            clf.fit(x_train, y[train_idx])
            scores = clf.predict_proba(x_test)[:, 1]
            fold_metrics.append(compute_metrics(y[test_idx], scores))
            if prediction_sink is not None:
                prediction_sink.extend(
                    {
                        "method": "attenmia", "repeat": repeat, "fold": split["fold"],
                        "uid": df.iloc[int(i)]["uid"], "group": df.iloc[int(i)]["group"],
                        "y_true": int(y[int(i)]), "score": float(score),
                    }
                    for i, score in zip(test_idx, scores)
                )
        rows.append(
            {
                "method": "attenmia",
                "repeat": repeat,
                "auc": float(np.mean([m["auc"] for m in fold_metrics])),
                "auprc": float(np.mean([m["auprc"] for m in fold_metrics])),
                "tpr_at_10_fpr": float(np.mean([m["tpr_at_10_fpr"] for m in fold_metrics])),
                "n_pos": int(y.sum()),
                "n_neg": int((1 - y).sum()),
                "n_features": len(feature_cols),
                "n_selected_mean": math.nan,
            }
        )
    return rows


def wilcoxon_p(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    diff = diff[np.abs(diff) > 1e-12]
    if len(diff) == 0:
        return 1.0
    if wilcoxon is not None:
        try:
            return float(wilcoxon(diff, alternative="two-sided", zero_method="wilcox").pvalue)
        except ValueError:
            return 1.0
    # Fallback exact sign-rank for small n
    n = len(diff)
    ranks = pd.Series(np.abs(diff)).rank(method="average").to_numpy(float)
    total = float(ranks.sum())
    observed = min(float(ranks[diff > 0].sum()), float(ranks[diff < 0].sum()))
    if n > 20:
        mean = total / 2.0
        var = n * (n + 1) * (2 * n + 1) / 24.0
        z = (observed - mean) / math.sqrt(max(var, 1e-12))
        return float(math.erfc(abs(z) / math.sqrt(2.0)))
    values = []
    for mask in range(1 << n):
        s = 0.0
        for i, r in enumerate(ranks):
            if (mask >> i) & 1:
                s += r
        values.append(min(s, total - s))
    values = np.asarray(values, dtype=float)
    return float((values <= observed + 1e-12).mean())


def paired_tests(auc_df: pd.DataFrame, proposed_method: str = "proposed_en") -> pd.DataFrame:
    rows = []
    for (model, comparison), sub in auc_df.groupby(["model", "comparison"]):
        proposed = sub[sub["method"] == proposed_method][["repeat", "auc"]].rename(columns={"auc": "proposed_auc"})
        for method in sorted(set(sub["method"]) - {proposed_method}):
            baseline = sub[sub["method"] == method][["repeat", "auc"]].rename(columns={"auc": "baseline_auc"})
            merged = proposed.merge(baseline, on="repeat", how="inner").sort_values("repeat")
            if merged.empty:
                continue
            diff = merged["proposed_auc"].to_numpy() - merged["baseline_auc"].to_numpy()
            rows.append(
                {
                    "model": model,
                    "comparison": comparison,
                    "proposed_method": proposed_method,
                    "baseline_method": method,
                    "n_repeats": int(len(merged)),
                    "proposed_auc_mean": float(merged["proposed_auc"].mean()),
                    "baseline_auc_mean": float(merged["baseline_auc"].mean()),
                    "mean_auc_diff": float(diff.mean()),
                    "std_auc_diff": float(diff.std(ddof=1)) if len(diff) > 1 else math.nan,
                    "wilcoxon_p": wilcoxon_p(diff),
                    "proposed_outperforms": bool(diff.mean() > 0),
                }
            )
    return pd.DataFrame(rows)


def summarize(auc_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-repeat metrics. Tolerates missing optional columns."""
    work = auc_df.copy()
    if "n_selected_mean" not in work.columns:
        work["n_selected_mean"] = float("nan")
    for col, default in (("n_pos", math.nan), ("n_neg", math.nan), ("n_features", math.nan)):
        if col not in work.columns:
            work[col] = default
    return (
        work.groupby(["model", "comparison", "method"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
            tpr_at_10_fpr_mean=("tpr_at_10_fpr", "mean"),
            tpr_at_10_fpr_std=("tpr_at_10_fpr", "std"),
            n_repeats=("repeat", "count"),
            n_pos=("n_pos", "first"),
            n_neg=("n_neg", "first"),
            n_features=("n_features", "first"),
            n_selected_mean=("n_selected_mean", "mean"),
        )
        .sort_values(["model", "comparison", "method"])
    )


def latex_table(summary_df: pd.DataFrame, path: Path) -> None:
    method_order = [
        "proposed_en",
        "proposed_all",
        "attenmia",
        "lora_leak",
        "initial_loss",
        "loss_decrease",
    ]
    headers = {
        "proposed_en": "Proposed+EN",
        "proposed_all": "All att.",
        "attenmia": "AttenMIA",
        "lora_leak": "LoRA-Leak",
        "initial_loss": "Init. loss",
        "loss_decrease": "Loss dec.",
    }
    present = [m for m in method_order if m in set(summary_df["method"])]
    lines = [
        "\\begin{table}[t]",
        "\\caption{Fixed-20-step comparison (paper-aligned evaluation). Each cell: AUC / TPR@10\\%FPR mean over 10 runs.}",
        "\\label{tab:baseline_comparison_reproduced}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}l" + "c" * len(present) + "@{}}",
        "\\toprule",
        "Comparison & " + " & ".join(headers[m] for m in present) + " \\\\",
        "\\midrule",
    ]
    for comparison in ["ft_vs_pt", "ft_vs_unseen", "pt_vs_unseen"]:
        sub = summary_df[summary_df["comparison"] == comparison]
        if sub.empty:
            continue
        vals = []
        for m in present:
            row = sub[sub["method"] == m]
            if row.empty:
                vals.append("--")
            else:
                r = row.iloc[0]
                vals.append(f"{r['auc_mean']:.3f} / {r['tpr_at_10_fpr_mean']:.3f}")
        label = COMPARISON_LABELS.get(comparison, comparison)
        lines.append(label + " & " + " & ".join(vals) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _groups_for_comparisons(comparisons: Sequence[str]) -> List[str]:
    """Map comparison names to the minimum set of group keys that must be loaded."""
    needed: set[str] = set()
    key_to_short = {
        "mimir_wikipedia_nonmember_ft": "ft",
        "mimir_wikipedia_member_pt": "pt",
        "mimir_wikipedia_nonmember_unseen": "unseen",
    }
    for comparison in comparisons:
        if comparison not in COMPARISONS:
            raise KeyError(f"Unknown comparison: {comparison}")
        pos, neg = COMPARISONS[comparison]
        for label in (pos, neg):
            short = key_to_short.get(label)
            if short is None:
                # Fall back to last underscore token when labels evolve.
                short = label.rsplit("_", 1)[-1]
            needed.add(short)
    order = [g for g in ("ft", "pt", "unseen") if g in needed]
    return order


def run_one_model(
    model: str,
    config: Dict[str, str],
    args: argparse.Namespace,
    output_dir: Path,
    prediction_rows: Optional[List[Dict]] = None,
) -> List[Dict]:
    log(f"Start {model}")
    proposed_root = resolve_path(config["proposed_root"])
    condition_prefix = str(getattr(args, "condition_prefix", "fixed_attention_20") or "fixed_attention_20")
    group_keys = _groups_for_comparisons(args.comparisons)
    log(f"{model}: loading groups={group_keys} under prefix={condition_prefix}")
    raw = read_group_files(
        proposed_root,
        "raw_experiment4_attention_shift.csv",
        condition_prefix=condition_prefix,
        groups=group_keys,
    )
    proposed_features = load_or_build_proposed_features(
        proposed_root,
        raw,
        refresh=bool(getattr(args, "refresh_feature_cache", False)),
        condition_prefix=condition_prefix,
    )
    loss_scores = load_loss_sample_scores(
        proposed_root, condition_prefix=condition_prefix, groups=group_keys
    )

    lora_df = None
    if config.get("lora_root"):
        lora_df = load_lora_scores(config["lora_root"])
    attenmia_root = config.get("attenmia_root") or ""

    all_rows: List[Dict] = []
    for comparison in args.comparisons:
        comparison_predictions: List[Dict] = []
        positive_group, negative_group = COMPARISONS[comparison]
        log(f"{model} {comparison}: common folds")
        common_splits, split_df = make_common_splits(
            proposed_features,
            positive_group,
            negative_group,
            repeats=args.repeats,
            cv_splits=args.cv_splits,
            seed=args.seed,
        )
        split_df.insert(0, "comparison", comparison)
        split_df.insert(0, "model", model)
        split_df.to_csv(output_dir / f"common_folds_{model}_{comparison}.csv", index=False)

        prepared = None
        if any(method in args.methods for method in ("proposed_all", "proposed_en")):
            prepared = prepare_feature_evaluation(
                proposed_features, positive_group, negative_group, common_splits
            )

        if "proposed_all" in args.methods:
            all_rows.extend(
                {"model": model, "comparison": comparison, **row}
                for row in run_proposed_all(
                    proposed_features, positive_group, negative_group, common_splits, args, prepared,
                    comparison_predictions,
                )
            )
        if "proposed_en" in args.methods:
            all_rows.extend(
                {"model": model, "comparison": comparison, **row}
                for row in run_proposed_en(
                    proposed_features, positive_group, negative_group, common_splits, args, prepared,
                    comparison_predictions,
                )
            )
        if "initial_loss" in args.methods:
            all_rows.extend(
                {"model": model, "comparison": comparison, **row}
                for row in run_scalar_lr(
                    loss_scores, "initial_loss", "initial_loss", positive_group, negative_group, common_splits, args,
                    comparison_predictions,
                )
            )
        if "loss_decrease" in args.methods:
            all_rows.extend(
                {"model": model, "comparison": comparison, **row}
                for row in run_scalar_lr(
                    loss_scores, "loss_decrease", "loss_decrease", positive_group, negative_group, common_splits, args,
                    comparison_predictions,
                )
            )
        if "lora_leak" in args.methods and lora_df is not None:
            score_col = args.lora_score_col
            if score_col not in lora_df.columns:
                # Prefer paper-like Min-K++ if present
                for cand in ["target_mink++_0.2", "target_mink++_0.5", "target_loss", "loss_refpt"]:
                    if cand in lora_df.columns:
                        score_col = cand
                        break
            log(f"LoRA-Leak using score_col={score_col}")
            all_rows.extend(
                {"model": model, "comparison": comparison, **row}
                for row in run_scalar_lr(
                    lora_df, score_col, "lora_leak", positive_group, negative_group, common_splits, args,
                    comparison_predictions,
                )
            )
        if "attenmia" in args.methods and attenmia_root:
            att = load_attenmia_features(attenmia_root, comparison)
            if att is not None:
                all_rows.extend(
                    {"model": model, "comparison": comparison, **row}
                    for row in run_attenmia_mlp(
                        att, positive_group, negative_group, common_splits, args, comparison_predictions
                    )
                )
            else:
                log(f"AttenMIA features missing for {comparison}; skipped")

        pd.DataFrame(all_rows).to_csv(output_dir / "auc_10runs.partial.csv", index=False)
        if prediction_rows is not None:
            prediction_rows.extend(
                {"model": model, "comparison": comparison, **row} for row in comparison_predictions
            )
            pd.DataFrame(prediction_rows).to_csv(output_dir / "oof_predictions.partial.csv", index=False)
    return all_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-aligned fixed-20 multi-model evaluation")
    parser.add_argument(
        "--output-dir",
        default=str(script_dir() / "results" / "strict_fixed20_method_comparison_10runs"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["pythia1b"],
        choices=list(MODEL_CONFIGS),
        help="One or more of: pythia1b, pythia410m, gptneo27b",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Target-level stratified bootstrap draws for method and paired-delta intervals.",
    )
    parser.add_argument("--selection-c", type=float, default=0.1)
    parser.add_argument("--classifier-c", type=float, default=1.0)
    parser.add_argument("--elasticnet-l1-ratio", type=float, default=0.7)
    parser.add_argument(
        "--elasticnet-max-iter",
        type=int,
        default=1000,
        help="Maximum EN selection iterations; numerical convergence only, not a paper variable.",
    )
    parser.add_argument(
        "--elasticnet-tol",
        type=float,
        default=5e-4,
        help="EN selection tolerance; train-fold selection remains unchanged in protocol.",
    )
    parser.add_argument(
        "--refresh-feature-cache",
        action="store_true",
        help="Rebuild proposed feature cache from raw CSVs.",
    )
    parser.add_argument(
        "--condition-prefix",
        default="fixed_attention_20",
        help="Directory prefix under proposed-root: fixed_attention_{N} or dynamic_attention "
        "(Exp.3 step ablation). Default matches the main fixed-20 setting.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=int(os.environ.get("EVAL_N_JOBS", str(min(4, os.cpu_count() or 1)))),
        help="Parallel workers for Proposed+EN repeats (threads; default up to 4). "
        "Set 1 for strictly serial execution.",
    )
    parser.add_argument("--attenmia-max-iter", type=int, default=300)
    parser.add_argument("--lora-score-col", default="auto")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["proposed_all", "proposed_en", "initial_loss", "loss_decrease", "lora_leak", "attenmia"],
        choices=["proposed_all", "proposed_en", "initial_loss", "loss_decrease", "lora_leak", "attenmia"],
    )
    parser.add_argument(
        "--comparisons",
        nargs="+",
        default=["ft_vs_pt", "ft_vs_unseen"],
        choices=list(COMPARISONS.keys()),
    )
    for model, config in MODEL_CONFIGS.items():
        parser.add_argument(f"--{model}-proposed-root", default=config["proposed_root"])
        parser.add_argument(f"--{model}-lora-root", default=config.get("lora_root", ""))
        parser.add_argument(f"--{model}-attenmia-root", default=config.get("attenmia_root", ""))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    prediction_rows: List[Dict] = []
    for model in args.models:
        config = dict(MODEL_CONFIGS[model])
        config["proposed_root"] = getattr(args, f"{model}_proposed_root")
        config["lora_root"] = getattr(args, f"{model}_lora_root")
        config["attenmia_root"] = getattr(args, f"{model}_attenmia_root")
        rows.extend(run_one_model(model, config, args, output_dir, prediction_rows))

    auc_df = pd.DataFrame(rows)
    auc_df.to_csv(output_dir / "auc_10runs.csv", index=False)
    predictions_df = pd.DataFrame(prediction_rows)
    predictions_df.to_csv(output_dir / "oof_predictions.csv", index=False)
    from reviewer_followup.analyze_oof_uncertainty import infer_oof_uncertainty
    method_intervals, paired_intervals = infer_oof_uncertainty(
        predictions_df,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        reference_method="proposed_en",
    )
    method_intervals.to_csv(output_dir / "method_target_bootstrap_auc.csv", index=False)
    paired_intervals.to_csv(output_dir / "paired_target_bootstrap_auc_deltas.csv", index=False)
    summary_df = summarize(auc_df)
    summary_df.to_csv(output_dir / "summary_auc.csv", index=False)
    tests_df = paired_tests(auc_df, "proposed_en")
    tests_df.to_csv(output_dir / "paired_auc_tests.csv", index=False)
    latex_table(summary_df, output_dir / "paper_table_reproduced.tex")

    config_out = vars(args).copy()
    condition_prefix = str(getattr(args, "condition_prefix", "fixed_attention_20") or "fixed_attention_20")
    # Infer numeric step when possible (fixed_attention_20 → 20); dynamic → null
    fixed_step: Any = None
    if condition_prefix.startswith("fixed_attention_"):
        tail = condition_prefix.split("fixed_attention_", 1)[-1]
        if tail.isdigit():
            fixed_step = int(tail)
    config_out.update(
        {
            "attention_definition": "paper_masked_rows_over_Q",
            "condition_prefix": condition_prefix,
            "fixed_step": fixed_step if fixed_step is not None else condition_prefix,
            "positive_group_by_comparison": {
                name: COMPARISONS[name][0] for name in args.comparisons
            },
            "ft_positive_class": all(name.startswith("ft_") for name in args.comparisons),
            "auc_posthoc_flip": False,
            "scalar_baseline": "1d_logistic_regression_train_direction",
            "metric_aggregation": "mean_over_folds_then_mean_over_repeats",
        }
    )
    with open(output_dir / "comparison_config.json", "w", encoding="utf-8") as f:
        json.dump(config_out, f, ensure_ascii=False, indent=2)

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("Paper-aligned fixed-20 evaluation\n")
        f.write(f"output_dir={output_dir}\n")
        f.write(f"repeats={args.repeats}, cv_splits={args.cv_splits}, seed={args.seed}\n")
        f.write(
            "Positive groups by comparison: "
            + json.dumps({name: COMPARISONS[name][0] for name in args.comparisons})
            + "; no test-set score flipping; train-only preprocessing.\n\n"
        )
        f.write(summary_df.to_string(index=False))
        f.write("\n\nWilcoxon tests vs Proposed+EN\n")
        f.write(tests_df.to_string(index=False))
        f.write("\n")

    print("\nSummary:")
    print(summary_df.round(6).to_string(index=False))
    print("\nPaired Wilcoxon tests:")
    print(tests_df.round(6).to_string(index=False))
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
