# -*- coding: utf-8 -*-
"""Strict fixed-20 comparison over Pythia-1B, Pythia-410M, and GPT-Neo-2.7B.

This script recomputes downstream 10 repeated 5-fold evaluations using the
already extracted original full-matrix attention-update features and existing
baseline score files.  It does not rerun model inference or attention
extraction.

Defaults:
  - Proposed+EN, AttenMIA, LoRA-Leak target_mink++_0.5, initial loss, loss delta
  - FT is the positive class
  - AUC is not flipped after observing the result
  - fixed_attention_20 only

Run on the server:
  python run_strict_fixed20_3model_comparison_10runs.py
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
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

MODEL_CONFIGS = {
    "pythia1b": {
        "label": "Pythia-1B",
        "proposed_root": "/workplace/FT/BlackNLP_2/results/experiment4_mimir_hardsplit_stopping_condition",
        "attenmia_dir": "/workplace/FT/BlackNLP_2/results/attenmia_official_mimir_hardsplit",
        "lora_dir": "/workplace/FT/BlackNLP_2/results/lora_leak_official_mimir_hardsplit",
    },
    "pythia410m": {
        "label": "Pythia-410M",
        "proposed_root": "/workplace/FT/BlackNLP_2/results/mimir_wikipedia_hardsplit_fixed20_pythia410m_rerun",
        "attenmia_dir": "/workplace/FT/BlackNLP_2/results/attenmia_official_mimir_hardsplit_pythia410m",
        "lora_dir": "/workplace/FT/BlackNLP_2/results/lora_leak_official_mimir_hardsplit_pythia410m",
    },
    "gptneo27b": {
        "label": "GPT-Neo-2.7B",
        "proposed_root": "/workplace/FT/BlackNLP_2/results/mimir_wikipedia_hardsplit_fixed20_gptneo27b",
        "attenmia_dir": "/workplace/FT/BlackNLP_2/results/attenmia_official_mimir_hardsplit_gptneo27b",
        "lora_dir": "/workplace/FT/BlackNLP_2/results/lora_leak_official_mimir_hardsplit_gptneo27b",
    },
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def resolve_path(path_like: str, required_files: Sequence[str] = ()) -> Path:
    """Resolve server absolute paths or local downloaded mirrors."""
    path = Path(path_like).expanduser()
    root = script_dir()
    candidates = [
        path,
        Path.cwd() / path,
        root / path.name,
        root / "results" / path.name,
        root / path,
        root.parent / path.name,
        root.parent / "results" / path.name,
    ]

    path_str = str(path_like)
    for prefix in [
        "/workplace/FT/BlackNLP_2/results/",
        "/workplace/FT/BlackNLP_2/",
        "/workplace/FT/BlackNLP/results/",
        "/workplace/FT/results/",
        "results/",
    ]:
        if path_str.startswith(prefix):
            suffix = path_str[len(prefix) :]
            candidates.extend([root / suffix, root / "results" / suffix])

    for nested in root.glob(f"**/{path.name}"):
        candidates.append(nested)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        if not candidate.exists():
            continue
        if required_files and not all((candidate / item).exists() for item in required_files):
            continue
        return candidate

    required = ", ".join(required_files) if required_files else "path exists"
    tried = "\n".join(f"  - {candidate}" for candidate in candidates[:40])
    raise FileNotFoundError(
        f"Could not resolve path: {path_like}\n"
        f"Required: {required}\n"
        f"Tried:\n{tried}"
    )


def ensure_uid(df: pd.DataFrame) -> pd.DataFrame:
    """Create a stable group-local UID for aligning methods.

    Several files use group-local sample_id while others use global sample_id.
    We therefore remap sample_id to 0..N-1 inside each group and use
    group::local_id as the cross-method key.
    """
    out = df.copy()
    if "sample_id" not in out.columns:
        out["sample_id"] = out.groupby("group").cumcount()
    out["sample_id"] = out["sample_id"].astype(int)
    out["_local_sample_id"] = 0
    for group, idx in out.groupby("group").groups.items():
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
        cv = StratifiedKFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=seed + repeat - 1,
        )
        for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(y)), y), start=1):
            train_uids = set(sub.iloc[train_idx]["uid"].tolist())
            test_uids = set(sub.iloc[test_idx]["uid"].tolist())
            splits.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "train_uids": train_uids,
                    "test_uids": test_uids,
                }
            )
            rows.extend(
                {"repeat": repeat, "fold": fold, "uid": uid, "split": "train"}
                for uid in sorted(train_uids)
            )
            rows.extend(
                {"repeat": repeat, "fold": fold, "uid": uid, "split": "test"}
                for uid in sorted(test_uids)
            )
    return splits, pd.DataFrame(rows)


def read_group_files(root: Path, filename: str) -> pd.DataFrame:
    all_candidates = list(root.glob(f"**/fixed_attention_20_all/{filename}"))
    if all_candidates:
        path = max(all_candidates, key=lambda item: item.stat().st_size)
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        return frame

    parts = []
    for group_key in ["ft", "pt", "unseen"]:
        candidates = list(root.glob(f"**/fixed_attention_20_{group_key}/{filename}"))
        if not candidates:
            raise FileNotFoundError(
                f"{filename} for fixed_attention_20_{group_key} not found under {root}"
            )
        path = max(candidates, key=lambda item: item.stat().st_size)
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def make_proposed_features(raw: pd.DataFrame, exclude_mse: bool) -> pd.DataFrame:
    metrics_found = [metric for metric in ATTENTION_METRICS if metric in raw.columns]
    if exclude_mse:
        metrics_found = [metric for metric in metrics_found if metric != "mse"]
    if not metrics_found:
        raise ValueError("No attention metric columns found in proposed raw CSVs.")
    long = raw[["sample_id", "group", "layer", "head"] + metrics_found].melt(
        id_vars=["sample_id", "group", "layer", "head"],
        value_vars=metrics_found,
        var_name="metric",
        value_name="value",
    )
    long["feature"] = (
        "attn_l"
        + long["layer"].astype(int).astype(str)
        + "_h"
        + long["head"].astype(int).astype(str)
        + "_"
        + long["metric"]
    )
    wide = long.pivot_table(
        index=["sample_id", "group"],
        columns="feature",
        values="value",
        aggfunc="mean",
    ).reset_index()
    wide.columns.name = None
    return wide


def clean_matrix(df: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    frame = df[list(feature_cols)].replace([np.inf, -np.inf], np.nan)
    valid_cols = [
        col
        for col in feature_cols
        if frame[col].notna().sum() >= 4 and frame[col].nunique(dropna=True) > 1
    ]
    frame = frame[valid_cols].fillna(frame[valid_cols].median(numeric_only=True))
    return frame.to_numpy(dtype=float), valid_cols


def run_proposed_en(
    features: pd.DataFrame,
    positive_group: str,
    negative_group: str,
    common_splits: List[Dict],
    args: argparse.Namespace,
) -> List[Dict]:
    feature_cols = [column for column in features.columns if column.startswith("attn_l")]
    df = ensure_uid(features)
    df = df[df["group"].isin([positive_group, negative_group])].drop_duplicates("uid")
    df = df.reset_index(drop=True)
    x, valid_cols = clean_matrix(df, feature_cols)
    y = (df["group"].to_numpy() == positive_group).astype(int)
    uid_to_idx = {uid: i for i, uid in enumerate(df["uid"].tolist())}
    rows = []

    for repeat in range(1, args.repeats + 1):
        log(f"Proposed+EN repeat {repeat}/{args.repeats}")
        oof = np.full(len(y), np.nan)
        selected_counts = []
        for split in [item for item in common_splits if item["repeat"] == repeat]:
            train_idx = np.array([uid_to_idx[uid] for uid in split["train_uids"] if uid in uid_to_idx])
            test_idx = np.array([uid_to_idx[uid] for uid in split["test_uids"] if uid in uid_to_idx])

            scaler = StandardScaler()
            x_train = scaler.fit_transform(x[train_idx])
            x_test = scaler.transform(x[test_idx])

            selector = LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=args.elasticnet_l1_ratio,
                C=args.selection_c,
                max_iter=args.elasticnet_max_iter,
                class_weight="balanced",
                random_state=args.seed + repeat * 100 + split["fold"],
            )
            selector.fit(x_train, y[train_idx])
            selected = np.flatnonzero(np.abs(selector.coef_[0]) > 1e-10)
            if len(selected) == 0:
                selected = np.arange(x_train.shape[1])
            selected_counts.append(len(selected))

            classifier = LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                C=args.classifier_c,
                max_iter=2000,
                class_weight="balanced",
                random_state=args.seed + repeat * 100 + split["fold"],
            )
            classifier.fit(x_train[:, selected], y[train_idx])
            oof[test_idx] = classifier.predict_proba(x_test[:, selected])[:, 1]

        row = compute_metrics(y, oof)
        row.update(
            {
                "method": "proposed_en",
                "repeat": repeat,
                "n_pos": int(y.sum()),
                "n_neg": int((1 - y).sum()),
                "n_features": len(valid_cols),
                "n_selected_mean": float(np.mean(selected_counts)),
            }
        )
        rows.append(row)
    return rows


def load_attenmia_features(root: Path, comparison: str) -> pd.DataFrame:
    candidates = [
        root / comparison / "attenmia_official_base_features.csv",
        root / root.name / comparison / "attenmia_official_base_features.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(candidate)
    matches = list(root.glob(f"**/{comparison}/attenmia_official_base_features.csv"))
    if matches:
        return pd.read_csv(matches[0])
    raise FileNotFoundError(f"AttenMIA features not found: {root} / {comparison}")


def run_attenmia_mlp(
    attenmia_root: Path,
    comparison: str,
    positive_group: str,
    negative_group: str,
    common_splits: List[Dict],
    args: argparse.Namespace,
) -> List[Dict]:
    df = ensure_uid(load_attenmia_features(attenmia_root, comparison))
    feature_cols = [c for c in df.columns if c.startswith(("trans_", "base_", "pert_"))]
    if not feature_cols:
        raise ValueError(f"No AttenMIA feature columns for {comparison}")
    df = df[df["group"].isin([positive_group, negative_group])].drop_duplicates("uid")
    df = df.reset_index(drop=True)
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    y = (df["group"].to_numpy() == positive_group).astype(int)
    uid_to_idx = {uid: i for i, uid in enumerate(df["uid"].tolist())}
    rows = []

    for repeat in range(1, args.repeats + 1):
        log(f"AttenMIA repeat {repeat}/{args.repeats}")
        oof = np.full(len(y), np.nan)
        for split in [item for item in common_splits if item["repeat"] == repeat]:
            train_idx = np.array([uid_to_idx[uid] for uid in split["train_uids"] if uid in uid_to_idx])
            test_idx = np.array([uid_to_idx[uid] for uid in split["test_uids"] if uid in uid_to_idx])
            classifier = Pipeline(
                [
                    ("scale", StandardScaler()),
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
                    ),
                ]
            )
            classifier.fit(x[train_idx], y[train_idx])
            oof[test_idx] = classifier.predict_proba(x[test_idx])[:, 1]

        row = compute_metrics(y, oof)
        row.update(
            {
                "method": "attenmia_mlp",
                "repeat": repeat,
                "n_pos": int(y.sum()),
                "n_neg": int((1 - y).sum()),
                "n_features": len(feature_cols),
                "n_selected_mean": math.nan,
            }
        )
        rows.append(row)
    return rows


def load_lora_scores(root: Path) -> pd.DataFrame:
    path = root / "lora_leak_scores.csv"
    if path.exists():
        return pd.read_csv(path)
    matches = list(root.glob("**/lora_leak_scores.csv"))
    if matches:
        return pd.read_csv(matches[0])
    raise FileNotFoundError(f"lora_leak_scores.csv not found under {root}")


def choose_lora_score(root: Path, comparison: str, score_col: str) -> str:
    if score_col != "auto":
        return score_col
    path = root / "lora_leak_pairwise_results.csv"
    if not path.exists():
        matches = list(root.glob("**/lora_leak_pairwise_results.csv"))
        if not matches:
            raise FileNotFoundError(f"lora_leak_pairwise_results.csv not found under {root}")
        path = matches[0]
    frame = pd.read_csv(path)
    subset = frame[
        (frame["comparison"] == comparison)
        & (frame["score_col"].astype(str) == "target_mink++_0.5")
    ]
    if subset.empty:
        raise ValueError(f"target_mink++_0.5 not found in {path} for {comparison}")
    return "target_mink++_0.5"


def run_fixed_score_method(
    df: pd.DataFrame,
    score_col: str,
    method: str,
    positive_group: str,
    negative_group: str,
    common_splits: List[Dict],
    args: argparse.Namespace,
) -> List[Dict]:
    df = ensure_uid(df)
    df = df[df["group"].isin([positive_group, negative_group])].dropna(subset=[score_col])
    df = df.drop_duplicates("uid").reset_index(drop=True)
    y = (df["group"].to_numpy() == positive_group).astype(int)
    score = df[score_col].to_numpy(float)
    uid_to_idx = {uid: i for i, uid in enumerate(df["uid"].tolist())}
    rows = []

    for repeat in range(1, args.repeats + 1):
        log(f"{method} repeat {repeat}/{args.repeats}")
        fold_auc = []
        fold_auprc = []
        fold_tpr = []
        for split in [item for item in common_splits if item["repeat"] == repeat]:
            test_idx = np.array([uid_to_idx[uid] for uid in split["test_uids"] if uid in uid_to_idx])
            fold_auc.append(float(roc_auc_score(y[test_idx], score[test_idx])))
            fold_auprc.append(float(average_precision_score(y[test_idx], score[test_idx])))
            fold_tpr.append(tpr_at_fpr(y[test_idx], score[test_idx]))
        rows.append(
            {
                "method": method,
                "repeat": repeat,
                "auc": float(np.mean(fold_auc)),
                "auprc": float(np.mean(fold_auprc)),
                "tpr_at_10_fpr": float(np.mean(fold_tpr)),
                "n_pos": int(y.sum()),
                "n_neg": int((1 - y).sum()),
                "n_features": 1,
                "n_selected_mean": math.nan,
            }
        )
    return rows


def signrank_exact_p(diff: Iterable[float]) -> float:
    diff = np.asarray(list(diff), dtype=float)
    diff = diff[np.isfinite(diff)]
    diff = diff[np.abs(diff) > 1e-12]
    n = len(diff)
    if n == 0:
        return 1.0
    ranks = pd.Series(np.abs(diff)).rank(method="average").to_numpy(dtype=float)
    total = float(ranks.sum())
    observed = min(float(ranks[diff > 0].sum()), float(ranks[diff < 0].sum()))
    if n > 20:
        mean = total / 2.0
        var = n * (n + 1) * (2 * n + 1) / 24.0
        z = (observed - mean) / math.sqrt(var)
        return float(math.erfc(abs(z) / math.sqrt(2.0)))
    values = []
    for mask in range(1 << n):
        signed_sum = 0.0
        for index, rank in enumerate(ranks):
            if (mask >> index) & 1:
                signed_sum += rank
        values.append(min(signed_sum, total - signed_sum))
    values = np.asarray(values, dtype=float)
    return float((values <= observed + 1e-12).mean())


def paired_tests(auc_df: pd.DataFrame, proposed_method: str = "proposed_en") -> pd.DataFrame:
    rows = []
    for (model, comparison), sub in auc_df.groupby(["model", "comparison"]):
        proposed = sub[sub["method"] == proposed_method][["repeat", "auc"]]
        proposed = proposed.rename(columns={"auc": "proposed_auc"})
        for method in sorted(set(sub["method"]) - {proposed_method}):
            baseline = sub[sub["method"] == method][["repeat", "auc"]]
            baseline = baseline.rename(columns={"auc": "baseline_auc"})
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
                    "wilcoxon_p": signrank_exact_p(diff),
                    "proposed_outperforms": bool(diff.mean() > 0),
                    "baseline_outperforms": bool(diff.mean() < 0),
                }
            )
    return pd.DataFrame(rows)


def summarize(auc_df: pd.DataFrame) -> pd.DataFrame:
    return (
        auc_df.groupby(["model", "comparison", "method"], as_index=False)
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


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    if value < 0.001:
        return "<.001"
    return f"{value:.3f}".lstrip("0")


def method_to_table_key(method: str) -> str:
    if method.startswith("lora_leak:"):
        return "lora_leak"
    return method


def latex_table(summary_df: pd.DataFrame, tests_df: pd.DataFrame, path: Path) -> None:
    method_order = [
        "proposed_en",
        "attenmia_mlp",
        "lora_leak",
        "initial_loss",
        "loss_delta",
    ]
    headers = {
        "proposed_en": "Proposed+EN",
        "attenmia_mlp": "AttenMIA",
        "lora_leak": "LoRA-Leak",
        "initial_loss": "\\shortstack{Initial\\\\loss}",
        "loss_delta": "\\shortstack{Loss\\\\decrease}",
    }
    rows = [
        "\\begin{table*}[t]",
        "\\caption{Mean AUC and TPR@10\\%FPR over 10 repeated runs. Each cell reports AUC / TPR@10\\%FPR. Parentheses show uncorrected two-sided Wilcoxon signed-rank test $p$-values against Proposed+EN based on AUC. $\\dagger$ and $\\ddagger$ indicate nominally significant improvement by Proposed+EN and by the baseline, respectively.}",
        "\\label{tab:baseline_comparison}",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{1.8pt}",
        "\\renewcommand{\\arraystretch}{1.15}",
        "\\begin{tabular}{@{}llccccc@{}}",
        "\\toprule",
        "\\shortstack{Model} & Comparison & "
        + " & ".join(headers[method] for method in method_order)
        + " \\\\",
        "\\midrule",
    ]

    for model in ["pythia1b", "pythia410m", "gptneo27b"]:
        model_label = MODEL_CONFIGS[model]["label"]
        for comparison in ["ft_vs_pt", "ft_vs_unseen"]:
            sub = summary_df[(summary_df["model"] == model) & (summary_df["comparison"] == comparison)]
            if sub.empty:
                continue
            values = {}
            for key in method_order:
                if key == "lora_leak":
                    row = sub[sub["method"].str.startswith("lora_leak:")]
                else:
                    row = sub[sub["method"] == key]
                if row.empty:
                    values[key] = "--"
                    continue
                item = row.iloc[0]
                cell = f"{item['auc_mean']:.3f} / {item['tpr_at_10_fpr_mean']:.3f}"
                if key != "proposed_en":
                    baseline_method = str(item["method"])
                    test = tests_df[
                        (tests_df["model"] == model)
                        & (tests_df["comparison"] == comparison)
                        & (tests_df["baseline_method"] == baseline_method)
                    ]
                    if not test.empty:
                        p_value = float(test.iloc[0]["wilcoxon_p"])
                        marker = ""
                        if p_value < 0.05 and bool(test.iloc[0]["proposed_outperforms"]):
                            marker = "$\\dagger$"
                        elif p_value < 0.05 and bool(test.iloc[0]["baseline_outperforms"]):
                            marker = "$\\ddagger$"
                        cell = (
                            f"{item['auc_mean']:.3f} ({format_p(p_value)}{marker})"
                            f" / {item['tpr_at_10_fpr_mean']:.3f}"
                        )
                values[key] = cell
            comp_label = "FT--PT" if comparison == "ft_vs_pt" else "FT--Unseen"
            rows.append(
                f"{model_label} & {comp_label} & "
                + " & ".join(values[method] for method in method_order)
                + " \\\\"
            )
            if comparison == "ft_vs_pt":
                rows.append("")

    rows.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    path.write_text("\n".join(rows), encoding="utf-8")


def run_one_model(
    model: str,
    config: Dict[str, str],
    args: argparse.Namespace,
    output_dir: Path,
) -> List[Dict]:
    log(f"Start {model}")
    proposed_root = resolve_path(config["proposed_root"])
    attenmia_root = resolve_path(config["attenmia_dir"])
    lora_root = resolve_path(config["lora_dir"], ["lora_leak_scores.csv"])

    raw = read_group_files(proposed_root, "raw_experiment4_attention_shift.csv")
    sample_level = read_group_files(proposed_root, "sample_level_experiment4.csv")
    proposed_features = make_proposed_features(raw, exclude_mse=args.exclude_mse)
    lora_scores = load_lora_scores(lora_root)
    all_rows = []

    for comparison in args.comparisons:
        positive_group, negative_group = COMPARISONS[comparison]
        log(f"{model} {comparison}: make common folds")
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

        if "proposed" in args.methods:
            all_rows.extend(
                {"model": model, "comparison": comparison, **row}
                for row in run_proposed_en(
                    proposed_features,
                    positive_group,
                    negative_group,
                    common_splits,
                    args,
                )
            )

        if "attenmia" in args.methods:
            all_rows.extend(
                {"model": model, "comparison": comparison, **row}
                for row in run_attenmia_mlp(
                    attenmia_root,
                    comparison,
                    positive_group,
                    negative_group,
                    common_splits,
                    args,
                )
            )

        if "lora_leak" in args.methods:
            score_col = choose_lora_score(lora_root, comparison, args.lora_score_col)
            all_rows.extend(
                {"model": model, "comparison": comparison, **row}
                for row in run_fixed_score_method(
                    lora_scores,
                    score_col,
                    f"lora_leak:{score_col}",
                    positive_group,
                    negative_group,
                    common_splits,
                    args,
                )
            )

        if "initial_loss" in args.methods:
            all_rows.extend(
                {"model": model, "comparison": comparison, **row}
                for row in run_fixed_score_method(
                    sample_level,
                    "before_loss",
                    "initial_loss",
                    positive_group,
                    negative_group,
                    common_splits,
                    args,
                )
            )

        if "loss_delta" in args.methods:
            all_rows.extend(
                {"model": model, "comparison": comparison, **row}
                for row in run_fixed_score_method(
                    sample_level,
                    "delta_loss_before_minus_after",
                    "loss_delta",
                    positive_group,
                    negative_group,
                    common_splits,
                    args,
                )
            )

        pd.DataFrame(all_rows).to_csv(output_dir / "auc_10runs.partial.csv", index=False)
    return all_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="/workplace/FT/BlackNLP_2/results/strict_fixed20_3model_method_comparison_10runs",
    )
    parser.add_argument("--models", nargs="+", default=["pythia1b", "pythia410m", "gptneo27b"])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-c", type=float, default=0.1)
    parser.add_argument("--classifier-c", type=float, default=1.0)
    parser.add_argument("--elasticnet-l1-ratio", type=float, default=0.7)
    parser.add_argument("--elasticnet-max-iter", type=int, default=5000)
    parser.add_argument("--attenmia-max-iter", type=int, default=500)
    parser.add_argument("--exclude-mse", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["proposed", "attenmia", "lora_leak", "initial_loss", "loss_delta"],
        choices=["proposed", "attenmia", "lora_leak", "initial_loss", "loss_delta"],
    )
    parser.add_argument(
        "--comparisons",
        nargs="+",
        default=["ft_vs_pt", "ft_vs_unseen"],
        choices=list(COMPARISONS.keys()),
    )
    parser.add_argument(
        "--lora-score-col",
        default="target_mink++_0.5",
        help="LoRA-Leak score column. Use 'auto' to require pairwise confirmation.",
    )
    for model, config in MODEL_CONFIGS.items():
        parser.add_argument(f"--{model}-proposed-root", default=config["proposed_root"])
        parser.add_argument(f"--{model}-attenmia-dir", default=config["attenmia_dir"])
        parser.add_argument(f"--{model}-lora-dir", default=config["lora_dir"])
    return parser.parse_args()


def args_to_config(args: argparse.Namespace) -> Dict:
    config = vars(args).copy()
    config["model_defaults"] = MODEL_CONFIGS
    config["attention_definition"] = "original_full_matrix"
    config["fixed_step"] = 20
    config["ft_positive_class"] = True
    config["auc_posthoc_flip"] = False
    return config


def main() -> None:
    args = parse_args()
    invalid = sorted(set(args.models) - set(MODEL_CONFIGS))
    if invalid:
        raise ValueError(f"Unknown models: {invalid}. Valid: {sorted(MODEL_CONFIGS)}")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in args.models:
        config = dict(MODEL_CONFIGS[model])
        config["proposed_root"] = getattr(args, f"{model}_proposed_root")
        config["attenmia_dir"] = getattr(args, f"{model}_attenmia_dir")
        config["lora_dir"] = getattr(args, f"{model}_lora_dir")
        rows.extend(run_one_model(model, config, args, output_dir))

    auc_df = pd.DataFrame(rows)
    auc_df.to_csv(output_dir / "auc_10runs.csv", index=False)

    summary_df = summarize(auc_df)
    summary_df.to_csv(output_dir / "summary_auc.csv", index=False)

    tests_df = paired_tests(auc_df, "proposed_en")
    tests_df.to_csv(output_dir / "paired_auc_tests.csv", index=False)

    latex_table(summary_df, tests_df, output_dir / "paper_baseline_comparison_table.tex")

    with open(output_dir / "comparison_config.json", "w", encoding="utf-8") as handle:
        json.dump(args_to_config(args), handle, ensure_ascii=False, indent=2)

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as handle:
        handle.write(
            "Strict fixed-20 full-matrix method comparison over 10 repeated runs\n"
            f"output_dir={output_dir}\n"
            f"models={','.join(args.models)}\n"
            f"repeats={args.repeats}, cv_splits={args.cv_splits}, seed={args.seed}\n"
            f"exclude_mse={args.exclude_mse}\n"
            "FT is always the positive class. AUC is not flipped after observing results.\n\n"
        )
        handle.write(summary_df.to_string(index=False))
        handle.write("\n\nPaired AUC tests vs Proposed+EN\n")
        handle.write(tests_df.to_string(index=False))
        handle.write("\n")

    print("\nSummary:")
    print(summary_df.round(6).to_string(index=False))
    print("\nPaired AUC tests:")
    print(tests_df.round(6).to_string(index=False))
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
