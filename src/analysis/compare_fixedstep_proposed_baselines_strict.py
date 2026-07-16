# -*- coding: utf-8 -*-
"""Strict repeated AUC comparison for fixed-step Proposed+EN and baselines.

This script is intended for the table comparing:

  Proposed+EN / AttenMIA / LoRA-Leak / Initial loss / Loss diff.

The important point is that every method is evaluated on the same samples and
the same repeated StratifiedKFold splits.  Wilcoxon signed-rank tests are then
run on paired AUC values from the same repeat index.

Proposed+EN follows the settings in analyze_mimir_fixed_steps_repeated_auc.py:
  - no layer/head averaging
  - Elastic Net feature selection inside each training fold only
  - L2 Logistic Regression classifier after feature selection
  - FT is the positive class
  - AUC is not flipped after observing the result

Default comparisons:
  - FT vs PT
  - FT vs Unseen

Default models:
  - Pythia-1B
  - Pythia-410M
"""

import argparse
import json
import math
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

DEFAULT_1B_PROPOSED_ROOT = (
    "results/"
    "experiment4_mimir_hardsplit_stopping_condition"
)
DEFAULT_1B_ATTENMIA_DIR = (
    "results/attenmia_official_mimir_hardsplit"
)
DEFAULT_1B_LORA_DIR = (
    "results/lora_leak_official_mimir_hardsplit"
)

DEFAULT_410M_PROPOSED_ROOT = (
    "results/"
    "mimir_wikipedia_hardsplit_fixed20_pythia410m_rerun"
)
DEFAULT_410M_ATTENMIA_DIR = (
    "results/"
    "attenmia_official_mimir_hardsplit_pythia410m"
)
DEFAULT_410M_LORA_DIR = (
    "results/"
    "lora_leak_official_mimir_hardsplit_pythia410m"
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def local_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_path(path_like: str, required_files: Sequence[str] = ()) -> Path:
    """Resolve original server paths or downloaded local mirrors."""
    root = local_root()
    path = Path(path_like)
    candidates = [path, root / path.name]

    path_str = str(path_like)
    prefixes = [
        "results/",
        "results/",
        "",
    ]
    for prefix in prefixes:
        if path_str.startswith(prefix):
            candidates.append(root / path_str.replace(prefix, ""))

    for nested in root.glob(f"**/{path.name}"):
        candidates.append(nested)

    seen = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        if not candidate.exists():
            continue
        if required_files and not all((candidate / f).exists() for f in required_files):
            continue
        return candidate

    if "experiment4_mimir_hardsplit_stopping_condition" in path_str:
        direct = [root / f"fixed_attention_20_{g}" for g in ["ft", "pt", "unseen"]]
        if all((d / "raw_experiment4_attention_shift.csv").exists() for d in direct):
            return root

    req = ", ".join(required_files) if required_files else "path"
    raise FileNotFoundError(f"Could not resolve {path_like}; required={req}")


def ensure_uid(df: pd.DataFrame) -> pd.DataFrame:
    """Add uid = group::local_sample_id.

    Some result files use group-local sample_id, while some use global ids.
    To align methods, sample_id is remapped to 0..N-1 within each group.
    """
    out = df.copy()
    if "sample_id" not in out.columns:
        out["sample_id"] = out.groupby("group").cumcount()
    out["sample_id"] = out["sample_id"].astype(int)
    out["_local_sample_id"] = 0
    for group, idx in out.groupby("group").groups.items():
        ids = sorted(out.loc[idx, "sample_id"].drop_duplicates().tolist())
        mapper = {sid: i for i, sid in enumerate(ids)}
        out.loc[idx, "_local_sample_id"] = out.loc[idx, "sample_id"].map(mapper)
    out["_local_sample_id"] = out["_local_sample_id"].astype(int)
    out["uid"] = out["group"].astype(str) + "::" + out["_local_sample_id"].astype(str)
    return out


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, max_fpr: float = 0.10) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = fpr <= max_fpr
    return float(np.max(tpr[valid])) if np.any(valid) else 0.0


def metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
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
    split_rows = []
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
            split_rows += [
                {"repeat": repeat, "fold": fold, "uid": uid, "split": "train"}
                for uid in sorted(train_uids)
            ]
            split_rows += [
                {"repeat": repeat, "fold": fold, "uid": uid, "split": "test"}
                for uid in sorted(test_uids)
            ]
    return splits, pd.DataFrame(split_rows)


def read_group_files(root: Path, filename: str) -> pd.DataFrame:
    parts = []
    for group_key in ["ft", "pt", "unseen"]:
        candidates = list(root.glob(f"**/fixed_attention_20_{group_key}/{filename}"))
        if not candidates:
            raise FileNotFoundError(f"{filename} for fixed_attention_20_{group_key} not found under {root}")
        path = max(candidates, key=lambda p: p.stat().st_size)
        parts.append(pd.read_csv(path))
    return pd.concat(parts, ignore_index=True)


def make_proposed_features(raw: pd.DataFrame) -> pd.DataFrame:
    metrics_found = [m for m in ATTENTION_METRICS if m in raw.columns]
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
    valid = [
        col
        for col in feature_cols
        if frame[col].notna().sum() >= 4 and frame[col].nunique(dropna=True) > 1
    ]
    frame = frame[valid].fillna(frame[valid].median(numeric_only=True))
    return frame.to_numpy(dtype=float), valid


def run_proposed_en(
    features: pd.DataFrame,
    positive_group: str,
    negative_group: str,
    common_splits: List[Dict],
    args: argparse.Namespace,
) -> List[Dict]:
    feature_cols = [c for c in features.columns if c.startswith("attn_l")]
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
        for split in [s for s in common_splits if s["repeat"] == repeat]:
            train_idx = np.array([uid_to_idx[u] for u in split["train_uids"] if u in uid_to_idx])
            test_idx = np.array([uid_to_idx[u] for u in split["test_uids"] if u in uid_to_idx])
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

            clf = LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                C=args.classifier_c,
                max_iter=2000,
                class_weight="balanced",
                random_state=args.seed + repeat * 100 + split["fold"],
            )
            clf.fit(x_train[:, selected], y[train_idx])
            oof[test_idx] = clf.predict_proba(x_test[:, selected])[:, 1]

        row = metrics(y, oof)
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
        for split in [s for s in common_splits if s["repeat"] == repeat]:
            train_idx = np.array([uid_to_idx[u] for u in split["train_uids"] if u in uid_to_idx])
            test_idx = np.array([uid_to_idx[u] for u in split["test_uids"] if u in uid_to_idx])
            clf = Pipeline(
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
            clf.fit(x[train_idx], y[train_idx])
            oof[test_idx] = clf.predict_proba(x[test_idx])[:, 1]
        row = metrics(y, oof)
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


def choose_lora_score(root: Path, comparison: str, mode: str) -> str:
    if mode.startswith("score:"):
        return mode.split(":", 1)[1]
    path = root / "lora_leak_pairwise_results.csv"
    if not path.exists():
        matches = list(root.glob("**/lora_leak_pairwise_results.csv"))
        if not matches:
            raise FileNotFoundError(f"lora_leak_pairwise_results.csv not found under {root}")
        path = matches[0]
    df = pd.read_csv(path)
    df = df[df["comparison"] == comparison].copy()
    if mode == "target_mink_family_best":
        df = df[df["score_col"].str.startswith(("target_mink_", "target_mink++_"))]
    elif mode == "target_minkpp_0.5":
        return "target_mink++_0.5"
    elif mode == "best_strict":
        pass
    else:
        raise ValueError(f"Unknown LoRA score mode: {mode}")
    if df.empty:
        raise ValueError(f"No LoRA score candidates for {comparison}, mode={mode}")
    return str(df.sort_values("auroc", ascending=False).iloc[0]["score_col"])


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
        for split in [s for s in common_splits if s["repeat"] == repeat]:
            test_idx = np.array([uid_to_idx[u] for u in split["test_uids"] if u in uid_to_idx])
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


def summarize(auc_df: pd.DataFrame) -> pd.DataFrame:
    return (
        auc_df.groupby(["model", "comparison", "method"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            auprc_mean=("auprc", "mean"),
            tpr_at_10_fpr_mean=("tpr_at_10_fpr", "mean"),
            n_repeats=("repeat", "count"),
            n_pos=("n_pos", "first"),
            n_neg=("n_neg", "first"),
            n_features=("n_features", "first"),
            n_selected_mean=("n_selected_mean", "mean"),
        )
        .sort_values(["model", "comparison", "method"])
    )


def paired_tests(auc_df: pd.DataFrame, proposed_method: str) -> pd.DataFrame:
    rows = []
    for (model, comparison), sub in auc_df.groupby(["model", "comparison"]):
        proposed = sub[sub["method"] == proposed_method][["repeat", "auc"]]
        proposed = proposed.rename(columns={"auc": "proposed_auc"})
        for method in sorted(set(sub["method"]) - {proposed_method}):
            base = sub[sub["method"] == method][["repeat", "auc"]]
            base = base.rename(columns={"auc": "baseline_auc"})
            merged = proposed.merge(base, on="repeat", how="inner").sort_values("repeat")
            diff = merged["proposed_auc"].to_numpy() - merged["baseline_auc"].to_numpy()
            try:
                w_p = float(wilcoxon(diff, alternative="two-sided", zero_method="wilcox").pvalue)
            except ValueError:
                w_p = math.nan
            t_p = float(ttest_rel(merged["proposed_auc"], merged["baseline_auc"]).pvalue)
            rows.append(
                {
                    "model": model,
                    "comparison": comparison,
                    "proposed_method": proposed_method,
                    "baseline_method": method,
                    "n_repeats": len(merged),
                    "proposed_auc_mean": float(merged["proposed_auc"].mean()),
                    "baseline_auc_mean": float(merged["baseline_auc"].mean()),
                    "mean_auc_diff": float(diff.mean()),
                    "std_auc_diff": float(diff.std(ddof=1)),
                    "wilcoxon_p": w_p,
                    "paired_t_p": t_p,
                    "proposed_outperforms": bool(diff.mean() > 0 and w_p < 0.05),
                }
            )
    return pd.DataFrame(rows)


def latex_table(summary_df: pd.DataFrame, tests_df: pd.DataFrame, path: Path) -> None:
    method_order = ["proposed_en", "attenmia_mlp", "lora_leak", "initial_loss", "loss_delta"]
    label = {
        "proposed_en": "Proposed+EN",
        "attenmia_mlp": "AttenMIA",
        "lora_leak": "LoRA-Leak",
        "initial_loss": "Initial loss",
        "loss_delta": "Loss diff.",
    }
    lines = [
        "\\begin{table*}[t]",
        "\\caption{Mean AUC over 10 repeated runs. Parentheses indicate Wilcoxon signed-rank test $p$-values comparing Proposed+EN with each baseline. A dagger ($\\dagger$) denotes that Proposed+EN significantly outperforms the baseline.}",
        "\\label{tab:exp3_auc_strict}",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Model & Comparison & Proposed+EN & AttenMIA & LoRA-Leak & Initial loss & Loss diff. \\\\",
        "\\midrule",
    ]
    for model in ["pythia1b", "pythia410m"]:
        for comparison in ["ft_vs_pt", "ft_vs_unseen"]:
            sub = summary_df[(summary_df["model"] == model) & (summary_df["comparison"] == comparison)]
            if sub.empty:
                continue
            values = {}
            for method in method_order:
                if method == "lora_leak":
                    row = sub[sub["method"].str.startswith("lora_leak:")]
                else:
                    row = sub[sub["method"] == method]
                if row.empty:
                    values[method] = "--"
                    continue
                auc = float(row.iloc[0]["auc_mean"])
                if method == "proposed_en":
                    values[method] = f"{auc:.3f}"
                else:
                    baseline_name = row.iloc[0]["method"]
                    test = tests_df[
                        (tests_df["model"] == model)
                        & (tests_df["comparison"] == comparison)
                        & (tests_df["baseline_method"] == baseline_name)
                    ]
                    if test.empty:
                        values[method] = f"{auc:.3f}"
                    else:
                        p = float(test.iloc[0]["wilcoxon_p"])
                        dagger = "$\\dagger$" if bool(test.iloc[0]["proposed_outperforms"]) else ""
                        values[method] = f"{auc:.3f} ({p:.3f}{dagger})"
            model_label = "Pythia-1B" if model == "pythia1b" else "Pythia-410M"
            comp_label = "FT--PT" if comparison == "ft_vs_pt" else "FT--Unseen"
            lines.append(
                f"{model_label} & {comp_label} & "
                f"{values['proposed_en']} & {values['attenmia_mlp']} & "
                f"{values['lora_leak']} & {values['initial_loss']} & "
                f"{values['loss_delta']} \\\\"
            )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_model(
    model: str,
    proposed_root: str,
    attenmia_root: str,
    lora_root: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> List[Dict]:
    log(f"Start {model}")
    proposed_root_p = resolve_path(proposed_root)
    attenmia_root_p = resolve_path(attenmia_root)
    lora_root_p = resolve_path(lora_root, ["lora_leak_scores.csv"])

    raw = read_group_files(proposed_root_p, "raw_experiment4_attention_shift.csv")
    sample_level = read_group_files(proposed_root_p, "sample_level_experiment4.csv")
    proposed_features = make_proposed_features(raw)
    lora_scores = load_lora_scores(lora_root_p)
    all_rows = []

    for comparison in args.comparisons:
        positive_group, negative_group = COMPARISONS[comparison]
        log(f"{model} {comparison}: make common folds")
        common_splits, split_df = make_common_splits(
            proposed_features,
            positive_group,
            negative_group,
            args.repeats,
            args.cv_splits,
            args.seed,
        )
        split_df.insert(0, "comparison", comparison)
        split_df.insert(0, "model", model)
        split_df.to_csv(output_dir / f"common_folds_{model}_{comparison}.csv", index=False)

        if "proposed" in args.methods:
            all_rows += [
                {"model": model, "comparison": comparison, **row}
                for row in run_proposed_en(
                    proposed_features, positive_group, negative_group, common_splits, args
                )
            ]

        if "attenmia" in args.methods:
            try:
                all_rows += [
                    {"model": model, "comparison": comparison, **row}
                    for row in run_attenmia_mlp(
                        attenmia_root_p,
                        comparison,
                        positive_group,
                        negative_group,
                        common_splits,
                        args,
                    )
                ]
            except FileNotFoundError as exc:
                log(f"[WARN] skip AttenMIA {model} {comparison}: {exc}")

        if "lora_leak" in args.methods:
            score_col = choose_lora_score(lora_root_p, comparison, args.lora_score_mode)
            all_rows += [
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
            ]

        if "initial_loss" in args.methods:
            all_rows += [
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
            ]

        if "loss_delta" in args.methods:
            all_rows += [
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
            ]

        pd.DataFrame(all_rows).to_csv(output_dir / "auc_10runs.partial.csv", index=False)
    return all_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="strict_fixedstep_method_comparison_10runs")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-c", type=float, default=0.1)
    parser.add_argument("--classifier-c", type=float, default=1.0)
    parser.add_argument("--elasticnet-l1-ratio", type=float, default=0.7)
    parser.add_argument("--elasticnet-max-iter", type=int, default=5000)
    parser.add_argument("--attenmia-max-iter", type=int, default=500)
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
        "--lora-score-mode",
        default="target_mink_family_best",
        help=(
            "LoRA-Leak score selection: target_mink_family_best, "
            "target_minkpp_0.5, best_strict, or score:<column>."
        ),
    )
    parser.add_argument("--run-1b", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-410m", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--proposed-1b-root", default=DEFAULT_1B_PROPOSED_ROOT)
    parser.add_argument("--attenmia-1b-dir", default=DEFAULT_1B_ATTENMIA_DIR)
    parser.add_argument("--lora-1b-dir", default=DEFAULT_1B_LORA_DIR)
    parser.add_argument("--proposed-410m-root", default=DEFAULT_410M_PROPOSED_ROOT)
    parser.add_argument("--attenmia-410m-dir", default=DEFAULT_410M_ATTENMIA_DIR)
    parser.add_argument("--lora-410m-dir", default=DEFAULT_410M_LORA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.run_1b:
        rows += run_model(
            "pythia1b",
            args.proposed_1b_root,
            args.attenmia_1b_dir,
            args.lora_1b_dir,
            args,
            output_dir,
        )
    if args.run_410m:
        rows += run_model(
            "pythia410m",
            args.proposed_410m_root,
            args.attenmia_410m_dir,
            args.lora_410m_dir,
            args,
            output_dir,
        )

    auc_df = pd.DataFrame(rows)
    auc_df.to_csv(output_dir / "auc_10runs.csv", index=False)
    summary_df = summarize(auc_df)
    summary_df.to_csv(output_dir / "summary_auc.csv", index=False)
    tests_df = paired_tests(auc_df, "proposed_en")
    tests_df.to_csv(output_dir / "paired_auc_tests.csv", index=False)
    latex_table(summary_df, tests_df, output_dir / "paper_auc_table.tex")
    with open(output_dir / "comparison_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print("\nSummary:")
    print(summary_df.round(6).to_string(index=False))
    print("\nPaired tests:")
    print(tests_df.round(6).to_string(index=False))
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
