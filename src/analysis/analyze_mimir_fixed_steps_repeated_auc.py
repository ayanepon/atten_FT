# -*- coding: utf-8 -*-
"""Repeated AUC analysis for MIMIR fixed 20/50/100-step experiments.

This script does not repeat attention extraction or model fine-tuning.
It repeats the downstream 5-fold OOF classification ten times with different
CV seeds using the already extracted fixed-step attention features.

FT is fixed as the positive class. AUC is never flipped after observing the
result. Elastic Net feature selection is performed only inside each training
fold to avoid information leakage.

Default comparisons:
  - FT vs PT
  - FT vs Unseen

Default conditions:
  - fixed_attention_20
  - fixed_attention_50
  - fixed_attention_100

Run:
  python analyze_mimir_fixed_steps_repeated_auc.py
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


GROUP_FT = "mimir_wikipedia_nonmember_ft"
GROUP_PT = "mimir_wikipedia_member_pt"
GROUP_UNSEEN = "mimir_wikipedia_nonmember_unseen"

PAIR_SPECS = [
    ("ft_vs_pt", GROUP_FT, GROUP_PT),
    ("ft_vs_unseen", GROUP_FT, GROUP_UNSEEN),
]

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        default="experiment4_mimir_hardsplit_stopping_condition",
        help="Root containing fixed_attention_{20,50,100}_{ft,pt,unseen}.",
    )
    parser.add_argument(
        "--output-dir",
        default="mimir_fixed_steps_repeated_auc_analysis",
    )
    parser.add_argument(
        "--steps",
        default="20,50,100",
        help="Comma-separated fixed optimizer-step conditions.",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--methods",
        default="elasticnet,all",
        help="Comma-separated methods: elasticnet,all.",
    )
    parser.add_argument("--selection-c", type=float, default=0.1)
    parser.add_argument("--classifier-c", type=float, default=1.0)
    parser.add_argument("--elasticnet-l1-ratio", type=float, default=0.7)
    parser.add_argument("--elasticnet-max-iter", type=int, default=5000)
    return parser.parse_args()


def resolve_input_root(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    candidates = [
        path,
        Path.cwd() / path,
        Path.cwd().parent / path,
        Path.cwd().parent / "results" / path.name,
        Path("BlackNLP_2/results") / path.name,
        Path("BlackNLP/results") / path.name,
        Path("results") / path.name,
    ]
    results_dir = os.environ.get("RESULTS_DIR")
    if results_dir:
        candidates.append(Path(results_dir) / path.name)

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists() and any(
            candidate.glob(
                "**/fixed_attention_20_*/"
                "raw_experiment4_attention_shift.csv"
            )
        ):
            return candidate

    tried = "\n".join(f"  - {candidate}" for candidate in unique_candidates)
    raise FileNotFoundError(
        "Fixed-step result root could not be resolved.\n"
        f"Requested: {path_like}\n"
        f"Tried:\n{tried}\n"
        "Pass the server result directory explicitly with --input-root."
    )


def choose_raw_file(root: Path, step: int, group_key: str) -> Path:
    run_name = f"fixed_attention_{step}_{group_key}"
    candidates = list(
        root.glob(f"**/{run_name}/raw_experiment4_attention_shift.csv")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No raw fixed-step result found for {run_name} under {root}"
        )
    # Duplicate result folders may exist. Prefer the largest, most complete CSV.
    return max(candidates, key=lambda path: path.stat().st_size)


def load_condition(root: Path, step: int) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    files = {
        "ft": choose_raw_file(root, step, "ft"),
        "pt": choose_raw_file(root, step, "pt"),
        "unseen": choose_raw_file(root, step, "unseen"),
    }
    parts = []
    for group_key, path in files.items():
        frame = pd.read_csv(path)
        frame["source_run"] = path.parent.name
        frame["source_group_key"] = group_key
        parts.append(frame)

    raw = pd.concat(parts, ignore_index=True)
    expected_groups = {GROUP_FT, GROUP_PT, GROUP_UNSEEN}
    raw = raw[raw["group"].isin(expected_groups)].copy()
    return raw, files


def make_nonaveraged_features(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = [metric for metric in ATTENTION_METRICS if metric in raw.columns]
    long = raw[["sample_id", "group", "layer", "head"] + metrics].melt(
        id_vars=["sample_id", "group", "layer", "head"],
        value_vars=metrics,
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


def clean_features(pair: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    feature_cols = [column for column in pair.columns if column.startswith("attn_l")]
    frame = pair[feature_cols].replace([np.inf, -np.inf], np.nan)
    valid_cols = [
        column
        for column in feature_cols
        if frame[column].notna().sum() >= 4
        and frame[column].nunique(dropna=True) > 1
    ]
    frame = frame[valid_cols]
    frame = frame.fillna(frame.median(numeric_only=True))
    return frame.to_numpy(dtype=float), valid_cols


def tpr_at_fpr(
    y_true: np.ndarray, scores: np.ndarray, max_fpr: float = 0.10
) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = fpr <= max_fpr
    return float(np.max(tpr[valid])) if np.any(valid) else 0.0


def select_elasticnet_features(
    x_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> np.ndarray:
    selector = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=args.elasticnet_l1_ratio,
        C=args.selection_c,
        max_iter=args.elasticnet_max_iter,
        class_weight="balanced",
        random_state=seed,
    )
    selector.fit(x_train, y_train)
    selected = np.flatnonzero(np.abs(selector.coef_[0]) > 1e-10)
    if len(selected) == 0:
        selected = np.arange(x_train.shape[1])
    return selected


def evaluate_one_repeat(
    pair: pd.DataFrame,
    pair_name: str,
    step: int,
    repeat_index: int,
    repeat_seed: int,
    method: str,
    args: argparse.Namespace,
) -> Tuple[Dict, List[Dict]]:
    x, feature_cols = clean_features(pair)
    # FT is always the positive class for both requested comparisons.
    y = (pair["group"] == GROUP_FT).astype(int).to_numpy()
    class_counts = np.bincount(y)
    if len(class_counts) != 2 or class_counts.min() < args.cv_splits:
        raise ValueError(
            f"Insufficient samples for {pair_name}, step={step}: "
            f"class counts={class_counts.tolist()}"
        )

    cv = StratifiedKFold(
        n_splits=args.cv_splits,
        shuffle=True,
        random_state=repeat_seed,
    )
    oof_scores = np.full(len(pair), np.nan)
    selected_counts = []
    selection_rows = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])

        if method == "elasticnet":
            selected = select_elasticnet_features(
                x_train,
                y[train_idx],
                args,
                seed=repeat_seed * 100 + fold,
            )
        elif method == "all":
            selected = np.arange(x.shape[1])
        else:
            raise ValueError(f"Unknown method: {method}")

        classifier = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=args.classifier_c,
            max_iter=2000,
            class_weight="balanced",
            random_state=repeat_seed * 100 + fold,
        )
        classifier.fit(x_train[:, selected], y[train_idx])
        scores = classifier.predict_proba(x_test[:, selected])[:, 1]
        oof_scores[test_idx] = scores
        selected_counts.append(len(selected))

        if method == "elasticnet":
            for feature_index in selected:
                selection_rows.append(
                    {
                        "step": step,
                        "comparison": pair_name,
                        "repeat": repeat_index,
                        "seed": repeat_seed,
                        "fold": fold,
                        "feature": feature_cols[feature_index],
                    }
                )

    return (
        {
            "step": step,
            "condition": f"fixed_attention_{step}",
            "comparison": pair_name,
            "method": method,
            "repeat": repeat_index,
            "seed": repeat_seed,
            "n_ft": int(np.sum(y == 1)),
            "n_other": int(np.sum(y == 0)),
            "n_features_total": len(feature_cols),
            "n_features_selected_mean": float(np.mean(selected_counts)),
            "n_features_selected_std": float(
                np.std(selected_counts, ddof=1)
            ),
            "oof_auc": float(roc_auc_score(y, oof_scores)),
            "oof_auprc": float(average_precision_score(y, oof_scores)),
            "oof_tpr_at_10_fpr": tpr_at_fpr(y, oof_scores),
        },
        selection_rows,
    )


def summarize_repeats(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["step", "condition", "comparison", "method"])
        .agg(
            n_repeats=("oof_auc", "size"),
            n_ft=("n_ft", "first"),
            n_other=("n_other", "first"),
            auc_mean=("oof_auc", "mean"),
            auc_std=("oof_auc", "std"),
            auc_min=("oof_auc", "min"),
            auc_max=("oof_auc", "max"),
            auprc_mean=("oof_auprc", "mean"),
            auprc_std=("oof_auprc", "std"),
            tpr_at_10_fpr_mean=("oof_tpr_at_10_fpr", "mean"),
            tpr_at_10_fpr_std=("oof_tpr_at_10_fpr", "std"),
            selected_features_mean=("n_features_selected_mean", "mean"),
            selected_features_std=("n_features_selected_mean", "std"),
        )
        .reset_index()
        .sort_values(["method", "step", "comparison"])
    )


def make_paper_table(summary: pd.DataFrame, method: str) -> pd.DataFrame:
    subset = summary[summary["method"] == method].copy()
    rows = []
    for step in sorted(subset["step"].unique()):
        row = {"Method": f"Fixed-step Attention ({step})"}
        for comparison, prefix in [
            ("ft_vs_unseen", "FT vs Unseen"),
            ("ft_vs_pt", "FT vs PT"),
        ]:
            match = subset[
                (subset["step"] == step)
                & (subset["comparison"] == comparison)
            ]
            if match.empty:
                row[prefix] = "--"
            else:
                item = match.iloc[0]
                row[prefix] = (
                    f"{item['auc_mean']:.3f} "
                    f"$\\pm$ {item['auc_std']:.3f}"
                )
        rows.append(row)
    return pd.DataFrame(rows)


def write_latex_table(table: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Method & FT vs Unseen AUC & FT vs PT AUC \\\\",
        "\\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{row['Method']} & {row['FT vs Unseen']} & "
            f"{row['FT vs PT']} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Mean $\\pm$ standard deviation of strict OOF AUC "
            "over repeated cross-validation runs. FT is the positive class.}",
            "\\label{tab:mimir_fixed_steps_repeated_auc}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = resolve_input_root(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = [int(value.strip()) for value in args.steps.split(",") if value.strip()]
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    invalid_methods = sorted(set(methods) - {"elasticnet", "all"})
    if invalid_methods:
        raise ValueError(f"Unsupported methods: {invalid_methods}")

    result_rows = []
    selection_rows = []
    input_rows = []

    for step in steps:
        print(f"\n[condition] fixed_attention_{step}", flush=True)
        raw, files = load_condition(root, step)
        features = make_nonaveraged_features(raw)

        group_counts = (
            features.groupby("group")["sample_id"].nunique().to_dict()
        )
        for key, path in files.items():
            input_rows.append(
                {
                    "step": step,
                    "group_key": key,
                    "path": str(path),
                    "file_size": path.stat().st_size,
                    "n_samples": group_counts.get(
                        {
                            "ft": GROUP_FT,
                            "pt": GROUP_PT,
                            "unseen": GROUP_UNSEEN,
                        }[key],
                        0,
                    ),
                }
            )

        for pair_name, group_ft, group_other in PAIR_SPECS:
            pair = features[
                features["group"].isin([group_ft, group_other])
            ].reset_index(drop=True)
            print(
                f"  {pair_name}: "
                f"{pair['group'].value_counts().to_dict()}",
                flush=True,
            )
            for method in methods:
                for repeat_index in range(1, args.repeats + 1):
                    repeat_seed = args.seed + repeat_index - 1
                    print(
                        f"    {method} repeat "
                        f"{repeat_index}/{args.repeats} seed={repeat_seed}",
                        flush=True,
                    )
                    result, selected = evaluate_one_repeat(
                        pair,
                        pair_name,
                        step,
                        repeat_index,
                        repeat_seed,
                        method,
                        args,
                    )
                    result_rows.append(result)
                    selection_rows.extend(selected)

    results = pd.DataFrame(result_rows)
    summary = summarize_repeats(results)
    inputs = pd.DataFrame(input_rows)
    selected = pd.DataFrame(selection_rows)

    results.to_csv(output_dir / "repeated_auc_per_run.csv", index=False)
    summary.to_csv(output_dir / "repeated_auc_summary.csv", index=False)
    inputs.to_csv(output_dir / "input_files_and_counts.csv", index=False)
    if not selected.empty:
        selected.to_csv(
            output_dir / "elasticnet_selected_features_by_run_fold.csv",
            index=False,
        )
        (
            selected.groupby(["step", "comparison", "feature"])
            .size()
            .reset_index(name="selected_count")
            .sort_values(
                ["step", "comparison", "selected_count"],
                ascending=[True, True, False],
            )
            .to_csv(
                output_dir / "elasticnet_selected_feature_frequency.csv",
                index=False,
            )
        )

    for method in methods:
        table = make_paper_table(summary, method)
        table.to_csv(
            output_dir / f"paper_auc_mean_std_{method}.csv", index=False
        )
        write_latex_table(
            table,
            output_dir / f"paper_auc_mean_std_{method}.tex",
        )

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as handle:
        handle.write(
            "Repeated fixed-step AUC analysis\n"
            f"input_root={root}\n"
            f"repeats={args.repeats}, cv_splits={args.cv_splits}, "
            f"base_seed={args.seed}\n"
            "FT is the positive class. AUC is not post-hoc flipped.\n"
            "The repeats vary downstream CV splits; attention extraction "
            "is not repeated.\n\n"
        )
        handle.write(summary.to_string(index=False))
        handle.write("\n")

    print("\nSaved summary:")
    print(summary.to_string(index=False))
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
