# -*- coding: utf-8 -*-
"""Strict Min-k% baseline comparison for MIMIR hard-split experiments.

This script evaluates Min-k% as a standalone baseline under the same repeated
5-fold splits used for the fixed-step Proposed+EN experiments.

It does not run model inference.  It reuses saved Min-k% scores from
LoRA-Leak output files, but only columns named ``target_mink_*`` are used.
Columns such as ``target_mink++_*`` are intentionally excluded so that this is
the plain Min-k% baseline.

Default comparisons:
  - FT vs PT
  - FT vs Unseen

Default outputs:
  - min_k_auc_10runs.csv
  - min_k_summary_auc.csv
  - min_k_best_by_comparison.csv
  - min_k_vs_proposed_paired_tests.csv, when Proposed+EN AUCs are available
  - paper_min_k_table.tex
"""

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


GROUP_FT = "mimir_wikipedia_nonmember_ft"
GROUP_PT = "mimir_wikipedia_member_pt"
GROUP_UNSEEN = "mimir_wikipedia_nonmember_unseen"

COMPARISONS = {
    "ft_vs_pt": (GROUP_FT, GROUP_PT),
    "ft_vs_unseen": (GROUP_FT, GROUP_UNSEEN),
}

DEFAULT_1B_PROPOSED_ROOT = (
    "results/"
    "experiment4_mimir_hardsplit_stopping_condition"
)
DEFAULT_1B_MINK_DIR = (
    "results/lora_leak_official_mimir_hardsplit"
)
DEFAULT_410M_PROPOSED_ROOT = (
    "results/"
    "mimir_wikipedia_hardsplit_fixed20_pythia410m_rerun"
)
DEFAULT_410M_MINK_DIR = (
    "results/"
    "lora_leak_official_mimir_hardsplit_pythia410m"
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def local_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_path(path_like: str, required_files: Sequence[str] = ()) -> Path:
    root = local_root()
    path = Path(path_like)
    candidates = [path, root / path.name]

    path_str = str(path_like)
    for prefix in [
        "results/",
        "results/",
        "",
    ]:
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
    """Attach uid = group::local_sample_id.

    This aligns files whose sample_id is group-local with files whose sample_id
    is global by remapping sorted sample_id values to 0..N-1 within each group.
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


def read_proposed_raw_for_folds(root: Path) -> pd.DataFrame:
    parts = []
    for group_key in ["ft", "pt", "unseen"]:
        matches = list(root.glob(f"**/fixed_attention_20_{group_key}/raw_experiment4_attention_shift.csv"))
        if not matches:
            raise FileNotFoundError(
                f"fixed_attention_20_{group_key}/raw_experiment4_attention_shift.csv not found under {root}"
            )
        path = max(matches, key=lambda p: p.stat().st_size)
        cols = ["sample_id", "group"]
        parts.append(pd.read_csv(path, usecols=lambda c: c in cols))
    return pd.concat(parts, ignore_index=True).drop_duplicates(["sample_id", "group"])


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
            splits.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "train_uids": train_uids,
                    "test_uids": test_uids,
                }
            )
            split_rows.extend(
                {"repeat": repeat, "fold": fold, "uid": uid, "split": "train"}
                for uid in sorted(train_uids)
            )
            split_rows.extend(
                {"repeat": repeat, "fold": fold, "uid": uid, "split": "test"}
                for uid in sorted(test_uids)
            )
    return splits, pd.DataFrame(split_rows)


def load_mink_scores(root: Path) -> pd.DataFrame:
    path = root / "lora_leak_scores.csv"
    if path.exists():
        return pd.read_csv(path)
    matches = list(root.glob("**/lora_leak_scores.csv"))
    if matches:
        return pd.read_csv(matches[0])
    raise FileNotFoundError(f"lora_leak_scores.csv not found under {root}")


def find_mink_columns(df: pd.DataFrame, requested: str) -> List[str]:
    if requested.strip().lower() != "all":
        cols = [c.strip() for c in requested.split(",") if c.strip()]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Requested Min-k columns not found: {missing}")
        return cols

    pattern = re.compile(r"^target_mink_\d+(?:\.\d+)?$")
    cols = [c for c in df.columns if pattern.match(c)]
    if not cols:
        raise ValueError("No plain Min-k% columns found. Expected columns like target_mink_0.2")
    return sorted(cols, key=lambda c: float(c.rsplit("_", 1)[1]))


def roc_auc_score_np(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return math.nan
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    sum_pos = float(np.sum(ranks[y_true == 1]))
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision_np(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    sorted_y = y_true[order]
    tp = np.cumsum(sorted_y == 1)
    rank = np.arange(1, len(sorted_y) + 1)
    precision = tp / rank
    return float(np.sum(precision[sorted_y == 1]) / n_pos)


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, max_fpr: float = 0.10) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    sorted_y = y_true[order]
    tp = np.cumsum(sorted_y == 1)
    fp = np.cumsum(sorted_y == 0)
    fpr = fp / n_neg
    tpr = tp / n_pos
    valid = fpr <= max_fpr
    return float(np.max(tpr[valid])) if np.any(valid) else 0.0


def evaluate_fixed_score(
    scores_df: pd.DataFrame,
    score_col: str,
    positive_group: str,
    negative_group: str,
    common_splits: List[Dict],
    repeats: int,
) -> List[Dict]:
    df = ensure_uid(scores_df)
    df = df[df["group"].isin([positive_group, negative_group])].dropna(subset=[score_col])
    df = df.drop_duplicates("uid").reset_index(drop=True)
    y = (df["group"].to_numpy() == positive_group).astype(int)
    scores = df[score_col].to_numpy(float)
    uid_to_idx = {uid: i for i, uid in enumerate(df["uid"].tolist())}
    rows = []

    for repeat in range(1, repeats + 1):
        fold_auc = []
        fold_auprc = []
        fold_tpr = []
        for split in [s for s in common_splits if s["repeat"] == repeat]:
            test_idx = np.array([uid_to_idx[u] for u in split["test_uids"] if u in uid_to_idx])
            fold_auc.append(float(roc_auc_score_np(y[test_idx], scores[test_idx])))
            fold_auprc.append(float(average_precision_np(y[test_idx], scores[test_idx])))
            fold_tpr.append(tpr_at_fpr(y[test_idx], scores[test_idx]))
        rows.append(
            {
                "method": f"min_k:{score_col}",
                "score_col": score_col,
                "repeat": repeat,
                "auc": float(np.mean(fold_auc)),
                "auprc": float(np.mean(fold_auprc)),
                "tpr_at_10_fpr": float(np.mean(fold_tpr)),
                "n_pos": int(y.sum()),
                "n_neg": int((1 - y).sum()),
            }
        )
    return rows


def summarize_auc(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["model", "comparison", "method", "score_col"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            auprc_mean=("auprc", "mean"),
            tpr_at_10_fpr_mean=("tpr_at_10_fpr", "mean"),
            n_repeats=("repeat", "count"),
            n_pos=("n_pos", "first"),
            n_neg=("n_neg", "first"),
        )
        .sort_values(["model", "comparison", "auc_mean"], ascending=[True, True, False])
    )


def best_by_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, comparison), sub in summary.groupby(["model", "comparison"]):
        rows.append(sub.sort_values("auc_mean", ascending=False).iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def load_proposed_auc(path_like: str) -> pd.DataFrame | None:
    if not path_like:
        return None
    path = Path(path_like)
    if not path.exists():
        candidate = local_root() / path_like
        path = candidate if candidate.exists() else path
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "method" in df.columns:
        df = df[df["method"].isin(["proposed_en", "proposed_l1_selected_layer_head"])].copy()
    return df


def rank_abs_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def wilcoxon_signed_rank_p(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[diff != 0]
    n = len(diff)
    if n == 0:
        return math.nan
    ranks = rank_abs_values(np.abs(diff))
    observed = min(float(np.sum(ranks[diff > 0])), float(np.sum(ranks[diff < 0])))
    total = 0
    extreme = 0
    # n is 10 in the default experiment, so exact enumeration is cheap.
    for mask in range(1 << n):
        plus = 0.0
        for i in range(n):
            if mask & (1 << i):
                plus += ranks[i]
        minus = float(np.sum(ranks) - plus)
        stat = min(plus, minus)
        total += 1
        if stat <= observed + 1e-12:
            extreme += 1
    return min(1.0, extreme / total)


def paired_t_p_placeholder(diff: np.ndarray) -> float:
    # The paper table uses Wilcoxon p-values.  We keep this column for schema
    # compatibility, but avoid a SciPy dependency in this standalone script.
    return math.nan


def paired_tests(min_k_auc: pd.DataFrame, proposed_auc: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "model",
        "comparison",
        "baseline_method",
        "n_repeats",
        "proposed_auc_mean",
        "baseline_auc_mean",
        "mean_auc_diff",
        "std_auc_diff",
        "wilcoxon_p",
        "paired_t_p",
        "proposed_outperforms",
    ]
    if proposed_auc is None or proposed_auc.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (model, comparison, method), sub in min_k_auc.groupby(["model", "comparison", "method"]):
        prop = proposed_auc[
            (proposed_auc["model"] == model)
            & (proposed_auc["comparison"] == comparison)
        ][["repeat", "auc"]].rename(columns={"auc": "proposed_auc"})
        if prop.empty:
            continue
        base = sub[["repeat", "auc"]].rename(columns={"auc": "baseline_auc"})
        merged = prop.merge(base, on="repeat", how="inner").sort_values("repeat")
        if len(merged) < 2:
            continue
        diff = merged["proposed_auc"].to_numpy() - merged["baseline_auc"].to_numpy()
        w_p = wilcoxon_signed_rank_p(diff)
        rows.append(
            {
                "model": model,
                "comparison": comparison,
                "baseline_method": method,
                "n_repeats": len(merged),
                "proposed_auc_mean": float(merged["proposed_auc"].mean()),
                "baseline_auc_mean": float(merged["baseline_auc"].mean()),
                "mean_auc_diff": float(diff.mean()),
                "std_auc_diff": float(diff.std(ddof=1)),
                "wilcoxon_p": w_p,
                "paired_t_p": paired_t_p_placeholder(diff),
                "proposed_outperforms": bool(diff.mean() > 0 and w_p < 0.05),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def write_latex(best: pd.DataFrame, tests: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Model & Comparison & Min-k\\% score & AUC & $p$ vs Proposed+EN \\\\",
        "\\midrule",
    ]
    for _, row in best.iterrows():
        test = tests[
            (tests["model"] == row["model"])
            & (tests["comparison"] == row["comparison"])
            & (tests["baseline_method"] == row["method"])
        ]
        p_text = "--" if test.empty else f"{float(test.iloc[0]['wilcoxon_p']):.3f}"
        model = "Pythia-1B" if row["model"] == "pythia1b" else "Pythia-410M"
        comp = "FT--PT" if row["comparison"] == "ft_vs_pt" else "FT--Unseen"
        score = str(row["score_col"]).replace("_", "\\_")
        lines.append(
            f"{model} & {comp} & {score} & {float(row['auc_mean']):.3f} & {p_text} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Plain Min-k\\% baseline evaluated on the same repeated folds as Proposed+EN.}",
            "\\label{tab:min_k_strict}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_model(model: str, proposed_root: str, mink_dir: str, args: argparse.Namespace, out: Path) -> List[Dict]:
    log(f"Start {model}")
    proposed_root_p = resolve_path(proposed_root)
    mink_dir_p = resolve_path(mink_dir, ["lora_leak_scores.csv"])
    base_for_folds = read_proposed_raw_for_folds(proposed_root_p)
    scores = load_mink_scores(mink_dir_p)
    mink_cols = find_mink_columns(scores, args.mink_columns)
    log(f"{model}: Min-k columns = {', '.join(mink_cols)}")

    rows = []
    for comparison in args.comparisons:
        positive, negative = COMPARISONS[comparison]
        splits, split_df = make_common_splits(
            base_for_folds,
            positive,
            negative,
            args.repeats,
            args.cv_splits,
            args.seed,
        )
        split_df.insert(0, "comparison", comparison)
        split_df.insert(0, "model", model)
        split_df.to_csv(out / f"common_folds_{model}_{comparison}.csv", index=False)

        for col in mink_cols:
            log(f"{model} {comparison}: evaluate {col}")
            rows.extend(
                {"model": model, "comparison": comparison, **r}
                for r in evaluate_fixed_score(scores, col, positive, negative, splits, args.repeats)
            )
        pd.DataFrame(rows).to_csv(out / "min_k_auc_10runs.partial.csv", index=False)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="min_k_strict_fixedstep_10runs")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mink-columns",
        default="all",
        help="Comma-separated plain Min-k columns, or all for target_mink_* columns.",
    )
    parser.add_argument(
        "--comparisons",
        nargs="+",
        default=["ft_vs_pt", "ft_vs_unseen"],
        choices=list(COMPARISONS.keys()),
    )
    parser.add_argument("--run-1b", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-410m", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--proposed-1b-root", default=DEFAULT_1B_PROPOSED_ROOT)
    parser.add_argument("--mink-1b-dir", default=DEFAULT_1B_MINK_DIR)
    parser.add_argument("--proposed-410m-root", default=DEFAULT_410M_PROPOSED_ROOT)
    parser.add_argument("--mink-410m-dir", default=DEFAULT_410M_MINK_DIR)
    parser.add_argument(
        "--proposed-auc-csv",
        default="strict_fixedstep_method_comparison_10runs/auc_10runs.csv",
        help="Optional Proposed+EN repeated AUC CSV for paired tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.run_1b:
        rows.extend(run_model("pythia1b", args.proposed_1b_root, args.mink_1b_dir, args, out))
    if args.run_410m:
        try:
            rows.extend(run_model("pythia410m", args.proposed_410m_root, args.mink_410m_dir, args, out))
        except FileNotFoundError as exc:
            log(f"[WARN] Skipping pythia410m: {exc}")

    auc = pd.DataFrame(rows)
    auc.to_csv(out / "min_k_auc_10runs.csv", index=False)
    summary = summarize_auc(auc)
    summary.to_csv(out / "min_k_summary_auc.csv", index=False)
    best = best_by_comparison(summary)
    best.to_csv(out / "min_k_best_by_comparison.csv", index=False)

    proposed = load_proposed_auc(args.proposed_auc_csv)
    tests = paired_tests(auc, proposed)
    tests.to_csv(out / "min_k_vs_proposed_paired_tests.csv", index=False)
    write_latex(best, tests, out / "paper_min_k_table.tex")
    with open(out / "min_k_comparison_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print("\nBest Min-k% by comparison:")
    print(best.round(6).to_string(index=False))
    if not tests.empty:
        print("\nPaired tests vs Proposed+EN:")
        print(tests.round(6).to_string(index=False))
    print(f"\nOutput directory: {out}")


if __name__ == "__main__":
    main()
