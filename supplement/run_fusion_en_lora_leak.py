#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Complementarity experiment: Proposed+EN ⊕ LoRA-Leak fusion.

Paper-facing question:
  If Proposed+EN and LoRA-Leak are near-tied in AUC, are they the *same*
  signal or complementary?  Train-fold-only fusion should improve if signals
  are partially orthogonal.

Fusion variants (no test leakage):
  1. ``fusion_2d``     — logistic regression on [s_EN, s_LoRA] fitted on train fold
  2. ``fusion_alpha``  — s = α·s_EN + (1-α)·s_LoRA; α chosen by train-fold AUC

Standalone baselines (same common folds):
  - proposed_en
  - lora_leak  (1D LR on the chosen LoRA-Leak score column)

Also reports Spearman ρ between OOF EN scores and LoRA-Leak raw scores.

Example:
  python run_fusion_en_lora_leak.py \\
    --proposed-root attention_features_mimir_hardsplit_legacy \\
    --lora-root results/lora_leak_pythia1b \\
    --output-dir results/fusion_en_lora_pythia1b \\
    --repeats 10 --n-jobs 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

# Reuse strict-eval helpers
from run_strict_fixed20_comparison_10runs import (
    COMPARISONS,
    compute_metrics,
    elastic_net_selector_kwargs,
    ensure_uid,
    fit_transform_train_only,
    load_or_build_proposed_features,
    make_common_splits,
    paired_tests,
    read_group_files,
    summarize,
    wilcoxon_p,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def resolve_local_path(path_like: str) -> Path:
    """Resolve paths accepted from project root or from ``data/``."""
    path = Path(path_like).expanduser()
    if path.is_absolute() and path.exists():
        return path
    candidates = [
        path,
        Path.cwd() / path,
        script_dir() / path,
        script_dir().parent / path,
        script_dir() / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve path: {path_like}")


def load_lora_scores_csv(lora_root: Path) -> pd.DataFrame:
    matches = list(lora_root.glob("**/lora_leak_scores.csv"))
    if not matches and (lora_root / "lora_leak_scores.csv").exists():
        matches = [lora_root / "lora_leak_scores.csv"]
    if not matches:
        raise FileNotFoundError(f"lora_leak_scores.csv not found under {lora_root}")
    path = max(matches, key=lambda p: p.stat().st_size)
    log(f"LoRA-Leak scores: {path}")
    return pd.read_csv(path)


def load_target_map(proposed_root: Path, condition_prefix: str) -> pd.DataFrame:
    """Map proposed (group, sample_id) → original_index / text for alignment."""
    parts = []
    for g in ("ft", "pt", "unseen"):
        cand = list(proposed_root.glob(f"**/{condition_prefix}_{g}/experiment4_target_samples.csv"))
        if not cand:
            raise FileNotFoundError(
                f"experiment4_target_samples.csv missing for {condition_prefix}_{g} under {proposed_root}"
            )
        path = max(cand, key=lambda p: p.stat().st_size)
        t = pd.read_csv(path)
        # proposed features use sample_id 0..n-1 in this file order
        if "sample_id" not in t.columns:
            t = t.copy()
            t["sample_id"] = np.arange(len(t), dtype=int)
        if "global_sample_id" in t.columns and t["sample_id"].nunique() < len(t):
            # prefer sequential local id matching feature extraction order
            t = t.copy()
            t["sample_id"] = np.arange(len(t), dtype=int)
        parts.append(t)
    out = pd.concat(parts, ignore_index=True)
    need = {"group", "sample_id", "original_index"}
    missing = need - set(out.columns)
    if missing:
        raise ValueError(f"target samples missing columns: {missing}")
    return out


def choose_lora_col(df: pd.DataFrame, preferred: str) -> str:
    if preferred and preferred != "auto" and preferred in df.columns:
        return preferred
    for cand in [
        "target_mink++_0.2",
        "mink++_0.2_refpt",
        "target_mink++_0.5",
        "target_loss",
        "loss_refpt",
        "target_mink_0.2",
    ]:
        if cand in df.columns and df[cand].notna().any():
            return cand
    raise ValueError(f"No usable LoRA-Leak score column in {list(df.columns)}")


def align_proposed_with_lora(
    proposed: pd.DataFrame,
    lora: pd.DataFrame,
    target_map: pd.DataFrame,
    lora_col: str,
) -> pd.DataFrame:
    """Attach LoRA score to each proposed sample via original_index (verified 1:1)."""
    prop = ensure_uid(proposed)
    tmap = target_map[["group", "sample_id", "original_index", "text"]].copy()
    tmap["sample_id"] = tmap["sample_id"].astype(int)
    # if multiple rows per group-sample_id, keep first
    tmap = tmap.drop_duplicates(["group", "sample_id"], keep="first")

    merged = prop.merge(tmap, on=["group", "sample_id"], how="inner", validate="one_to_one")
    lora_sub = lora[["group", "original_index", lora_col]].copy()
    lora_sub = lora_sub.drop_duplicates(["group", "original_index"], keep="first")
    merged = merged.merge(lora_sub, on=["group", "original_index"], how="inner", validate="one_to_one")
    if len(merged) != len(prop):
        log(
            f"WARNING: alignment kept {len(merged)}/{len(prop)} rows "
            f"(group counts: {merged.groupby('group').size().to_dict()})"
        )
    else:
        log(f"Aligned proposed ↔ LoRA-Leak: {len(merged)} samples (1:1 on original_index)")
    merged = merged.rename(columns={lora_col: "lora_raw"})
    return merged


def _fit_en_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    args: argparse.Namespace,
    rng: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (train_proba, test_proba, n_selected)."""
    selector = LogisticRegression(
        solver="saga",
        l1_ratio=args.elasticnet_l1_ratio,
        C=args.selection_c,
        tol=args.elasticnet_tol,
        max_iter=args.elasticnet_max_iter,
        class_weight="balanced",
        random_state=rng,
        **elastic_net_selector_kwargs(),
    )
    selector.fit(x_train, y_train)
    selected = np.flatnonzero(np.abs(selector.coef_[0]) > 1e-10)
    if len(selected) == 0:
        selected = np.arange(x_train.shape[1])
    clf = LogisticRegression(
        solver="lbfgs",
        C=args.classifier_c,
        max_iter=2000,
        class_weight="balanced",
        random_state=rng,
    )
    clf.fit(x_train[:, selected], y_train)
    s_tr = clf.predict_proba(x_train[:, selected])[:, 1]
    s_te = clf.predict_proba(x_test[:, selected])[:, 1]
    return s_tr, s_te, int(len(selected))


def _fit_1d_lr_scores(
    raw_train: np.ndarray,
    y_train: np.ndarray,
    raw_test: np.ndarray,
    rng: int,
) -> Tuple[np.ndarray, np.ndarray]:
    clf = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=rng,
    )
    clf.fit(raw_train.reshape(-1, 1), y_train)
    return clf.predict_proba(raw_train.reshape(-1, 1))[:, 1], clf.predict_proba(raw_test.reshape(-1, 1))[:, 1]


def _pick_alpha(s_en: np.ndarray, s_ll: np.ndarray, y: np.ndarray, grid: Sequence[float]) -> float:
    best_a, best_auc = 0.5, -1.0
    for a in grid:
        s = a * s_en + (1.0 - a) * s_ll
        try:
            auc = float(roc_auc_score(y, s))
        except ValueError:
            continue
        if auc > best_auc:
            best_auc = auc
            best_a = float(a)
    return best_a


def _run_fusion_repeat(
    repeat: int,
    *,
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    x_all: np.ndarray,
    y: np.ndarray,
    lora_raw: np.ndarray,
    uid_to_idx: Dict[str, int],
    common_splits: List[Dict],
    args: argparse.Namespace,
) -> Tuple[List[Dict], List[Dict]]:
    fold_m = {k: [] for k in ("proposed_en", "lora_leak", "fusion_2d", "fusion_alpha")}
    selected_counts = []
    alphas = []
    oof_rows: List[Dict] = []

    for split in [s for s in common_splits if s["repeat"] == repeat]:
        train_idx = np.asarray(
            split.get("train_idx", [uid_to_idx[u] for u in sorted(split["train_uids"])]),
            dtype=int,
        )
        test_idx = np.asarray(
            split.get("test_idx", [uid_to_idx[u] for u in sorted(split["test_uids"])]),
            dtype=int,
        )
        rng = args.seed + repeat * 100 + split["fold"]

        x_train, x_test = fit_transform_train_only(x_all[train_idx], x_all[test_idx])
        y_tr, y_te = y[train_idx], y[test_idx]

        s_en_tr, s_en_te, n_sel = _fit_en_scores(x_train, y_tr, x_test, args, rng)
        selected_counts.append(n_sel)
        s_ll_tr, s_ll_te = _fit_1d_lr_scores(lora_raw[train_idx], y_tr, lora_raw[test_idx], rng)

        fuse = LogisticRegression(
            solver="lbfgs",
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            random_state=rng,
        )
        Xtr = np.column_stack([s_en_tr, s_ll_tr])
        Xte = np.column_stack([s_en_te, s_ll_te])
        fuse.fit(Xtr, y_tr)
        s_2d_te = fuse.predict_proba(Xte)[:, 1]

        a = _pick_alpha(s_en_tr, s_ll_tr, y_tr, [i / 10.0 for i in range(11)])
        alphas.append(a)
        s_a_te = a * s_en_te + (1.0 - a) * s_ll_te

        fold_m["proposed_en"].append(compute_metrics(y_te, s_en_te))
        fold_m["lora_leak"].append(compute_metrics(y_te, s_ll_te))
        fold_m["fusion_2d"].append(compute_metrics(y_te, s_2d_te))
        fold_m["fusion_alpha"].append(compute_metrics(y_te, s_a_te))

        for j, idx in enumerate(test_idx):
            oof_rows.append(
                {
                    "repeat": repeat,
                    "fold": split["fold"],
                    "uid": df.iloc[idx]["uid"],
                    "group": df.iloc[idx]["group"],
                    "y": int(y[idx]),
                    "s_en": float(s_en_te[j]),
                    "s_lora": float(s_ll_te[j]),
                    "s_fusion_2d": float(s_2d_te[j]),
                    "s_fusion_alpha": float(s_a_te[j]),
                    "alpha": float(a),
                    "lora_raw": float(lora_raw[idx]),
                }
            )

    rows = []
    for method, metrics_list in fold_m.items():
        rows.append(
            {
                "method": method,
                "repeat": repeat,
                "auc": float(np.mean([m["auc"] for m in metrics_list])),
                "auprc": float(np.mean([m["auprc"] for m in metrics_list])),
                "tpr_at_10_fpr": float(np.mean([m["tpr_at_10_fpr"] for m in metrics_list])),
                "n_pos": int(y.sum()),
                "n_neg": int((1 - y).sum()),
                "n_features": len(feature_cols) if method == "proposed_en" else (2 if method.startswith("fusion") else 1),
                "n_selected_mean": float(np.mean(selected_counts)) if method == "proposed_en" else (
                    float(np.mean(alphas)) if method == "fusion_alpha" else math.nan
                ),
            }
        )
    return rows, oof_rows


def run_fusion_suite(
    aligned: pd.DataFrame,
    positive_group: str,
    negative_group: str,
    common_splits: List[Dict],
    args: argparse.Namespace,
) -> Tuple[List[Dict], pd.DataFrame]:
    """Run EN, LoRA-Leak, fusion_2d, fusion_alpha on shared folds.

    Also returns OOF score frame for correlation analysis.
    """
    feature_cols = [c for c in aligned.columns if c.startswith("attn_l")]
    df = aligned[aligned["group"].isin([positive_group, negative_group])].drop_duplicates("uid").reset_index(drop=True)
    x_all = df[feature_cols].to_numpy(dtype=float)
    y = (df["group"].to_numpy() == positive_group).astype(int)
    lora_raw = df["lora_raw"].to_numpy(dtype=float)
    uid_to_idx = {uid: i for i, uid in enumerate(df["uid"].tolist())}

    def run_repeat(repeat: int) -> Tuple[List[Dict], List[Dict]]:
        log(f"  repeat {repeat}/{args.repeats}")
        return _run_fusion_repeat(
            repeat,
            df=df,
            feature_cols=feature_cols,
            x_all=x_all,
            y=y,
            lora_raw=lora_raw,
            uid_to_idx=uid_to_idx,
            common_splits=common_splits,
            args=args,
        )

    if int(args.n_jobs or 1) != 1:
        from hardsplit.parallel import map_repeats

        parts = map_repeats(run_repeat, args.repeats, n_jobs=int(args.n_jobs), prefer="threads")
    else:
        parts = [run_repeat(repeat) for repeat in range(1, args.repeats + 1)]

    rows: List[Dict] = []
    oof_rows: List[Dict] = []
    for repeat_rows, repeat_oof in parts:
        rows.extend(repeat_rows)
        oof_rows.extend(repeat_oof)
    return rows, pd.DataFrame(oof_rows)


def score_correlations(oof: pd.DataFrame) -> pd.DataFrame:
    """Spearman between OOF EN and LoRA scores (mean over repeats)."""
    rows = []
    if oof.empty:
        return pd.DataFrame()
    for repeat, sub in oof.groupby("repeat"):
        # average duplicate uids across folds shouldn't happen (each uid once per repeat)
        rho_en_ll = sub["s_en"].corr(sub["s_lora"], method="spearman")
        rho_en_raw = sub["s_en"].corr(sub["lora_raw"], method="spearman")
        rows.append(
            {
                "repeat": int(repeat),
                "spearman_s_en_vs_s_lora": float(rho_en_ll) if pd.notna(rho_en_ll) else math.nan,
                "spearman_s_en_vs_lora_raw": float(rho_en_raw) if pd.notna(rho_en_raw) else math.nan,
                "n": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, corr: pd.DataFrame, paired: pd.DataFrame, path: Path) -> None:
    lines = [
        "Proposed+EN ⊕ LoRA-Leak fusion experiment",
        "=" * 60,
        "",
        "Summary (mean over repeats):",
        summary.to_string(index=False),
        "",
    ]
    if not corr.empty:
        lines.append("Score correlation (OOF Spearman, mean±std over repeats):")
        for col in ["spearman_s_en_vs_s_lora", "spearman_s_en_vs_lora_raw"]:
            if col in corr.columns:
                lines.append(
                    f"  {col}: {corr[col].mean():.4f} ± {corr[col].std(ddof=1) if len(corr)>1 else 0:.4f}"
                )
        lines.append("")
    if not paired.empty:
        lines.append("Paired AUC tests (fusion_2d vs baselines):")
        sub = paired[paired["proposed_method"] == "fusion_2d"]
        lines.append(sub.to_string(index=False) if not sub.empty else "(none)")
        lines.append("")
        lines.append("Paired AUC tests (proposed_en vs lora_leak / fusions):")
        sub2 = paired[paired["proposed_method"] == "proposed_en"]
        lines.append(sub2.to_string(index=False) if not sub2.empty else "(none)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Proposed+EN ⊕ LoRA-Leak fusion experiment")
    p.add_argument(
        "--proposed-root",
        default=str(script_dir() / "attention_features_mimir_hardsplit_legacy"),
    )
    p.add_argument(
        "--lora-root",
        default=str(script_dir() / "results" / "lora_leak_pythia1b"),
    )
    p.add_argument(
        "--output-dir",
        default=str(script_dir() / "results" / "fusion_en_lora_pythia1b"),
    )
    p.add_argument("--condition-prefix", default="fixed_attention_20")
    p.add_argument("--model-label", default="pythia1b")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--selection-c", type=float, default=0.1)
    p.add_argument("--classifier-c", type=float, default=1.0)
    p.add_argument("--elasticnet-l1-ratio", type=float, default=0.7)
    p.add_argument("--elasticnet-max-iter", type=int, default=1000)
    p.add_argument("--elasticnet-tol", type=float, default=5e-4)
    p.add_argument(
        "--n-jobs",
        type=int,
        default=int(os.environ.get("EVAL_N_JOBS", str(min(4, os.cpu_count() or 1)))),
    )
    p.add_argument("--lora-score-col", default="auto")
    p.add_argument(
        "--comparisons",
        nargs="+",
        default=["ft_vs_pt", "ft_vs_unseen"],
        choices=list(COMPARISONS.keys()),
    )
    p.add_argument("--refresh-feature-cache", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    proposed_root = resolve_local_path(args.proposed_root)
    lora_root = resolve_local_path(args.lora_root)

    log("=" * 72)
    log("FUSION EXPERIMENT: Proposed+EN ⊕ LoRA-Leak")
    log(f"proposed_root={proposed_root}")
    log(f"lora_root={lora_root}")
    log(f"repeats={args.repeats} seed={args.seed}")
    log("=" * 72)

    raw = read_group_files(
        proposed_root,
        "raw_experiment4_attention_shift.csv",
        condition_prefix=args.condition_prefix,
    )
    proposed = load_or_build_proposed_features(
        proposed_root,
        raw,
        refresh=args.refresh_feature_cache,
        condition_prefix=args.condition_prefix,
    )
    lora = load_lora_scores_csv(lora_root)
    lora_col = choose_lora_col(lora, args.lora_score_col)
    log(f"Using LoRA-Leak score column: {lora_col}")
    target_map = load_target_map(proposed_root, args.condition_prefix)
    aligned = align_proposed_with_lora(proposed, lora, target_map, lora_col)

    all_rows: List[Dict] = []
    all_oof: List[pd.DataFrame] = []

    for comparison in args.comparisons:
        pos, neg = COMPARISONS[comparison]
        log(f"\n### {comparison} ({pos} vs {neg})")
        common_splits, split_df = make_common_splits(
            aligned,
            pos,
            neg,
            repeats=args.repeats,
            cv_splits=args.cv_splits,
            seed=args.seed,
        )
        split_df.insert(0, "comparison", comparison)
        split_df.to_csv(out / f"common_folds_{comparison}.csv", index=False)

        rows, oof = run_fusion_suite(aligned, pos, neg, common_splits, args)
        for r in rows:
            all_rows.append({"model": args.model_label, "comparison": comparison, **r})
        oof = oof.copy()
        oof.insert(0, "comparison", comparison)
        all_oof.append(oof)

        # per-comparison correlation
        corr = score_correlations(oof)
        if not corr.empty:
            corr.to_csv(out / f"score_correlation_{comparison}.csv", index=False)
            log(
                f"  Spearman(s_en, s_lora) = {corr['spearman_s_en_vs_s_lora'].mean():.4f} "
                f"± {corr['spearman_s_en_vs_s_lora'].std(ddof=1) if len(corr)>1 else 0:.4f}"
            )

    auc_df = pd.DataFrame(all_rows)
    auc_df.to_csv(out / "auc_10runs.csv", index=False)
    summary = summarize(auc_df)
    summary.to_csv(out / "summary_auc.csv", index=False)

    oof_all = pd.concat(all_oof, ignore_index=True) if all_oof else pd.DataFrame()
    if not oof_all.empty:
        oof_all.to_csv(out / "oof_scores.csv", index=False)

    # Paired tests: fusion_2d vs each; proposed_en vs each
    paired_parts = []
    for ref in ("fusion_2d", "proposed_en", "fusion_alpha"):
        if ref in set(auc_df["method"]):
            paired_parts.append(paired_tests(auc_df, proposed_method=ref))
    paired = pd.concat(paired_parts, ignore_index=True) if paired_parts else pd.DataFrame()
    if not paired.empty:
        paired.to_csv(out / "paired_auc_tests.csv", index=False)

    corr_rows = []
    for comparison in args.comparisons:
        cpath = out / f"score_correlation_{comparison}.csv"
        if cpath.exists():
            c = pd.read_csv(cpath)
            c.insert(0, "comparison", comparison)
            corr_rows.append(c)
    corr_all = pd.concat(corr_rows, ignore_index=True) if corr_rows else pd.DataFrame()
    if not corr_all.empty:
        corr_all.to_csv(out / "score_correlation_all.csv", index=False)

    write_report(summary, corr_all, paired, out / "summary.txt")

    # Compact paper-style table
    pivot_lines = [
        r"\begin{table}[t]",
        r"\caption{Proposed+EN, LoRA-Leak, and train-fold fusions (mean AUC / TPR@10\%FPR).}",
        r"\label{tab:fusion_en_lora}",
        r"\centering\small",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Comparison & Method & AUC & TPR@10\%FPR \\",
        r"\midrule",
    ]
    method_order = ["proposed_en", "lora_leak", "fusion_alpha", "fusion_2d"]
    method_names = {
        "proposed_en": "Proposed+EN",
        "lora_leak": "LoRA-Leak",
        "fusion_alpha": r"Fusion $\alpha$",
        "fusion_2d": "Fusion 2D-LR",
    }
    for comparison in args.comparisons:
        label = "FT--PT" if comparison == "ft_vs_pt" else "FT--Unseen"
        for m in method_order:
            sub = summary[(summary["comparison"] == comparison) & (summary["method"] == m)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            pivot_lines.append(
                f"{label} & {method_names[m]} & {r['auc_mean']:.3f} & {r['tpr_at_10_fpr_mean']:.3f} \\\\"
            )
        if comparison != args.comparisons[-1]:
            pivot_lines.append(r"\midrule")
    pivot_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (out / "paper_table_fusion.tex").write_text("\n".join(pivot_lines), encoding="utf-8")

    config = vars(args).copy()
    config["lora_score_col_resolved"] = lora_col
    config["n_aligned"] = int(len(aligned))
    with open(out / "fusion_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    log("\n" + "=" * 72)
    log("DONE")
    log(f"summary → {out / 'summary_auc.csv'}")
    log(f"report  → {out / 'summary.txt'}")
    log(f"latex   → {out / 'paper_table_fusion.tex'}")
    print(summary.to_string(index=False))
    log("=" * 72)


if __name__ == "__main__":
    main()
