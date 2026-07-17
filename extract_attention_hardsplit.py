# -*- coding: utf-8 -*-
"""
Experiment 4: stopping-condition ablation for MIMIR Wikipedia hard split.

Purpose:
  Check whether attention-update features reflect training-history information,
  rather than merely the number of additional overfitting steps.

Conditions:
  - dynamic_attention: early stopping by loss/accuracy no-improvement patience.
  - fixed_attention_20/50/100: same number of optimization steps for every sample.
  - dynamic_steps_only: classify with only the dynamic early-stopping step count.

Default target model:
  results/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2

Default groups:
  PT      = MIMIR Wikipedia member
  FT      = MIMIR Wikipedia non-member FT split
  Unseen  = MIMIR Wikipedia non-member unseen split
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

CPU_THREAD_LIMIT = os.environ.get("CPU_THREAD_LIMIT", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", CPU_THREAD_LIMIT)
os.environ.setdefault("MKL_NUM_THREADS", CPU_THREAD_LIMIT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", CPU_THREAD_LIMIT)
os.environ.setdefault("NUMEXPR_NUM_THREADS", CPU_THREAD_LIMIT)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", CPU_THREAD_LIMIT)

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:  # pragma: no cover
    multipletests = None

import mimir_hardsplit_attention_common as common


try:
    from model_registry import PYTHIA1B_FEATURES_ROOT, PYTHIA1B_RUN_DIR, resolve_adapter_dir as _resolve_adapter_dir
except ImportError:  # pragma: no cover
    PYTHIA1B_RUN_DIR = "mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2"
    PYTHIA1B_FEATURES_ROOT = "attention_features_mimir_hardsplit"
    _resolve_adapter_dir = None

DEFAULT_RUN_DIR = PYTHIA1B_RUN_DIR
DEFAULT_OUTPUT_DIR = PYTHIA1B_FEATURES_ROOT

GROUP_KEYS = ["ft", "pt", "unseen"]
PAIRWISE_COMPARISONS = [
    ("ft_vs_pt", common.GROUP_SPECS["ft"]["group"], common.GROUP_SPECS["pt"]["group"]),
    ("ft_vs_unseen", common.GROUP_SPECS["ft"]["group"], common.GROUP_SPECS["unseen"]["group"]),
    ("pt_vs_unseen", common.GROUP_SPECS["pt"]["group"], common.GROUP_SPECS["unseen"]["group"]),
]


def resolve_adapter_dir(run_dir: str, adapter_dir: str | None) -> Path:
    if _resolve_adapter_dir is not None:
        try:
            return _resolve_adapter_dir(adapter_dir or run_dir, run_dir=run_dir)
        except FileNotFoundError:
            pass
    # Fallback: resolve via common path search
    candidates = []
    if adapter_dir:
        candidates.append(Path(adapter_dir))
    run = Path(run_dir)
    candidates.extend([run / "adapter", run])
    for candidate in candidates:
        try:
            return common.resolve_path(str(candidate))
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"LoRA adapter not found. run_dir={run_dir}, adapter_dir={adapter_dir}")


def load_all_samples(args) -> pd.DataFrame:
    targets_csv = str(getattr(args, "targets_csv", "") or "")
    if targets_csv:
        path = Path(targets_csv).expanduser()
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if not {"text", "group"}.issubset(df.columns):
            raise ValueError(f"{path} must contain text and group columns")
        requested = list(getattr(args, "target_groups", []) or [])
        if requested:
            missing = sorted(set(requested) - set(df["group"]))
            if missing:
                raise ValueError(f"Requested target groups absent from {path}: {missing}")
            df = df[df["group"].isin(requested)].copy()
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"].str.len() > 0].reset_index(drop=True)
        if args.n_per_group > 0:
            sampled = [
                part.sample(n=min(args.n_per_group, len(part)), random_state=args.seed)
                for _, part in df.groupby("group", sort=False)
            ]
            df = pd.concat(sampled, ignore_index=True) if sampled else df.iloc[0:0].copy()
        if "source" not in df.columns:
            df["source"] = "followup_targets"
        if "label" not in df.columns:
            if "ft_exposed" in df.columns:
                df["label"] = pd.to_numeric(df["ft_exposed"], errors="coerce").fillna(0).astype(int)
            else:
                df["label"] = 0
        else:
            df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        if "sample_id" in df.columns:
            df["source_sample_id"] = df["sample_id"].astype(str)
        df["global_sample_id"] = np.arange(len(df), dtype=int)
        return df
    parts = []
    group_keys = getattr(args, "groups", GROUP_KEYS)
    for group_key in group_keys:
        parts.append(common.load_group_samples(group_key, args.run_dir, args.n_per_group, args.seed))
    samples = pd.concat(parts, ignore_index=True)
    samples["global_sample_id"] = np.arange(len(samples), dtype=int)
    return samples


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_conditions(fixed_steps: Sequence[int]) -> List[Tuple[str, int | None]]:
    conditions: List[Tuple[str, int | None]] = []
    for step in fixed_steps:
        conditions.append((f"fixed_attention_{int(step)}", int(step)))
    return conditions


def make_conditions_from_args(args) -> List[Tuple[str, int | None]]:
    conditions: List[Tuple[str, int | None]] = []
    if args.run_dynamic:
        conditions.append(("dynamic_attention", None))
    for step in args.fixed_steps:
        conditions.append((f"fixed_attention_{int(step)}", int(step)))
    if not conditions:
        raise ValueError("No condition selected. Enable --run-dynamic or pass at least one --fixed-steps value.")
    return conditions


def make_query_protocols_from_args(args) -> List[Tuple[str, int, str, int]]:
    values = list(getattr(args, "query_protocol", []) or [])
    if not values:
        return [
            (
                "",
                int(getattr(args, "query_position_offset", 1)),
                str(getattr(args, "query_selection", "top_loss")),
                int(getattr(args, "topk_loss_percent", common.TOPK_LOSS_PERCENT)),
            )
        ]
    protocols: List[Tuple[str, int, str, int]] = []
    names = set()
    allowed = {"top_loss", "low_loss", "random", "all_valid", "gradient_logit"}
    for name, offset_raw, selection, rho_raw in values:
        offset, rho = int(offset_raw), int(rho_raw)
        if not name or name in names:
            raise ValueError(f"Query protocol names must be unique and non-empty: {name!r}")
        if offset not in {0, 1} or selection not in allowed or not 1 <= rho <= 100:
            raise ValueError(f"Invalid query protocol: {(name, offset, selection, rho)}")
        names.add(name)
        protocols.append((name, offset, selection, rho))
    return protocols


def train_accuracy_from_outputs(outputs, batch: Dict[str, torch.Tensor]) -> float:
    """Token accuracy on device (avoids host sync until the final scalar)."""
    try:
        with torch.no_grad():
            pred = outputs.logits[:, :-1, :].detach().argmax(dim=-1)
            labels = batch["labels"]
            shifted_labels = labels[:, 1:]
            mask = shifted_labels != -100
            denom = mask.sum()
            if int(denom.item()) == 0:
                return float("nan")
            correct = ((pred == shifted_labels) & mask).sum()
            return float((correct.float() / denom.float()).item())
    except Exception:
        return float("nan")


def overfit_fixed_steps(
    model,
    train_enc: Dict[str, torch.Tensor],
    steps: int,
    lr: float,
    *,
    record_train_curve: bool = False,
    use_amp: bool = True,
) -> Tuple[List[float], List[float], int]:
    """Sample-wise additional training for a fixed number of steps.

    By default only first/last loss are recorded (no per-step accuracy), which
    matches paper table needs for fixed-20 and is much faster.
    Uses bf16/fp16 autocast on CUDA when enabled.
    """
    from hardsplit.amp_utils import autocast_context, maybe_scaler

    model.train()
    batch = common.move_batch_to_device(train_enc, common.model_device(model))
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found. Check LoRA adapter loading.")

    optimizer = torch.optim.AdamW(params, lr=lr)
    scaler = maybe_scaler(use_amp)
    losses: List[float] = []
    accuracies: List[float] = []
    first_loss = float("nan")
    last_loss = float("nan")
    n_steps = int(steps)
    for step_i in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(use_amp):
            outputs = model(**batch, output_attentions=False, use_cache=False)
            loss = outputs.loss
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if record_train_curve:
            losses.append(float(loss.detach().float().item()))
            accuracies.append(train_accuracy_from_outputs(outputs, batch))
        else:
            # Minimal host sync: first and last loss only
            if step_i == 0:
                first_loss = float(loss.detach().float().item())
            if step_i == n_steps - 1:
                last_loss = float(loss.detach().float().item())

    if not record_train_curve:
        losses = [first_loss, last_loss]
        accuracies = [float("nan"), float("nan")]

    model.eval()
    return losses, accuracies, n_steps


def overfit_dynamic_steps(model, train_enc: Dict[str, torch.Tensor], args) -> Tuple[List[float], List[float], int]:
    from hardsplit.amp_utils import autocast_context, maybe_scaler

    model.train()
    batch = common.move_batch_to_device(train_enc, common.model_device(model))
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found. Check LoRA adapter loading.")

    optimizer = torch.optim.AdamW(params, lr=args.lr)
    use_amp = bool(getattr(args, "amp", True))
    scaler = maybe_scaler(use_amp)
    losses: List[float] = []
    accuracies: List[float] = []
    best_loss = float("inf")
    best_acc = -float("inf")
    patience_counter = 0
    step = 0
    record_curve = bool(getattr(args, "record_train_curve", False))
    use_acc_stop = bool(getattr(args, "early_stop_on_accuracy", True))

    while True:
        if step >= args.max_overfit_steps:
            break

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(use_amp):
            outputs = model(**batch, output_attentions=False, use_cache=False)
            loss = outputs.loss
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        cur_loss = float(loss.detach().float().item())
        # Accuracy only when needed for early-stop / curve logging
        if use_acc_stop or record_curve:
            cur_acc = train_accuracy_from_outputs(outputs, batch)
        else:
            cur_acc = float("nan")
        # Loss trajectory needed for early-stopping diagnostics
        losses.append(cur_loss)
        accuracies.append(cur_acc)

        improved = False
        if best_loss - cur_loss > args.early_stopping_tol:
            best_loss = cur_loss
            improved = True
        if not math.isnan(cur_acc) and cur_acc - best_acc > args.early_stopping_tol:
            best_acc = cur_acc
            improved = True

        patience_counter = 0 if improved else patience_counter + 1
        step += 1
        if step >= args.early_stopping_min_steps and patience_counter >= args.early_stopping_patience:
            break

    model.eval()
    return losses, accuracies, int(step)


def save_progress(output_dir: Path, raw_rows: List[Dict], sample_rows: List[Dict]) -> None:
    """Full rewrite (used for finalization / legacy callers)."""
    common.atomic_to_csv(pd.DataFrame(raw_rows), output_dir / "raw_experiment4_attention_shift.csv")
    common.atomic_to_csv(pd.DataFrame(sample_rows), output_dir / "sample_level_experiment4.csv")


def summarize_sample_attention(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()
    metric_cols = [m for m in common.ATTENTION_METRICS if m in raw_df.columns]
    return (
        raw_df.groupby(["condition", "sample_id", "group"], as_index=False)[metric_cols]
        .mean()
        .rename(columns={m: f"sample_mean_{m}" for m in metric_cols})
    )


def pivot_layer_head_features(raw_df: pd.DataFrame, condition: str, metric_cols: Sequence[str]) -> pd.DataFrame:
    df = raw_df[raw_df["condition"] == condition].copy()
    pieces = []
    base = df[["sample_id", "group"]].drop_duplicates().reset_index(drop=True)
    for metric in metric_cols:
        piv = df.pivot_table(index=["sample_id", "group"], columns=["layer", "head"], values=metric, aggfunc="mean")
        piv.columns = [f"{metric}_L{int(layer):02d}_H{int(head):02d}" for layer, head in piv.columns]
        pieces.append(piv.reset_index())
    out = base
    for piece in pieces:
        out = out.merge(piece, on=["sample_id", "group"], how="left")
    return out


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.10) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    ok = np.where(fpr <= target_fpr)[0]
    if len(ok) == 0:
        return 0.0
    return float(np.max(tpr[ok]))


def run_cv_classifier(df: pd.DataFrame, feature_cols: Sequence[str], positive_group: str, negative_group: str, seed: int) -> Dict | None:
    sub = df[df["group"].isin([positive_group, negative_group])].copy()
    sub = sub.dropna(subset=list(feature_cols))
    y = (sub["group"].to_numpy() == positive_group).astype(int)
    X = sub[list(feature_cols)].to_numpy(dtype=float)
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 2:
        return None

    n_splits = min(5, int(np.min(np.bincount(y))))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = np.zeros(len(y), dtype=float)
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced", random_state=seed),
        )
        clf.fit(X[train_idx], y[train_idx])
        score = clf.decision_function(X[test_idx])
        scores[test_idx] = score
        fold_rows.append({
            "fold": fold,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "auc": float(roc_auc_score(y[test_idx], score)),
            "auprc": float(average_precision_score(y[test_idx], score)),
            "tpr_at_10_fpr": tpr_at_fpr(y[test_idx], score, 0.10),
        })

    return {
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "n_features": int(len(feature_cols)),
        "auc": float(roc_auc_score(y, scores)),
        "auprc": float(average_precision_score(y, scores)),
        "tpr_at_10_fpr": tpr_at_fpr(y, scores, 0.10),
        "fold_auc_mean": float(np.mean([r["auc"] for r in fold_rows])),
        "fold_auc_std": float(np.std([r["auc"] for r in fold_rows], ddof=1)) if len(fold_rows) > 1 else 0.0,
        "fold_rows": fold_rows,
    }


def cliffs_delta(x: Iterable[float], y: Iterable[float]) -> float:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    gt = 0
    lt = 0
    for xv in x:
        gt += int(np.sum(xv > y))
        lt += int(np.sum(xv < y))
    return float((gt - lt) / (len(x) * len(y)))


def pairwise_tests(sample_df: pd.DataFrame, condition: str, metrics: Sequence[str]) -> pd.DataFrame:
    rows = []
    df = sample_df[sample_df["condition"] == condition].copy()
    for comp_name, positive, negative in PAIRWISE_COMPARISONS:
        for metric in metrics:
            x = df.loc[df["group"] == positive, metric].dropna().to_numpy(float)
            y = df.loc[df["group"] == negative, metric].dropna().to_numpy(float)
            if len(x) == 0 or len(y) == 0:
                continue
            _, p = mannwhitneyu(x, y, alternative="two-sided")
            rows.append({
                "condition": condition,
                "comparison": comp_name,
                "positive_group": positive,
                "negative_group": negative,
                "metric": metric,
                "positive_mean": float(np.mean(x)),
                "negative_mean": float(np.mean(y)),
                "p": float(p),
                "cliffs_delta": cliffs_delta(x, y),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        p = out["p"].to_numpy(float)
        if multipletests is not None:
            out["p_adj_bh"] = multipletests(p, method="fdr_bh")[1]
        else:
            # Benjamini--Hochberg fallback
            n = len(p)
            order = np.argsort(p)
            ranked = p[order]
            q = ranked * n / np.arange(1, n + 1)
            q = np.minimum.accumulate(q[::-1])[::-1]
            adj = np.empty(n, dtype=float)
            adj[order] = np.clip(q, 0, 1)
            out["p_adj_bh"] = adj
    return out


def make_plots(output_dir: Path, sample_feature_df: pd.DataFrame, classification_df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots_experiment4"
    plot_dir.mkdir(parents=True, exist_ok=True)

    metrics = [f"sample_mean_{m}" for m in ["l1_mean", "l2_rms", "js_div", "max_shift"] if f"sample_mean_{m}" in sample_feature_df.columns]
    group_order = [common.GROUP_SPECS[k]["group"] for k in GROUP_KEYS]
    for condition in sample_feature_df["condition"].dropna().unique():
        sub = sample_feature_df[sample_feature_df["condition"] == condition]
        for metric in metrics:
            data = [sub.loc[sub["group"] == g, metric].dropna().to_numpy(float) for g in group_order]
            fig, ax = plt.subplots(figsize=(7.2, 4.4))
            try:
                ax.boxplot(data, tick_labels=["FT", "PT", "Unseen"], showfliers=False)
            except TypeError:
                ax.boxplot(data, labels=["FT", "PT", "Unseen"], showfliers=False)
            ax.set_title(f"{condition}: {metric}")
            ax.set_ylabel(metric)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(plot_dir / f"{condition}_{metric}_boxplot.png", dpi=180)
            plt.close(fig)

    if not classification_df.empty:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        plot_df = classification_df[classification_df["comparison"].isin(["ft_vs_pt", "ft_vs_unseen"])].copy()
        plot_df = plot_df.sort_values(["comparison", "feature_set"])
        labels = [f"{r.comparison}\n{r.feature_set}" for r in plot_df.itertuples()]
        ax.bar(np.arange(len(plot_df)), plot_df["auc"].to_numpy(float), color="#4c78a8")
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(np.arange(len(plot_df)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("AUC")
        ax.set_title("Experiment 4 classification AUC")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "classification_auc_bar.png", dpi=180)
        plt.close(fig)


def analyze_outputs(output_dir: Path, seed: int) -> None:
    raw_path = output_dir / "raw_experiment4_attention_shift.csv"
    sample_path = output_dir / "sample_level_experiment4.csv"
    if not raw_path.exists() or not sample_path.exists():
        return

    raw_df = pd.read_csv(raw_path)
    sample_df = pd.read_csv(sample_path)
    metric_cols = [m for m in common.ATTENTION_METRICS if m in raw_df.columns]

    sample_attention = summarize_sample_attention(raw_df)
    sample_features = sample_df.merge(sample_attention, on=["condition", "sample_id", "group"], how="left")
    common.atomic_to_csv(sample_features, output_dir / "sample_level_experiment4_features.csv")

    tests = []
    sample_metrics = [f"sample_mean_{m}" for m in metric_cols] + [
        "before_loss",
        "after_loss",
        "delta_loss_before_minus_after",
        "overfit_steps",
    ]
    for condition in sample_features["condition"].dropna().unique():
        tests.append(pairwise_tests(sample_features, condition, [m for m in sample_metrics if m in sample_features.columns]))
    test_df = pd.concat(tests, ignore_index=True) if tests else pd.DataFrame()
    common.atomic_to_csv(test_df, output_dir / "experiment4_pairwise_tests.csv")

    class_rows = []
    fold_rows_all = []
    conditions = sorted(raw_df["condition"].dropna().unique())
    for condition in conditions:
        layer_head = pivot_layer_head_features(raw_df, condition, metric_cols)
        summary_cols = [f"sample_mean_{m}" for m in metric_cols if f"sample_mean_{m}" in sample_features.columns]
        summary_df = sample_features[sample_features["condition"] == condition][["sample_id", "group"] + summary_cols].copy()
        feature_sets = [
            ("attention_summary", summary_df, summary_cols),
            ("attention_layer_head_all", layer_head, [c for c in layer_head.columns if c not in {"sample_id", "group"}]),
        ]
        for comp_name, positive, negative in PAIRWISE_COMPARISONS:
            for feature_set, df, cols in feature_sets:
                if not cols:
                    continue
                result = run_cv_classifier(df, cols, positive, negative, seed)
                if result is None:
                    continue
                row = {
                    "condition": condition,
                    "comparison": comp_name,
                    "feature_set": feature_set,
                    "positive_group": positive,
                    "negative_group": negative,
                    **{k: v for k, v in result.items() if k != "fold_rows"},
                }
                class_rows.append(row)
                for fr in result["fold_rows"]:
                    fold_rows_all.append({"condition": condition, "comparison": comp_name, "feature_set": feature_set, **fr})

    dynamic_steps = sample_features[sample_features["condition"] == "dynamic_attention"][["sample_id", "group", "overfit_steps"]].copy()
    if not dynamic_steps.empty:
        for comp_name, positive, negative in PAIRWISE_COMPARISONS:
            result = run_cv_classifier(dynamic_steps, ["overfit_steps"], positive, negative, seed)
            if result is None:
                continue
            class_rows.append({
                "condition": "dynamic_attention",
                "comparison": comp_name,
                "feature_set": "dynamic_steps_only",
                "positive_group": positive,
                "negative_group": negative,
                **{k: v for k, v in result.items() if k != "fold_rows"},
            })
            for fr in result["fold_rows"]:
                fold_rows_all.append({"condition": "dynamic_attention", "comparison": comp_name, "feature_set": "dynamic_steps_only", **fr})

    classification_df = pd.DataFrame(class_rows)
    fold_df = pd.DataFrame(fold_rows_all)
    common.atomic_to_csv(classification_df, output_dir / "experiment4_classification_results.csv")
    common.atomic_to_csv(fold_df, output_dir / "experiment4_classification_fold_results.csv")
    make_plots(output_dir, sample_features, classification_df)

    with (output_dir / "experiment4_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Experiment 4: stopping-condition ablation\n")
        f.write(f"raw_rows={len(raw_df)}\n")
        f.write(f"sample_rows={len(sample_df)}\n\n")
        if not classification_df.empty:
            show = classification_df.sort_values(["comparison", "condition", "feature_set"])
            f.write(show[["condition", "comparison", "feature_set", "n_positive", "n_negative", "n_features", "auc", "auprc", "tpr_at_10_fpr"]].to_string(index=False))
            f.write("\n")


def load_existing_progress(output_dir: Path) -> Tuple[int, int, set]:
    """Sanitize partial CSVs and return row counts plus finished sample keys.

    A sample is committed only when its sample-level row exists. Raw attention
    conditions may carry a ``__QUERY_PROTOCOL`` suffix, so their base condition
    is matched against the committed sample row. This removes orphan rows left
    by an interruption between raw and sample-level appends.
    """
    raw_path = output_dir / "raw_experiment4_attention_shift.csv"
    sample_path = output_dir / "sample_level_experiment4.csv"
    done: set = set()
    sample_count = 0
    if sample_path.exists() and sample_path.stat().st_size > 1:
        sample_df = pd.read_csv(sample_path)
        if {"condition", "sample_id"}.issubset(sample_df.columns):
            sample_df = sample_df.drop_duplicates(["condition", "sample_id"], keep="last").reset_index(drop=True)
            done = set(zip(sample_df["condition"].astype(str), sample_df["sample_id"].astype(int)))
            common.atomic_to_csv(sample_df, sample_path)
            sample_count = len(sample_df)
    raw_count = 0
    if raw_path.exists() and raw_path.stat().st_size > 1:
        raw_df = pd.read_csv(raw_path)
        if done and {"condition", "sample_id"}.issubset(raw_df.columns):
            cond = raw_df["condition"].astype(str).str.split("__", n=1).str[0]
            sid = raw_df["sample_id"].astype(int)
            keep = [(c, s) in done for c, s in zip(cond.tolist(), sid.tolist())]
            raw_df = raw_df.loc[keep].drop_duplicates().reset_index(drop=True)
        else:
            raw_df = raw_df.iloc[0:0].copy()
        common.atomic_to_csv(raw_df, raw_path)
        raw_count = len(raw_df)

    update_path = output_dir / "raw_update_baseline_features.csv"
    if update_path.exists() and update_path.stat().st_size > 1:
        update_df = pd.read_csv(update_path)
        if done and {"condition", "sample_id"}.issubset(update_df.columns):
            cond = update_df["condition"].astype(str)
            sid = update_df["sample_id"].astype(int)
            keep = [(c, s) in done for c, s in zip(cond.tolist(), sid.tolist())]
            update_df = update_df.loc[keep].drop_duplicates().reset_index(drop=True)
        else:
            update_df = update_df.iloc[0:0].copy()
        common.atomic_to_csv(update_df, update_path)
    return raw_count, sample_count, done


def run_extraction(args) -> None:
    from hardsplit.amp_utils import enable_tf32
    from hardsplit.progress import ExtractProgressStore, parse_shard_spec

    enable_tf32()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    common.set_seed(args.seed)

    adapter_dir = resolve_adapter_dir(args.run_dir, args.adapter_dir)
    model_name = common.get_model_name(model_name=getattr(args, "model_name", None), adapter_dir=str(adapter_dir))
    tokenizer = common.load_tokenizer(model_name=model_name, adapter_dir=str(adapter_dir))
    samples = load_all_samples(args)

    # Optional sample sharding for multi-GPU: keep only indices i % N == K
    shard_index, shard_total = parse_shard_spec(getattr(args, "shard", "") or "")
    if shard_total > 1:
        keep = [i for i in range(len(samples)) if i % shard_total == shard_index]
        samples = samples.iloc[keep].reset_index(drop=True)
        print(f"Shard {shard_index}/{shard_total}: keeping {len(samples)} samples")

    common.atomic_to_csv(samples, output_dir / "experiment4_target_samples.csv")

    conditions = make_conditions_from_args(args)
    if args.resume:
        raw_row_count, sample_row_count, done_keys = load_existing_progress(output_dir)
    else:
        raw_row_count, sample_row_count, done_keys = 0, 0, set()
        # Fresh run: truncate progress files so incremental append starts clean
        for name in (
            "raw_experiment4_attention_shift.csv",
            "sample_level_experiment4.csv",
            "raw_update_baseline_features.csv",
        ):
            p = output_dir / name
            if p.exists():
                p.unlink()
    # Claim output_dir before loading the (expensive) model so a colliding
    # host fails fast instead of after paying for the model load.
    progress = ExtractProgressStore(output_dir)

    config = vars(args).copy()
    config["resolved_adapter_dir"] = str(adapter_dir)
    config["resolved_model_name"] = model_name
    config["conditions"] = [name for name, _ in conditions]
    config["additional_training_lr"] = float(args.lr)
    config["resume"] = bool(args.resume)
    config["shard"] = f"{shard_index}/{shard_total}"
    config["amp"] = bool(getattr(args, "amp", True))
    adapter_config_path = adapter_dir / "adapter_config.json"
    config["adapter_config_sha256"] = sha256_file(adapter_config_path) if adapter_config_path.exists() else ""
    targets_csv = Path(args.targets_csv).expanduser() if getattr(args, "targets_csv", "") else None
    config["targets_csv_sha256"] = sha256_file(targets_csv) if targets_csv and targets_csv.exists() else ""
    (output_dir / "experiment4_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Target samples:")
    print(samples["group"].value_counts().to_string())
    print(f"Model: {model_name}")
    print(f"Adapter: {adapter_dir}")
    print(f"Output: {output_dir}")
    print(f"Resume: {args.resume}, already_done={len(done_keys)}")
    print(f"Dynamic: lr={args.lr}, patience={args.early_stopping_patience}, max_steps={args.max_overfit_steps}")
    print(f"Fixed steps: {args.fixed_steps}")
    print(f"AMP: {config['amp']}, shard={config['shard']}")

    # Load once; restore LoRA weights after each sample (paper: independent per-sample updates)
    model = common.load_before_model_trainable(str(adapter_dir), model_name=model_name)
    adapter_state = common.snapshot_trainable_state(model)
    # progress (ExtractProgressStore) was already constructed above, before model load.
    # If resuming with existing rows already loaded into memory lists for done_keys,
    # new rows are append-only via progress store.
    processed_since_gc = 0
    record_curve = bool(getattr(args, "record_train_curve", False))
    record_update_baselines = bool(getattr(args, "record_update_baselines", False))
    query_protocols = make_query_protocols_from_args(args)
    update_baseline_path = output_dir / "raw_update_baseline_features.csv"
    use_amp = bool(getattr(args, "amp", True))
    status_every = max(1, int(getattr(args, "flush_every", 25)))
    newly_done = 0

    for condition, fixed_step in conditions:
        for row in tqdm(samples.itertuples(index=False), total=len(samples), desc=condition):
            sample_id = int(row.global_sample_id)
            key = (condition, sample_id)
            if key in done_keys:
                continue
            text = str(row.text)
            if getattr(args, "verbose_samples", False):
                print(f"\n=== condition={condition}, sample_id={sample_id}, group={row.group} ===")

            # Deterministic per-sample RNG (dropout during train, etc.)
            common.set_seed(int(args.seed) + int(sample_id))
            common.restore_trainable_state(model, adapter_state)
            train_enc = common.make_lm_encoding(tokenizer, text)
            # One forward: before loss + token losses + attention snapshot.
            before_loss, token_losses, token_mask, attn_before, attn_mask = (
                common.compute_diagnostics_and_attentions(model, train_enc)
            )
            gradient_scores = None
            if any(selection == "gradient_logit" for _, _, selection, _ in query_protocols):
                gradient_scores, _ = common.compute_token_logit_gradient_norms(model, train_enc)
            selected_by_protocol = {}
            for protocol_name, query_position_offset, query_selection, topk_percent in query_protocols:
                selection_scores = gradient_scores if query_selection == "gradient_logit" else token_losses
                selection_mode = "top_loss" if query_selection == "gradient_logit" else query_selection
                selected_by_protocol[protocol_name] = common.select_query_positions(
                    selection_scores,
                    token_mask,
                    topk_percent,
                    query_position_offset=query_position_offset,
                    selection_mode=selection_mode,
                    random_seed=int(args.seed) + sample_id,
                )

            update_rows = []
            initial_gradients = {}
            if record_update_baselines:
                from reviewer_followup.update_features import initial_gradient_features

                update_rows, initial_gradients = initial_gradient_features(
                    model,
                    train_enc,
                    model_device_fn=common.model_device,
                    move_batch_fn=common.move_batch_to_device,
                )

            if fixed_step is None:
                losses, accuracies, actual_steps = overfit_dynamic_steps(model, train_enc, args)
                gradient_curve = []
            elif record_update_baselines:
                from reviewer_followup.update_features import overfit_fixed_steps_with_gradient_curve

                losses, accuracies, actual_steps, gradient_curve = overfit_fixed_steps_with_gradient_curve(
                    model,
                    train_enc,
                    steps=fixed_step,
                    lr=args.lr,
                    model_device_fn=common.model_device,
                    move_batch_fn=common.move_batch_to_device,
                    use_amp=use_amp,
                )
            else:
                losses, accuracies, actual_steps = overfit_fixed_steps(
                    model,
                    train_enc,
                    fixed_step,
                    args.lr,
                    record_train_curve=record_curve,
                    use_amp=use_amp,
                )
                gradient_curve = []

            if record_update_baselines:
                from reviewer_followup.common import append_csv_rows
                from reviewer_followup.update_features import curve_summary, parameter_delta_features

                update_rows.extend(parameter_delta_features(model, adapter_state, initial_gradients))
                metadata = {
                    "condition": condition,
                    "sample_id": sample_id,
                    "group": row.group,
                    "source": row.source,
                }
                for update_row in update_rows:
                    update_row.update(metadata)
                append_csv_rows(update_baseline_path, update_rows)
                curve_fields = {
                    **curve_summary(losses, "train_loss_curve"),
                    **curve_summary(gradient_curve, "train_gradient_curve"),
                }
            else:
                curve_fields = {}

            # One forward: after loss + attention snapshot.
            after_loss, attn_after, _ = common.compute_sequence_loss_and_attentions(model, train_enc)
            metric_rows = []
            for protocol_name, query_position_offset, query_selection, topk_percent in query_protocols:
                protocol_rows = common.attention_shift_metrics(
                    attn_before,
                    attn_after,
                    selected_by_protocol[protocol_name],
                    attention_mask=attn_mask,
                )
                output_condition = f"{condition}__{protocol_name}" if protocol_name else condition
                for mr in protocol_rows:
                    mr.update({
                        "condition": output_condition,
                        "query_protocol": protocol_name or "default",
                        "sample_id": sample_id,
                        "group": row.group,
                        "source": row.source,
                        "label": int(row.label),
                        "before_loss": before_loss,
                        "after_loss": after_loss,
                        "delta_loss_before_minus_after": before_loss - after_loss,
                        "train_loss_first": losses[0] if losses else np.nan,
                        "train_loss_last": losses[-1] if losses else np.nan,
                        "train_acc_first": accuracies[0] if accuracies else np.nan,
                        "train_acc_last": accuracies[-1] if accuracies else np.nan,
                        "overfit_steps": int(actual_steps),
                        "fixed_steps": int(fixed_step) if fixed_step is not None else np.nan,
                        "topk_loss_percent": topk_percent,
                        "query_position_offset": query_position_offset,
                        "query_selection": query_selection,
                    })
                metric_rows.extend(protocol_rows)

            first_name, query_position_offset, query_selection, topk_percent = query_protocols[0]
            selected_positions = selected_by_protocol[first_name]

            sample_row = {
                "condition": condition,
                "sample_id": sample_id,
                "group": row.group,
                "source": row.source,
                "label": int(row.label),
                "before_loss": before_loss,
                "after_loss": after_loss,
                "delta_loss_before_minus_after": before_loss - after_loss,
                "train_loss_first": losses[0] if losses else np.nan,
                "train_loss_last": losses[-1] if losses else np.nan,
                "train_acc_first": accuracies[0] if accuracies else np.nan,
                "train_acc_last": accuracies[-1] if accuracies else np.nan,
                "num_topk_loss_queries": len(selected_positions),
                "analysis_text_char_len": len(text),
                "overfit_steps": int(actual_steps),
                "fixed_steps": int(fixed_step) if fixed_step is not None else np.nan,
                "topk_loss_percent": topk_percent,
                "query_position_offset": query_position_offset,
                "query_selection": query_selection,
                "query_protocol_count": len(query_protocols),
                **curve_fields,
            }
            # Incremental disk write (no full-table rewrite)
            progress.append_sample(metric_rows, sample_row)
            raw_row_count += len(metric_rows)
            sample_row_count += 1
            done_keys.add(key)

            del attn_before, attn_after
            processed_since_gc += 1
            newly_done += 1
            if torch.cuda.is_available() and processed_since_gc % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            if newly_done % status_every == 0:
                progress.flush_status(
                    "running\n"
                    f"done_keys={len(done_keys)}\n"
                    f"raw_attention_rows≈{raw_row_count}\n"
                    f"shard={shard_index}/{shard_total}\n"
                )

    if not args.skip_analyze:
        try:
            analyze_outputs(output_dir, args.seed)
            progress.flush_status("completed\n")
        except Exception as exc:
            progress.flush_status(
                "extraction_completed_analyze_failed\n"
                f"error={exc}\n"
                f"shard={shard_index}/{shard_total}\n"
            )
            print(f"Analysis failed after successful extraction: {exc}")
    else:
        progress.flush_status(
            "extraction_completed_skip_analyze\n"
            f"raw_attention_rows={raw_row_count}\n"
            f"sample_rows={sample_row_count}\n"
            f"shard={shard_index}/{shard_total}\n"
        )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 4 stopping-condition ablation on MIMIR hard split.")
    parser.add_argument("--run-dir", default=os.environ.get("MIMIR_HARDSPLIT_RUN_DIR", DEFAULT_RUN_DIR))
    parser.add_argument("--adapter-dir", default=os.environ.get("MIMIR_HARDSPLIT_ADAPTER_DIR"))
    parser.add_argument(
        "--model-name",
        default=os.environ.get("BASE_MODEL_NAME", ""),
        help="HF model id or preset key (pythia-1b / pythia-410m / gpt-neo-2.7b). "
        "If empty, inferred from adapter_config.json.",
    )
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-per-group", type=int, default=int(os.environ.get("N_PER_GROUP", "500")))
    parser.add_argument(
        "--targets-csv",
        default="",
        help="Generic target CSV with text/group columns; enables crossed and controlled-stage experiments.",
    )
    parser.add_argument(
        "--target-groups",
        nargs="*",
        default=[],
        help="Optional group values to keep from --targets-csv.",
    )
    parser.add_argument("--fixed-steps", type=int, nargs="*", default=[20, 50, 100])
    parser.add_argument("--run-dynamic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--groups", nargs="+", choices=GROUP_KEYS, default=GROUP_KEYS)
    parser.add_argument(
        "--query-position-offset",
        type=int,
        choices=[0, 1],
        default=int(os.environ.get("QUERY_POSITION_OFFSET", "1")),
        help="Map selected next-token loss index t to attention query t+offset. "
        "Default 1 is the paper condition; 0 is the predictor-state ablation.",
    )
    parser.add_argument(
        "--query-selection",
        choices=["top_loss", "low_loss", "random", "all_valid", "gradient_logit"],
        default=os.environ.get("QUERY_SELECTION", "top_loss"),
        help="Query selection control for ablations; default top_loss matches the paper.",
    )
    parser.add_argument(
        "--topk-loss-percent",
        type=int,
        choices=range(1, 101),
        default=int(os.environ.get("TOPK_LOSS_PERCENT", "10")),
        metavar="PCT",
        help="Percentage used by top/low/random/gradient token selection.",
    )
    parser.add_argument(
        "--query-protocol",
        nargs=4,
        action="append",
        metavar=("NAME", "OFFSET", "SELECTION", "RHO"),
        default=[],
        help="Evaluate multiple query protocols from the same attention before/after pair; may be repeated.",
    )
    # Sample-wise additional training lr = 1e-5 (not LoRA FT lr 1e-4)
    parser.add_argument("--lr", type=float, default=float(os.environ.get("OVERFIT_LR", "1e-5")))
    parser.add_argument("--early-stopping-patience", type=int, default=int(os.environ.get("EARLY_STOPPING_PATIENCE", "50")))
    parser.add_argument("--early-stopping-tol", type=float, default=float(os.environ.get("EARLY_STOPPING_TOL", "1e-6")))
    parser.add_argument("--early-stopping-min-steps", type=int, default=int(os.environ.get("EARLY_STOPPING_MIN_STEPS", "1")))
    parser.add_argument("--max-overfit-steps", type=int, default=int(os.environ.get("MAX_OVERFIT_STEPS", "5000")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    parser.add_argument("--analyze-only", action="store_true", help="Skip extraction and recompute CSV summaries/classification from existing raw outputs.")
    parser.add_argument("--skip-analyze", action="store_true", help="Run extraction only; skip post-hoc classification/plots.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Resume from partial CSVs (default: on).")
    parser.add_argument(
        "--flush-every",
        type=int,
        default=int(os.environ.get("FLUSH_EVERY", "25")),
        help="Update run_status.txt every N newly finished samples (CSVs append each sample).",
    )
    parser.add_argument(
        "--record-train-curve",
        action="store_true",
        help="Record per-step train loss/accuracy during fixed-step overfit (slower).",
    )
    parser.add_argument(
        "--record-update-baselines",
        action="store_true",
        help="Record matched pre-update gradient, LoRA parameter-delta, and gradient-trajectory features.",
    )
    parser.add_argument(
        "--verbose-samples",
        action="store_true",
        help="Print a line for every sample (default: quiet tqdm only).",
    )
    parser.add_argument(
        "--early-stop-on-accuracy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dynamic early-stop also watches train accuracy (default: on).",
    )
    parser.add_argument(
        "--shard",
        default=os.environ.get("EXTRACT_SHARD", ""),
        help="Sample shard K/N (0-based), e.g. 0/4 keeps indices i%%4==0.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("EXTRACT_AMP", "1") not in {"0", "false", "False", "no"},
        help="Use CUDA autocast (bf16/fp16) during additional training (default: on).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.analyze_only:
        analyze_outputs(Path(args.output_dir), args.seed)
        (Path(args.output_dir) / "run_status.txt").write_text("completed\n", encoding="utf-8")
    else:
        run_extraction(args)


if __name__ == "__main__":
    main()
