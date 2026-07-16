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

import argparse
import gc
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
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

import mimir_hardsplit_attention_common as common


DEFAULT_RUN_DIR = "results/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2"
DEFAULT_OUTPUT_DIR = "results/experiment4_mimir_hardsplit_stopping_condition"

GROUP_KEYS = ["ft", "pt", "unseen"]
PAIRWISE_COMPARISONS = [
    ("ft_vs_pt", common.GROUP_SPECS["ft"]["group"], common.GROUP_SPECS["pt"]["group"]),
    ("ft_vs_unseen", common.GROUP_SPECS["ft"]["group"], common.GROUP_SPECS["unseen"]["group"]),
    ("pt_vs_unseen", common.GROUP_SPECS["pt"]["group"], common.GROUP_SPECS["unseen"]["group"]),
]


def resolve_adapter_dir(run_dir: str, adapter_dir: str | None) -> Path:
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
    parts = []
    group_keys = getattr(args, "groups", GROUP_KEYS)
    for group_key in group_keys:
        parts.append(common.load_group_samples(group_key, args.run_dir, args.n_per_group, args.seed))
    samples = pd.concat(parts, ignore_index=True)
    samples["global_sample_id"] = np.arange(len(samples), dtype=int)
    return samples


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


def train_accuracy_from_outputs(outputs, batch: Dict[str, torch.Tensor]) -> float:
    try:
        with torch.no_grad():
            pred = outputs.logits[:, :-1, :].detach().cpu().argmax(dim=-1)
            labels = batch["labels"].detach().cpu()
            shifted_labels = labels[:, 1:]
            mask = shifted_labels != -100
            if mask.sum().item() == 0:
                return float("nan")
            return float(((pred == shifted_labels) & mask).sum().item()) / float(mask.sum().item())
    except Exception:
        return float("nan")


def overfit_fixed_steps(model, train_enc: Dict[str, torch.Tensor], steps: int, lr: float) -> Tuple[List[float], List[float], int]:
    model.train()
    batch = common.move_batch_to_device(train_enc, common.model_device(model))
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found. Check LoRA adapter loading.")

    optimizer = torch.optim.AdamW(params, lr=lr)
    losses: List[float] = []
    accuracies: List[float] = []
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch, output_attentions=False, use_cache=False)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().float().cpu()))
        accuracies.append(train_accuracy_from_outputs(outputs, batch))

    model.eval()
    return losses, accuracies, int(steps)


def overfit_dynamic_steps(model, train_enc: Dict[str, torch.Tensor], args) -> Tuple[List[float], List[float], int]:
    model.train()
    batch = common.move_batch_to_device(train_enc, common.model_device(model))
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found. Check LoRA adapter loading.")

    optimizer = torch.optim.AdamW(params, lr=args.lr)
    losses: List[float] = []
    accuracies: List[float] = []
    best_loss = float("inf")
    best_acc = -float("inf")
    patience_counter = 0
    step = 0

    while True:
        if step >= args.max_overfit_steps:
            break

        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch, output_attentions=False, use_cache=False)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        cur_loss = float(loss.detach().float().cpu())
        cur_acc = train_accuracy_from_outputs(outputs, batch)
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
        out["p_adj_bh"] = multipletests(out["p"].to_numpy(float), method="fdr_bh")[1]
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


def run_extraction(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    common.set_seed(args.seed)

    adapter_dir = resolve_adapter_dir(args.run_dir, args.adapter_dir)
    tokenizer = common.load_tokenizer()
    samples = load_all_samples(args)
    common.atomic_to_csv(samples, output_dir / "experiment4_target_samples.csv")

    conditions = make_conditions_from_args(args)
    raw_rows: List[Dict] = []
    sample_rows: List[Dict] = []

    config = vars(args).copy()
    config["resolved_adapter_dir"] = str(adapter_dir)
    config["conditions"] = [name for name, _ in conditions]
    (output_dir / "experiment4_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Target samples:")
    print(samples["group"].value_counts().to_string())
    print(f"Adapter: {adapter_dir}")
    print(f"Output: {output_dir}")
    print(f"Dynamic: lr={args.lr}, patience={args.early_stopping_patience}, max_steps={args.max_overfit_steps}")
    print(f"Fixed steps: {args.fixed_steps}")

    for condition, fixed_step in conditions:
        for row in tqdm(samples.itertuples(index=False), total=len(samples), desc=condition):
            sample_id = int(row.global_sample_id)
            text = str(row.text)
            print(f"\n=== condition={condition}, sample_id={sample_id}, group={row.group} ===")

            model = common.load_before_model_trainable(str(adapter_dir))
            train_enc = common.make_lm_encoding(tokenizer, text)
            before_loss = common.compute_sequence_loss(model, train_enc)
            token_losses, token_mask = common.compute_token_losses(model, tokenizer, text)
            selected_positions = common.select_topk_query_positions(token_losses, token_mask, common.TOPK_LOSS_PERCENT)
            attention_key_mask = common.get_attention_key_mask(tokenizer, text)
            attn_before = common.get_attentions(model, tokenizer, text)

            if fixed_step is None:
                losses, accuracies, actual_steps = overfit_dynamic_steps(model, train_enc, args)
            else:
                losses, accuracies, actual_steps = overfit_fixed_steps(model, train_enc, fixed_step, args.lr)

            after_loss = common.compute_sequence_loss(model, train_enc)
            attn_after = common.get_attentions(model, tokenizer, text)
            metric_rows = common.attention_shift_metrics(
                attn_before,
                attn_after,
                selected_positions,
                attention_key_mask,
            )

            for mr in metric_rows:
                mr.update({
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
                    "overfit_steps": int(actual_steps),
                    "fixed_steps": int(fixed_step) if fixed_step is not None else np.nan,
                    "topk_loss_percent": common.TOPK_LOSS_PERCENT,
                })
            raw_rows.extend(metric_rows)

            sample_rows.append({
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
            })

            del model, attn_before, attn_after
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            save_progress(output_dir, raw_rows, sample_rows)
            status = pd.DataFrame(sample_rows).groupby(["condition", "group"]).size().to_dict()
            (output_dir / "run_status.txt").write_text(
                "running\n"
                f"processed_sample_counts={status}\n"
                f"raw_attention_rows={len(raw_rows)}\n",
                encoding="utf-8",
            )

    save_progress(output_dir, raw_rows, sample_rows)
    analyze_outputs(output_dir, args.seed)
    (output_dir / "run_status.txt").write_text("completed\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 4 stopping-condition ablation on MIMIR hard split.")
    parser.add_argument("--run-dir", default=os.environ.get("MIMIR_HARDSPLIT_RUN_DIR", DEFAULT_RUN_DIR))
    parser.add_argument("--adapter-dir", default=os.environ.get("MIMIR_HARDSPLIT_ADAPTER_DIR"))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-per-group", type=int, default=int(os.environ.get("N_PER_GROUP", "500")))
    parser.add_argument("--fixed-steps", type=int, nargs="*", default=[20, 50, 100])
    parser.add_argument("--run-dynamic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--groups", nargs="+", choices=GROUP_KEYS, default=GROUP_KEYS)
    parser.add_argument("--lr", type=float, default=float(os.environ.get("OVERFIT_LR", "1e-5")))
    parser.add_argument("--early-stopping-patience", type=int, default=int(os.environ.get("EARLY_STOPPING_PATIENCE", "50")))
    parser.add_argument("--early-stopping-tol", type=float, default=float(os.environ.get("EARLY_STOPPING_TOL", "1e-6")))
    parser.add_argument("--early-stopping-min-steps", type=int, default=int(os.environ.get("EARLY_STOPPING_MIN_STEPS", "1")))
    parser.add_argument("--max-overfit-steps", type=int, default=int(os.environ.get("MAX_OVERFIT_STEPS", "5000")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    parser.add_argument("--analyze-only", action="store_true", help="Skip extraction and recompute CSV summaries/classification from existing raw outputs.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.analyze_only:
        analyze_outputs(Path(args.output_dir), args.seed)
    else:
        run_extraction(args)


if __name__ == "__main__":
    main()
