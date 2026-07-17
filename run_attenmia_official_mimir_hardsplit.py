# -*- coding: utf-8 -*-
"""
Official-style AttenMIA for the MIMIR Wikipedia hard-split experiment.

Paper target:
  AttenMIA: LLM Membership Inference Attack through Attention Signals

This standalone script evaluates the MIMIR hard split created by
`train_mimir_wikipedia_hardsplit_lora.py`:
  PT      = MIMIR Wikipedia member
  FT      = MIMIR Wikipedia nonmember used for LoRA FT
  Unseen  = MIMIR Wikipedia nonmember not used for LoRA FT

Default comparisons:
  - FT vs PT
  - FT vs Unseen
  - PT vs Unseen

Official-style implementation details:
  - transitional attention features across adjacent layers:
      correlation, Frobenius distance, KL divergence, barycenter mean/variance
  - perturbation features:
      token drop, token replace, prefix insertion
  - lightweight MLP classifier
  - 5-fold Stratified CV

To avoid test-fold leakage while keeping the paper's "unrelated non-member
prefix" idea, prefix pools are built from each fold's training negative group.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple  # noqa: F401

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    from model_registry import (
        DEFAULT_ATTENMIA_DIR,
        DEFAULT_HF_ID,
        PYTHIA1B_RUN_DIR,
        PYTHIA1B_RUN_DIR_RESULTS,
        add_model_arguments,
        apply_model_namespace,
        resolve_adapter_dir as registry_resolve_adapter_dir,
        resolve_from_args,
    )
except ImportError:  # pragma: no cover
    DEFAULT_HF_ID = "EleutherAI/pythia-1b"
    DEFAULT_ATTENMIA_DIR = "results/attenmia_official_mimir_hardsplit"
    PYTHIA1B_RUN_DIR = "mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2"
    PYTHIA1B_RUN_DIR_RESULTS = f"results/{PYTHIA1B_RUN_DIR}"
    add_model_arguments = None
    apply_model_namespace = None
    registry_resolve_adapter_dir = None
    resolve_from_args = None

BASE_MODEL_NAME = DEFAULT_HF_ID
DEFAULT_RUN_DIR = PYTHIA1B_RUN_DIR_RESULTS
DEFAULT_OUTPUT_DIR = DEFAULT_ATTENMIA_DIR
SEED = 42
EPS = 1e-12

GROUP_PT = "mimir_wikipedia_member_pt"
GROUP_FT = "mimir_wikipedia_nonmember_ft"
GROUP_UNSEEN = "mimir_wikipedia_nonmember_unseen"


@dataclass
class ComparisonSpec:
    name: str
    positive_group: str
    negative_group: str


DEFAULT_COMPARISONS = [
    ComparisonSpec("ft_vs_pt", GROUP_FT, GROUP_PT),
    ComparisonSpec("ft_vs_unseen", GROUP_FT, GROUP_UNSEEN),
    ComparisonSpec("pt_vs_unseen", GROUP_PT, GROUP_UNSEEN),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_existing_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.exists():
        return path
    local_candidates = [
        Path(path_like.replace("results/", "")),
        Path(path_like.replace("results/", "")),
        Path(path.name),
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Path not found: {path_like}")


def resolve_adapter_dir(path_like: str) -> Path:
    if registry_resolve_adapter_dir is not None:
        try:
            return registry_resolve_adapter_dir(path_like)
        except FileNotFoundError:
            pass
    path = resolve_existing_path(path_like)
    if (path / "adapter_config.json").exists():
        return path
    if (path / "adapter" / "adapter_config.json").exists():
        return path / "adapter"
    raise FileNotFoundError(
        f"LoRA adapter_config.json not found in {path} or {path / 'adapter'}"
    )


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def model_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_attention_model(args):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    adapter_dir = resolve_adapter_dir(args.adapter_dir)
    if resolve_from_args is not None:
        model_name = resolve_from_args(args, adapter_dir=adapter_dir, default=BASE_MODEL_NAME)
    else:
        model_name = args.model_name or BASE_MODEL_NAME
    args.model_name = model_name
    print(f"Loading base model: {model_name}")
    print(f"Loading adapter: {adapter_dir}")

    tokenizer = load_tokenizer(model_name)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    kwargs = {"output_attentions": True}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            attn_implementation="eager",
            **kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            **kwargs,
        )

    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    model.config.output_attentions = True
    model.config.use_cache = False
    return model, tokenizer, adapter_dir


@torch.no_grad()
def get_attentions(model, tokenizer, text: str, args) -> List[np.ndarray]:
    device = model_device(model)
    enc = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
        padding=False,
    ).to(device)
    outputs = model(**enc, output_attentions=True, use_cache=False)
    if outputs.attentions is None:
        raise RuntimeError("Model did not return attentions.")

    attns = []
    for a in outputs.attentions:
        arr = a[0].detach().float().cpu().numpy()
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        row_sum = arr.sum(axis=-1, keepdims=True)
        arr = arr / np.maximum(row_sum, EPS)
        attns.append(arr)
    return attns


def row_kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0)
    q = np.clip(q, EPS, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


def kl_to_uniform(a: np.ndarray) -> np.ndarray:
    t = a.shape[-1]
    u = np.full_like(a, 1.0 / max(1, t))
    return row_kl(a, u)


def barycenter(a: np.ndarray) -> np.ndarray:
    t = a.shape[-1]
    positions = np.arange(t, dtype=np.float64)
    return np.sum(a * positions[None, None, :], axis=-1)


def transitional_features(attns: List[np.ndarray], prefix: str = "trans") -> Dict[str, float]:
    feats: Dict[str, float] = {}
    for layer in range(len(attns) - 1):
        a0 = attns[layer]
        a1 = attns[layer + 1]
        heads = min(a0.shape[0], a1.shape[0])
        t = min(a0.shape[-1], a1.shape[-1])
        a0 = a0[:heads, :t, :t]
        a1 = a1[:heads, :t, :t]
        c0 = barycenter(a0)
        c1 = barycenter(a1)
        drift = np.abs(c1 - c0)
        for h in range(heads):
            v0 = a0[h].reshape(-1)
            v1 = a1[h].reshape(-1)
            corr = 0.0 if np.std(v0) < EPS or np.std(v1) < EPS else float(np.corrcoef(v0, v1)[0, 1])
            feats[f"{prefix}_corr_l{layer}_h{h}"] = corr
            feats[f"{prefix}_frob_l{layer}_h{h}"] = float(np.linalg.norm(a1[h] - a0[h], ord="fro") / max(1, t * t))
            feats[f"{prefix}_kl_l{layer}_h{h}"] = float(row_kl(a0[h], a1[h]).mean())
            feats[f"{prefix}_bary_mean_l{layer}_h{h}"] = float(drift[h].mean())
            feats[f"{prefix}_bary_var_l{layer}_h{h}"] = float(drift[h].var())
    return feats


def attention_concentration(attns: List[np.ndarray], prefix: str) -> Dict[str, float]:
    feats = {}
    for layer, a in enumerate(attns):
        for h in range(a.shape[0]):
            feats[f"{prefix}_kappa_l{layer}_h{h}"] = float(kl_to_uniform(a[h]).mean())
    return feats


def fixed_token_positions(n: int, k: int) -> List[int]:
    if n <= 2:
        return []
    k = min(k, max(1, n - 1))
    return sorted(set(np.linspace(1, n - 1, num=k, dtype=int).tolist()))


def perturb_text(text: str, tokenizer, strategy: str, args, prefix_pool: List[str]) -> str:
    ids = tokenizer.encode(str(text))
    if args.max_length and len(ids) > args.max_length:
        ids = ids[: args.max_length]
    if len(ids) < 4:
        return str(text)

    positions = fixed_token_positions(len(ids), args.num_perturb_tokens)
    if strategy == "drop":
        remove = set(positions)
        new_ids = [tok for i, tok in enumerate(ids) if i not in remove]
        return tokenizer.decode(new_ids, skip_special_tokens=True)

    if strategy == "replace":
        new_ids = list(ids)
        vocab = max(100, int(getattr(tokenizer, "vocab_size", 50000)))
        rng = np.random.default_rng(args.seed + len(ids))
        for pos in positions:
            new_ids[pos] = int(rng.integers(low=0, high=vocab))
        return tokenizer.decode(new_ids, skip_special_tokens=True)

    if strategy == "prefix":
        if not prefix_pool:
            return str(text)
        prefix = str(random.choice(prefix_pool))
        prefix_ids = tokenizer.encode(prefix)[: args.prefix_tokens]
        new_ids = (prefix_ids + ids)[: args.max_length]
        return tokenizer.decode(new_ids, skip_special_tokens=True)

    raise ValueError(f"Unknown perturbation strategy: {strategy}")


def perturbation_features(base_attns: List[np.ndarray], pert_attns: List[np.ndarray], strategy: str) -> Dict[str, float]:
    base = attention_concentration(base_attns, "base")
    pert = attention_concentration(pert_attns, "pert")
    feats = {}
    for key, base_val in base.items():
        suffix = key.replace("base_kappa_", "")
        pkey = "pert_kappa_" + suffix
        if pkey in pert:
            feats[f"pert_{strategy}_delta_kappa_{suffix}"] = float(abs(base_val - pert[pkey]))
    return feats


def extract_features_for_sample(
    model,
    tokenizer,
    text: str,
    args,
    prefix_pool: List[str],
    perturbations: Optional[List[str]] = None,
) -> Dict[str, float]:
    base_attns = get_attentions(model, tokenizer, text, args)
    feats: Dict[str, float] = {}
    feats.update(transitional_features(base_attns))
    feats.update(attention_concentration(base_attns, "base"))

    active = args.perturbations if perturbations is None else perturbations
    for strategy in active:
        ptext = perturb_text(text, tokenizer, strategy, args, prefix_pool)
        pattns = get_attentions(model, tokenizer, ptext, args)
        feats.update(perturbation_features(base_attns, pattns, strategy))
    return feats


def load_hardsplit_targets(args) -> pd.DataFrame:
    data_dir = Path(args.data_dir) if args.data_dir else Path(args.run_dir) / "data"
    data_dir = resolve_existing_path(str(data_dir))
    files = [
        data_dir / "mimir_wikipedia_pt_member.csv",
        data_dir / "mimir_wikipedia_ft_nonmember.csv",
        data_dir / "mimir_wikipedia_unseen_nonmember.csv",
    ]
    parts = []
    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"Required split CSV not found: {path}")
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "group" not in df.columns or "text" not in df.columns:
            raise ValueError(f"{path} must contain group and text columns.")
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"].str.len() > 0].drop_duplicates("text").reset_index(drop=True)
        parts.append(df[["group", "source", "text", "original_index"] if "source" in df.columns else ["group", "text", "original_index"]])

    targets = pd.concat(parts, ignore_index=True)
    if "source" not in targets.columns:
        targets["source"] = targets["group"]
    targets["original_index"] = pd.to_numeric(targets.get("original_index", pd.Series(np.arange(len(targets)))), errors="coerce").fillna(-1).astype(int)

    if args.n_per_group > 0:
        sampled = []
        for _, gdf in targets.groupby("group"):
            sampled.append(gdf.sample(n=min(args.n_per_group, len(gdf)), random_state=args.seed))
        targets = pd.concat(sampled, ignore_index=True)
    return targets.reset_index(drop=True)


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return 0.0, float("inf")
    idx = valid[np.argmax(tpr[valid])]
    return float(tpr[idx]), float(thresholds[idx])


def add_fold_prefix_features(feature_df: pd.DataFrame, data: pd.DataFrame, spec: ComparisonSpec, args, model, tokenizer, train_idx: np.ndarray) -> pd.DataFrame:
    if "prefix" not in args.perturbations:
        return feature_df

    train_data = data.iloc[train_idx]
    prefix_pool = train_data.loc[train_data["group"] == spec.negative_group, "text"].astype(str).tolist()
    if not prefix_pool:
        prefix_pool = train_data["text"].astype(str).tolist()

    rows = []
    for row in data.itertuples(index=False):
        feats = extract_features_for_sample(
            model,
            tokenizer,
            str(getattr(row, "text")),
            args,
            prefix_pool=prefix_pool,
            perturbations=["prefix"],
        )
        prefix_only = {k: v for k, v in feats.items() if k.startswith("pert_prefix_")}
        prefix_only["sample_id"] = int(getattr(row, "sample_id"))
        rows.append(prefix_only)
    return feature_df.merge(pd.DataFrame(rows), on="sample_id", how="left")


def run_mlp_cv(feature_df: pd.DataFrame, spec: ComparisonSpec, args, model, tokenizer) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base_feature_cols = [c for c in feature_df.columns if c.startswith(("trans_", "base_", "pert_"))]
    data = feature_df[feature_df["group"].isin([spec.positive_group, spec.negative_group])].copy()
    data = data.dropna(subset=base_feature_cols, how="all").reset_index(drop=True)
    y = (data["group"] == spec.positive_group).astype(int).to_numpy()
    n_splits = min(args.cv_splits, int(np.bincount(y).min()))
    if n_splits < 2:
        raise ValueError(f"Not enough samples for CV in {spec.name}: {np.bincount(y)}")

    clf = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=tuple(args.hidden_layers),
            activation="relu",
            solver="adam",
            alpha=args.mlp_alpha,
            learning_rate_init=args.mlp_lr,
            max_iter=args.mlp_max_iter,
            early_stopping=True,
            random_state=args.seed,
        )),
    ])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    fold_rows = []
    pred_rows = []
    split_anchor = np.zeros((len(y), 1), dtype=float)
    for fold, (tr, te) in enumerate(cv.split(split_anchor, y), start=1):
        fold_features = add_fold_prefix_features(feature_df, data, spec, args, model, tokenizer, tr)
        fold_data = fold_features[fold_features["group"].isin([spec.positive_group, spec.negative_group])].copy().reset_index(drop=True)
        feature_cols = [c for c in fold_data.columns if c.startswith(("trans_", "base_", "pert_"))]
        X = fold_data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:, 1]

        tpr01, thr01 = tpr_at_fpr(y[te], prob, 0.01)
        tpr10, thr10 = tpr_at_fpr(y[te], prob, 0.10)
        fold_rows.append({
            "comparison": spec.name,
            "fold": fold,
            "positive_group": spec.positive_group,
            "negative_group": spec.negative_group,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "n_features": int(len(feature_cols)),
            "auc": float(roc_auc_score(y[te], prob)),
            "auprc": float(average_precision_score(y[te], prob)),
            "tpr_at_1_fpr": tpr01,
            "threshold_at_1_fpr": thr01,
            "tpr_at_10_fpr": tpr10,
            "threshold_at_10_fpr": thr10,
        })
        for meta_row, yy, p in zip(fold_data.iloc[te].itertuples(index=False), y[te], prob):
            pred_rows.append({
                "comparison": spec.name,
                "fold": fold,
                "sample_id": int(getattr(meta_row, "sample_id")),
                "group": getattr(meta_row, "group"),
                "label_positive": int(yy),
                "membership_probability": float(p),
            })
    return pd.DataFrame(fold_rows), pd.DataFrame(pred_rows)


def summarize_cv(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comp, sub in folds.groupby("comparison"):
        row = {"comparison": comp, "n_folds": int(len(sub))}
        for metric in ["auc", "auprc", "tpr_at_1_fpr", "tpr_at_10_fpr"]:
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_std"] = float(sub[metric].std(ddof=0))
        row["n_features"] = int(sub["n_features"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_roc(preds: pd.DataFrame, spec: ComparisonSpec, out_dir: Path) -> None:
    y = preds["label_positive"].to_numpy(int)
    p = preds["membership_probability"].to_numpy(float)
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    plt = setup_matplotlib()
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(spec.name)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "attenmia_roc.png", dpi=180)
    plt.close(fig)


def comparison_specs(args) -> List[ComparisonSpec]:
    all_specs = {s.name: s for s in DEFAULT_COMPARISONS}
    selected_names = args.experiments or list(all_specs.keys())
    unknown = [name for name in selected_names if name not in all_specs]
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}. Available: {list(all_specs)}")
    return [all_specs[name] for name in selected_names]


def _feature_cache_key(row) -> tuple:
    """Stable key so FT features are reused across ft_vs_pt and ft_vs_unseen."""
    group = str(row["group"])
    text = str(row["text"])
    if "original_index" in row and str(row["original_index"]).strip() not in {"", "nan", "None"}:
        return (group, f"orig:{row['original_index']}", text)
    return (group, text)


def precompute_non_prefix_features(
    targets: pd.DataFrame,
    model,
    tokenizer,
    args,
) -> Dict[tuple, Dict[str, float]]:
    """Compute drop/replace (etc.) features once per unique sample (shared by comparisons)."""
    non_prefix = [p for p in args.perturbations if p != "prefix"]
    cache: Dict[tuple, Dict[str, float]] = {}
    if not non_prefix:
        return cache

    # Unique samples across all groups used in any comparison
    uniq = targets.drop_duplicates(subset=["group", "text"]).reset_index(drop=True)
    for _, row in tqdm(uniq.iterrows(), total=len(uniq), desc="AttenMIA shared base features"):
        key = _feature_cache_key(row)
        if key in cache:
            continue
        feats = extract_features_for_sample(
            model,
            tokenizer,
            str(row["text"]),
            args,
            prefix_pool=[],
            perturbations=non_prefix,
        )
        cache[key] = feats
    return cache


def run_one(
    spec: ComparisonSpec,
    targets: pd.DataFrame,
    model,
    tokenizer,
    args,
    feature_cache: Optional[Dict[tuple, Dict[str, float]]] = None,
) -> pd.DataFrame:
    out_dir = Path(args.output_dir) / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_targets = targets[targets["group"].isin([spec.positive_group, spec.negative_group])].copy().reset_index(drop=True)
    sub_targets.to_csv(out_dir / "attenmia_target_samples.csv", index=False)

    non_prefix_perturbations = [p for p in args.perturbations if p != "prefix"]
    feature_rows = []
    for sample_id, row in tqdm(sub_targets.iterrows(), total=len(sub_targets), desc=f"{spec.name} base features"):
        key = _feature_cache_key(row)
        if feature_cache is not None and key in feature_cache:
            feats = dict(feature_cache[key])
        else:
            feats = extract_features_for_sample(
                model,
                tokenizer,
                str(row["text"]),
                args,
                prefix_pool=[],
                perturbations=non_prefix_perturbations,
            )
            if feature_cache is not None:
                feature_cache[key] = dict(feats)
        feats.update({
            "sample_id": int(sample_id),
            "group": row["group"],
            "source": row.get("source", ""),
            "original_index": int(row.get("original_index", sample_id)),
            "text_char_len": int(len(str(row["text"]))),
            "text": str(row["text"]),
        })
        feature_rows.append(feats)
        if args.save_every and len(feature_rows) % args.save_every == 0:
            pd.DataFrame(feature_rows).to_csv(out_dir / "attenmia_base_features.partial.csv", index=False)

    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(out_dir / "attenmia_official_base_features.csv", index=False)
    folds, preds = run_mlp_cv(feature_df, spec, args, model, tokenizer)
    folds.to_csv(out_dir / "attenmia_official_mlp_cv_fold_results.csv", index=False)
    preds.to_csv(out_dir / "attenmia_official_mlp_cv_predictions.csv", index=False)
    summary = summarize_cv(folds)
    summary.to_csv(out_dir / "attenmia_official_mlp_results.csv", index=False)
    plot_roc(preds, spec, out_dir)

    with open(out_dir / "attenmia_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"comparison: {spec.name}\n")
        f.write(f"positive_group: {spec.positive_group}\n")
        f.write(f"negative_group: {spec.negative_group}\n")
        f.write(f"classifier: lightweight MLP\n")
        f.write("prefix_source: train-fold negative-group samples only\n")
        f.write(f"perturbations: {args.perturbations}\n")
        f.write(sub_targets["group"].value_counts().to_string())
        f.write("\n\n")
        f.write(summary.to_string(index=False))
        f.write("\n")
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    if add_model_arguments is not None:
        add_model_arguments(parser)
    else:
        parser.add_argument("--model", default="")
        parser.add_argument("--model-name", default="")
    parser.add_argument("--run-dir", default=os.environ.get("MIMIR_HARDSPLIT_RUN_DIR", DEFAULT_RUN_DIR))
    parser.add_argument("--adapter-dir", default=os.environ.get("MIMIR_HARDSPLIT_ADAPTER_DIR", DEFAULT_RUN_DIR))
    parser.add_argument("--data-dir", default=os.environ.get("MIMIR_HARDSPLIT_DATA_DIR", ""))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-per-group", type=int, default=int(os.environ.get("N_PER_GROUP", "500")))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--experiments", nargs="*", default=[], help="Default: ft_vs_pt ft_vs_unseen pt_vs_unseen")

    parser.add_argument("--max-length", type=int, default=int(os.environ.get("MAX_LENGTH", "256")))
    parser.add_argument("--perturbations", nargs="+", default=["drop", "replace", "prefix"])
    parser.add_argument("--num-perturb-tokens", type=int, default=7)
    parser.add_argument("--prefix-tokens", type=int, default=16)
    parser.add_argument("--hidden-layers", nargs="+", type=int, default=[128, 64])
    parser.add_argument("--mlp-alpha", type=float, default=1e-4)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-max-iter", type=int, default=500)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if apply_model_namespace is not None:
        args = apply_model_namespace(args, profile="attenmia", log=print)
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    targets = load_hardsplit_targets(args)
    targets.to_csv(out / "attenmia_all_target_samples.csv", index=False)
    model, tokenizer, resolved_adapter = load_attention_model(args)

    with open(out / "attenmia_config.json", "w", encoding="utf-8") as f:
        config = vars(args).copy()
        config["resolved_adapter_dir"] = str(resolved_adapter)
        config["target_counts"] = targets["group"].value_counts().to_dict()
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("Target counts:")
    print(targets["group"].value_counts().to_string())
    print(f"Resolved adapter: {resolved_adapter}")

    # Shared non-prefix features across comparisons (FT appears in both paper tables)
    specs = comparison_specs(args)
    used_groups = set()
    for spec in specs:
        used_groups.add(spec.positive_group)
        used_groups.add(spec.negative_group)
    cache_targets = targets[targets["group"].isin(used_groups)].copy()
    feature_cache = precompute_non_prefix_features(cache_targets, model, tokenizer, args)
    print(f"Shared AttenMIA feature cache size: {len(feature_cache)}")

    summaries = []
    for spec in specs:
        print(f"\n=== AttenMIA official MIMIR hard split: {spec.name} ===")
        summaries.append(run_one(spec, targets, model, tokenizer, args, feature_cache=feature_cache))

    merged = pd.concat(summaries, ignore_index=True)
    merged.to_csv(out / "attenmia_official_mimir_hardsplit_results.csv", index=False)
    with open(out / "attenmia_summary.txt", "w", encoding="utf-8") as f:
        f.write(merged.to_string(index=False))
        f.write("\n")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\n=== AttenMIA official MIMIR hard split results ===")
    print(merged.to_string(index=False))


if __name__ == "__main__":
    main()
