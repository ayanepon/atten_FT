# -*- coding: utf-8 -*-
"""GDS baseline on the MIMIR Wikipedia hard-split data.

Paper: "From Unfamiliar to Familiar: Detecting Pre-training Data via
Gradient Deviations in Large Language Models".

This adapts GDS to the same MIMIR hard split used by the proposed method:

  PT      = MIMIR Wikipedia member / pre-training group
  FT      = MIMIR Wikipedia nonmember used for LoRA fine-tuning
  Unseen  = MIMIR Wikipedia nonmember not used for fine-tuning

GDS itself is fine-tuning-free: it inserts a fresh LoRA adapter into the target
base model, performs a forward/backward pass for each sample, and extracts
eight gradient-deviation features from LoRA_B gradient matrices. No optimizer
step is performed.

Default evaluation:
  - FT is the positive class for ft_vs_* comparisons
  - comparisons: FT vs PT, FT vs Unseen
  - 10 repeated 5-fold CV with an MLP classifier
  - AUC is not post-hoc flipped
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


GROUP_PT = "mimir_wikipedia_member_pt"
GROUP_FT = "mimir_wikipedia_nonmember_ft"
GROUP_UNSEEN = "mimir_wikipedia_nonmember_unseen"

DEFAULT_MODEL_NAME = "EleutherAI/pythia-1b"
DEFAULT_DATA_DIR = "/workplace/FT/BlackNLP_2/results/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2/data"
DEFAULT_OUTPUT_DIR = "/workplace/FT/BlackNLP_2/results/gds_mimir_hardsplit"


@dataclass(frozen=True)
class ComparisonSpec:
    name: str
    positive_group: str
    negative_group: str


DEFAULT_COMPARISONS = [
    ComparisonSpec("ft_vs_pt", GROUP_FT, GROUP_PT),
    ComparisonSpec("ft_vs_unseen", GROUP_FT, GROUP_UNSEEN),
    ComparisonSpec("pt_vs_unseen", GROUP_PT, GROUP_UNSEEN),
]


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def resolve_existing_path(path_like: str, required_files: Sequence[str] = ()) -> Path:
    root = script_dir()
    path = Path(path_like).expanduser()
    candidates = [
        path,
        Path.cwd() / path,
        root / path.name,
        root / "results" / path.name,
        root / path,
    ]
    path_str = str(path_like)
    for prefix in [
        "/workplace/FT/BlackNLP_2/results/",
        "/workplace/FT/BlackNLP_2/models/",
        "/workplace/FT/BlackNLP_2/",
        "/workplace/FT/BlackNLP/results/",
        "/workplace/FT/results/",
        "results/",
    ]:
        if path_str.startswith(prefix):
            suffix = path_str[len(prefix) :]
            candidates.extend([root / suffix, root / "results" / suffix])

    for candidate in candidates:
        if not candidate.exists():
            continue
        if required_files and not all((candidate / item).exists() for item in required_files):
            continue
        return candidate
    tried = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve path: {path_like}\nTried:\n{tried}")


def load_targets(data_dir: Path, n_per_group: int, seed: int) -> pd.DataFrame:
    specs = [
        ("mimir_wikipedia_pt_member.csv", GROUP_PT),
        ("mimir_wikipedia_ft_nonmember.csv", GROUP_FT),
        ("mimir_wikipedia_unseen_nonmember.csv", GROUP_UNSEEN),
    ]
    parts = []
    for filename, group in specs:
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required split CSV not found: {path}")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "text" not in frame.columns:
            raise ValueError(f"{path} must contain a text column.")
        frame["text"] = frame["text"].astype(str).str.strip()
        frame = frame[frame["text"].str.len() > 0].drop_duplicates("text").reset_index(drop=True)
        frame["group"] = group
        frame["sample_id"] = np.arange(len(frame), dtype=int)
        if n_per_group > 0:
            frame = (
                frame.sample(n=min(n_per_group, len(frame)), random_state=seed)
                .sort_index()
                .reset_index(drop=True)
            )
            frame["sample_id"] = np.arange(len(frame), dtype=int)
        keep = [c for c in ["group", "sample_id", "source", "original_index", "text"] if c in frame.columns]
        parts.append(frame[keep])
    return pd.concat(parts, ignore_index=True)


def infer_target_modules(model, explicit: str) -> List[str]:
    if explicit:
        return [x.strip() for x in explicit.split(",") if x.strip()]

    linear_suffixes = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        lname = name.lower()
        if not any(token in lname for token in ["attn", "attention", "mlp", "ffn"]):
            continue
        linear_suffixes.add(name.split(".")[-1])

    preferred = [
        "query_key_value",
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
        "c_fc",
        "c_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "o_proj",
    ]
    selected = [item for item in preferred if item in linear_suffixes]
    if selected:
        return selected
    if linear_suffixes:
        return sorted(linear_suffixes)
    raise RuntimeError("Could not infer LoRA target modules.")


def load_model_and_tokenizer(args):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": None,
    }[args.dtype]
    kwargs = {}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    elif device.type == "cuda":
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    base = AutoModelForCausalLM.from_pretrained(args.model_name, **kwargs)
    target_modules = infer_target_modules(base, args.target_modules)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(base, config)
    model.to(device)
    model.config.use_cache = False
    model.train()
    return model, tokenizer, target_modules


def encode_text(tokenizer, text: str, max_length: int, device: torch.device) -> Dict[str, torch.Tensor] | None:
    encoded = tokenizer(
        str(text),
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)
    if input_ids.shape[1] < 2:
        return None
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def module_group(name: str) -> str:
    lname = name.lower()
    if any(token in lname for token in ["attn", "attention", "query", "key", "value", "q_proj", "k_proj", "v_proj"]):
        return "attn"
    if any(token in lname for token in ["mlp", "ffn", "dense_h_to_4h", "dense_4h_to_h", "c_fc", "c_proj", "gate", "up", "down"]):
        return "ffn"
    return "other"


def clean_feature_name(name: str) -> str:
    name = name.replace("base_model.model.", "")
    name = name.replace(".default", "")
    name = name.replace(".weight", "")
    name = name.replace(".", "_")
    name = re.sub(r"[^0-9A-Za-z_]+", "_", name)
    return name.strip("_")


def gradient_features(matrix: torch.Tensor, top_ratio: float, sparse_threshold: float) -> Dict[str, float]:
    g = matrix.detach().float().abs().cpu()
    if g.ndim != 2:
        g = g.reshape(g.shape[0], -1)
    r, h = g.shape
    flat = g.flatten()
    n = int(flat.numel())
    if n == 0:
        return {}
    k = max(1, int(math.ceil(n * top_ratio)))
    top_vals, top_idx = torch.topk(flat, k=k, largest=True)
    rows = torch.div(top_idx, h, rounding_mode="floor").float()
    cols = (top_idx % h).float()

    abs_mean = float(flat.mean().item())
    row_means = g.mean(dim=1)
    row_mean_max = float(row_means.max().item())
    total_l1 = float(flat.sum().item())
    top10_ratio = float(top_vals.sum().item() / max(total_l1, 1e-12))
    sparsity = float((flat < sparse_threshold).float().mean().item())
    std = float(torch.sqrt(torch.mean((flat - abs_mean) ** 2)).item())
    row_mean_std = float(torch.sqrt(torch.mean((row_means - row_means.mean()) ** 2)).item())

    if r > 1:
        row_ecc = torch.abs((2.0 * (rows + 1.0) - (r + 1.0)) / (r - 1.0)).mean()
        row_ecc_value = float(row_ecc.item())
    else:
        row_ecc_value = 0.0
    if h > 1:
        col_ecc = torch.abs((2.0 * (cols + 1.0) - (h + 1.0)) / (h - 1.0)).mean()
        col_ecc_value = float(col_ecc.item())
    else:
        col_ecc_value = 0.0

    return {
        "abs_mean": abs_mean,
        "row_mean_max": row_mean_max,
        "row_ecc": row_ecc_value,
        "col_ecc": col_ecc_value,
        "top10_ratio": top10_ratio,
        "sparsity": sparsity,
        "std": std,
        "row_mean_std": row_mean_std,
    }


def named_lora_b_grads(model) -> Iterable[Tuple[str, torch.Tensor]]:
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        lname = name.lower()
        if "lora_b" not in lname:
            continue
        yield name, param.grad


def score_one_sample(model, tokenizer, text: str, args: argparse.Namespace) -> Dict[str, float]:
    device = next(model.parameters()).device
    batch = encode_text(tokenizer, text, args.max_length, device)
    if batch is None:
        return {"valid": 0, "num_tokens": 0}

    model.train()
    model.zero_grad(set_to_none=True)
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["input_ids"],
        use_cache=False,
    )
    loss = outputs.loss
    loss.backward()

    row: Dict[str, float] = {
        "valid": 1,
        "num_tokens": int(batch["input_ids"].shape[1]),
        "loss": float(loss.detach().float().cpu().item()),
    }
    module_feature_values = []
    attn_feature_values = []
    ffn_feature_values = []
    for name, grad in named_lora_b_grads(model):
        feats = gradient_features(grad, args.top_ratio, args.sparse_threshold)
        if not feats:
            continue
        prefix = clean_feature_name(name)
        group = module_group(name)
        for key, value in feats.items():
            row[f"gds_{prefix}_{key}"] = value
            row[f"gds_{group}_{key}_values_count"] = row.get(f"gds_{group}_{key}_values_count", 0.0) + 1.0
            row[f"gds_{group}_{key}_sum"] = row.get(f"gds_{group}_{key}_sum", 0.0) + value
            module_feature_values.append((key, value))
            if group == "attn":
                attn_feature_values.append((key, value))
            elif group == "ffn":
                ffn_feature_values.append((key, value))

    # Aggregate summaries are helpful when models have different module layouts.
    for key in ["abs_mean", "row_mean_max", "row_ecc", "col_ecc", "top10_ratio", "sparsity", "std", "row_mean_std"]:
        vals = [v for k, v in module_feature_values if k == key]
        row[f"gds_all_{key}_mean"] = float(np.mean(vals)) if vals else np.nan
        row[f"gds_all_{key}_std"] = float(np.std(vals)) if vals else np.nan
        attn_vals = [v for k, v in attn_feature_values if k == key]
        ffn_vals = [v for k, v in ffn_feature_values if k == key]
        row[f"gds_attn_{key}_mean"] = float(np.mean(attn_vals)) if attn_vals else np.nan
        row[f"gds_ffn_{key}_mean"] = float(np.mean(ffn_vals)) if ffn_vals else np.nan

    for col in list(row):
        if col.endswith("_values_count") or col.endswith("_sum"):
            continue
    for col in [c for c in list(row) if c.endswith("_sum")]:
        base = col[:-4]
        count = row.get(f"{base}_values_count", 0.0)
        if count:
            row[f"{base}_mean"] = row[col] / count
        del row[col]
        row.pop(f"{base}_values_count", None)

    model.zero_grad(set_to_none=True)
    return row


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, max_fpr: float = 0.10) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = fpr <= max_fpr
    return float(np.max(tpr[valid])) if np.any(valid) else 0.0


def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "tpr_at_5_fpr": tpr_at_fpr(y_true, scores, 0.05),
        "tpr_at_10_fpr": tpr_at_fpr(y_true, scores, 0.10),
    }


def evaluate_gds(scores: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [c for c in scores.columns if c.startswith("gds_")]
    rows = []
    pred_rows = []
    for spec in DEFAULT_COMPARISONS:
        if spec.name not in args.comparisons:
            continue
        sub = scores[(scores["group"].isin([spec.positive_group, spec.negative_group])) & (scores["valid"] == 1)]
        sub = sub.drop_duplicates(["group", "sample_id"]).reset_index(drop=True)
        x_frame = sub[feature_cols].replace([np.inf, -np.inf], np.nan)
        valid_cols = [
            c for c in feature_cols
            if x_frame[c].notna().sum() >= 4 and x_frame[c].nunique(dropna=True) > 1
        ]
        x_frame = x_frame[valid_cols].fillna(x_frame[valid_cols].median(numeric_only=True))
        x = x_frame.to_numpy(float)
        y = (sub["group"].to_numpy() == spec.positive_group).astype(int)

        for repeat in range(1, args.repeats + 1):
            cv = StratifiedKFold(
                n_splits=args.cv_splits,
                shuffle=True,
                random_state=args.seed + repeat - 1,
            )
            oof = np.full(len(y), np.nan)
            for fold, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
                clf = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "mlp",
                            MLPClassifier(
                                hidden_layer_sizes=(128, 64),
                                activation="relu",
                                solver="adam",
                                alpha=args.mlp_alpha,
                                learning_rate_init=args.mlp_lr,
                                max_iter=args.mlp_max_iter,
                                early_stopping=True,
                                random_state=args.seed + repeat * 100 + fold,
                            ),
                        ),
                    ]
                )
                clf.fit(x[train_idx], y[train_idx])
                oof[test_idx] = clf.predict_proba(x[test_idx])[:, 1]
                for idx in test_idx:
                    pred_rows.append(
                        {
                            "comparison": spec.name,
                            "repeat": repeat,
                            "fold": fold,
                            "group": sub.loc[idx, "group"],
                            "sample_id": int(sub.loc[idx, "sample_id"]),
                            "y_true": int(y[idx]),
                            "score": float(oof[idx]),
                        }
                    )
            metric = compute_metrics(y, oof)
            metric.update(
                {
                    "comparison": spec.name,
                    "method": "gds_mlp",
                    "repeat": repeat,
                    "n_positive": int(y.sum()),
                    "n_negative": int((1 - y).sum()),
                    "n_features": len(valid_cols),
                }
            )
            rows.append(metric)
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def summarize(perf: pd.DataFrame) -> pd.DataFrame:
    return (
        perf.groupby(["comparison", "method"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
            tpr_at_5_fpr_mean=("tpr_at_5_fpr", "mean"),
            tpr_at_5_fpr_std=("tpr_at_5_fpr", "std"),
            tpr_at_10_fpr_mean=("tpr_at_10_fpr", "mean"),
            tpr_at_10_fpr_std=("tpr_at_10_fpr", "std"),
            n_repeats=("repeat", "count"),
            n_positive=("n_positive", "first"),
            n_negative=("n_negative", "first"),
            n_features=("n_features", "first"),
        )
        .sort_values(["comparison", "method"])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME))
    parser.add_argument("--data-dir", default=os.environ.get("MIMIR_HARDSPLIT_DATA_DIR", DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-per-group", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--target-modules", default="", help="Comma-separated LoRA target module names. Default: infer.")
    parser.add_argument("--top-ratio", type=float, default=0.10)
    parser.add_argument("--sparse-threshold", type=float, default=1e-6)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-alpha", type=float, default=1e-4)
    parser.add_argument("--mlp-max-iter", type=int, default=500)
    parser.add_argument("--comparisons", nargs="+", default=["ft_vs_pt", "ft_vs_unseen"])
    parser.add_argument("--skip-scoring", action="store_true", help="Reuse gds_scores.csv if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = resolve_existing_path(
        args.data_dir,
        [
            "mimir_wikipedia_pt_member.csv",
            "mimir_wikipedia_ft_nonmember.csv",
            "mimir_wikipedia_unseen_nonmember.csv",
        ],
    )
    score_path = output_dir / "gds_scores.csv"

    if args.skip_scoring and score_path.exists():
        log(f"Reuse scores: {score_path}")
        scores = pd.read_csv(score_path)
        target_modules: List[str] = []
    else:
        targets = load_targets(data_dir, args.n_per_group, args.seed)
        targets.to_csv(output_dir / "gds_target_samples.csv", index=False)
        model, tokenizer, target_modules = load_model_and_tokenizer(args)
        rows = []
        for row in tqdm(targets.itertuples(index=False), total=len(targets), desc="GDS gradient features"):
            result = score_one_sample(model, tokenizer, row.text, args)
            result.update(
                {
                    "group": row.group,
                    "sample_id": int(row.sample_id),
                    "source": getattr(row, "source", ""),
                    "original_index": getattr(row, "original_index", ""),
                    "text_char_len": len(str(row.text)),
                }
            )
            rows.append(result)
        scores = pd.DataFrame(rows)
        scores.to_csv(score_path, index=False)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    perf, oof = evaluate_gds(scores, args)
    summary = summarize(perf)
    perf.to_csv(output_dir / "gds_auc_10runs.csv", index=False)
    oof.to_csv(output_dir / "gds_oof_predictions.csv", index=False)
    summary.to_csv(output_dir / "gds_summary_auc.csv", index=False)

    with open(output_dir / "gds_config.json", "w", encoding="utf-8") as handle:
        config = vars(args).copy()
        config["resolved_data_dir"] = str(data_dir)
        config["target_modules"] = target_modules
        config["feature_definition"] = {
            "per_matrix": [
                "abs_mean",
                "row_mean_max",
                "row_ecc",
                "col_ecc",
                "top10_ratio",
                "sparsity",
                "std",
                "row_mean_std",
            ],
            "notes": "Fresh LoRA is inserted into the base model; LoRA_B gradients are collected after one CLM backward pass; no optimizer step is performed.",
        }
        json.dump(config, handle, ensure_ascii=False, indent=2)

    with open(output_dir / "gds_summary.txt", "w", encoding="utf-8") as handle:
        handle.write("GDS on MIMIR hard split\n")
        handle.write(f"model_name={args.model_name}\n")
        handle.write(f"data_dir={data_dir}\n")
        handle.write(f"target_modules={target_modules}\n")
        handle.write("FT is the positive class for ft_vs_* comparisons. AUC is not flipped.\n\n")
        handle.write(summary.to_string(index=False))
        handle.write("\n")

    print("\nSummary:")
    print(summary.round(6).to_string(index=False))
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
