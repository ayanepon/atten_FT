# -*- coding: utf-8 -*-
"""G-DriftMIA on the MIMIR Wikipedia hard-split data.

This script adapts G-DriftMIA to the same MIMIR hard split used by the
proposed attention-update experiments:

  PT      = MIMIR Wikipedia member / pre-training group
  FT      = MIMIR Wikipedia nonmember used for LoRA fine-tuning
  Unseen  = MIMIR Wikipedia nonmember not used for fine-tuning

The original G-DriftMIA paper is phrased for QA pairs and a single answer
token. Here, for plain article text, we use causal-LM next-token positions:
loss is the mean next-token loss, the true-token logit is averaged over valid
next-token positions, and the final-layer hidden state is pooled over valid
prediction positions.

Default evaluation:
  - FT is the positive class
  - comparisons: FT vs PT, FT vs Unseen
  - 10 repeated 5-fold CV logistic regression
  - AUC is not post-hoc flipped

Run:
  python run_g_driftmia_mimir_hardsplit.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


GROUP_PT = "mimir_wikipedia_member_pt"
GROUP_FT = "mimir_wikipedia_nonmember_ft"
GROUP_UNSEEN = "mimir_wikipedia_nonmember_unseen"

DEFAULT_MODEL_NAME = "EleutherAI/pythia-1b"
DEFAULT_RUN_DIR = "/workplace/FT/BlackNLP_2/results/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2"
DEFAULT_DATA_DIR = "/workplace/FT/BlackNLP_2/results/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2/data"
DEFAULT_OUTPUT_DIR = "/workplace/FT/BlackNLP_2/results/g_driftmia_mimir_hardsplit"


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


def resolve_existing_path(path_like: str | None, required_files: Sequence[str] = ()) -> Path:
    if not path_like:
        raise FileNotFoundError("Empty path.")
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

    required = ", ".join(required_files) if required_files else "path exists"
    tried = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not resolve path: {path_like}\nRequired: {required}\nTried:\n{tried}"
    )


def resolve_adapter_dir(run_dir: str | None, adapter_dir: str | None) -> Path | None:
    candidates = []
    if adapter_dir:
        candidates.append(adapter_dir)
    if run_dir:
        candidates.extend([run_dir, str(Path(run_dir) / "adapter")])
    for item in candidates:
        try:
            path = resolve_existing_path(item)
        except FileNotFoundError:
            continue
        if (path / "adapter_config.json").exists():
            return path
        if (path / "adapter" / "adapter_config.json").exists():
            return path / "adapter"
    return None


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


def load_model_and_tokenizer(args):
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
    kwargs = {"output_hidden_states": True}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    elif device.type == "cuda":
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **kwargs)
    adapter_dir = resolve_adapter_dir(args.run_dir, args.adapter_dir)
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)

    model.to(device)
    model.config.use_cache = False
    return model, tokenizer, adapter_dir


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


def prediction_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    # Causal LM prediction at position i predicts token i+1.
    return (attention_mask[:, :-1] > 0) & (attention_mask[:, 1:] > 0)


def forward_gdrift_features(
    model,
    batch: Dict[str, torch.Tensor],
    direction: torch.Tensor,
    hidden_pool: str,
    require_grad: bool,
) -> Tuple[torch.Tensor, Dict[str, float | torch.Tensor]]:
    context = torch.enable_grad() if require_grad else torch.no_grad()
    with context:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["input_ids"],
            output_hidden_states=True,
            use_cache=False,
        )
        logits = outputs.logits
        hidden = outputs.hidden_states[-1]

        labels = batch["input_ids"][:, 1:]
        pred_logits = logits[:, :-1, :]
        valid = prediction_mask(batch["attention_mask"])
        token_losses = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape_as(labels)
        loss = token_losses[valid].mean()

        true_logits = pred_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        mean_true_logit = true_logits[valid].mean()

        pred_hidden = hidden[:, :-1, :]
        if hidden_pool == "last":
            valid_positions = torch.nonzero(valid[0], as_tuple=False).flatten()
            idx = valid_positions[-1]
            pooled_hidden = pred_hidden[0, idx, :]
        elif hidden_pool == "mean":
            pooled_hidden = pred_hidden[valid].mean(dim=0)
        else:
            raise ValueError(f"Unknown hidden_pool: {hidden_pool}")

        projection = torch.dot(pooled_hidden.float(), direction.float())
        out = {
            "loss": loss,
            "loss_value": float(loss.detach().float().cpu().item()),
            "true_logit_mean": float(mean_true_logit.detach().float().cpu().item()),
            "projection": float(projection.detach().float().cpu().item()),
            "hidden": pooled_hidden.detach().float().cpu(),
        }
        return loss, out


def trainable_parameters(model, scope: str) -> List[torch.nn.Parameter]:
    if scope == "all":
        params = [p for p in model.parameters()]
        for p in params:
            p.requires_grad_(True)
        return params
    if scope == "lora":
        params = []
        for name, p in model.named_parameters():
            use = "lora_" in name or "modules_to_save" in name
            p.requires_grad_(use)
            if use:
                params.append(p)
        if not params:
            raise RuntimeError("No LoRA trainable parameters found. Use --trainable-scope all for a full model.")
        return params
    raise ValueError(f"Unknown trainable scope: {scope}")


def clone_params(params: Sequence[torch.nn.Parameter]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]


@torch.no_grad()
def restore_params(params: Sequence[torch.nn.Parameter], snapshot: Sequence[torch.Tensor]) -> None:
    for p, old in zip(params, snapshot):
        p.copy_(old)
        p.grad = None


@torch.no_grad()
def gradient_ascent_step(params: Sequence[torch.nn.Parameter], lr: float) -> None:
    for p in params:
        if p.grad is not None:
            p.add_(p.grad, alpha=lr)
            p.grad = None


def score_one_sample(
    model,
    tokenizer,
    text: str,
    direction: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
    base_snapshot: Sequence[torch.Tensor],
    args: argparse.Namespace,
) -> Dict[str, float]:
    device = direction.device
    batch = encode_text(tokenizer, text, args.max_length, device)
    if batch is None:
        return {"num_tokens": 0, "valid": 0}

    restore_params(params, base_snapshot)
    model.eval()
    _, before = forward_gdrift_features(
        model, batch, direction, hidden_pool=args.hidden_pool, require_grad=False
    )

    model.train()
    model.zero_grad(set_to_none=True)
    loss_for_update, _ = forward_gdrift_features(
        model, batch, direction, hidden_pool=args.hidden_pool, require_grad=True
    )
    loss_for_update.backward()
    gradient_ascent_step(params, args.ascent_lr)

    model.eval()
    _, after = forward_gdrift_features(
        model, batch, direction, hidden_pool=args.hidden_pool, require_grad=False
    )

    h0 = before["hidden"]
    h1 = after["hidden"]
    hidden_drift = float(torch.linalg.vector_norm(h1 - h0).item())
    projection_drift = float(after["projection"] - before["projection"])
    loss_drift = float(after["loss_value"] - before["loss_value"])
    logit_drift = float(after["true_logit_mean"] - before["true_logit_mean"])

    restore_params(params, base_snapshot)
    num_tokens = int(batch["input_ids"].shape[1])
    num_valid = int(prediction_mask(batch["attention_mask"]).sum().detach().cpu().item())
    return {
        "num_tokens": num_tokens,
        "num_valid_next_tokens": num_valid,
        "valid": 1,
        "loss_before": before["loss_value"],
        "true_logit_before": before["true_logit_mean"],
        "projection_before": before["projection"],
        "loss_after": after["loss_value"],
        "true_logit_after": after["true_logit_mean"],
        "projection_after": after["projection"],
        "loss_drift": loss_drift,
        "true_logit_drift": logit_drift,
        "projection_drift": projection_drift,
        "hidden_l2_drift": hidden_drift,
        "abs_projection_drift": abs(projection_drift),
        "abs_logit_drift": abs(logit_drift),
        "abs_loss_drift": abs(loss_drift),
    }


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, max_fpr: float = 0.10) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = fpr <= max_fpr
    return float(np.max(tpr[valid])) if np.any(valid) else 0.0


def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "tpr_at_10_fpr": tpr_at_fpr(y_true, scores, 0.10),
    }


def evaluate_gdrift(scores: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [
        "loss_before",
        "true_logit_before",
        "projection_before",
        "loss_after",
        "true_logit_after",
        "projection_after",
        "hidden_l2_drift",
    ]
    if args.include_drift_features:
        feature_cols.extend(
            [
                "loss_drift",
                "true_logit_drift",
                "projection_drift",
                "abs_loss_drift",
                "abs_logit_drift",
                "abs_projection_drift",
            ]
        )

    rows = []
    split_rows = []
    for spec in DEFAULT_COMPARISONS:
        if spec.name not in args.comparisons:
            continue
        sub = scores[(scores["group"].isin([spec.positive_group, spec.negative_group])) & (scores["valid"] == 1)]
        sub = sub.drop_duplicates(["group", "sample_id"]).reset_index(drop=True)
        x_frame = sub[feature_cols].replace([np.inf, -np.inf], np.nan)
        valid_cols = [c for c in feature_cols if x_frame[c].notna().sum() >= 4 and x_frame[c].nunique(dropna=True) > 1]
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
                            "clf",
                            LogisticRegression(
                                penalty="l2",
                                solver="lbfgs",
                                C=args.classifier_c,
                                max_iter=2000,
                                class_weight="balanced",
                                random_state=args.seed + repeat * 100 + fold,
                            ),
                        ),
                    ]
                )
                clf.fit(x[train_idx], y[train_idx])
                oof[test_idx] = clf.predict_proba(x[test_idx])[:, 1]
                for idx in test_idx:
                    split_rows.append(
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
                    "method": "g_driftmia",
                    "repeat": repeat,
                    "n_positive": int(y.sum()),
                    "n_negative": int((1 - y).sum()),
                    "n_features": len(valid_cols),
                    "feature_cols": ",".join(valid_cols),
                }
            )
            rows.append(metric)

    return pd.DataFrame(rows), pd.DataFrame(split_rows)


def summarize(perf: pd.DataFrame) -> pd.DataFrame:
    return (
        perf.groupby(["comparison", "method"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
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
    parser.add_argument("--run-dir", default=os.environ.get("MIMIR_HARDSPLIT_RUN_DIR", DEFAULT_RUN_DIR))
    parser.add_argument("--adapter-dir", default=os.environ.get("MIMIR_HARDSPLIT_ADAPTER_DIR", ""))
    parser.add_argument("--data-dir", default=os.environ.get("MIMIR_HARDSPLIT_DATA_DIR", DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-per-group", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--ascent-lr", type=float, default=1e-2)
    parser.add_argument("--trainable-scope", choices=["lora", "all"], default="lora")
    parser.add_argument("--hidden-pool", choices=["mean", "last"], default="mean")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--classifier-c", type=float, default=1.0)
    parser.add_argument("--comparisons", nargs="+", default=["ft_vs_pt", "ft_vs_unseen"])
    parser.add_argument("--include-drift-features", action="store_true")
    parser.add_argument("--skip-scoring", action="store_true", help="Reuse g_driftmia_scores.csv if it exists.")
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
    score_path = output_dir / "g_driftmia_scores.csv"

    if args.skip_scoring and score_path.exists():
        log(f"Reuse scores: {score_path}")
        scores = pd.read_csv(score_path)
        adapter_dir = resolve_adapter_dir(args.run_dir, args.adapter_dir)
    else:
        targets = load_targets(data_dir, args.n_per_group, args.seed)
        targets.to_csv(output_dir / "g_driftmia_target_samples.csv", index=False)
        model, tokenizer, adapter_dir = load_model_and_tokenizer(args)
        params = trainable_parameters(model, args.trainable_scope)
        base_snapshot = clone_params(params)

        hidden_dim = int(model.get_input_embeddings().embedding_dim)
        generator = torch.Generator(device=args.device)
        generator.manual_seed(args.seed)
        direction = torch.randn(hidden_dim, generator=generator, device=args.device)
        direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-12)

        rows = []
        for row in tqdm(targets.itertuples(index=False), total=len(targets), desc="G-DriftMIA scoring"):
            result = score_one_sample(
                model,
                tokenizer,
                row.text,
                direction,
                params,
                base_snapshot,
                args,
            )
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
        restore_params(params, base_snapshot)

    perf, oof = evaluate_gdrift(scores, args)
    summary = summarize(perf)
    perf.to_csv(output_dir / "g_driftmia_auc_10runs.csv", index=False)
    oof.to_csv(output_dir / "g_driftmia_oof_predictions.csv", index=False)
    summary.to_csv(output_dir / "g_driftmia_summary_auc.csv", index=False)

    with open(output_dir / "g_driftmia_config.json", "w", encoding="utf-8") as handle:
        config = vars(args).copy()
        config["resolved_data_dir"] = str(data_dir)
        config["resolved_adapter_dir"] = str(adapter_dir) if adapter_dir is not None else None
        config["feature_vector"] = [
            "loss_before",
            "true_logit_before",
            "projection_before",
            "loss_after",
            "true_logit_after",
            "projection_after",
            "hidden_l2_drift",
        ]
        config["text_adaptation"] = (
            "QA single-answer-token G-Drift is adapted to causal-LM article text by "
            "averaging loss/logit over non-padding next-token positions and pooling "
            "final-layer hidden states over valid prediction positions."
        )
        json.dump(config, handle, ensure_ascii=False, indent=2)

    with open(output_dir / "g_driftmia_summary.txt", "w", encoding="utf-8") as handle:
        handle.write("G-DriftMIA on MIMIR hard split\n")
        handle.write(f"model_name={args.model_name}\n")
        handle.write(f"data_dir={data_dir}\n")
        handle.write(f"adapter_dir={adapter_dir}\n")
        handle.write(f"trainable_scope={args.trainable_scope}, ascent_lr={args.ascent_lr}\n")
        handle.write("FT is the positive class for ft_vs_* comparisons. AUC is not flipped.\n\n")
        handle.write(summary.to_string(index=False))
        handle.write("\n")

    print("\nSummary:")
    print(summary.round(6).to_string(index=False))
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
