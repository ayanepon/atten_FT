# -*- coding: utf-8 -*-
"""
Official-style LoRA-Leak for the MIMIR Wikipedia hard-split experiment.

Paper target:
  LoRA-Leak: Membership Inference Attacks Against LoRA Fine-tuned Language Models

This standalone script evaluates the MIMIR hard split:
  PT      = MIMIR Wikipedia member
  FT      = MIMIR Wikipedia nonmember used for LoRA FT
  Unseen  = MIMIR Wikipedia nonmember not used for LoRA FT

Default comparisons:
  - FT vs PT
  - FT vs Unseen
  - PT vs Unseen

Implemented LoRA-Leak-style scores:
  - LOSS: target negative loss score
  - zlib-normalized loss score
  - Min-K%
  - Min-K%++
  - GradNormx
  - pre-trained-reference variants: S_ref(x)=S(Mpt,x)-S(Mft,x)

GradNormx is enabled by default because it is one of the practical LoRA-Leak
signals. It can be disabled with --no-gradnormx when speed matters.
"""

import argparse
import gc
import json
import os
import random
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from tqdm import tqdm


BASE_MODEL_NAME = "EleutherAI/pythia-1b"
DEFAULT_RUN_DIR = "results/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2"
DEFAULT_OUTPUT_DIR = "results/lora_leak_official_mimir_hardsplit"
DEFAULT_DATA_DIR = "/workplace/FT/BlackNLP_2/results/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2/data"
SEED = 42

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
    path = resolve_existing_path(path_like)
    if (path / "adapter_config.json").exists():
        return path
    if (path / "adapter" / "adapter_config.json").exists():
        return path / "adapter"
    raise FileNotFoundError(f"adapter_config.json not found in {path} or {path / 'adapter'}")


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


def load_models(args):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = load_tokenizer(args.model_name)

    try:
        target_base = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype)
        pretrained = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype)
    except TypeError:
        target_base = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
        pretrained = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)

    adapter_dir = resolve_adapter_dir(args.adapter_dir)
    target = PeftModel.from_pretrained(target_base, str(adapter_dir), is_trainable=False)
    target = target.to(device).eval()
    pretrained = pretrained.to(device).eval()
    target.config.use_cache = False
    pretrained.config.use_cache = False
    return target, pretrained, tokenizer, adapter_dir


def encode_text(tokenizer, text: str, max_length: int, device: torch.device):
    ids = tokenizer.encode(str(text))
    if max_length and len(ids) > max_length:
        ids = ids[:max_length]
    if len(ids) < 2:
        return None
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


def basic_score_names(args) -> List[str]:
    names = ["loss", "zlib_loss"]
    for ratio in args.min_k_ratios:
        names.append(f"mink_{ratio:.1f}")
        names.append(f"mink++_{ratio:.1f}")
    if args.include_gradnormx:
        names.append("gradnormx")
    return names


@torch.no_grad()
def forward_scores(model, tokenizer, text: str, args) -> Dict[str, float]:
    device = model_device(model)
    input_ids = encode_text(tokenizer, text, args.max_length, device)
    if input_ids is None:
        return {"num_tokens": 0, "text_char_len": len(str(text))}

    outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    loss_value = float(outputs.loss.detach().float().cpu().item())
    logits = outputs.logits

    # Higher score is intended to mean stronger membership evidence.
    s_loss = -loss_value
    zlen = max(1, len(zlib.compress(bytes(str(text), "utf-8"))))
    scores = {
        "num_tokens": int(input_ids.shape[1]),
        "text_char_len": int(len(str(text))),
        "loss": s_loss,
        "raw_loss": loss_value,
        "zlib_loss": -loss_value / float(zlen),
    }

    shifted_ids = input_ids[0][1:].unsqueeze(-1)
    log_probs = F.log_softmax(logits[0, :-1].float(), dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=shifted_ids).squeeze(-1)

    probs = torch.exp(log_probs)
    mu = (probs * log_probs).sum(dim=-1)
    sigma2 = (probs * torch.square(log_probs)).sum(dim=-1) - torch.square(mu)
    sigma = torch.sqrt(torch.clamp(sigma2, min=1e-12))
    standardized = (token_log_probs - mu) / sigma

    token_log_probs_np = token_log_probs.detach().cpu().numpy()
    standardized_np = standardized.detach().cpu().numpy()
    order = np.argsort(token_log_probs_np)
    for ratio in args.min_k_ratios:
        k = max(1, int(len(token_log_probs_np) * ratio))
        idx = order[:k]
        scores[f"mink_{ratio:.1f}"] = float(token_log_probs_np[idx].mean())
        scores[f"mink++_{ratio:.1f}"] = float(standardized_np[idx].mean())
    return scores


def gradnormx_score(model, tokenizer, text: str, args) -> float:
    device = model_device(model)
    input_ids = encode_text(tokenizer, text, args.max_length, device)
    if input_ids is None:
        return np.nan

    model.zero_grad(set_to_none=True)
    embeds = model.get_input_embeddings()(input_ids).detach()
    embeds.requires_grad_(True)
    attention_mask = torch.ones_like(input_ids, device=device)
    outputs = model(inputs_embeds=embeds, attention_mask=attention_mask, labels=input_ids, use_cache=False)
    loss = outputs.loss
    loss.backward()
    score = -float(torch.linalg.vector_norm(embeds.grad.detach().float()).cpu().item())
    model.zero_grad(set_to_none=True)
    return score


def score_one_text(target_model, pretrained_model, tokenizer, text: str, args) -> Dict[str, float]:
    target = forward_scores(target_model, tokenizer, text, args)
    pretrained = forward_scores(pretrained_model, tokenizer, text, args)

    if args.include_gradnormx:
        target["gradnormx"] = gradnormx_score(target_model, tokenizer, text, args)
        pretrained["gradnormx"] = gradnormx_score(pretrained_model, tokenizer, text, args)

    out = {
        "num_tokens": target.get("num_tokens", np.nan),
        "text_char_len": target.get("text_char_len", len(str(text))),
    }
    for name in basic_score_names(args):
        out[f"target_{name}"] = target.get(name, np.nan)
        out[f"pretrained_{name}"] = pretrained.get(name, np.nan)
        out[f"{name}_refpt"] = pretrained.get(name, np.nan) - target.get(name, np.nan)
    return out


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
        if args.n_per_group > 0:
            df = df.sample(n=min(args.n_per_group, len(df)), random_state=args.seed).reset_index(drop=True)
        parts.append(df)
    return pd.concat(parts, ignore_index=True).reset_index(drop=True)


def tpr_at_fpr(y_true: np.ndarray, score: np.ndarray, target_fpr: float) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, score)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return 0.0, float("inf")
    idx = valid[np.argmax(tpr[valid])]
    return float(tpr[idx]), float(thresholds[idx])


def evaluate_score(scores: pd.DataFrame, spec: ComparisonSpec, score_col: str) -> Dict[str, object]:
    sub = scores[scores["group"].isin([spec.positive_group, spec.negative_group])].copy()
    sub = sub[np.isfinite(sub[score_col].to_numpy(float))]
    if sub.empty or sub["group"].nunique() != 2:
        return {}

    y = (sub["group"] == spec.positive_group).astype(int).to_numpy()
    raw = sub[score_col].to_numpy(float)
    pos_mean = float(sub.loc[sub["group"] == spec.positive_group, score_col].mean())
    neg_mean = float(sub.loc[sub["group"] == spec.negative_group, score_col].mean())

    auc = float(roc_auc_score(y, raw))
    auprc = float(average_precision_score(y, raw))
    tpr01, thr01 = tpr_at_fpr(y, raw, 0.01)
    tpr10, thr10 = tpr_at_fpr(y, raw, 0.10)

    direction = "higher" if pos_mean >= neg_mean else "lower"
    eff = raw if direction == "higher" else -raw
    eff_auc = float(roc_auc_score(y, eff))
    eff_auprc = float(average_precision_score(y, eff))
    eff_tpr01, eff_thr01 = tpr_at_fpr(y, eff, 0.01)
    eff_tpr10, eff_thr10 = tpr_at_fpr(y, eff, 0.10)

    return {
        "comparison": spec.name,
        "positive_group": spec.positive_group,
        "negative_group": spec.negative_group,
        "score_col": score_col,
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "positive_mean": pos_mean,
        "negative_mean": neg_mean,
        "strict_direction": "higher_score_means_positive",
        "score_direction_for_positive_by_mean": direction,
        "auroc": auc,
        "auprc": auprc,
        "tpr_at_1_fpr": tpr01,
        "threshold_at_1_fpr": thr01,
        "tpr_at_10_fpr": tpr10,
        "threshold_at_10_fpr": thr10,
        "effective_auroc": eff_auc,
        "effective_auprc": eff_auprc,
        "effective_tpr_at_1_fpr": eff_tpr01,
        "effective_threshold_at_1_fpr": eff_thr01,
        "effective_tpr_at_10_fpr": eff_tpr10,
        "effective_threshold_at_10_fpr": eff_thr10,
        "note": "Use auroc as strict fixed-direction metric; effective_* flips direction by observed group means and is diagnostic.",
    }


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_roc(scores: pd.DataFrame, spec: ComparisonSpec, score_col: str, out_dir: Path) -> None:
    sub = scores[scores["group"].isin([spec.positive_group, spec.negative_group])].copy()
    sub = sub[np.isfinite(sub[score_col].to_numpy(float))]
    if sub.empty or sub["group"].nunique() != 2:
        return
    y = (sub["group"] == spec.positive_group).astype(int).to_numpy()
    raw = sub[score_col].to_numpy(float)
    if sub.loc[sub["group"] == spec.positive_group, score_col].mean() < sub.loc[sub["group"] == spec.negative_group, score_col].mean():
        raw = -raw
    fpr, tpr, _ = roc_curve(y, raw)
    auc = roc_auc_score(y, raw)
    plt = setup_matplotlib()
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, label=f"effective AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"{spec.name}: {score_col}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    safe = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in score_col)
    fig.savefig(plot_dir / f"lora_leak_roc_{safe}.png", dpi=180)
    plt.close(fig)


def comparison_specs(args) -> List[ComparisonSpec]:
    all_specs = {s.name: s for s in DEFAULT_COMPARISONS}
    selected_names = args.experiments or list(all_specs.keys())
    unknown = [name for name in selected_names if name not in all_specs]
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}. Available: {list(all_specs)}")
    return [all_specs[name] for name in selected_names]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=BASE_MODEL_NAME)
    parser.add_argument("--run-dir", default=os.environ.get("MIMIR_HARDSPLIT_RUN_DIR", DEFAULT_RUN_DIR))
    parser.add_argument("--adapter-dir", default=os.environ.get("MIMIR_HARDSPLIT_ADAPTER_DIR", DEFAULT_RUN_DIR))
    parser.add_argument("--data-dir", default=os.environ.get("MIMIR_HARDSPLIT_DATA_DIR", DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-per-group", type=int, default=int(os.environ.get("N_PER_GROUP", "500")))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--experiments", nargs="*", default=[], help="Default: ft_vs_pt ft_vs_unseen pt_vs_unseen")

    parser.add_argument("--max-length", type=int, default=int(os.environ.get("MAX_LENGTH", "256")))
    parser.add_argument("--min-k-ratios", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--include-gradnormx", action="store_true", default=True)
    parser.add_argument("--no-gradnormx", dest="include_gradnormx", action="store_false")
    parser.add_argument("--plot-scores", nargs="+", default=["target_loss", "loss_refpt", "target_mink++_0.2", "mink++_0.2_refpt", "target_gradnormx", "gradnormx_refpt"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    targets = load_hardsplit_targets(args)
    targets.to_csv(out / "lora_leak_target_samples.csv", index=False)
    target_model, pretrained_model, tokenizer, adapter_dir = load_models(args)

    with open(out / "lora_leak_config.json", "w", encoding="utf-8") as f:
        config = vars(args).copy()
        config["resolved_adapter_dir"] = str(adapter_dir)
        config["target_counts"] = targets["group"].value_counts().to_dict()
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("Target counts:")
    print(targets["group"].value_counts().to_string())
    print(f"Resolved adapter: {adapter_dir}")

    rows = []
    for sample_id, row in tqdm(targets.iterrows(), total=len(targets), desc="LoRA-Leak scoring"):
        score = score_one_text(target_model, pretrained_model, tokenizer, str(row["text"]), args)
        score.update({
            "sample_id": int(sample_id),
            "group": row["group"],
            "source": row.get("source", ""),
            "original_index": int(row.get("original_index", sample_id)),
            "text": row["text"],
        })
        rows.append(score)

    scores_df = pd.DataFrame(rows)
    scores_df.to_csv(out / "lora_leak_scores.csv", index=False)

    score_cols = []
    for name in basic_score_names(args):
        score_cols.extend([f"target_{name}", f"pretrained_{name}", f"{name}_refpt"])
    score_cols = [c for c in score_cols if c in scores_df.columns and scores_df[c].notna().sum() > 0]

    result_rows = []
    for spec in comparison_specs(args):
        comp_dir = out / spec.name
        comp_dir.mkdir(parents=True, exist_ok=True)
        for col in score_cols:
            res = evaluate_score(scores_df, spec, col)
            if res:
                result_rows.append(res)
                if col in args.plot_scores:
                    plot_roc(scores_df, spec, col, comp_dir)

    results = pd.DataFrame(result_rows)
    results.to_csv(out / "lora_leak_pairwise_results.csv", index=False)
    strict_ranked = results.sort_values(["comparison", "auroc"], ascending=[True, False])
    effective_ranked = results.sort_values(["comparison", "effective_auroc"], ascending=[True, False])
    strict_ranked.to_csv(out / "lora_leak_all_strict_ranked_results.csv", index=False)
    effective_ranked.to_csv(out / "lora_leak_all_effective_ranked_results.csv", index=False)

    with open(out / "lora_leak_summary.txt", "w", encoding="utf-8") as f:
        f.write("Strict ranking by fixed score direction: higher score = positive group\n")
        f.write(strict_ranked.groupby("comparison").head(8).to_string(index=False))
        f.write("\n\nDiagnostic ranking with direction flipped by observed group means\n")
        f.write(effective_ranked.groupby("comparison").head(8).to_string(index=False))
        f.write("\n")

    del target_model, pretrained_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n=== LoRA-Leak strict top results ===")
    print(strict_ranked.groupby("comparison").head(8).to_string(index=False))
    print("\n=== LoRA-Leak effective top results ===")
    print(effective_ranked.groupby("comparison").head(8).to_string(index=False))


if __name__ == "__main__":
    main()
