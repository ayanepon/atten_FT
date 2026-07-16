# -*- coding: utf-8 -*-
"""
Standalone attention-update extraction for the MIMIR Wikipedia hard split.

This file does not import any previous experiment script.  It implements the
same core procedure directly:
  1. load the FT LoRA adapter,
  2. extract attentions before per-sample overfitting,
  3. overfit one target sample with early stopping,
  4. extract attentions after overfitting,
  5. save layer/head attention-shift metrics and loss changes.
"""

import gc
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CPU_THREAD_LIMIT = os.environ.get("CPU_THREAD_LIMIT", "1")
GPU_DEVICE_INDEX = os.environ.get("GPU_DEVICE_INDEX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", CPU_THREAD_LIMIT)
os.environ.setdefault("MKL_NUM_THREADS", CPU_THREAD_LIMIT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", CPU_THREAD_LIMIT)
os.environ.setdefault("NUMEXPR_NUM_THREADS", CPU_THREAD_LIMIT)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", CPU_THREAD_LIMIT)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

try:
    torch.set_num_threads(int(CPU_THREAD_LIMIT))
    torch.set_num_interop_threads(1)
except Exception:
    pass


BASE_MODEL_NAME = os.environ.get("BASE_MODEL_NAME", "EleutherAI/pythia-1b")
DEFAULT_BASE_DIR = os.environ.get(
    "MIMIR_HARDSPLIT_BASE_DIR",
    "results/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2",
)
DEFAULT_OUTPUT_ROOT = os.environ.get(
    "MIMIR_HARDSPLIT_ATTENTION_OUTPUT_ROOT",
    "results/mimir_wikipedia_hardsplit_attention",
)
DEFAULT_ADAPTER_DIR = os.environ.get(
    "MIMIR_HARDSPLIT_ADAPTER_DIR",
    f"{DEFAULT_BASE_DIR}/adapter",
)

SEED = int(os.environ.get("SEED", "42"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "256"))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0"))

LR = float(os.environ.get("OVERFIT_LR", "1e-5"))
EARLY_STOPPING_PATIENCE = int(os.environ.get("EARLY_STOPPING_PATIENCE", "50"))
EARLY_STOPPING_TOL = float(os.environ.get("EARLY_STOPPING_TOL", "1e-6"))
EARLY_STOPPING_MIN_STEPS = int(os.environ.get("EARLY_STOPPING_MIN_STEPS", "1"))
MAX_OVERFIT_STEPS = int(os.environ.get("MAX_OVERFIT_STEPS", "5000"))

TOPK_LOSS_PERCENT = int(os.environ.get("TOPK_LOSS_PERCENT", "10"))
TOP_SHIFT_PERCENTS = [1, 5, 10]
ATTENTION_METRICS = [
    "l1_mean",
    "l2_rms",
    "js_div",
    "entropy_delta",
    "max_shift",
] + [f"top{pct}_shift_mean" for pct in TOP_SHIFT_PERCENTS]

DEVICE = f"cuda:{GPU_DEVICE_INDEX}" if torch.cuda.is_available() else "cpu"

GROUP_SPECS = {
    "pt": {
        "group": "mimir_wikipedia_member_pt",
        "csv": "data/mimir_wikipedia_pt_member.csv",
        "output": "pt_member_attention",
        "label": 1,
    },
    "ft": {
        "group": "mimir_wikipedia_nonmember_ft",
        "csv": "data/mimir_wikipedia_ft_nonmember.csv",
        "output": "ft_nonmember_attention",
        "label": 0,
    },
    "unseen": {
        "group": "mimir_wikipedia_nonmember_unseen",
        "csv": "data/mimir_wikipedia_unseen_nonmember.csv",
        "output": "unseen_nonmember_attention",
        "label": 0,
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def write_status(output_dir: Path, message: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_status.txt").write_text(message.rstrip() + "\n", encoding="utf-8")


def resolve_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.exists():
        return path

    # Useful when copying scripts/results between a server workspace and a local mirror.
    local_candidates = [
        Path(path_like.replace("results/", "")),
        Path(path_like.replace("results/", "")),
        Path(path.name),
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Path not found: {path_like}")


def load_group_samples(group_key: str, base_dir: str, max_samples: int, seed: int) -> pd.DataFrame:
    spec = GROUP_SPECS[group_key]
    csv_path = resolve_path(str(Path(base_dir) / spec["csv"]))
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if "text" not in df.columns:
        raise ValueError(f"CSV must contain a text column: {csv_path}")
    if "group" not in df.columns:
        df["group"] = spec["group"]
    df = df[df["group"] == spec["group"]].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].drop_duplicates("text").reset_index(drop=True)
    if max_samples > 0:
        df = df.sample(n=min(max_samples, len(df)), random_state=seed).reset_index(drop=True)

    return pd.DataFrame({
        "group": spec["group"],
        "source": df["source"] if "source" in df.columns else spec["group"],
        "label": spec["label"],
        "wikimia_config": "",
        "text": df["text"],
        "label_text": "",
        "original_index": df["original_index"] if "original_index" in df.columns else np.arange(len(df), dtype=int),
    })


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_before_model_trainable(adapter_dir: str):
    if PeftModel is None:
        raise ImportError("peft is required to load the LoRA adapter.")

    cfg = AutoConfig.from_pretrained(BASE_MODEL_NAME)
    try:
        cfg.attn_implementation = "eager"
    except Exception:
        pass
    cfg.output_attentions = True

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        config=cfg,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        base = base.to(DEVICE)
    base.config.use_cache = False

    model = PeftModel.from_pretrained(base, str(resolve_path(adapter_dir)), is_trainable=True)
    model.config.use_cache = False
    model.eval()
    return model


def model_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def make_lm_encoding(tokenizer, text: str) -> Dict[str, torch.Tensor]:
    enc = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    enc["labels"] = labels
    return enc


def move_batch_to_device(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def compute_sequence_loss(model, train_enc: Dict[str, torch.Tensor]) -> float:
    batch = move_batch_to_device(train_enc, model_device(model))
    outputs = model(**batch, output_attentions=False, use_cache=False)
    return float(outputs.loss.detach().float().cpu())


@torch.no_grad()
def compute_token_losses(model, tokenizer, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )
    enc_dev = {k: v.to(model_device(model)) for k, v in enc.items()}
    outputs = model(
        input_ids=enc_dev["input_ids"],
        attention_mask=enc_dev["attention_mask"],
        output_attentions=False,
        use_cache=False,
    )
    logits = outputs.logits.float()
    input_ids = enc_dev["input_ids"]
    attention_mask = enc_dev["attention_mask"]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]
    loss_per_token = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    ).reshape(1, -1)
    return loss_per_token[0].detach().cpu(), shift_mask[0].detach().cpu()


def select_topk_query_positions(losses: torch.Tensor, mask: torch.Tensor, topk_percent: int) -> List[int]:
    valid = torch.where(mask == 1)[0]
    if len(valid) == 0:
        return []
    valid_losses = losses[valid]
    k = max(1, int(len(valid) * topk_percent / 100))
    top_idx = torch.topk(valid_losses, k=k).indices
    selected = valid[top_idx] + 1
    selected = selected[selected < MAX_LENGTH]
    return selected.tolist()


@torch.no_grad()
def get_attentions(model, tokenizer, text: str):
    enc = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )
    enc_dev = {k: v.to(model_device(model)) for k, v in enc.items()}
    outputs = model(
        input_ids=enc_dev["input_ids"],
        attention_mask=enc_dev["attention_mask"],
        output_attentions=True,
        use_cache=False,
    )
    return getattr(outputs, "attentions", None)


def get_attention_key_mask(tokenizer, text: str) -> torch.Tensor:
    enc = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )
    return enc["attention_mask"][0].detach().cpu().bool()


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.float().clamp(min=eps)
    q = q.float().clamp(min=eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.float().clamp(min=eps)
    p = p / p.sum(dim=-1, keepdim=True)
    return -(p * p.log()).sum(dim=-1)


def _valid_key_mask_for_query(query_position: int, key_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
    causal = torch.arange(seq_len, dtype=torch.long) <= int(query_position)
    return causal & key_mask[:seq_len].bool()


def attention_shift_metrics(
    attn_before,
    attn_after,
    selected_positions: List[int],
    key_mask: Optional[torch.Tensor] = None,
) -> List[Dict]:
    rows = []
    if attn_before is None or attn_after is None:
        return rows

    for layer in range(len(attn_before)):
        a_b = attn_before[layer][0].detach().cpu()
        a_a = attn_after[layer][0].detach().cpu()
        seq_len = int(a_b.shape[-1])
        if key_mask is None:
            effective_key_mask = torch.ones(seq_len, dtype=torch.bool)
        else:
            effective_key_mask = key_mask[:seq_len].detach().cpu().bool()

        for head in range(a_b.shape[0]):
            if selected_positions:
                query_positions = [int(pos) for pos in selected_positions if int(pos) < seq_len]
            else:
                query_positions = [
                    int(pos)
                    for pos in torch.where(effective_key_mask)[0].tolist()
                    if int(pos) < seq_len
                ]

            p_valid = []
            q_valid = []
            js_values = []
            entropy_before_values = []
            entropy_after_values = []
            valid_key_counts = []
            for query_position in query_positions:
                valid_keys = _valid_key_mask_for_query(
                    query_position, effective_key_mask, seq_len
                )
                if not bool(valid_keys.any()):
                    continue
                p_vec = a_b[head, query_position, valid_keys]
                q_vec = a_a[head, query_position, valid_keys]
                p_valid.append(p_vec.float())
                q_valid.append(q_vec.float())
                js_values.append(js_divergence(p_vec, q_vec))
                entropy_before_values.append(entropy(p_vec))
                entropy_after_values.append(entropy(q_vec))
                valid_key_counts.append(int(valid_keys.sum().item()))

            if not p_valid:
                continue

            p_flat = torch.cat(p_valid)
            q_flat = torch.cat(q_valid)

            abs_diff = (q_flat - p_flat).abs()
            sq_diff = (q_flat - p_flat) ** 2
            flat = abs_diff.reshape(-1)
            entropy_before_mean = torch.stack(entropy_before_values).mean()
            entropy_after_mean = torch.stack(entropy_after_values).mean()
            row = {
                "layer": int(layer),
                "head": int(head),
                "l1_mean": float(abs_diff.mean()),
                "l2_rms": float(torch.sqrt(sq_diff.mean())),
                "js_div": float(torch.stack(js_values).mean()),
                "entropy_before": float(entropy_before_mean),
                "entropy_after": float(entropy_after_mean),
                "entropy_delta": float(entropy_after_mean - entropy_before_mean),
                "max_shift": float(flat.max()),
                "num_topk_loss_queries": int(len(selected_positions)),
                "num_attention_queries_used": int(len(valid_key_counts)),
                "num_valid_attention_entries": int(flat.numel()),
                "mean_valid_keys_per_query": float(np.mean(valid_key_counts)),
            }
            for pct in TOP_SHIFT_PERCENTS:
                k = max(1, int(flat.numel() * pct / 100))
                row[f"top{pct}_shift_mean"] = float(torch.topk(flat, k=k).values.mean())
            rows.append(row)
    return rows


def overfit_one_sample(model, train_enc: Dict[str, torch.Tensor]) -> Tuple[List[float], int]:
    model.train()
    batch = move_batch_to_device(train_enc, model_device(model))
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found. Check LoRA adapter loading.")
    optimizer = torch.optim.AdamW(params, lr=LR)

    losses: List[float] = []
    best_loss = float("inf")
    best_acc = -float("inf")
    patience_counter = 0
    step = 0

    while True:
        if step >= MAX_OVERFIT_STEPS:
            print(f"Reached safety max steps: {MAX_OVERFIT_STEPS}. Stopping.")
            break

        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch, output_attentions=False, use_cache=False)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        cur_loss = float(loss.detach().float().cpu())
        losses.append(cur_loss)

        cur_acc = float("nan")
        try:
            with torch.no_grad():
                pred = outputs.logits[:, :-1, :].detach().cpu().argmax(dim=-1)
                labels = batch["labels"].detach().cpu()
                shifted_labels = labels[:, 1:]
                mask = shifted_labels != -100
                if mask.sum().item() > 0:
                    cur_acc = float(((pred == shifted_labels) & mask).sum().item()) / float(mask.sum().item())
        except Exception:
            pass

        improved = False
        if best_loss - cur_loss > EARLY_STOPPING_TOL:
            best_loss = cur_loss
            improved = True
        if not math.isnan(cur_acc) and cur_acc - best_acc > EARLY_STOPPING_TOL:
            best_acc = cur_acc
            improved = True

        patience_counter = 0 if improved else patience_counter + 1
        step += 1

        if step >= EARLY_STOPPING_MIN_STEPS and patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at step {step}: no improvement for {patience_counter} consecutive steps")
            break

    model.eval()
    return losses, step


def make_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()
    agg = {m: ["mean", "std", "median"] for m in ATTENTION_METRICS}
    summary = raw_df.groupby(["group", "layer", "head"]).agg(agg)
    summary.columns = ["_".join(c).strip() for c in summary.columns.values]
    return summary.reset_index()


def save_progress(output_dir: Path, all_rows: List[Dict], sample_rows: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = pd.DataFrame(all_rows)
    sample_df = pd.DataFrame(sample_rows)
    atomic_to_csv(raw_df, output_dir / "raw_samplewise_overfit_attention_shift.csv")
    atomic_to_csv(sample_df, output_dir / "sample_level_overfit_loss.csv")
    return raw_df, sample_df


def plot_single_group_distributions(output_dir: Path, raw_df: pd.DataFrame) -> None:
    if raw_df.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots_attention"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for metric in ATTENTION_METRICS:
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        vals = raw_df[metric].dropna().to_numpy(float)
        ax.boxplot([vals], labels=[raw_df["group"].iloc[0]], showfliers=False)
        ax.set_title(f"{metric} distribution")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{metric}_distribution.png", dpi=180)
        plt.close(fig)


def reset_outputs(output_dir: Path) -> None:
    for name in [
        "target_samples_used.csv",
        "raw_samplewise_overfit_attention_shift.csv",
        "sample_level_overfit_loss.csv",
        "summary_by_group_layer_head.csv",
        "run_status.txt",
    ]:
        path = output_dir / name
        if path.exists():
            path.unlink()


def run_group(
    group_key: str,
    base_dir: str = DEFAULT_BASE_DIR,
    adapter_dir: str = DEFAULT_ADAPTER_DIR,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    max_samples: int = MAX_SAMPLES,
) -> None:
    if group_key not in GROUP_SPECS:
        raise ValueError(f"Unknown group_key={group_key}. Available: {list(GROUP_SPECS)}")

    spec = GROUP_SPECS[group_key]
    output_dir = Path(output_root) / spec["output"]
    output_dir.mkdir(parents=True, exist_ok=True)
    reset_outputs(output_dir)
    write_status(output_dir, "running")
    set_seed(SEED)

    tokenizer = load_tokenizer()
    samples = load_group_samples(group_key, base_dir, max_samples, SEED)
    atomic_to_csv(samples, output_dir / "target_samples_used.csv")

    resolved_adapter = str(resolve_path(adapter_dir))
    print("Target samples:")
    print(samples["group"].value_counts().to_string())
    print(f"Adapter: {resolved_adapter}")
    print(f"Output: {output_dir}")
    print(
        "Overfit condition: "
        f"LR={LR}, max_steps={MAX_OVERFIT_STEPS}, patience={EARLY_STOPPING_PATIENCE}, "
        f"max_length={MAX_LENGTH}, topk_loss_percent={TOPK_LOSS_PERCENT}"
    )

    all_rows: List[Dict] = []
    sample_rows: List[Dict] = []

    for sample_id, row in tqdm(samples.iterrows(), total=len(samples), desc=f"{group_key} overfit"):
        print(f"\n=== sample_id={sample_id}, group={row['group']} ===")
        model = load_before_model_trainable(resolved_adapter)
        train_enc = make_lm_encoding(tokenizer, str(row["text"]))
        analysis_text = str(row["text"])

        before_loss = compute_sequence_loss(model, train_enc)
        token_losses, token_mask = compute_token_losses(model, tokenizer, analysis_text)
        selected_positions = select_topk_query_positions(token_losses, token_mask, TOPK_LOSS_PERCENT)
        attention_key_mask = get_attention_key_mask(tokenizer, analysis_text)
        attn_before = get_attentions(model, tokenizer, analysis_text)

        train_losses, actual_steps = overfit_one_sample(model, train_enc)

        after_loss = compute_sequence_loss(model, train_enc)
        attn_after = get_attentions(model, tokenizer, analysis_text)

        metric_rows = attention_shift_metrics(
            attn_before,
            attn_after,
            selected_positions,
            attention_key_mask,
        )
        for mr in metric_rows:
            mr.update({
                "sample_id": int(sample_id),
                "group": row["group"],
                "source": row["source"],
                "label": int(row["label"]),
                "wikimia_config": row.get("wikimia_config", ""),
                "before_loss": before_loss,
                "after_loss": after_loss,
                "delta_loss_before_minus_after": before_loss - after_loss,
                "train_loss_first": train_losses[0] if train_losses else np.nan,
                "train_loss_last": train_losses[-1] if train_losses else np.nan,
                "overfit_steps": int(actual_steps),
                "topk_loss_percent": TOPK_LOSS_PERCENT,
            })
        all_rows.extend(metric_rows)

        sample_rows.append({
            "sample_id": int(sample_id),
            "group": row["group"],
            "source": row["source"],
            "label": int(row["label"]),
            "wikimia_config": row.get("wikimia_config", ""),
            "before_loss": before_loss,
            "after_loss": after_loss,
            "delta_loss_before_minus_after": before_loss - after_loss,
            "train_loss_first": train_losses[0] if train_losses else np.nan,
            "train_loss_last": train_losses[-1] if train_losses else np.nan,
            "num_topk_loss_queries": len(selected_positions),
            "analysis_text_char_len": len(analysis_text),
            "overfit_steps": int(actual_steps),
        })

        del model, attn_before, attn_after
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        raw_df, sample_df = save_progress(output_dir, all_rows, sample_rows)
        write_status(
            output_dir,
            "running\n"
            f"group_key={group_key}\n"
            f"processed_sample_counts={sample_df['group'].value_counts().to_dict() if 'group' in sample_df.columns else {}}\n"
            f"raw_attention_rows={len(raw_df)}",
        )

    raw_df, sample_df = save_progress(output_dir, all_rows, sample_rows)
    summary = make_summary(raw_df)
    atomic_to_csv(summary, output_dir / "summary_by_group_layer_head.csv")
    plot_single_group_distributions(output_dir, raw_df)
    write_status(
        output_dir,
        "completed\n"
        f"group_key={group_key}\n"
        f"processed_sample_counts={sample_df['group'].value_counts().to_dict() if 'group' in sample_df.columns else {}}\n"
        f"raw_attention_rows={len(raw_df)}",
    )
    print(f"Saved to: {output_dir}")
