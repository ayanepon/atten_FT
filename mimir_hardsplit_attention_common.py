# -*- coding: utf-8 -*-
"""
Standalone attention-update extraction for the MIMIR Wikipedia hard split.

Paper-aligned procedure (acl_latex.tex):
  1. load the FT LoRA adapter,
  2. select top-rho% highest next-token loss positions as query set Q,
  3. extract attentions before target-specific additional training (eval, no dropout),
  4. optimize only LoRA parameters on that sample (fixed steps or early stopping),
  5. extract attentions after training with the same Q (eval, no dropout),
  6. compute layer/head attention-update features over masked attention rows.

Attention features use causal + padding masks: for each query i in Q, only valid
key positions j <= i with attention_mask[j]=1 are used (paper Sec. Proposed Method).
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
    from model_registry import (
        DEFAULT_HF_ID,
        read_base_model_from_adapter,
        resolve_adapter_dir,
        resolve_model_name,
    )
except ImportError:  # pragma: no cover
    DEFAULT_HF_ID = "EleutherAI/pythia-1b"
    read_base_model_from_adapter = None
    resolve_adapter_dir = None
    resolve_model_name = None

try:
    torch.set_num_threads(int(CPU_THREAD_LIMIT))
    torch.set_num_interop_threads(1)
except Exception:
    pass


BASE_MODEL_NAME = os.environ.get("BASE_MODEL_NAME", DEFAULT_HF_ID)
DEFAULT_BASE_DIR = os.environ.get(
    "MIMIR_HARDSPLIT_BASE_DIR",
    "results/mimir_wikipedia_hardsplit_lora_ft",
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

# Sample-wise additional training (paper experiments / Japanese draft): lr = 1e-5.
# Note: English body text incorrectly says 1e-4; LoRA FT itself uses 1e-4.
# Override with OVERFIT_LR if needed.
LR = float(os.environ.get("OVERFIT_LR", "1e-5"))
EARLY_STOPPING_PATIENCE = int(os.environ.get("EARLY_STOPPING_PATIENCE", "50"))
EARLY_STOPPING_TOL = float(os.environ.get("EARLY_STOPPING_TOL", "1e-6"))
EARLY_STOPPING_MIN_STEPS = int(os.environ.get("EARLY_STOPPING_MIN_STEPS", "1"))
MAX_OVERFIT_STEPS = int(os.environ.get("MAX_OVERFIT_STEPS", "5000"))

# Paper: rho = 10 (top 10% highest token losses as queries)
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


def get_model_name(model_name: Optional[str] = None, adapter_dir: Optional[str] = None) -> str:
    """Resolve HF model id: explicit > adapter_config > env BASE_MODEL_NAME."""
    if resolve_model_name is not None:
        return resolve_model_name(explicit=model_name, adapter_dir=adapter_dir, default=BASE_MODEL_NAME)
    if model_name:
        return model_name
    if adapter_dir and read_base_model_from_adapter is not None:
        inferred = read_base_model_from_adapter(adapter_dir)
        if inferred:
            return inferred
    return BASE_MODEL_NAME


def load_tokenizer(model_name: Optional[str] = None, adapter_dir: Optional[str] = None):
    name = get_model_name(model_name=model_name, adapter_dir=adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_before_model_trainable(adapter_dir: str, model_name: Optional[str] = None):
    """Load base model + LoRA once (prefer reusing with snapshot/restore)."""
    if PeftModel is None:
        raise ImportError("peft is required to load the LoRA adapter.")

    resolved_adapter = str(resolve_path(adapter_dir))
    name = get_model_name(model_name=model_name, adapter_dir=resolved_adapter)
    print(f"Loading base model: {name}")
    print(f"Loading adapter: {resolved_adapter}")

    cfg = AutoConfig.from_pretrained(name)
    try:
        cfg.attn_implementation = "eager"
    except Exception:
        pass
    cfg.output_attentions = True

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    load_kwargs = {"config": cfg}
    try:
        base = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **load_kwargs)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, **load_kwargs)
    if torch.cuda.is_available():
        base = base.to(DEVICE)
    base.config.use_cache = False
    base.config.output_attentions = True

    model = PeftModel.from_pretrained(base, resolved_adapter, is_trainable=True)
    model.config.use_cache = False
    model.config.output_attentions = True
    model.eval()
    return model


def snapshot_trainable_state(model, on_cpu: Optional[bool] = None) -> Dict[str, torch.Tensor]:
    """Clone trainable (LoRA) parameters for per-sample reset.

    Default keeps tensors on the model device (faster restore). Set
    ``SNAPSHOT_ON_CPU=1`` or ``on_cpu=True`` to store on CPU (saves VRAM).
    """
    if on_cpu is None:
        on_cpu = os.environ.get("SNAPSHOT_ON_CPU", "0").strip() in {"1", "true", "True", "yes"}
    state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            t = param.detach().clone()
            if on_cpu:
                t = t.cpu()
            state[name] = t
    if not state:
        raise RuntimeError("No trainable parameters found for snapshot.")
    return state


@torch.no_grad()
def restore_trainable_state(model, state: Dict[str, torch.Tensor]) -> None:
    """Restore LoRA weights to the FT checkpoint after sample-wise updates."""
    name_to_param = {name: param for name, param in model.named_parameters() if param.requires_grad}
    missing = [k for k in state if k not in name_to_param]
    if missing:
        raise RuntimeError(f"Missing trainable params during restore: {missing[:5]}")
    for name, tensor in state.items():
        dst = name_to_param[name]
        if tensor.device == dst.device and tensor.dtype == dst.dtype:
            dst.copy_(tensor)
        else:
            dst.copy_(tensor.to(device=dst.device, dtype=dst.dtype, non_blocking=True))
    model.eval()


def model_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def make_lm_encoding(tokenizer, text: str) -> Dict[str, torch.Tensor]:
    """Tokenize one sample without right-padding to max_length.

    Paper uses max length 256 with causal LM loss over non-padding next-token
    positions. For single-sample training, padding is unnecessary and would
    dilute attention statistics if not masked carefully.
    """
    enc = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
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
    """Per next-token CE losses and validity mask (paper: non-padding positions)."""
    enc = make_lm_encoding(tokenizer, text)
    _, token_losses, token_mask = compute_before_diagnostics(model, enc)
    return token_losses, token_mask


@torch.no_grad()
def compute_token_logit_gradient_norms(
    model,
    train_enc: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact per-token norm of d(CE)/d(logits), used for query ablation.

    This avoids one backward pass per token.  For cross entropy, the gradient
    with respect to the logit vector is ``softmax(logits) - one_hot(label)``;
    its L2 norm is therefore available from a single forward pass.  It is a
    token-selection control, not a parameter-gradient attribution method.
    """
    model.eval()
    batch = move_batch_to_device(train_enc, model_device(model))
    outputs = model(**batch, output_attentions=False, use_cache=False)
    logits = outputs.logits[:, :-1, :].float()
    labels = batch["input_ids"][:, 1:]
    mask = batch["attention_mask"][:, 1:]
    probs = torch.softmax(logits, dim=-1)
    true_prob = probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    norm_sq = probs.square().sum(dim=-1) - 2.0 * true_prob + 1.0
    norms = norm_sq.clamp_min(0.0).sqrt()
    return norms[0].detach().cpu(), mask[0].detach().cpu().long()


@torch.no_grad()
def compute_before_diagnostics(
    model,
    train_enc: Dict[str, torch.Tensor],
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """One forward for sequence loss + per-token CE losses (CPU tensors for token stats).

    Returns
    -------
    sequence_loss : float
    token_losses : 1D CPU float tensor over next-token positions
    token_mask : 1D CPU long tensor (1 = valid non-pad next-token position)
    """
    batch = move_batch_to_device(train_enc, model_device(model))
    outputs = model(**batch, output_attentions=False, use_cache=False)
    sequence_loss = float(outputs.loss.detach().float().cpu())

    logits = outputs.logits.float()
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]
    loss_per_token = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    ).reshape(1, -1)
    return (
        sequence_loss,
        loss_per_token[0].detach().cpu(),
        shift_mask[0].detach().cpu(),
    )


@torch.no_grad()
def compute_diagnostics_and_attentions(
    model,
    train_enc: Dict[str, torch.Tensor],
) -> Tuple[float, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor]:
    """Compute before-loss diagnostics and attention weights in one forward.

    The extraction pipeline needs the same pre-update forward for sequence
    loss, top-loss query selection, and attention snapshots.  Keeping these in
    one call removes a redundant model pass per sample without changing the
    masks or the loss definition used by :func:`compute_before_diagnostics`.
    """
    model.eval()
    batch = move_batch_to_device(train_enc, model_device(model))
    outputs = model(**batch, output_attentions=True, use_cache=False)
    attns = getattr(outputs, "attentions", None)
    if attns is None:
        raise RuntimeError(
            "Model returned no attentions. Use attn_implementation='eager' and "
            "output_attentions=True (paper requires attention weights)."
        )

    sequence_loss = float(outputs.loss.detach().float().cpu())
    logits = outputs.logits.float()
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]
    loss_per_token = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    ).reshape(1, -1)
    token_losses = loss_per_token[0].detach().cpu()
    token_mask = shift_mask[0].detach().cpu()
    mask = attention_mask[0].detach().cpu().long()
    # Drop the large logits/model-output references before returning the
    # attention tuple, which remains live on the model device.
    del outputs, logits, shift_logits, loss_per_token
    return (
        sequence_loss,
        token_losses,
        token_mask,
        tuple(attns),
        mask,
    )


@torch.no_grad()
def compute_sequence_loss_and_attentions(
    model,
    train_enc: Dict[str, torch.Tensor],
) -> Tuple[float, Tuple[torch.Tensor, ...], torch.Tensor]:
    """Compute post-update sequence loss and attention weights in one forward."""
    model.eval()
    batch = move_batch_to_device(train_enc, model_device(model))
    outputs = model(**batch, output_attentions=True, use_cache=False)
    attns = getattr(outputs, "attentions", None)
    if attns is None:
        raise RuntimeError(
            "Model returned no attentions. Use attn_implementation='eager' and "
            "output_attentions=True (paper requires attention weights)."
        )
    sequence_loss = float(outputs.loss.detach().float().cpu())
    mask = batch["attention_mask"][0].detach().cpu().long()
    del outputs
    return sequence_loss, tuple(attns), mask


def select_query_positions(
    losses: torch.Tensor,
    mask: torch.Tensor,
    topk_percent: int,
    *,
    query_position_offset: int = 1,
    selection_mode: str = "top_loss",
    random_seed: int = 42,
) -> List[int]:
    """Select attention query positions for position-selection ablations.

    ``losses`` and ``mask`` are over next-token indices ``t=0..T-2``.
    The paper condition maps a selected loss index to query position ``t+1``.
    ``query_position_offset=0`` provides the predictor-state alternative, while
    ``selection_mode`` supports deterministic ``top_loss``, ``low_loss``,
    ``random``, and ``all_valid`` controls.  The default is exactly the paper
    procedure, so existing runs remain unchanged.
    """
    if selection_mode not in {"top_loss", "low_loss", "random", "all_valid"}:
        raise ValueError(
            "selection_mode must be one of top_loss, low_loss, random, all_valid"
        )
    if query_position_offset not in {0, 1}:
        raise ValueError("query_position_offset must be 0 or 1")
    valid = torch.where(mask == 1)[0]
    if len(valid) == 0:
        return []
    k = max(1, int(math.ceil(len(valid) * topk_percent / 100.0)))
    k = min(k, len(valid))
    if selection_mode == "all_valid":
        selected = valid
    else:
        valid_losses = losses[valid]
        if selection_mode == "top_loss":
            selected_idx = torch.topk(valid_losses, k=k).indices
        elif selection_mode == "low_loss":
            selected_idx = torch.topk(valid_losses, k=k, largest=False).indices
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(random_seed))
            selected_idx = torch.randperm(len(valid), generator=generator)[:k]
        selected = valid[selected_idx]
    # query position = next-token index + configured offset
    selected = selected + int(query_position_offset)
    seq_len = int(mask.numel()) + 1  # original length estimate
    selected = selected[selected < seq_len]
    return sorted(set(int(x) for x in selected.tolist()))


def select_topk_query_positions(
    losses: torch.Tensor, mask: torch.Tensor, topk_percent: int
) -> List[int]:
    """Backward-compatible wrapper for the canonical top-loss, t+1 mapping."""
    return select_query_positions(
        losses,
        mask,
        topk_percent,
        query_position_offset=1,
        selection_mode="top_loss",
    )


@torch.no_grad()
def get_attentions(model, tokenizer, text: str, enc: Optional[Dict[str, torch.Tensor]] = None):
    """Return (attentions tuple, attention_mask 1D long CPU tensor).

    Prefer passing the same `enc` used for loss/training so token alignment is exact.
    """
    model.eval()
    if enc is None:
        enc = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
            return_tensors="pt",
        )
    # Only forward ids/mask (labels not needed for attention)
    fwd = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    }
    enc_dev = move_batch_to_device(fwd, model_device(model))
    outputs = model(
        input_ids=enc_dev["input_ids"],
        attention_mask=enc_dev["attention_mask"],
        output_attentions=True,
        use_cache=False,
    )
    attns = getattr(outputs, "attentions", None)
    if attns is None:
        raise RuntimeError(
            "Model returned no attentions. Use attn_implementation='eager' and "
            "output_attentions=True (paper requires attention weights)."
        )
    mask = enc["attention_mask"][0].detach().cpu().long()
    return attns, mask


def _normalize_dist(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.float().clamp(min=0.0)
    s = p.sum(dim=-1, keepdim=True).clamp(min=eps)
    return p / s


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """JSD over the last dimension; p,q already restricted to valid keys."""
    p = _normalize_dist(p, eps)
    q = _normalize_dist(q, eps)
    m = 0.5 * (p + q)
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    m = m.clamp(min=eps)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = _normalize_dist(p, eps).clamp(min=eps)
    return -(p * p.log()).sum(dim=-1)


def _valid_key_index(query_pos: int, attention_mask_1d: torch.Tensor) -> torch.Tensor:
    """Keys j with padding mask and causal mask: j <= query_pos and mask[j]=1."""
    t = attention_mask_1d.numel()
    j = torch.arange(t, device=attention_mask_1d.device)
    return torch.where((attention_mask_1d == 1) & (j <= int(query_pos)))[0]


def attention_shift_metrics(
    attn_before,
    attn_after,
    selected_positions: List[int],
    attention_mask: Optional[torch.Tensor] = None,
) -> List[Dict]:
    """Paper features over selected queries and masked key positions.

    MeanDiff / RMSE / MaxShift / TopK use element-wise Delta A on valid (i,j).
    JSD / Entropy use renormalized distributions on the same valid keys.
    Entropy diff is signed: H(after) - H(before).

    Performance: keeps tensors on the attention device (GPU when available),
    processes all heads for each query together, and only syncs scalars.
    """
    rows = []
    if attn_before is None or attn_after is None or not attn_before:
        return rows

    # Default mask: all positions valid up to sequence length of attention.
    # The vectorized implementation below retains the exact causal + padding
    # restriction while eliminating Python loops over queries and heads.
    device = attn_before[0].device
    seq_len = int(attn_before[0].shape[-1])
    if attention_mask is None:
        attention_mask = torch.ones(seq_len, dtype=torch.long, device=device)
    else:
        attention_mask = attention_mask.detach().to(device=device, dtype=torch.long).view(-1)
        if attention_mask.numel() < seq_len:
            pad = torch.zeros(seq_len - attention_mask.numel(), dtype=torch.long, device=device)
            attention_mask = torch.cat([attention_mask, pad], dim=0)
        attention_mask = attention_mask[:seq_len]

    queries = list(selected_positions) if selected_positions else list(range(seq_len))
    queries = [
        int(q)
        for q in queries
        if 0 <= int(q) < seq_len and int(attention_mask[int(q)].item()) == 1
    ]
    if not queries:
        return rows

    query_idx = torch.as_tensor(queries, dtype=torch.long, device=device)
    key_idx = torch.arange(seq_len, dtype=torch.long, device=device)
    valid = (attention_mask[key_idx].bool().unsqueeze(0)) & (key_idx.unsqueeze(0) <= query_idx.unsqueeze(1))
    if not bool(valid.any()):
        return rows
    valid_h = valid.unsqueeze(0)
    n_queries = int(query_idx.numel())
    n_valid = int(valid.sum().item())

    for layer in range(len(attn_before)):
        # [H, Q, T] on device; invalid keys are excluded from every metric.
        a_b = attn_before[layer][0].detach().float()[:, query_idx, :]
        a_a = attn_after[layer][0].detach().float()[:, query_idx, :]
        d = a_a - a_b
        abs_d = d.abs()
        sq_d = d * d
        abs_flat = abs_d.masked_select(valid_h).reshape(abs_d.shape[0], n_valid)
        sq_flat = sq_d.masked_select(valid_h).reshape(sq_d.shape[0], n_valid)

        # Renormalization is over valid keys per query, matching the original
        # per-query implementation before JSD and entropy are computed.
        p = a_b.masked_fill(~valid_h, 0.0)
        r = a_a.masked_fill(~valid_h, 0.0)
        js = js_divergence(p, r).mean(dim=1)
        ent_b = entropy(p).mean(dim=1)
        ent_a = entropy(r).mean(dim=1)

        metric_values = [
            abs_flat.mean(dim=1),
            torch.sqrt(sq_flat.mean(dim=1)),
            js,
            ent_b,
            ent_a,
            ent_a - ent_b,
            abs_flat.max(dim=1).values,
        ]
        for pct in TOP_SHIFT_PERCENTS:
            k = max(1, min(n_valid, int(math.floor(n_valid * pct / 100.0))))
            metric_values.append(torch.topk(abs_flat, k=k, dim=1).values.mean(dim=1))

        values = torch.stack(metric_values, dim=1).detach().cpu().numpy()
        for head, row_values in enumerate(values):
            row = {
                "layer": int(layer),
                "head": int(head),
                "l1_mean": float(row_values[0]),
                "l2_rms": float(row_values[1]),
                "js_div": float(row_values[2]),
                "entropy_before": float(row_values[3]),
                "entropy_after": float(row_values[4]),
                "entropy_delta": float(row_values[5]),
                "max_shift": float(row_values[6]),
                "num_topk_loss_queries": n_queries,
                "num_valid_attention_elements": n_valid,
            }
            for i, pct in enumerate(TOP_SHIFT_PERCENTS, start=7):
                row[f"top{pct}_shift_mean"] = float(row_values[i])
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
        try:
            ax.boxplot([vals], tick_labels=[raw_df["group"].iloc[0]], showfliers=False)
        except TypeError:
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


def load_resume_state(
    raw_path: Path,
    sample_path: Path,
) -> Tuple[List[Dict], List[Dict], set]:
    """Load partial outputs and return rows plus finished sample_id set."""
    all_rows: List[Dict] = []
    sample_rows: List[Dict] = []
    done: set = set()
    if sample_path.exists() and sample_path.stat().st_size > 1:
        sample_df = pd.read_csv(sample_path)
        sample_rows = sample_df.to_dict(orient="records")
        if "sample_id" in sample_df.columns:
            done = set(int(x) for x in sample_df["sample_id"].tolist())
    if raw_path.exists() and raw_path.stat().st_size > 1:
        raw_df = pd.read_csv(raw_path)
        if done and "sample_id" in raw_df.columns:
            raw_df = raw_df[raw_df["sample_id"].isin(done)]
        all_rows = raw_df.to_dict(orient="records")
    return all_rows, sample_rows, done


def run_group(
    group_key: str,
    base_dir: str = DEFAULT_BASE_DIR,
    adapter_dir: str = DEFAULT_ADAPTER_DIR,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    max_samples: int = MAX_SAMPLES,
    resume: bool = True,
    model_name: Optional[str] = None,
) -> None:
    if group_key not in GROUP_SPECS:
        raise ValueError(f"Unknown group_key={group_key}. Available: {list(GROUP_SPECS)}")

    spec = GROUP_SPECS[group_key]
    output_dir = Path(output_root) / spec["output"]
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_samplewise_overfit_attention_shift.csv"
    sample_path = output_dir / "sample_level_overfit_loss.csv"

    if not resume:
        reset_outputs(output_dir)
        all_rows, sample_rows, done_ids = [], [], set()
    else:
        all_rows, sample_rows, done_ids = load_resume_state(raw_path, sample_path)

    write_status(output_dir, "running")
    set_seed(SEED)

    resolved_adapter = str(resolve_path(adapter_dir))
    resolved_model = get_model_name(model_name=model_name, adapter_dir=resolved_adapter)
    tokenizer = load_tokenizer(model_name=resolved_model, adapter_dir=resolved_adapter)
    samples = load_group_samples(group_key, base_dir, max_samples, SEED)
    atomic_to_csv(samples, output_dir / "target_samples_used.csv")

    print("Target samples:")
    print(samples["group"].value_counts().to_string())
    print(f"Model: {resolved_model}")
    print(f"Adapter: {resolved_adapter}")
    print(f"Output: {output_dir}")
    print(f"Resume: {resume}, already_done={len(done_ids)}")
    print(
        "Overfit condition: "
        f"LR={LR}, max_steps={MAX_OVERFIT_STEPS}, patience={EARLY_STOPPING_PATIENCE}, "
        f"max_length={MAX_LENGTH}, topk_loss_percent={TOPK_LOSS_PERCENT}"
    )

    # Load base+LoRA once; reset trainable weights after each sample.
    model = load_before_model_trainable(resolved_adapter, model_name=resolved_model)
    adapter_state = snapshot_trainable_state(model)

    for sample_id, row in tqdm(samples.iterrows(), total=len(samples), desc=f"{group_key} overfit"):
        sid = int(sample_id)
        if sid in done_ids:
            continue
        print(f"\n=== sample_id={sid}, group={row['group']} ===")
        set_seed(SEED + sid)
        restore_trainable_state(model, adapter_state)
        train_enc = make_lm_encoding(tokenizer, str(row["text"]))
        analysis_text = str(row["text"])

        before_loss, token_losses, token_mask, attn_before, attn_mask = (
            compute_diagnostics_and_attentions(model, train_enc)
        )
        selected_positions = select_topk_query_positions(token_losses, token_mask, TOPK_LOSS_PERCENT)

        train_losses, actual_steps = overfit_one_sample(model, train_enc)

        after_loss, attn_after, _ = compute_sequence_loss_and_attentions(model, train_enc)

        metric_rows = attention_shift_metrics(
            attn_before, attn_after, selected_positions, attention_mask=attn_mask
        )
        for mr in metric_rows:
            mr.update({
                "sample_id": sid,
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
            "sample_id": sid,
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
        done_ids.add(sid)

        del attn_before, attn_after
        if torch.cuda.is_available() and (len(done_ids) % 20 == 0):
            gc.collect()
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
    try:
        plot_single_group_distributions(output_dir, raw_df)
    except Exception as exc:
        print(f"Plotting failed (extraction still complete): {exc}")
    write_status(
        output_dir,
        "completed\n"
        f"group_key={group_key}\n"
        f"processed_sample_counts={sample_df['group'].value_counts().to_dict() if 'group' in sample_df.columns else {}}\n"
        f"raw_attention_rows={len(raw_df)}",
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Saved to: {output_dir}")
