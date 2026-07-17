# -*- coding: utf-8 -*-
"""Paper-aligned LoRA fine-tuning on MIMIR Wikipedia hard split.

Reproduces the FT setup in acl_latex.tex:
  - Models: pythia-1b | pythia-410m | gpt-neo-2.7b (via --model)
  - Data: MIMIR wikipedia_(en) hard split ngram_13_0.8
  - PT = member, FT/Unseen = random split of nonmember
  - LoRA r=8, alpha=16, dropout=0.05 (architecture-specific modules)
  - AdamW lr=1e-4, epochs=5, micro-batch=1, grad_accum=16
  - max length 256, seed 42, 500 samples per group by default

Usage:
  # Pythia-1B from prepared CSVs:
  python train_mimir_wikipedia_hardsplit_lora.py \\
    --model pythia-1b \\
    --from-csv-dir mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2/data

  # Pythia-410M / GPT-Neo (reuse the same hard-split CSVs):
  python train_mimir_wikipedia_hardsplit_lora.py \\
    --model pythia-410m \\
    --from-csv-dir mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2/data

  python train_mimir_wikipedia_hardsplit_lora.py \\
    --model gpt-neo-2.7b \\
    --from-csv-dir mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2/data

  # Or download MIMIR and create splits:
  python train_mimir_wikipedia_hardsplit_lora.py --model pythia-1b
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


GROUP_PT = "mimir_wikipedia_member_pt"
GROUP_FT = "mimir_wikipedia_nonmember_ft"
GROUP_UNSEEN = "mimir_wikipedia_nonmember_unseen"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TextLMDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        labels = item["input_ids"].clone()
        labels[item["attention_mask"] == 0] = -100
        item["labels"] = labels
        return item


def prepare_from_mimir(args) -> Dict[str, pd.DataFrame]:
    from datasets import load_dataset

    ds = load_dataset(args.mimir_name, args.mimir_config, split=args.mimir_split, token=args.hf_token or None)
    # MIMIR configs expose member / nonmember columns (lists of strings or rows)
    if args.member_col not in ds.column_names or args.nonmember_col not in ds.column_names:
        raise ValueError(f"Expected columns {args.member_col}/{args.nonmember_col}, got {ds.column_names}")

    def flatten_col(col_name: str) -> List[str]:
        vals = []
        for row in ds[col_name]:
            if isinstance(row, list):
                vals.extend([str(x) for x in row if str(x).strip()])
            else:
                if str(row).strip():
                    vals.append(str(row))
        # unique preserve order
        seen = set()
        out = []
        for t in vals:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    members = flatten_col(args.member_col)
    nonmembers = flatten_col(args.nonmember_col)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(nonmembers)

    n_pt = min(args.max_pt_samples, len(members)) if args.max_pt_samples > 0 else len(members)
    nonmember_pool = nonmembers
    if args.max_nonmember_samples > 0:
        nonmember_pool = nonmembers[: args.max_nonmember_samples]
    half = len(nonmember_pool) // 2
    ft_pool = nonmember_pool[:half]
    unseen_pool = nonmember_pool[half:]
    n_ft = min(args.max_ft_samples, len(ft_pool)) if args.max_ft_samples > 0 else len(ft_pool)
    n_unseen = min(args.max_unseen_samples, len(unseen_pool)) if args.max_unseen_samples > 0 else len(unseen_pool)

    pt_texts = list(rng.choice(members, size=n_pt, replace=False)) if n_pt < len(members) else members[:n_pt]
    ft_texts = ft_pool[:n_ft]
    unseen_texts = unseen_pool[:n_unseen]

    def make_df(group: str, source: str, texts: List[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "group": group,
                "source": source,
                "text": texts,
                "original_index": np.arange(len(texts), dtype=int),
            }
        )

    return {
        "pt": make_df(GROUP_PT, f"mimir_{args.mimir_split}_member", pt_texts),
        "ft": make_df(GROUP_FT, f"mimir_{args.mimir_split}_nonmember_first_half", ft_texts),
        "unseen": make_df(GROUP_UNSEEN, f"mimir_{args.mimir_split}_nonmember_second_half", unseen_texts),
    }


def prepare_from_csv_dir(csv_dir: Path) -> Dict[str, pd.DataFrame]:
    files = {
        "pt": csv_dir / "mimir_wikipedia_pt_member.csv",
        "ft": csv_dir / "mimir_wikipedia_ft_nonmember.csv",
        "unseen": csv_dir / "mimir_wikipedia_unseen_nonmember.csv",
    }
    out = {}
    for key, path in files.items():
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "text" not in df.columns:
            raise ValueError(f"{path} needs text column")
        if "group" not in df.columns:
            df["group"] = {"pt": GROUP_PT, "ft": GROUP_FT, "unseen": GROUP_UNSEEN}[key]
        out[key] = df
    return out


def save_splits(splits: Dict[str, pd.DataFrame], data_dir: Path) -> Dict[str, str]:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pt": data_dir / "mimir_wikipedia_pt_member.csv",
        "ft": data_dir / "mimir_wikipedia_ft_nonmember.csv",
        "unseen": data_dir / "mimir_wikipedia_unseen_nonmember.csv",
        "all": data_dir / "mimir_wikipedia_pt_ft_unseen_targets.csv",
    }
    for key in ["pt", "ft", "unseen"]:
        splits[key].to_csv(paths[key], index=False)
    pd.concat([splits["pt"], splits["ft"], splits["unseen"]], ignore_index=True).to_csv(paths["all"], index=False)
    return {k: str(v) for k, v in paths.items()}


def train_lora(args, ft_df: pd.DataFrame, output_dir: Path) -> Path:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        default_data_collator,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32 if args.fp32 else (
        torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
    model.config.use_cache = False

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    train_ds = TextLMDataset(ft_df["text"].astype(str).tolist(), tokenizer, args.max_length)
    train_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_strategy="epoch" if args.save_each_epoch else "no",
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=[],
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        data_collator=default_data_collator,
    )
    trainer.train()

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    return adapter_dir


def parse_args() -> argparse.Namespace:
    from model_registry import DEFAULT_HF_ID, DEFAULT_PYTHIA_LORA_CSV, PYTHIA1B_RUN_DIR, add_model_arguments

    p = argparse.ArgumentParser(description="Paper LoRA FT on MIMIR Wikipedia hard split")
    add_model_arguments(p, model_name_default=DEFAULT_HF_ID)
    p.add_argument("--output-dir", default=PYTHIA1B_RUN_DIR)
    p.add_argument("--from-csv-dir", default="", help="Use existing split CSVs instead of downloading MIMIR")
    p.add_argument("--mimir-name", default="iamgroot42/mimir")
    p.add_argument("--mimir-config", default="wikipedia_(en)")
    p.add_argument("--mimir-split", default="ngram_13_0.8")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--member-col", default="member")
    p.add_argument("--nonmember-col", default="nonmember")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-pt-samples", type=int, default=500)
    p.add_argument("--max-nonmember-samples", type=int, default=0)
    p.add_argument("--max-ft-samples", type=int, default=500)
    p.add_argument("--max-unseen-samples", type=int, default=500)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--num-train-epochs", type=float, default=5.0)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lr-scheduler-type", default="cosine")
    p.add_argument("--optim", default="adamw_torch")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-each-epoch", action="store_true")
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--dataloader-num-workers", type=int, default=0)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default=DEFAULT_PYTHIA_LORA_CSV)
    return p.parse_args()


def apply_model_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Fill model_name / target_modules / output_dir from --model preset."""
    from model_registry import apply_model_namespace

    return apply_model_namespace(args, profile="train", log=print)


def main() -> None:
    args = parse_args()
    args = apply_model_preset(args)
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"

    if args.from_csv_dir:
        splits = prepare_from_csv_dir(Path(args.from_csv_dir))
    else:
        splits = prepare_from_mimir(args)
    paths = save_splits(splits, data_dir)

    config = vars(args).copy()
    config["groups"] = {"pt": GROUP_PT, "ft": GROUP_FT, "unseen": GROUP_UNSEEN}
    config["counts"] = {k: int(len(v)) for k, v in splits.items()}
    config["split_files"] = paths
    config["model_name"] = args.model_name
    config["target_modules"] = args.target_modules

    if args.prepare_only:
        config["adapter_output_dir"] = ""
        (output_dir / "train_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Prepared splits only under {data_dir}")
        return

    adapter_dir = train_lora(args, splits["ft"], output_dir)
    config["adapter_output_dir"] = str(adapter_dir)
    (output_dir / "train_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved adapter to {adapter_dir}")
    print(f"Saved splits to {data_dir}")


if __name__ == "__main__":
    main()
