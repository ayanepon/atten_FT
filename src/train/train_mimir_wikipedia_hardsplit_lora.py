# -*- coding: utf-8 -*-
"""
LoRA fine-tuning on a strict MIMIR Wikipedia hard split.

Experiment design:
  PT      = MIMIR Wikipedia member
  FT      = first half of MIMIR Wikipedia nonmember
  Unseen  = second half of MIMIR Wikipedia nonmember

This script first saves the three splits, then fine-tunes Pythia-1B only on
the FT split. The saved split CSVs are intended to be reused by the later
attention-update, Initial Loss, LoRA-Leak, AttenMIA, and Proposed-method
comparisons.

Example:
  HF_TOKEN=... CUDA_VISIBLE_DEVICES=0 python3 train_mimir_wikipedia_hardsplit_lora.py

If MIMIR is already downloaded/exported locally:
  python3 train_mimir_wikipedia_hardsplit_lora.py --mimir-csv /path/to/mimir_wikipedia_13_0.8.csv
"""

import argparse
import inspect
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


BASE_MODEL_NAME = "EleutherAI/pythia-410m"
MIMIR_NAME = "iamgroot42/mimir"
MIMIR_CONFIG = "wikipedia_(en)"
MIMIR_SPLIT = "ngram_13_0.8"

MEMBER_COL = "member"
NONMEMBER_COL = "nonmember"

GROUP_PT = "mimir_wikipedia_member_pt"
GROUP_FT = "mimir_wikipedia_nonmember_ft"
GROUP_UNSEEN = "mimir_wikipedia_nonmember_unseen"

DEFAULT_OUTPUT_DIR = "models/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_pythia410m"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_text_series(values) -> pd.Series:
    s = pd.Series(values, dtype="string").fillna("").astype(str).str.strip()
    s = s[s.str.len() > 0]
    return s.drop_duplicates().reset_index(drop=True)


def read_local_mimir_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if MEMBER_COL not in df.columns or NONMEMBER_COL not in df.columns:
        raise ValueError(
            f"Local MIMIR CSV must contain '{MEMBER_COL}' and '{NONMEMBER_COL}' columns. "
            f"Available columns: {list(df.columns)}"
        )
    return df


def load_mimir_dataframe(args) -> pd.DataFrame:
    if args.mimir_csv:
        return read_local_mimir_csv(args.mimir_csv)

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("datasets is required to load iamgroot42/mimir from Hugging Face.") from e

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    try:
        ds = load_dataset(
            args.mimir_name,
            args.mimir_config,
            split=args.mimir_split,
            token=hf_token,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to load iamgroot42/mimir. This dataset may be gated. "
            "Set HF_TOKEN/HUGGINGFACE_HUB_TOKEN or pass --mimir-csv with member/nonmember columns."
        ) from e

    missing = [c for c in (args.member_col, args.nonmember_col) if c not in ds.column_names]
    if missing:
        raise ValueError(f"MIMIR columns not found: {missing}. Available columns: {ds.column_names}")
    return ds.to_pandas()


def make_mimir_hard_splits(args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_mimir_dataframe(args)
    member = clean_text_series(df[args.member_col])
    nonmember = clean_text_series(df[args.nonmember_col])

    rng = np.random.default_rng(args.seed)
    member_idx = rng.permutation(len(member))
    nonmember_idx = rng.permutation(len(nonmember))

    member = member.iloc[member_idx].reset_index(drop=True)
    nonmember = nonmember.iloc[nonmember_idx].reset_index(drop=True)

    if args.max_pt_samples > 0:
        member = member.iloc[: args.max_pt_samples].reset_index(drop=True)
    if args.max_nonmember_samples > 0:
        nonmember = nonmember.iloc[: args.max_nonmember_samples].reset_index(drop=True)

    split_at = len(nonmember) // 2
    ft_text = nonmember.iloc[:split_at].reset_index(drop=True)
    unseen_text = nonmember.iloc[split_at:].reset_index(drop=True)

    if args.max_ft_samples > 0:
        ft_text = ft_text.iloc[: args.max_ft_samples].reset_index(drop=True)
    if args.max_unseen_samples > 0:
        unseen_text = unseen_text.iloc[: args.max_unseen_samples].reset_index(drop=True)

    pt = pd.DataFrame({
        "group": GROUP_PT,
        "source": "mimir_wikipedia_13_0.8_member",
        "text": member,
        "original_index": np.arange(len(member), dtype=int),
    })
    ft = pd.DataFrame({
        "group": GROUP_FT,
        "source": "mimir_wikipedia_13_0.8_nonmember_first_half",
        "text": ft_text,
        "original_index": np.arange(len(ft_text), dtype=int),
    })
    unseen = pd.DataFrame({
        "group": GROUP_UNSEEN,
        "source": "mimir_wikipedia_13_0.8_nonmember_second_half",
        "text": unseen_text,
        "original_index": np.arange(len(unseen_text), dtype=int),
    })
    return pt, ft, unseen


def save_splits(pt: pd.DataFrame, ft: pd.DataFrame, unseen: pd.DataFrame, out_dir: Path) -> None:
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pt.to_csv(data_dir / "mimir_wikipedia_pt_member.csv", index=False)
    ft.to_csv(data_dir / "mimir_wikipedia_ft_nonmember.csv", index=False)
    unseen.to_csv(data_dir / "mimir_wikipedia_unseen_nonmember.csv", index=False)
    all_targets = pd.concat([pt, ft, unseen], ignore_index=True)
    all_targets.to_csv(data_dir / "mimir_wikipedia_pt_ft_unseen_targets.csv", index=False)


def tokenize_texts(examples, tokenizer, max_length: int):
    enc = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    enc["labels"] = [ids.copy() for ids in enc["input_ids"]]
    return enc


def train_lora_on_ft_split(ft: pd.DataFrame, args, out_dir: Path) -> None:
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if args.fp32:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable") and args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ds = Dataset.from_pandas(ft[["text"]], preserve_index=False)
    tokenized = ds.map(
        lambda batch: tokenize_texts(batch, tokenizer, args.max_length),
        batched=True,
        remove_columns=["text"],
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    adapter_dir = out_dir / "adapter"
    logs_dir = out_dir / "logs"
    training_kwargs = {
        "output_dir": str(out_dir / "checkpoints"),
        "overwrite_output_dir": True,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "logging_dir": str(logs_dir),
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch" if args.save_each_epoch else "no",
        "save_total_limit": args.save_total_limit,
        "bf16": (dtype == torch.bfloat16),
        "fp16": (dtype == torch.float16),
        "report_to": [],
        "seed": args.seed,
        "data_seed": args.seed,
        "dataloader_num_workers": args.dataloader_num_workers,
        "optim": args.optim,
        "max_grad_norm": args.max_grad_norm,
    }
    supported_args = set(inspect.signature(TrainingArguments.__init__).parameters)
    training_kwargs = {k: v for k, v in training_kwargs.items() if k in supported_args}
    training_args = TrainingArguments(**training_kwargs)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized,
        "data_collator": collator,
        "tokenizer": tokenizer,
        "processing_class": tokenizer,
    }
    supported_trainer_args = set(inspect.signature(Trainer.__init__).parameters)
    trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if k in supported_trainer_args}
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=BASE_MODEL_NAME)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mimir-name", default=MIMIR_NAME)
    parser.add_argument("--mimir-config", default=MIMIR_CONFIG)
    parser.add_argument("--mimir-split", default=MIMIR_SPLIT)
    parser.add_argument("--mimir-csv", default=os.environ.get("MIMIR_CSV", ""))
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--member-col", default=MEMBER_COL)
    parser.add_argument("--nonmember-col", default=NONMEMBER_COL)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-pt-samples", type=int, default=0, help="0 means use all available member texts.")
    parser.add_argument("--max-nonmember-samples", type=int, default=0, help="0 means use all available nonmember texts.")
    parser.add_argument("--max-ft-samples", type=int, default=0, help="0 means use the full nonmember first half.")
    parser.add_argument("--max-unseen-samples", type=int, default=0, help="0 means use the full nonmember second half.")
    parser.add_argument("--prepare-only", action="store_true", help="Only save PT/FT/Unseen CSVs; do not train.")

    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-train-epochs", type=float, default=5.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-each-epoch", action="store_true")
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--fp32", action="store_true")

    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="query_key_value,dense,dense_h_to_4h,dense_4h_to_h",
        help="Comma-separated LoRA target modules for Pythia/GPT-NeoX.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pt, ft, unseen = make_mimir_hard_splits(args)
    save_splits(pt, ft, unseen, out_dir)

    config: Dict[str, object] = vars(args).copy()
    config.update({
        "groups": {
            "pt": GROUP_PT,
            "ft": GROUP_FT,
            "unseen": GROUP_UNSEEN,
        },
        "counts": {
            "pt": int(len(pt)),
            "ft": int(len(ft)),
            "unseen": int(len(unseen)),
        },
        "adapter_output_dir": str(out_dir / "adapter"),
        "split_files": {
            "pt": str(out_dir / "data" / "mimir_wikipedia_pt_member.csv"),
            "ft": str(out_dir / "data" / "mimir_wikipedia_ft_nonmember.csv"),
            "unseen": str(out_dir / "data" / "mimir_wikipedia_unseen_nonmember.csv"),
            "all": str(out_dir / "data" / "mimir_wikipedia_pt_ft_unseen_targets.csv"),
        },
    })
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("MIMIR hard split counts:")
    print(pd.Series(config["counts"]).to_string())
    print(f"Saved split CSVs to: {out_dir / 'data'}")

    if args.prepare_only:
        print("prepare-only mode: skipping LoRA fine-tuning.")
        return

    train_lora_on_ft_split(ft, args, out_dir)
    print(f"Saved LoRA adapter to: {out_dir / 'adapter'}")


if __name__ == "__main__":
    main()
