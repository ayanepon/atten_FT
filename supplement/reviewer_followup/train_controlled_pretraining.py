#!/usr/bin/env python3
"""Run a researcher-controlled full-parameter continued-pretraining stage."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from reviewer_followup.common import atomic_write_json, base_manifest, sha256_file
from train_mimir_wikipedia_hardsplit_lora import TextLMDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True, help="HF id or local causal-LM checkpoint")
    parser.add_argument("--revision", default="", help="Optional immutable Hugging Face model revision/commit")
    parser.add_argument("--train-csv", default="", help="Controlled-PT CSV with text column")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--archive-base-only",
        action="store_true",
        help="Archive the immutable pre-continued-pretraining checkpoint and exit without training.",
    )
    return parser.parse_args(argv)


def archive_base_checkpoint(model, tokenizer, output_dir: Path, model_name: str, requested_revision: str = "") -> dict:
    base_dir = output_dir / "base_model_before_controlled_pt"
    base_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(base_dir)
    tokenizer.save_pretrained(base_dir)
    weight_files = sorted(
        path.name for pattern in ("*.safetensors", "pytorch_model*.bin") for path in base_dir.glob(pattern)
    )
    manifest = base_manifest(experiment="e11_base_checkpoint_archive", command=sys.argv)
    manifest.update(
        {
            "status": "completed",
            "model_name": model_name,
            "requested_revision": requested_revision,
            "resolved_model_commit": str(getattr(model.config, "_commit_hash", "") or ""),
            "base_model_output_dir": str(base_dir),
            "model_config_sha256": sha256_file(base_dir / "config.json"),
            "weight_files": weight_files,
        }
    )
    atomic_write_json(output_dir / "base_checkpoint_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.train_csv and not args.archive_base_only:
        raise ValueError("--train-csv is required unless --archive-base-only is used")
    train_csv = Path(args.train_csv) if args.train_csv else None
    frame = pd.DataFrame()
    if train_csv is not None:
        frame = pd.read_csv(train_csv, dtype=str, keep_default_na=False)
        if "text" not in frame.columns:
            raise ValueError(f"{train_csv} needs text column")
        frame = frame[frame["text"].astype(str).str.strip().str.len() > 0].reset_index(drop=True)
        if frame.empty:
            raise ValueError("Controlled-PT training CSV is empty")
    manifest = base_manifest(experiment="e11_controlled_pretraining", command=sys.argv)
    manifest.update(
        {
            "model_name": args.model_name,
            "requested_revision": args.revision,
            "train_csv": str(train_csv) if train_csv is not None else "",
            "train_csv_sha256": sha256_file(train_csv) if train_csv is not None else "",
            "n_train": int(len(frame)),
            "seed": args.seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "dry_run": bool(args.dry_run),
        }
    )
    if args.dry_run and not args.archive_base_only:
        manifest["status"] = "dry_run_validated"
        atomic_write_json(output_dir / "controlled_pretraining_manifest.json", manifest)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator

    set_seed(args.seed)
    revision_args = {"revision": args.revision} if args.revision else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **revision_args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32 if args.fp32 else (
        torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype, **revision_args)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, **revision_args)
    manifest["resolved_model_commit"] = str(getattr(model.config, "_commit_hash", "") or "")
    base_manifest_payload = archive_base_checkpoint(model, tokenizer, output_dir, args.model_name, args.revision)
    if args.archive_base_only:
        print(json.dumps(base_manifest_payload, indent=2, ensure_ascii=False))
        return
    manifest["base_model_output_dir"] = base_manifest_payload["base_model_output_dir"]
    manifest["base_checkpoint_manifest"] = str(output_dir / "base_checkpoint_manifest.json")
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    dataset = TextLMDataset(frame["text"].astype(str).tolist(), tokenizer, args.max_length)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_strategy="no",
        logging_steps=10,
        report_to=[],
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        remove_unused_columns=False,
    )
    Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    ).train()
    model_dir = output_dir / "controlled_pt_model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    manifest["status"] = "completed"
    manifest["model_output_dir"] = str(model_dir)
    manifest["model_config_sha256"] = sha256_file(model_dir / "config.json")
    atomic_write_json(output_dir / "controlled_pretraining_manifest.json", manifest)
    print(f"Saved controlled-PT checkpoint to {model_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
