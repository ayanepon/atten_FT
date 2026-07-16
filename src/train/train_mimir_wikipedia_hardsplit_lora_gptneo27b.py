# -*- coding: utf-8 -*-
"""
LoRA fine-tuning on the MIMIR Wikipedia hard split for EleutherAI/gpt-neo-2.7B.

This script reuses the same shared split CSVs as the Pythia and GPT-Neo-1.3B
experiments, so the FT/PT/Unseen examples remain identical across models.
"""

from pathlib import Path
import sys

import pandas as pd

import train_mimir_wikipedia_hardsplit_lora as base


BASE_MODEL_NAME = "EleutherAI/gpt-neo-2.7B"
DEFAULT_OUTPUT_DIR = "models/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_gptneo27b"
DEFAULT_PYTHIA_SPLIT_DATA_DIR_CANDIDATES = [
    "data/mimir_hardsplit",
    "results/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2/data",
    "mimir_wikipedia_hardsplit_lora_ft/data",
    "results/mimir_wikipedia_hardsplit_lora_ft/data",
    "models/mimir_wikipedia_hardsplit_lora_ft/data",
]

# GPT-Neo uses separate projection modules rather than Pythia/GPT-NeoX's
# query_key_value module.
GPT_NEO_TARGET_MODULES = "q_proj,k_proj,v_proj,out_proj,c_fc,c_proj"


def pop_arg_value(argv, name: str, default: str = ""):
    if name not in argv:
        return argv, default
    i = argv.index(name)
    if i + 1 >= len(argv):
        raise ValueError(f"{name} requires a value")
    value = argv[i + 1]
    return argv[:i] + argv[i + 2 :], value


def resolve_pythia_split_data_dir(explicit_dir: str) -> Path:
    candidates = [explicit_dir] if explicit_dir else DEFAULT_PYTHIA_SPLIT_DATA_DIR_CANDIDATES
    required = [
        "mimir_wikipedia_pt_member.csv",
        "mimir_wikipedia_ft_nonmember.csv",
        "mimir_wikipedia_unseen_nonmember.csv",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if all((path / name).exists() for name in required):
            return path
    searched = "\n  - ".join(str(Path(c).expanduser()) for c in candidates if c)
    raise FileNotFoundError(
        "Shared split CSVs were not found. Pass --pythia-split-data-dir with the directory "
        "containing mimir_wikipedia_pt_member.csv, mimir_wikipedia_ft_nonmember.csv, and "
        f"mimir_wikipedia_unseen_nonmember.csv.\nSearched:\n  - {searched}"
    )


def read_split_csv(path: Path, expected_group: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "text" not in df.columns:
        raise ValueError(f"Split CSV must contain a text column: {path}")
    df = df.copy()
    if "group" not in df.columns:
        df["group"] = expected_group
    if "source" not in df.columns:
        df["source"] = path.stem
    if "original_index" not in df.columns:
        df["original_index"] = range(len(df))
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    return df


def make_existing_pythia_splits(split_data_dir: Path):
    pt = read_split_csv(split_data_dir / "mimir_wikipedia_pt_member.csv", base.GROUP_PT)
    ft = read_split_csv(split_data_dir / "mimir_wikipedia_ft_nonmember.csv", base.GROUP_FT)
    unseen = read_split_csv(split_data_dir / "mimir_wikipedia_unseen_nonmember.csv", base.GROUP_UNSEEN)
    print(f"Using shared split CSVs from: {split_data_dir}")
    return pt, ft, unseen


def main() -> None:
    base.BASE_MODEL_NAME = BASE_MODEL_NAME
    base.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR

    argv = sys.argv[1:]
    argv, pythia_split_data_dir_arg = pop_arg_value(argv, "--pythia-split-data-dir", "")
    split_data_dir = resolve_pythia_split_data_dir(pythia_split_data_dir_arg)
    base.make_mimir_hard_splits = lambda args: make_existing_pythia_splits(split_data_dir)

    if "--target-modules" not in argv:
        argv = argv + ["--target-modules", GPT_NEO_TARGET_MODULES]
    if "--learning-rate" not in argv:
        argv = argv + ["--learning-rate", "1e-4"]
    if "--num-train-epochs" not in argv:
        argv = argv + ["--num-train-epochs", "5"]

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + argv
        base.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
