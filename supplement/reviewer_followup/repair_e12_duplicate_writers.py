#!/usr/bin/env python3
"""Repair E12 shard CSVs after two writers appended the same sample keys."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


SAMPLE_KEYS = ["condition", "sample_id"]
RAW_ID_KEYS = ["condition", "sample_id", "layer", "head"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".repair_tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def base_condition(series: pd.Series) -> pd.Series:
    return series.astype(str).str.split("__", n=1).str[0]


def repair_shard(shard_dir: Path, backup_root: Path, expected_rows_per_key: int) -> dict:
    sample_path = shard_dir / "sample_level_experiment4.csv"
    raw_path = shard_dir / "raw_experiment4_attention_shift.csv"
    if expected_rows_per_key < 1:
        raise ValueError("expected_rows_per_key must be positive")
    for path in (sample_path, raw_path):
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing non-empty shard artifact: {path}")

    sample = pd.read_csv(sample_path)
    raw = pd.read_csv(raw_path)
    if not set(SAMPLE_KEYS).issubset(sample.columns):
        raise ValueError(f"{sample_path} is missing {sorted(set(SAMPLE_KEYS) - set(sample.columns))}")
    if not set(RAW_ID_KEYS).issubset(raw.columns):
        raise ValueError(f"{raw_path} is missing {sorted(set(RAW_ID_KEYS) - set(raw.columns))}")

    sample["sample_id"] = sample["sample_id"].astype(int)
    raw["sample_id"] = raw["sample_id"].astype(int)
    duplicate_sample_rows = int(sample.duplicated(SAMPLE_KEYS).sum())
    duplicate_keys = {
        (str(row.condition), int(row.sample_id))
        for row in sample.loc[sample.duplicated(SAMPLE_KEYS, keep=False), SAMPLE_KEYS].itertuples(index=False)
    }
    sample_clean = sample.drop_duplicates(SAMPLE_KEYS, keep="last").reset_index(drop=True)
    committed = {(str(row.condition), int(row.sample_id)) for row in sample_clean[SAMPLE_KEYS].itertuples(index=False)}

    raw = raw.copy()
    raw["_base_condition"] = base_condition(raw["condition"])
    raw["_source_order"] = range(len(raw))
    retained_parts: list[pd.DataFrame] = []
    removed_raw_rows = 0
    orphan_raw_rows = 0
    repaired_keys: list[dict] = []
    for (condition, sample_id), part in raw.groupby(["_base_condition", "sample_id"], sort=False):
        key = (str(condition), int(sample_id))
        part = part.sort_values("_source_order")
        if key not in committed:
            orphan_raw_rows += len(part)
            removed_raw_rows += len(part)
            continue
        if len(part) == expected_rows_per_key:
            retained_parts.append(part)
            continue
        if key not in duplicate_keys or len(part) % expected_rows_per_key:
            raise ValueError(
                f"Unsafe raw block for {key}: rows={len(part)}, expected={expected_rows_per_key}"
            )
        blocks = [
            part.iloc[offset : offset + expected_rows_per_key]
            for offset in range(0, len(part), expected_rows_per_key)
        ]
        reference = blocks[-1][RAW_ID_KEYS].reset_index(drop=True)
        if not all(block[RAW_ID_KEYS].reset_index(drop=True).equals(reference) for block in blocks[:-1]):
            raise ValueError(f"Raw identifier order differs across duplicate blocks for {key}")
        retained_parts.append(blocks[-1])
        removed_raw_rows += len(part) - expected_rows_per_key
        repaired_keys.append({"condition": key[0], "sample_id": key[1], "blocks_found": len(blocks)})

    raw_clean = pd.concat(retained_parts, ignore_index=True).sort_values("_source_order")
    raw_clean = raw_clean.drop(columns=["_base_condition", "_source_order"]).reset_index(drop=True)
    raw_clean_base = base_condition(raw_clean["condition"])
    counts = (
        raw_clean.assign(_base_condition=raw_clean_base)
        .groupby(["_base_condition", "sample_id"], sort=False)
        .size()
    )
    bad_counts = counts[counts != expected_rows_per_key]
    if not bad_counts.empty:
        raise ValueError(f"Post-repair raw block counts are invalid: {bad_counts.to_dict()}")
    raw_duplicates_after = int(raw_clean.duplicated(RAW_ID_KEYS).sum())
    if raw_duplicates_after:
        raise ValueError(f"Post-repair raw identifier duplicates remain: {raw_duplicates_after}")
    if sample_clean.duplicated(SAMPLE_KEYS).any():
        raise ValueError("Post-repair sample duplicates remain")

    shard_backup = backup_root / shard_dir.name
    shard_backup.mkdir(parents=True, exist_ok=False)
    before = {}
    for path in (sample_path, raw_path, shard_dir / "run_status.txt", shard_dir / ".extract_owner.lock"):
        if path.exists():
            before[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
            shutil.copy2(path, shard_backup / path.name)

    atomic_csv(sample_clean, sample_path)
    atomic_csv(raw_clean, raw_path)
    after = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in (sample_path, raw_path)
    }
    report = {
        "shard_dir": str(shard_dir),
        "backup_dir": str(shard_backup),
        "expected_rows_per_key": expected_rows_per_key,
        "sample_rows_before": len(sample),
        "sample_rows_after": len(sample_clean),
        "duplicate_sample_rows_removed": duplicate_sample_rows,
        "raw_rows_before": len(raw),
        "raw_rows_after": len(raw_clean),
        "duplicate_raw_rows_removed": removed_raw_rows - orphan_raw_rows,
        "orphan_raw_rows_removed": orphan_raw_rows,
        "repaired_keys": repaired_keys,
        "before": before,
        "after": after,
        "status": "completed",
    }
    (shard_backup / "repair_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", action="append", required=True)
    parser.add_argument("--backup-root", required=True)
    parser.add_argument("--expected-rows-per-key", type=int, default=2560)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    backup_root = Path(args.backup_root).expanduser()
    backup_root.mkdir(parents=True, exist_ok=False)
    reports = [
        repair_shard(Path(path).expanduser(), backup_root, args.expected_rows_per_key)
        for path in args.shard_dir
    ]
    manifest = {
        "status": "completed",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "reports": reports,
    }
    (backup_root / "repair_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
