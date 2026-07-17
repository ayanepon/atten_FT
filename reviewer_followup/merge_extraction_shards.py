#!/usr/bin/env python3
"""Validate and stream-merge completed attention-extraction shards."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

from reviewer_followup.common import atomic_write_json, base_manifest


FILES = (
    "raw_experiment4_attention_shift.csv",
    "sample_level_experiment4.csv",
    "experiment4_target_samples.csv",
)


def stream_merge(paths: list[Path], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    header: list[str] | None = None
    count = 0
    with tmp.open("w", encoding="utf-8", newline="") as destination:
        writer = None
        for path in paths:
            with path.open("r", encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                if reader.fieldnames is None:
                    raise ValueError(f"Missing CSV header: {path}")
                if header is None:
                    header = list(reader.fieldnames)
                    writer = csv.DictWriter(destination, fieldnames=header, extrasaction="raise")
                    writer.writeheader()
                elif list(reader.fieldnames) != header:
                    raise ValueError(f"Shard header mismatch: {path}")
                assert writer is not None
                for row in reader:
                    writer.writerow(row)
                    count += 1
    tmp.replace(output)
    return count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-targets", type=int, required=True)
    parser.add_argument("--expected-conditions", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    shards = [Path(path) for path in args.shard_dir]
    if len({path.resolve() for path in shards}) != len(shards):
        raise ValueError("Duplicate shard directories")
    for shard in shards:
        status = shard / "run_status.txt"
        if not status.exists() or "completed" not in status.read_text(encoding="utf-8", errors="replace"):
            raise RuntimeError(f"Shard is not complete: {shard}")
        for filename in FILES:
            if not (shard / filename).exists():
                raise FileNotFoundError(shard / filename)

    targets = pd.concat(
        [pd.read_csv(shard / "experiment4_target_samples.csv") for shard in shards], ignore_index=True
    )
    if len(targets) != args.expected_targets:
        raise ValueError(f"Expected {args.expected_targets} targets, found {len(targets)}")
    if targets["global_sample_id"].duplicated().any():
        raise ValueError("global_sample_id overlaps across shards")
    expected_ids = set(range(args.expected_targets))
    actual_ids = set(pd.to_numeric(targets["global_sample_id"]).astype(int))
    if actual_ids != expected_ids:
        raise ValueError(f"Target ID coverage mismatch: missing={sorted(expected_ids - actual_ids)[:10]}")

    sample_parts = [pd.read_csv(shard / "sample_level_experiment4.csv") for shard in shards]
    samples = pd.concat(sample_parts, ignore_index=True)
    required_sample_columns = {"condition", "sample_id"}
    if not required_sample_columns.issubset(samples.columns):
        raise ValueError(f"Missing sample-level columns: {sorted(required_sample_columns - set(samples.columns))}")
    sample_condition_count = int(samples["condition"].nunique())
    if sample_condition_count <= 0:
        raise ValueError("No sample-level conditions found")
    expected_sample_rows = args.expected_targets * sample_condition_count
    if len(samples) != expected_sample_rows:
        raise ValueError(f"Expected {expected_sample_rows} sample-condition rows, found {len(samples)}")
    if samples[["condition", "sample_id"]].duplicated().any():
        raise ValueError("Duplicate condition/sample_id rows across shards")
    condition_counts = samples.groupby("condition", dropna=False)["sample_id"].nunique()
    if len(condition_counts) != sample_condition_count or not (condition_counts == args.expected_targets).all():
        raise ValueError(f"Incomplete sample coverage by condition: {condition_counts.to_dict()}")
    sample_counts = samples.groupby("sample_id", dropna=False)["condition"].nunique()
    if len(sample_counts) != args.expected_targets or not (sample_counts == sample_condition_count).all():
        raise ValueError("Incomplete condition coverage by sample")

    output = Path(args.output_dir)
    counts = {
        filename: stream_merge([shard / filename for shard in shards], output / filename)
        for filename in FILES
    }
    (output / "run_status.txt").write_text(
        "extraction_completed_shard_merge\n"
        f"shards={len(shards)}\n"
        f"raw_attention_rows={counts['raw_experiment4_attention_shift.csv']}\n"
        f"sample_rows={counts['sample_level_experiment4.csv']}\n",
        encoding="utf-8",
    )
    manifest = base_manifest(experiment="e12_extraction_shard_merge", command=sys.argv)
    manifest.update(
        {
            "status": "completed",
            "shards": [str(path) for path in shards],
            "expected_targets": args.expected_targets,
            "expected_conditions": args.expected_conditions,
            "sample_condition_count": sample_condition_count,
            "row_counts": counts,
        }
    )
    atomic_write_json(output / "shard_merge_manifest.json", manifest)
    print(counts)


if __name__ == "__main__":
    main(sys.argv[1:])
