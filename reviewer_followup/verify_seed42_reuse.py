#!/usr/bin/env python3
"""Verify that the frozen E7 seed-42 result is reusable for E13."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from reviewer_followup.common import atomic_write_json, base_manifest, sha256_file


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--targets-csv", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--expected-seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = {name: Path(getattr(args, name)) for name in ("train_csv", "targets_csv", "train_config", "predictions_csv")}
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    config = json.loads(paths["train_config"].read_text(encoding="utf-8"))
    if int(config.get("seed", -1)) != args.expected_seed:
        raise ValueError(f"E7 train seed is {config.get('seed')}, expected {args.expected_seed}")
    hashes = config.get("input_sha256", {})
    expected_train = sha256_file(paths["train_csv"])
    expected_targets = sha256_file(paths["targets_csv"])
    if hashes.get("ft") != expected_train or hashes.get("all") != expected_targets:
        raise ValueError("E7 train_config input hashes do not match the frozen E13 inputs")
    predictions = pd.read_csv(paths["predictions_csv"])
    required = {"comparison", "method", "repeat", "sample_id", "y_true", "score"}
    if required - set(predictions.columns) or predictions.empty:
        raise ValueError("E7 OOF prediction artifact is incomplete")
    manifest = base_manifest(experiment="e13_verified_seed42_reuse", command=sys.argv)
    manifest.update(
        {
            "status": "completed", "reused_seed": args.expected_seed,
            "verification": "exact_input_hash_and_train_seed_match",
            "artifacts": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
        }
    )
    atomic_write_json(Path(args.output_json), manifest)
    print(f"Verified E7 seed {args.expected_seed} for E13 reuse")


if __name__ == "__main__":
    main(sys.argv[1:])
