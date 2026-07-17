#!/usr/bin/env python3
"""Wait for long GPU stages and launch their CPU-only downstream analyses."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from reviewer_followup.common import atomic_write_json, base_manifest
from reviewer_followup.controller import E12_SHARDS


def status_is_complete(path: Path) -> bool:
    return path.exists() and "completed" in path.read_text(encoding="utf-8", errors="replace")


def wait_for(label: str, predicate, poll_seconds: int) -> None:
    while not predicate():
        print(f"[{label}] waiting", flush=True)
        time.sleep(poll_seconds)
    print(f"[{label}] prerequisites complete", flush=True)


def run_controller(data_dir: Path, output_root: Path, *arguments: str) -> None:
    command = [
        sys.executable,
        "-m",
        "reviewer_followup.controller",
        *arguments,
        "--output-root",
        str(output_root),
    ]
    print(f"[downstream] running: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=data_dir, check=True)


def run_e8(data_dir: Path, output_root: Path, poll_seconds: int) -> None:
    prerequisite = output_root / "e7_crossed_2x2" / "evaluation" / "factorial_contrast_summary.csv"
    wait_for("e8", prerequisite.exists, poll_seconds)
    run_controller(data_dir, output_root, "run-stage", "--stage", "e8")


def run_e10(data_dir: Path, output_root: Path, poll_seconds: int) -> None:
    prerequisites = (
        output_root / "e9_multiseed" / "checkpoint_aggregation" / "checkpoint_summary.csv",
        output_root / "e9_multiseed" / "sample_seed_aggregation" / "sample_seed_summary.csv",
    )
    wait_for("e10", lambda: all(path.exists() for path in prerequisites), poll_seconds)
    run_controller(data_dir, output_root, "run-stage", "--stage", "e10")


def run_e12(data_dir: Path, output_root: Path, poll_seconds: int) -> None:
    statuses = [
        output_root
        / "e12_nested_protocol"
        / "features"
        / f"all_protocols_shard_{index}_of_{E12_SHARDS}"
        / "run_status.txt"
        for index in range(E12_SHARDS)
    ]
    wait_for("e12", lambda: all(status_is_complete(path) for path in statuses), poll_seconds)
    for command in (
        "merge_all_query_protocol_shards",
        "nested_select_ft_vs_pt",
        "nested_select_ft_vs_unseen",
    ):
        run_controller(data_dir, output_root, "run-command", "--command", command)


def run_final_audit(data_dir: Path, output_root: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "reviewer_followup.audit_results",
        "--output-root",
        str(output_root),
    ]
    print(f"[downstream] running final audit: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=data_dir, check=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="results/reviewer_followup_20260716")
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.poll_seconds < 1:
        raise ValueError("--poll-seconds must be positive")
    data_dir = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (data_dir / output_root).resolve()
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(worker, data_dir, output_root, args.poll_seconds)
            for worker in (run_e8, run_e10, run_e12)
        ]
        for future in futures:
            future.result()
    run_final_audit(data_dir, output_root)
    manifest = base_manifest(experiment="reviewer_followup_downstream_watcher", command=sys.argv)
    manifest["status"] = "completed"
    atomic_write_json(output_root / "downstream_watcher_manifest.json", manifest)
    print("All downstream analyses completed", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
