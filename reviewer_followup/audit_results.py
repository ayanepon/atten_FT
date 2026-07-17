#!/usr/bin/env python3
"""Produce the strict, submission-facing audit for reviewer follow-up experiments."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from reviewer_followup.common import atomic_write_json, sha256_file
from reviewer_followup.controller import build_plan, command_completion_report


HASH_LIMIT_BYTES = 128 * 1024 * 1024
PACKAGE_NAMES = ("torch", "transformers", "peft", "numpy", "pandas", "scikit-learn", "scipy", "statsmodels")


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str]:
    if not path.exists():
        return None, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"unreadable:{type(exc).__name__}"
    if not isinstance(payload, dict):
        return None, "not_an_object"
    return payload, "ok"


def _file_evidence(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists() or not path.is_file():
        return result
    stat = path.stat()
    result.update({"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    if stat.st_size <= HASH_LIMIT_BYTES:
        result["sha256"] = sha256_file(path)
    else:
        result["sha256"] = None
        result["hash_omitted_reason"] = f"larger_than_{HASH_LIMIT_BYTES}_bytes"
    return result


def _run_git(data_dir: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=data_dir, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def environment_evidence(data_dir: Path) -> dict[str, Any]:
    packages = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    dirty_lines = _run_git(data_dir, "status", "--porcelain").splitlines()
    return {
        "hostname": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "git_head": _run_git(data_dir, "rev-parse", "HEAD") or None,
        "git_dirty": bool(dirty_lines),
        "git_changed_path_count": len(dirty_lines),
    }


def _required_manifests(root: Path) -> list[Path]:
    return [
        root / "experiment_plan.json",
        root / "e7_crossed_2x2/data/factorial_manifest.json",
        root / "e7_crossed_2x2/evaluation/factorial_evaluation_manifest.json",
        root / "e8_update_baselines/update_baseline_manifest.json",
        root / "e9_multiseed/checkpoint_aggregation/checkpoint_manifest.json",
        root / "e9_multiseed/sample_seed_aggregation/checkpoint_manifest.json",
        root / "e10_head_stability/ft_vs_pt/head_stability_manifest.json",
        root / "e10_head_stability/ft_vs_unseen/head_stability_manifest.json",
        root / "e11_controlled_family/data/factorial_manifest.json",
        root / "e11_controlled_family/controlled_pt/controlled_pretraining_manifest.json",
        root / "e11_controlled_family/controlled_pt/base_checkpoint_manifest.json",
        root / "e11_controlled_family/evaluation/factorial_evaluation_manifest.json",
        root / "e12_nested_protocol/features/all_protocols/shard_merge_manifest.json",
        root / "e12_nested_protocol/evaluation/ft_vs_pt/nested_protocol_manifest.json",
        root / "e12_nested_protocol/evaluation/ft_vs_unseen/nested_protocol_manifest.json",
    ]


def _factorial_check(path: Path) -> dict[str, Any]:
    payload, reason = _read_json(path)
    result: dict[str, Any] = {"path": str(path), "valid": False, "reason": reason}
    if payload is None:
        return result
    validation = payload.get("validation", {})
    counts = validation.get("group_counts", {}) if isinstance(validation, dict) else {}
    duplicates = validation.get("exact_duplicate_count") if isinstance(validation, dict) else None
    values = [int(value) for value in counts.values()] if isinstance(counts, dict) and counts else []
    result.update(
        {
            "group_counts": counts,
            "exact_duplicate_count": duplicates,
            "valid": len(values) == 4 and len(set(values)) == 1 and duplicates == 0,
            "reason": "ok" if len(values) == 4 and len(set(values)) == 1 and duplicates == 0 else "invalid_balance_or_duplicates",
        }
    )
    return result


def _e12_merge_check(path: Path) -> dict[str, Any]:
    payload, reason = _read_json(path)
    result: dict[str, Any] = {"path": str(path), "valid": False, "reason": reason}
    if payload is None:
        return result
    counts = payload.get("row_counts", {})
    valid = (
        payload.get("status") == "completed"
        and payload.get("expected_targets") == 1500
        and payload.get("expected_conditions") == 80
        and payload.get("sample_condition_count") == 4
        and len(payload.get("shards", [])) == 10
        and counts.get("experiment4_target_samples.csv") == 1500
        and counts.get("sample_level_experiment4.csv") == 6000
        and counts.get("raw_experiment4_attention_shift.csv") == 15360000
    )
    result.update({"valid": valid, "reason": "ok" if valid else "merge_invariants_failed", "row_counts": counts})
    return result


def _declared_input_hash_checks(manifest_paths: list[Path]) -> list[dict[str, Any]]:
    checks = []
    for path in manifest_paths:
        payload, _ = _read_json(path)
        if payload is None:
            continue
        inputs = payload.get("inputs", {})
        if isinstance(inputs, dict):
            for input_path, expected_hash in inputs.items():
                source = Path(input_path)
                actual = sha256_file(source) if source.exists() and source.is_file() else None
                checks.append(
                    {
                        "manifest": str(path),
                        "input": str(source),
                        "expected_sha256": expected_hash,
                        "actual_sha256": actual,
                        "valid": actual == expected_hash,
                    }
                )
        for prefix in ("train_csv", "targets_csv"):
            source_value = payload.get(prefix)
            expected_hash = payload.get(f"{prefix}_sha256")
            if source_value and expected_hash:
                source = Path(source_value)
                actual = sha256_file(source) if source.exists() and source.is_file() else None
                checks.append(
                    {
                        "manifest": str(path),
                        "input": str(source),
                        "expected_sha256": expected_hash,
                        "actual_sha256": actual,
                        "valid": actual == expected_hash,
                    }
                )
    return checks


def build_audit(data_dir: Path, output_root: Path) -> dict[str, Any]:
    frozen, frozen_reason = _read_json(output_root / "experiment_plan.json")
    controlled_model = str((frozen or {}).get("controlled_model", "EleutherAI/gpt-neo-125m"))
    sample_seed = int((frozen or {}).get("sample_seed", 4242))
    stages = build_plan(data_dir, output_root, controlled_model, sample_seed)
    stage_reports = {}
    all_commands = []
    for stage, commands in stages.items():
        reports = [command_completion_report(command, output_root) for command in commands]
        stage_reports[stage] = {
            "completed": sum(report["completed"] for report in reports),
            "total": len(reports),
            "commands": reports,
        }
        all_commands.extend(zip(commands, reports))

    manifest_paths = _required_manifests(output_root)
    manifest_reports = []
    for path in manifest_paths:
        payload, reason = _read_json(path)
        valid = payload is not None
        if valid and path.name != "experiment_plan.json":
            if path.name == "factorial_manifest.json":
                valid = _factorial_check(path)["valid"]
            else:
                valid = payload.get("status") == "completed"
            if not valid:
                reason = "status_not_completed"
        manifest_reports.append({"path": str(path), "valid": valid, "reason": "ok" if valid else reason})

    factorial_reports = [
        _factorial_check(output_root / "e7_crossed_2x2/data/factorial_manifest.json"),
        _factorial_check(output_root / "e11_controlled_family/data/factorial_manifest.json"),
    ]
    e12_report = _e12_merge_check(output_root / "e12_nested_protocol/features/all_protocols/shard_merge_manifest.json")
    input_hashes = _declared_input_hash_checks(manifest_paths)
    summary_outputs = [
        Path(path)
        for command, report in all_commands
        if report["completed"]
        for path in command.expected_outputs
        if Path(path).suffix.lower() in {".csv", ".json"}
    ]
    output_evidence = [_file_evidence(path) for path in sorted(set(summary_outputs))]
    commands_complete = all(report["completed"] for _, report in all_commands)
    manifests_valid = all(report["valid"] for report in manifest_reports)
    scientific_checks_valid = all(report["valid"] for report in factorial_reports) and e12_report["valid"]
    hashes_valid = all(report["valid"] for report in input_hashes)
    complete = commands_complete and manifests_valid and scientific_checks_valid and hashes_valid
    return {
        "schema_version": 1,
        "status": "completed" if complete else "incomplete",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output_root": str(output_root),
        "frozen_plan_valid": frozen is not None,
        "frozen_plan_reason": frozen_reason,
        "frozen_plan_sha256": sha256_file(output_root / "experiment_plan.json") if frozen else None,
        "environment": environment_evidence(data_dir),
        "stages": stage_reports,
        "required_manifests": manifest_reports,
        "factorial_integrity": factorial_reports,
        "e12_merge_integrity": e12_report,
        "declared_input_hashes": input_hashes,
        "output_evidence": output_evidence,
        "checks": {
            "commands_complete": commands_complete,
            "manifests_valid": manifests_valid,
            "scientific_checks_valid": scientific_checks_valid,
            "input_hashes_valid": hashes_valid,
        },
    }


def write_summary(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Reviewer follow-up final audit",
        "",
        f"- Status: **{audit['status']}**",
        f"- Generated: {audit['created_at']}",
        f"- Frozen plan SHA-256: `{audit.get('frozen_plan_sha256') or 'unavailable'}`",
        "",
        "## Stage completion",
        "",
        "| Stage | Complete | Total |",
        "|---|---:|---:|",
    ]
    for stage, report in audit["stages"].items():
        lines.append(f"| {stage} | {report['completed']} | {report['total']} |")
    lines.extend(["", "## Gates", ""])
    for name, valid in audit["checks"].items():
        lines.append(f"- {name}: {'PASS' if valid else 'PENDING/FAIL'}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="results/reviewer_followup_20260716")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--wait", action="store_true")
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
    while True:
        audit = build_audit(data_dir, output_root)
        pending = sum(report["total"] - report["completed"] for report in audit["stages"].values())
        if audit["status"] == "completed" or not args.wait:
            break
        print(f"[final-audit] waiting: {pending} controller commands remain", flush=True)
        time.sleep(args.poll_seconds)
    atomic_write_json(output_root / "final_audit.json", audit)
    write_summary(output_root / "final_audit_summary.md", audit)
    print(f"Final audit status: {audit['status']} (pending_commands={pending})")
    if audit["status"] != "completed" and not args.allow_incomplete and not args.wait:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
