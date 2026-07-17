#!/usr/bin/env python3
"""Plan, prepare, execute, and audit all reviewer-follow-up experiments.

The controller never writes into the paper's canonical result directories.
GPU stages require an explicit opt-in and a successful lab GPU-status API
snapshot immediately before execution.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from reviewer_followup.common import atomic_write_json, base_manifest


# Use a site-local endpoint in real deployments. The public supplement keeps
# only an anonymized default and permits an explicit environment override.
GPU_STATUS_URL = os.environ.get("GPU_STATUS_URL", "https://example.edu/api/gpu/status")
FT_GROUP = "mimir_wikipedia_nonmember_ft"
PT_GROUP = "mimir_wikipedia_member_pt"
UNSEEN_GROUP = "mimir_wikipedia_nonmember_unseen"
CHECKPOINT_SEEDS = (42, 123, 2026, 3407, 7919)
SAMPLE_SEEDS = (4242, 4343, 4444)
E12_SHARDS = 10


@dataclass(frozen=True)
class Command:
    name: str
    argv: list[str]
    expected_outputs: list[str]
    gpu: bool = False


def py(module: str, *args: object) -> list[str]:
    return [sys.executable, "-m", module, *(str(value) for value in args)]


def script(name: str, *args: object) -> list[str]:
    return [sys.executable, name, *(str(value) for value in args)]


def extraction_command(
    *,
    run_dir: Path,
    output_dir: Path,
    seed: int,
    targets_csv: Path | None = None,
    model_name: str = "",
    query_offset: int = 1,
    query_selection: str = "top_loss",
    rho: int = 10,
    fixed_steps: Iterable[int] = (20,),
    run_dynamic: bool = False,
    record_updates: bool = False,
    query_protocols: Iterable[tuple[str, int, str, int]] = (),
    shard: str = "",
    resume: bool = True,
) -> list[str]:
    argv = script(
        "extract_attention_hardsplit.py",
        "--run-dir",
        run_dir,
        "--adapter-dir",
        run_dir / "adapter",
        "--output-dir",
        output_dir,
        "--n-per-group",
        500 if targets_csv is None else 250,
        "--seed",
        seed,
        "--query-position-offset",
        query_offset,
        "--query-selection",
        query_selection,
        "--topk-loss-percent",
        rho,
        "--fixed-steps",
        *fixed_steps,
        "--run-dynamic" if run_dynamic else "--no-run-dynamic",
        "--skip-analyze",
        "--resume" if resume else "--no-resume",
    )
    if targets_csv is not None:
        argv.extend(["--targets-csv", str(targets_csv), "--target-groups", "p0f0", "p0f1", "p1f0", "p1f1"])
    if model_name:
        argv.extend(["--model-name", model_name])
    if record_updates:
        argv.append("--record-update-baselines")
    for name, offset, selection, protocol_rho in query_protocols:
        argv.extend(["--query-protocol", name, str(offset), selection, str(protocol_rho)])
    if shard:
        argv.extend(["--shard", shard])
    return argv


def evaluation_args(raw: Path, output: Path) -> list[str]:
    return py(
        "reviewer_followup.evaluate_attention_features",
        "--attention-csv",
        raw,
        "--output-dir",
        output,
        "--comparison",
        f"ft_vs_pt={FT_GROUP},{PT_GROUP}",
        "--comparison",
        f"ft_vs_unseen={FT_GROUP},{UNSEEN_GROUP}",
    )


def build_plan(data_dir: Path, output_root: Path, controlled_model: str, sample_seed: int) -> dict[str, list[Command]]:
    canonical_run = data_dir / "mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2"
    canonical_data = canonical_run / "data"
    member = canonical_data / "mimir_wikipedia_pt_member.csv"
    ft = canonical_data / "mimir_wikipedia_ft_nonmember.csv"
    unseen = canonical_data / "mimir_wikipedia_unseen_nonmember.csv"

    e7_data = output_root / "e7_crossed_2x2" / "data"
    e7_run = output_root / "e7_crossed_2x2" / "lora_ft"
    e7_features = output_root / "e7_crossed_2x2" / "features"
    e7_raw = e7_features / "raw_experiment4_attention_shift.csv"
    e7_samples = e7_features / "sample_level_experiment4.csv"
    e7_updates = e7_features / "raw_update_baseline_features.csv"
    targets = e7_data / "factorial_targets.csv"

    controlled_data = output_root / "e11_controlled_family" / "data"
    controlled_targets = controlled_data / "factorial_targets.csv"
    controlled_pt = output_root / "e11_controlled_family" / "controlled_pt"
    controlled_model_dir = controlled_pt / "controlled_pt_model"
    controlled_ft = output_root / "e11_controlled_family" / "lora_ft"
    controlled_features = output_root / "e11_controlled_family" / "features"

    stages: dict[str, list[Command]] = {
        "prepare": [
            Command(
                "build_mimir_crossed_dataset",
                py(
                    "reviewer_followup.build_factorial_dataset",
                    "--mode",
                    "mimir-membership",
                    "--member-csv",
                    member,
                    "--nonmember-csv",
                    ft,
                    "--nonmember-extra-csv",
                    unseen,
                    "--output-dir",
                    e7_data,
                    "--n-per-cell",
                    250,
                    "--seed",
                    42,
                ),
                [str(targets), str(e7_data / "factorial_ft_train.csv")],
            ),
            Command(
                "build_controlled_crossed_dataset",
                py(
                    "reviewer_followup.build_factorial_dataset",
                    "--mode",
                    "controlled-exposure",
                    "--nonmember-csv",
                    ft,
                    "--nonmember-extra-csv",
                    unseen,
                    "--output-dir",
                    controlled_data,
                    "--n-per-cell",
                    250,
                    "--seed",
                    31415,
                ),
                [str(controlled_targets), str(controlled_data / "factorial_controlled_pt_train.csv")],
            ),
        ],
        "e7": [
            Command(
                "train_crossed_lora",
                script(
                    "train_mimir_wikipedia_hardsplit_lora.py",
                    "--model",
                    "pythia-1b",
                    "--output-dir",
                    e7_run,
                    "--train-csv",
                    e7_data / "factorial_ft_train.csv",
                    "--targets-csv",
                    targets,
                    "--seed",
                    42,
                ),
                [str(e7_run / "adapter" / "adapter_config.json")],
                gpu=True,
            ),
            Command(
                "extract_crossed_attention_and_updates",
                extraction_command(
                    run_dir=e7_run,
                    output_dir=e7_features,
                    seed=4242,
                    targets_csv=targets,
                    fixed_steps=(20,),
                    record_updates=True,
                ),
                [str(e7_raw), str(e7_samples), str(e7_updates)],
                gpu=True,
            ),
            Command(
                "evaluate_crossed_factorial",
                py(
                    "reviewer_followup.evaluate_crossed_2x2",
                    "--attention-csv",
                    e7_raw,
                    "--targets-csv",
                    targets,
                    "--output-dir",
                    output_root / "e7_crossed_2x2" / "evaluation",
                ),
                [str(output_root / "e7_crossed_2x2" / "evaluation" / "factorial_contrast_summary.csv")],
            ),
        ],
        "e8": [
            Command(
                "evaluate_update_baselines",
                py(
                    "reviewer_followup.evaluate_update_baselines",
                    "--attention-csv",
                    e7_raw,
                    "--update-csv",
                    e7_updates,
                    "--sample-csv",
                    e7_samples,
                    "--output-dir",
                    output_root / "e8_update_baselines",
                    *sum(
                        (
                            ["--comparison", f"ft_effect_pt1=p1f1,p1f0"],
                            ["--comparison", f"ft_effect_pt0=p0f1,p0f0"],
                            ["--comparison", f"pt_effect_ft1=p1f1,p0f1"],
                            ["--comparison", f"pt_effect_ft0=p1f0,p0f0"],
                        ),
                        [],
                    ),
                ),
                [str(output_root / "e8_update_baselines" / "update_baseline_summary.csv")],
            )
        ],
        "e9": [],
        "e10": [],
        "e11": [],
        "e12": [],
    }

    seed_summaries: list[str] = []
    effects_by_comparison: dict[str, list[str]] = {"ft_vs_pt": [], "ft_vs_unseen": []}
    selections: list[str] = []
    for ft_seed in CHECKPOINT_SEEDS:
        seed_root = output_root / "e9_multiseed" / f"ft_seed_{ft_seed}"
        run = seed_root / "checkpoint"
        features_root = seed_root / "features"
        feature_dir = features_root / "fixed_attention_20_all"
        raw = feature_dir / "raw_experiment4_attention_shift.csv"
        evaluation = seed_root / "evaluation"
        stats = seed_root / "layer_head_stats"
        stages["e9"].extend(
            [
                Command(
                    f"train_ft_seed_{ft_seed}",
                    script(
                        "train_mimir_wikipedia_hardsplit_lora.py",
                        "--model",
                        "pythia-1b",
                        "--output-dir",
                        run,
                        "--from-csv-dir",
                        canonical_data,
                        "--seed",
                        ft_seed,
                    ),
                    [str(run / "adapter" / "adapter_config.json")],
                    gpu=True,
                ),
                Command(
                    f"extract_ft_seed_{ft_seed}",
                    extraction_command(run_dir=run, output_dir=feature_dir, seed=sample_seed),
                    [str(raw)],
                    gpu=True,
                ),
                Command(
                    f"evaluate_ft_seed_{ft_seed}",
                    evaluation_args(raw, evaluation),
                    [str(evaluation / "attention_summary.csv"), str(evaluation / "attention_selected_features.csv")],
                ),
                Command(
                    f"layer_head_ft_seed_{ft_seed}",
                    script("analyze_exp1_layer_head_significance.py", "--root", features_root, "--output-dir", stats),
                    [str(stats / "ft_vs_pt_layer_head_tests.csv"), str(stats / "ft_vs_unseen_layer_head_tests.csv")],
                ),
            ]
        )
        seed_summaries.extend(["--result", f"{ft_seed}={evaluation / 'attention_summary.csv'}"])
        selections.extend(["--selection", f"{ft_seed}={evaluation / 'attention_selected_features.csv'}"])
        for comparison in effects_by_comparison:
            effects_by_comparison[comparison].extend(
                ["--effect", f"{ft_seed}={stats / f'{comparison}_layer_head_tests.csv'}"]
            )
    stages["e9"].append(
        Command(
            "aggregate_checkpoint_seeds",
            py(
                "reviewer_followup.aggregate_multiseed",
                *seed_summaries,
                "--output-dir",
                output_root / "e9_multiseed" / "checkpoint_aggregation",
                "--method",
                "proposed_en",
            ),
            [str(output_root / "e9_multiseed" / "checkpoint_aggregation" / "checkpoint_summary.csv")],
        )
    )
    sample_seed_results: list[str] = [
        "--result",
        f"{sample_seed}={output_root / 'e9_multiseed' / 'ft_seed_42' / 'evaluation' / 'attention_summary.csv'}",
    ]
    seed42_run = output_root / "e9_multiseed" / "ft_seed_42" / "checkpoint"
    for auxiliary_seed in SAMPLE_SEEDS:
        if auxiliary_seed == sample_seed:
            continue
        auxiliary_root = output_root / "e9_multiseed" / "sample_seed_sensitivity" / f"sample_seed_{auxiliary_seed}"
        feature_dir = auxiliary_root / "fixed_attention_20_all"
        raw = feature_dir / "raw_experiment4_attention_shift.csv"
        evaluation = auxiliary_root / "evaluation"
        stages["e9"].extend(
            [
                Command(
                    f"extract_sample_seed_{auxiliary_seed}",
                    extraction_command(run_dir=seed42_run, output_dir=feature_dir, seed=auxiliary_seed),
                    [str(raw)],
                    gpu=True,
                ),
                Command(
                    f"evaluate_sample_seed_{auxiliary_seed}",
                    evaluation_args(raw, evaluation),
                    [str(evaluation / "attention_summary.csv")],
                ),
            ]
        )
        sample_seed_results.extend(["--result", f"{auxiliary_seed}={evaluation / 'attention_summary.csv'}"])
    stages["e9"].append(
        Command(
            "aggregate_sample_seeds",
            py(
                "reviewer_followup.aggregate_multiseed",
                *sample_seed_results,
                "--output-dir",
                output_root / "e9_multiseed" / "sample_seed_aggregation",
                "--method",
                "proposed_en",
                "--seed-axis",
                "sample",
            ),
            [str(output_root / "e9_multiseed" / "sample_seed_aggregation" / "sample_seed_summary.csv")],
        )
    )
    for comparison in ("ft_vs_pt", "ft_vs_unseen"):
        out = output_root / "e10_head_stability" / comparison
        stages["e10"].append(
            Command(
                f"head_stability_{comparison}",
                py(
                    "reviewer_followup.analyze_head_stability",
                    *effects_by_comparison[comparison],
                    *selections,
                    "--comparison",
                    comparison,
                    "--output-dir",
                    out,
                ),
                [str(out / "head_pairwise_stability.csv"), str(out / "selected_feature_frequency_across_seeds.csv")],
            )
        )

    controlled_raw = controlled_features / "raw_experiment4_attention_shift.csv"
    stages["e11"] = [
        Command(
            "controlled_full_parameter_pretraining",
            py(
                "reviewer_followup.train_controlled_pretraining",
                "--model-name",
                controlled_model,
                "--train-csv",
                controlled_data / "factorial_controlled_pt_train.csv",
                "--output-dir",
                controlled_pt,
                "--seed",
                2718,
            ),
            [str(controlled_model_dir / "config.json")],
            gpu=True,
        ),
        Command(
            "controlled_family_lora_ft",
            script(
                "train_mimir_wikipedia_hardsplit_lora.py",
                "--model-name",
                controlled_model_dir,
                "--target-modules",
                "q_proj,k_proj,v_proj,out_proj,c_fc,c_proj",
                "--output-dir",
                controlled_ft,
                "--train-csv",
                controlled_data / "factorial_ft_train.csv",
                "--targets-csv",
                controlled_targets,
                "--seed",
                2718,
            ),
            [str(controlled_ft / "adapter" / "adapter_config.json")],
            gpu=True,
        ),
        Command(
            "extract_controlled_family",
            extraction_command(
                run_dir=controlled_ft,
                output_dir=controlled_features,
                seed=1618,
                targets_csv=controlled_targets,
                model_name=str(controlled_model_dir),
                fixed_steps=(20,),
                record_updates=True,
            ),
            [str(controlled_raw)],
            gpu=True,
        ),
        Command(
            "evaluate_controlled_factorial",
            py(
                "reviewer_followup.evaluate_crossed_2x2",
                "--attention-csv",
                controlled_raw,
                "--targets-csv",
                controlled_targets,
                "--output-dir",
                output_root / "e11_controlled_family" / "evaluation",
            ),
            [str(output_root / "e11_controlled_family" / "evaluation" / "factorial_contrast_summary.csv")],
        ),
    ]

    candidate_args: list[str] = []
    e12_protocols: list[tuple[str, int, str, int]] = []
    for offset in (0, 1):
        query_settings = [(selection, rho) for selection in ("top_loss", "random", "gradient_logit") for rho in (5, 10, 20)]
        query_settings.append(("all_valid", 100))
        for selection, rho in query_settings:
            protocol = f"q{offset}_{selection}_r{rho}"
            e12_protocols.append((protocol, offset, selection, rho))
    e12_features = output_root / "e12_nested_protocol" / "features" / "all_protocols"
    e12_raw = e12_features / "raw_experiment4_attention_shift.csv"
    shard_dirs = []
    for shard_index in range(E12_SHARDS):
        shard_dir = output_root / "e12_nested_protocol" / "features" / f"all_protocols_shard_{shard_index}_of_{E12_SHARDS}"
        shard_dirs.append(shard_dir)
        stages["e12"].append(
            Command(
                f"extract_all_query_protocols_shard_{shard_index}",
                extraction_command(
                    run_dir=canonical_run,
                    output_dir=shard_dir,
                    seed=sample_seed,
                    fixed_steps=(20, 50, 100),
                    run_dynamic=True,
                    query_protocols=e12_protocols,
                    shard=f"{shard_index}/{E12_SHARDS}",
                ),
                [str(shard_dir / "raw_experiment4_attention_shift.csv")],
                gpu=True,
            )
        )
    merge_args: list[str] = []
    for shard_dir in shard_dirs:
        merge_args.extend(["--shard-dir", str(shard_dir)])
    stages["e12"].append(
        Command(
            "merge_all_query_protocol_shards",
            py(
                "reviewer_followup.merge_extraction_shards",
                *merge_args,
                "--output-dir",
                e12_features,
                "--expected-targets",
                1500,
                "--expected-conditions",
                len(e12_protocols) * 4,
            ),
            [str(e12_raw), str(e12_features / "shard_merge_manifest.json")],
        )
    )
    for protocol, _, _, _ in e12_protocols:
        for condition in ("fixed_attention_20", "fixed_attention_50", "fixed_attention_100", "dynamic_attention"):
            combined_condition = f"{condition}__{protocol}"
            candidate_args.extend(["--candidate", f"{protocol}_{condition}={e12_raw}@{combined_condition}"])
    for name, positive, negative in (
        ("ft_vs_pt", FT_GROUP, PT_GROUP),
        ("ft_vs_unseen", FT_GROUP, UNSEEN_GROUP),
    ):
        out = output_root / "e12_nested_protocol" / "evaluation" / name
        stages["e12"].append(
            Command(
                f"nested_select_{name}",
                py(
                    "reviewer_followup.run_nested_protocol_selection",
                    *candidate_args,
                    "--positive-group",
                    positive,
                    "--negative-group",
                    negative,
                    "--output-dir",
                    out,
                ),
                [str(out / "nested_protocol_repeat_auc.csv"), str(out / "nested_protocol_selection_counts.csv")],
            )
        )
    return stages


def snapshot_gpu(output_root: Path) -> dict:
    snapshot_path = os.environ.get("GPU_STATUS_SNAPSHOT", "").strip()
    if snapshot_path:
        source = Path(snapshot_path).expanduser()
        if not source.exists():
            raise FileNotFoundError(f"GPU_STATUS_SNAPSHOT does not exist: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(GPU_STATUS_URL, timeout=20) as response:
            payload = json.load(response)
    generated_at = str(payload.get("generated_at", "")) if isinstance(payload, dict) else ""
    if not generated_at:
        raise RuntimeError("GPU status payload has no generated_at timestamp")
    generated = dt.datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    age_seconds = (dt.datetime.now(dt.timezone.utc) - generated.astimezone(dt.timezone.utc)).total_seconds()
    max_age_seconds = int(os.environ.get("GPU_STATUS_MAX_AGE_SECONDS", "600"))
    if max_age_seconds < 60 or max_age_seconds > 3600:
        raise RuntimeError("GPU_STATUS_MAX_AGE_SECONDS must be between 60 and 3600")
    if age_seconds < -60 or age_seconds > max_age_seconds:
        raise RuntimeError(f"GPU status snapshot is stale or invalid: age_seconds={age_seconds:.1f}")
    if not any(server.get("status") == "ok" and float(server.get("age_seconds", 9999)) <= 300 for server in payload.get("servers", [])):
        raise RuntimeError("GPU status payload has no fresh healthy server")
    path = output_root / "provenance" / f"gpu_status_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json"
    compact = dict(payload)
    compact["servers"] = [
        {**server, "gpus": [{key: value for key, value in gpu.items() if key != "history"} for gpu in server.get("gpus", [])]}
        for server in payload.get("servers", [])
    ]
    compact["validated_age_seconds"] = age_seconds
    compact["accepted_max_age_seconds"] = max_age_seconds
    compact["source_snapshot"] = snapshot_path or GPU_STATUS_URL
    atomic_write_json(path, compact)
    print(f"Saved validated GPU status snapshot: {path} (age={age_seconds:.1f}s)")
    return payload


def write_plan(stages: dict[str, list[Command]], output_root: Path, args: argparse.Namespace) -> Path:
    payload = base_manifest(experiment="all_reviewer_followup_experiments", command=sys.argv)
    payload.update(
        {
            "status": "implemented",
            "output_root": str(output_root),
            "ft_checkpoint_seeds": list(CHECKPOINT_SEEDS),
            "sample_sensitivity_seeds": list(SAMPLE_SEEDS),
            "sample_seed": args.sample_seed,
            "controlled_model": args.controlled_model,
            "stages": {name: [asdict(command) for command in commands] for name, commands in stages.items()},
        }
    )
    path = output_root / "experiment_plan.json"
    atomic_write_json(path, payload)
    return path


def _validate_expected_output(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if not path.is_file():
        return False, "not_a_file"
    if path.stat().st_size <= 0:
        return False, "empty"
    try:
        if path.suffix.lower() == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
                first_row = next(reader, None)
            if not header:
                return False, "missing_csv_header"
            if first_row is None:
                return False, "csv_has_no_data_rows"
    except (OSError, UnicodeError, ValueError, csv.Error, json.JSONDecodeError) as exc:
        return False, f"unreadable:{type(exc).__name__}"
    return True, "ok"


def _extract_status_files(command: Command) -> list[Path]:
    return sorted({Path(path).parent / "run_status.txt" for path in command.expected_outputs})


def _valid_completed_status(path: Path) -> bool:
    if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
        return False
    return "completed" in path.read_text(encoding="utf-8", errors="replace")


def command_completion_report(command: Command, output_root: Path) -> dict:
    output_checks = {
        str(path): {"valid": valid, "reason": reason}
        for path in map(Path, command.expected_outputs)
        for valid, reason in [_validate_expected_output(path)]
    }
    reasons = [f"output:{path}:{check['reason']}" for path, check in output_checks.items() if not check["valid"]]
    marker = output_root / ".controller_done" / f"{command.name}.json"
    marker_payload = None
    if not marker.exists():
        reasons.append("controller_marker:missing")
    else:
        try:
            marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            reasons.append(f"controller_marker:unreadable:{type(exc).__name__}")
        if isinstance(marker_payload, dict):
            if marker_payload.get("name") != command.name:
                reasons.append("controller_marker:name_mismatch")
            recorded_outputs = [str(Path(path)) for path in marker_payload.get("outputs", [])]
            expected_outputs = [str(Path(path)) for path in command.expected_outputs]
            if recorded_outputs != expected_outputs:
                reasons.append("controller_marker:outputs_mismatch")
        elif marker_payload is not None:
            reasons.append("controller_marker:not_an_object")

    status_checks = {}
    if command.name.startswith("extract_"):
        status_checks = {str(path): _valid_completed_status(path) for path in _extract_status_files(command)}
        reasons.extend(f"run_status:{path}:not_completed" for path, valid in status_checks.items() if not valid)
    return {
        "name": command.name,
        "completed": not reasons,
        "reasons": reasons,
        "marker": str(marker),
        "outputs": output_checks,
        "run_status": status_checks,
    }


def command_is_complete(command: Command, output_root: Path) -> bool:
    return bool(command_completion_report(command, output_root)["completed"])


def _legacy_supporting_artifacts(command: Command) -> list[Path]:
    first = Path(command.expected_outputs[0]) if command.expected_outputs else Path()
    name = command.name
    if name.startswith("build_"):
        return [first.parent / "factorial_manifest.json"]
    if name == "controlled_full_parameter_pretraining":
        return [first.parent.parent / "controlled_pretraining_manifest.json"]
    if name.startswith("train_") or name == "controlled_family_lora_ft":
        return [first.parent.parent / "train_config.json"]
    return []


def reconcile_existing_commands(stages: dict[str, list[Command]], output_root: Path) -> tuple[int, list[str]]:
    """Create strict markers for verified legacy outputs produced before markers were mandatory."""
    reconciled = 0
    skipped: list[str] = []
    for commands in stages.values():
        for command in commands:
            marker = output_root / ".controller_done" / f"{command.name}.json"
            if marker.exists():
                continue
            output_checks = [_validate_expected_output(Path(path)) for path in command.expected_outputs]
            if not command.expected_outputs or not all(valid for valid, _ in output_checks):
                skipped.append(command.name)
                continue
            if command.name.startswith("extract_"):
                if not all(_valid_completed_status(path) for path in _extract_status_files(command)):
                    skipped.append(command.name)
                    continue
            else:
                supporting = _legacy_supporting_artifacts(command)
                if not supporting or not all(_validate_expected_output(path)[0] for path in supporting):
                    skipped.append(command.name)
                    continue
            atomic_write_json(
                marker,
                {
                    "name": command.name,
                    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "outputs": command.expected_outputs,
                    "reconciled_from_verified_legacy_outputs": True,
                    "supporting_artifacts": [str(path) for path in _legacy_supporting_artifacts(command)],
                },
            )
            reconciled += 1
    return reconciled, skipped


def run_commands(commands: list[Command], data_dir: Path, *, gpu_allowed: bool, output_root: Path) -> None:
    if any(command.gpu for command in commands):
        if not gpu_allowed:
            raise SystemExit("This stage contains GPU jobs. Re-run with --yes-really-run-gpu after reviewing experiment_plan.json.")
        snapshot_gpu(output_root)
    log_path = output_root / "execution_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for command in commands:
        if command_is_complete(command, output_root):
            print(f"\n[{command.name}] already complete; skipping", flush=True)
            continue
        started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        print(f"\n[{command.name}] {shlex.join(command.argv)}", flush=True)
        result = subprocess.run(command.argv, cwd=data_dir, check=False)
        record = {"name": command.name, "started_at": started, "returncode": result.returncode, "argv": command.argv}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if result.returncode != 0:
            raise SystemExit(f"Stage stopped at {command.name} (exit={result.returncode})")
        invalid = {
            path: reason
            for path in command.expected_outputs
            for valid, reason in [_validate_expected_output(Path(path))]
            if not valid
        }
        if invalid:
            raise SystemExit(f"{command.name} exited successfully but expected outputs are invalid: {invalid}")
        if command.name.startswith("extract_"):
            incomplete_status = [str(path) for path in _extract_status_files(command) if not _valid_completed_status(path)]
            if incomplete_status:
                raise SystemExit(f"{command.name} exited successfully but run status is incomplete: {incomplete_status}")
        marker = output_root / ".controller_done" / f"{command.name}.json"
        atomic_write_json(
            marker,
            {
                "name": command.name,
                "status": "completed",
                "returncode": 0,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "outputs": command.expected_outputs,
                "output_sizes": {path: Path(path).stat().st_size for path in command.expected_outputs},
            },
        )


def status(stages: dict[str, list[Command]], output_root: Path) -> None:
    rows = []
    for stage, commands in stages.items():
        completed = sum(command_is_complete(command, output_root) for command in commands)
        rows.append((stage, completed, len(commands)))
    for stage, completed, total in rows:
        print(f"{stage:8s} {completed:3d}/{total:3d} commands complete")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["plan", "prepare", "run-stage", "run-command", "run-sequence", "status", "reconcile"]
    )
    parser.add_argument("--stage", choices=["prepare", "e7", "e8", "e9", "e10", "e11", "e12"])
    parser.add_argument(
        "--command",
        action="append",
        default=[],
        help="Exact command name from experiment_plan.json; repeat for run-sequence",
    )
    parser.add_argument("--output-root", default="results/reviewer_followup_20260716")
    parser.add_argument("--controlled-model", default="EleutherAI/gpt-neo-125m")
    parser.add_argument("--sample-seed", type=int, default=4242)
    parser.add_argument("--yes-really-run-gpu", action="store_true")
    parser.add_argument("--yes-reconcile-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_dir = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (data_dir / output_root).resolve()
    stages = build_plan(data_dir, output_root, args.controlled_model, args.sample_seed)
    plan_path = output_root / "experiment_plan.json"
    if args.action == "plan":
        plan_path = write_plan(stages, output_root, args)
        print(f"Wrote {plan_path}")
    elif args.action == "status":
        status(stages, output_root)
    elif args.action == "reconcile":
        if not args.yes_reconcile_existing:
            raise SystemExit("reconcile writes strict completion markers; re-run with --yes-reconcile-existing")
        reconciled, skipped = reconcile_existing_commands(stages, output_root)
        print(f"Reconciled {reconciled} legacy commands")
        if skipped:
            print("Not reconciled (missing or insufficient evidence): " + ", ".join(skipped))
    elif args.action == "prepare":
        write_plan(stages, output_root, args)
        run_commands(stages["prepare"], data_dir, gpu_allowed=False, output_root=output_root)
    else:
        if args.action == "run-stage" and not args.stage:
            raise SystemExit("--stage is required with run-stage")
        if args.action in {"run-command", "run-sequence"} and not args.command:
            raise SystemExit("--command is required with run-command/run-sequence")
        if args.action == "run-command" and len(args.command) != 1:
            raise SystemExit("run-command accepts exactly one --command")
        if not plan_path.exists():
            raise SystemExit("experiment_plan.json is missing; run the controller's plan action first")
        frozen = json.loads(plan_path.read_text(encoding="utf-8"))
        if frozen.get("controlled_model") != args.controlled_model or int(frozen.get("sample_seed", -1)) != args.sample_seed:
            raise SystemExit("CLI model/seed differs from the frozen plan; regenerate it with the plan action first")
        requested = set(args.command)
        selected = stages[args.stage] if args.action == "run-stage" else [
            command for commands in stages.values() for command in commands if command.name in requested
        ]
        if args.action == "run-sequence":
            selected = sorted(selected, key=lambda command: args.command.index(command.name))
        if not selected:
            raise SystemExit(f"Unknown command: {args.command}")
        if args.action != "run-stage" and {command.name for command in selected} != requested:
            missing = sorted(requested - {command.name for command in selected})
            raise SystemExit(f"Unknown command(s): {missing}")
        run_commands(selected, data_dir, gpu_allowed=args.yes_really_run_gpu, output_root=output_root)


if __name__ == "__main__":
    main(sys.argv[1:])
