#!/usr/bin/env python3
"""Frozen controller for reviewer-revision analyses E13 and E14.

This controller is intentionally separate from the completed E7--E12 plan.
GPU actions require the same live lab-status snapshot and explicit opt-in as
the original controller.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from reviewer_followup.common import atomic_write_json, base_manifest
from reviewer_followup.controller import Command, extraction_command, py, run_commands, script, status


FT_SEEDS = (42, 123, 2026, 3407, 7919)
NEW_FT_SEEDS = FT_SEEDS[1:]


def build_plan(data_dir: Path, source_root: Path, output_root: Path) -> dict[str, list[Command]]:
    e7 = source_root / "e7_crossed_2x2"
    e11 = source_root / "e11_controlled_family"
    targets = e7 / "data" / "factorial_targets.csv"
    ft_train = e7 / "data" / "factorial_ft_train.csv"
    seed42_predictions = e7 / "evaluation" / "factorial_outer_predictions.csv"
    seed42_reuse = output_root / "e13_crossed_multicheckpoint" / "seed_42_reuse.json"

    stages: dict[str, list[Command]] = {
        "prepare": [
            Command(
                "verify_e13_seed42_reuse",
                py(
                    "reviewer_followup.verify_seed42_reuse",
                    "--train-csv", ft_train,
                    "--targets-csv", targets,
                    "--train-config", e7 / "lora_ft" / "train_config.json",
                    "--predictions-csv", seed42_predictions,
                    "--output-json", seed42_reuse,
                ),
                [str(seed42_reuse)],
            )
        ],
        "e13_train": [],
        "e13_extract": [],
        "e13_evaluate": [],
        "e14": [],
        "uncertainty": [],
    }

    prediction_args: list[object] = ["--prediction", f"42={seed42_predictions}"]
    uncertainty_args: list[object] = ["--input", f"pythia_seed42={seed42_predictions}"]
    for ft_seed in NEW_FT_SEEDS:
        seed_root = output_root / "e13_crossed_multicheckpoint" / f"ft_seed_{ft_seed}"
        run_dir = seed_root / "lora_ft"
        features = seed_root / "features"
        raw = features / "raw_experiment4_attention_shift.csv"
        evaluation = seed_root / "evaluation"
        prediction = evaluation / "factorial_outer_predictions.csv"
        stages["e13_train"].append(
            Command(
                f"e13_train_ft_seed_{ft_seed}",
                script(
                    "train_mimir_wikipedia_hardsplit_lora.py",
                    "--model", "pythia-1b",
                    "--output-dir", run_dir,
                    "--train-csv", ft_train,
                    "--targets-csv", targets,
                    "--seed", ft_seed,
                ),
                [str(run_dir / "adapter" / "adapter_config.json")],
                gpu=True,
            )
        )
        stages["e13_extract"].append(
            Command(
                f"extract_e13_ft_seed_{ft_seed}",
                extraction_command(
                    run_dir=run_dir, output_dir=features, seed=4242,
                    targets_csv=targets, fixed_steps=(20,),
                ),
                [str(raw)],
                gpu=True,
            )
        )
        stages["e13_evaluate"].append(
            Command(
                f"e13_evaluate_ft_seed_{ft_seed}",
                py(
                    "reviewer_followup.evaluate_crossed_2x2",
                    "--attention-csv", raw, "--targets-csv", targets,
                    "--output-dir", evaluation,
                ),
                [str(prediction), str(evaluation / "factorial_contrast_summary.csv")],
            )
        )
        prediction_args.extend(["--prediction", f"{ft_seed}={prediction}"])
        uncertainty_args.extend(["--input", f"pythia_seed{ft_seed}={prediction}"])

    aggregation = output_root / "e13_crossed_multicheckpoint" / "aggregation"
    stages["e13_evaluate"].append(
        Command(
            "e13_aggregate_crossed_checkpoints",
            py(
                "reviewer_followup.analyze_crossed_multicheckpoint",
                *prediction_args, "--output-dir", aggregation,
            ),
            [str(aggregation / "crossed_checkpoint_summary.csv")],
        )
    )

    e14 = output_root / "e14_controlled_pythia160m"
    e14_data = e11 / "data"
    e14_targets = e14_data / "factorial_targets.csv"
    controlled_pt = e14 / "controlled_pt"
    controlled_model = controlled_pt / "controlled_pt_model"
    controlled_ft = e14 / "lora_ft"
    features = e14 / "features"
    raw = features / "raw_experiment4_attention_shift.csv"
    evaluation = e14 / "evaluation"
    e14_predictions = evaluation / "factorial_outer_predictions.csv"
    stages["e14"] = [
        Command(
            "e14_pythia160m_controlled_pretraining",
            py(
                "reviewer_followup.train_controlled_pretraining",
                "--model-name", "EleutherAI/pythia-160m",
                "--train-csv", e14_data / "factorial_controlled_pt_train.csv",
                "--output-dir", controlled_pt, "--seed", 2718,
            ),
            [str(controlled_model / "config.json")],
            gpu=True,
        ),
        Command(
            "e14_pythia160m_lora_ft",
            script(
                "train_mimir_wikipedia_hardsplit_lora.py",
                "--model-name", controlled_model,
                "--target-modules", "query_key_value,dense,dense_h_to_4h,dense_4h_to_h",
                "--output-dir", controlled_ft,
                "--train-csv", e14_data / "factorial_ft_train.csv",
                "--targets-csv", e14_targets, "--seed", 2718,
            ),
            [str(controlled_ft / "adapter" / "adapter_config.json")],
            gpu=True,
        ),
        Command(
            "extract_e14_pythia160m",
            extraction_command(
                run_dir=controlled_ft, output_dir=features, seed=1618,
                targets_csv=e14_targets, model_name=str(controlled_model),
                fixed_steps=(20,), record_updates=True,
            ),
            [str(raw), str(features / "raw_update_baseline_features.csv")],
            gpu=True,
        ),
        Command(
            "e14_evaluate_pythia160m",
            py(
                "reviewer_followup.evaluate_crossed_2x2",
                "--attention-csv", raw, "--targets-csv", e14_targets,
                "--output-dir", evaluation,
            ),
            [str(e14_predictions), str(evaluation / "factorial_contrast_summary.csv")],
        ),
    ]

    neo_predictions = e11 / "evaluation" / "factorial_outer_predictions.csv"
    factorial_uncertainty = output_root / "factorial_uncertainty"
    stages["uncertainty"] = [
        Command(
            "analyze_e13_per_checkpoint_uncertainty",
            py(
                "reviewer_followup.analyze_factorial_uncertainty",
                *uncertainty_args, "--output-dir", output_root / "e13_crossed_multicheckpoint" / "uncertainty",
            ),
            [str(output_root / "e13_crossed_multicheckpoint" / "uncertainty" / "factorial_target_bootstrap.csv")],
        ),
        Command(
            "analyze_controlled_model_uncertainty",
            py(
                "reviewer_followup.analyze_factorial_uncertainty",
                "--input", f"gptneo125m={neo_predictions}",
                "--input", f"pythia160m={e14_predictions}",
                "--output-dir", factorial_uncertainty,
            ),
            [str(factorial_uncertainty / "factorial_target_bootstrap.csv")],
        ),
    ]
    return stages


def write_plan(stages: dict[str, list[Command]], source_root: Path, output_root: Path) -> Path:
    payload = base_manifest(experiment="reviewer_revision_e13_e14", command=sys.argv)
    payload.update(
        {
            "status": "implemented", "source_root": str(source_root), "output_root": str(output_root),
            "ft_checkpoint_seeds": list(FT_SEEDS), "controlled_model": "EleutherAI/pythia-160m",
            "seed42_reuse_policy": "exact input hashes and seed must match frozen E7",
            "stages": {stage: [asdict(command) for command in commands] for stage, commands in stages.items()},
        }
    )
    path = output_root / "revision_experiment_plan.json"
    atomic_write_json(path, payload)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "run-stage", "run-command", "status"])
    parser.add_argument("--stage", choices=["prepare", "e13_train", "e13_extract", "e13_evaluate", "e14", "uncertainty"])
    parser.add_argument("--command", action="append", default=[])
    parser.add_argument("--source-root", default="results/reviewer_followup_20260716")
    parser.add_argument("--output-root", default="results/reviewer_revision_20260718")
    parser.add_argument("--yes-really-run-gpu", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_dir = Path(__file__).resolve().parents[1]
    source_root = Path(args.source_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    if not source_root.is_absolute():
        source_root = (data_dir / source_root).resolve()
    if not output_root.is_absolute():
        output_root = (data_dir / output_root).resolve()
    stages = build_plan(data_dir, source_root, output_root)
    plan = output_root / "revision_experiment_plan.json"
    if args.action == "plan":
        print(f"Wrote {write_plan(stages, source_root, output_root)}")
        return
    if args.action == "status":
        status(stages, output_root)
        return
    if not plan.exists():
        raise SystemExit("revision_experiment_plan.json is missing; run plan first")
    frozen = json.loads(plan.read_text(encoding="utf-8"))
    if frozen.get("source_root") != str(source_root) or frozen.get("output_root") != str(output_root):
        raise SystemExit("CLI roots differ from the frozen revision plan")
    if args.action == "run-stage":
        if not args.stage:
            raise SystemExit("--stage is required")
        selected = stages[args.stage]
    else:
        requested = set(args.command)
        selected = [command for commands in stages.values() for command in commands if command.name in requested]
        if {command.name for command in selected} != requested:
            raise SystemExit("Unknown command name")
    run_commands(selected, data_dir, gpu_allowed=args.yes_really_run_gpu, output_root=output_root)


if __name__ == "__main__":
    main(sys.argv[1:])
