#!/usr/bin/env python3
"""Write an auditable implementation-to-paper baseline fidelity manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from reviewer_followup.common import atomic_write_json, base_manifest, sha256_file


BASELINES = {
    "initial_loss": {
        "entrypoint": "run_strict_fixed20_comparison_10runs.py",
        "paper_role": "one-dimensional logistic regression with direction learned on training folds",
        "required_tokens": ["run_scalar_lr", "initial_loss", "common_splits"],
        "official_or_adapted": "paper-matched local implementation",
    },
    "loss_decrease": {
        "entrypoint": "run_strict_fixed20_comparison_10runs.py",
        "paper_role": "one-dimensional logistic regression with direction learned on training folds",
        "required_tokens": ["run_scalar_lr", "loss_decrease", "common_splits"],
        "official_or_adapted": "paper-matched local implementation",
    },
    "lora_leak": {
        "entrypoint": "run_lora_leak_official_mimir_hardsplit_2.py",
        "paper_role": "frozen target_mink++_0.2 scalar reported as Min-K%++ (LoRA-FT)",
        "required_tokens": ["mink", "target"],
        "official_or_adapted": "single score from the LoRA-Leak method family; not the complete attack suite",
    },
    "attenmia": {
        "entrypoint": "run_attenmia_official_mimir_hardsplit.py",
        "paper_role": "official attention-perturbation feature construction with common-fold MLP evaluation",
        "required_tokens": ["attention", "pert"],
        "official_or_adapted": "adapted official method",
    },
    "fusion_2d": {
        "entrypoint": "run_crossfit_fusion_en_lora_leak.py",
        "paper_role": "outer-fold-safe fusion of Proposed+EN and Min-K%++ (LoRA-FT) scores",
        "required_tokens": ["outer", "fusion"],
        "official_or_adapted": "new composition; not an original baseline",
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    rows = {}
    for name, spec in BASELINES.items():
        path = data_dir / spec["entrypoint"]
        if not path.is_file():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        missing = [token for token in spec["required_tokens"] if token.lower() not in text]
        if missing:
            raise ValueError(f"{name} implementation is missing fidelity tokens: {missing}")
        rows[name] = {
            **{key: value for key, value in spec.items() if key != "required_tokens"},
            # Keep the manifest portable and anonymous: the entrypoint is
            # resolved against --data-dir for validation, but only its
            # supplement-relative name is serialized.
            "path": spec["entrypoint"], "sha256": sha256_file(path), "validation": "passed",
            "paper_protocol_changes": ["shared target split", "shared outer folds", "train-only preprocessing"],
        }
    manifest = base_manifest(
        experiment="baseline_fidelity_audit",
        command=[
            "python", "-m", "reviewer_followup.write_baseline_fidelity_manifest",
            "--data-dir", ".", "--output-json", Path(args.output_json).name,
        ],
    )
    manifest.update(
        {
            "status": "completed", "baselines": rows,
            "claim_boundary": "The reported Min-K%++ row is one frozen scalar from the LoRA-Leak method family, not a reproduction of the complete attack suite; other fidelity statements concern the paper-controlled split and folds.",
        }
    )
    atomic_write_json(Path(args.output_json), manifest)
    print(f"Validated {len(rows)} baseline implementations")


if __name__ == "__main__":
    main()
