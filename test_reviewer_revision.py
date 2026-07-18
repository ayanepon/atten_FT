"""CPU-only tests for the 2026-07-18 reviewer-revision implementation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class TestReviewerRevisionStatistics(unittest.TestCase):
    @staticmethod
    def predictions(methods=("attention_en",), repeats=3, n_per_class=20):
        rows = []
        rng = np.random.default_rng(7)
        for comparison in ("ft_effect_pt1", "pt_effect_ft0"):
            for method in methods:
                for repeat in range(1, repeats + 1):
                    for y in (0, 1):
                        for index in range(n_per_class):
                            rows.append(
                                {
                                    "comparison": comparison, "method": method, "repeat": repeat,
                                    "sample_id": f"{comparison}-{y}-{index}", "y_true": y,
                                    "score": float(0.15 + 0.7 * y + rng.normal(0, 0.02)),
                                }
                            )
        return pd.DataFrame(rows)

    def test_factorial_bootstrap_averages_repeats_before_resampling(self):
        from reviewer_followup.analyze_factorial_uncertainty import average_target_scores, bootstrap_auc_rows

        averaged = average_target_scores(self.predictions())
        self.assertEqual(averaged["n_repeats"].min(), 3)
        self.assertEqual(len(averaged), 80)
        result = bootstrap_auc_rows(averaged, source="synthetic", n_bootstrap=100, seed=1)
        self.assertTrue((result["n_targets"] == 40).all())
        self.assertTrue((result["chance_classification"] == "above_chance").all())

    def test_strict_oof_reports_paired_target_delta(self):
        from reviewer_followup.analyze_oof_uncertainty import infer_oof_uncertainty

        source = self.predictions(methods=("proposed_en", "baseline"), repeats=2)
        source["model"] = "model"
        source["uid"] = source["sample_id"]
        source.loc[source["method"] == "baseline", "score"] = 0.5
        methods, deltas = infer_oof_uncertainty(source, n_bootstrap=100, seed=2)
        self.assertEqual(len(methods), 4)
        self.assertEqual(len(deltas), 2)
        self.assertTrue((deltas["delta_auc"] > 0).all())

    def test_update_inference_uses_target_scores(self):
        from reviewer_followup.evaluate_update_baselines import target_score_inference

        boot, perm = target_score_inference(self.predictions(methods=("gradient",)), n_bootstrap=50, n_permutations=50, seed=3)
        self.assertEqual(set(boot["n_targets"]), {40})
        self.assertTrue((perm["test"] == "fixed_oof_score_label_permutation").all())

    def test_fusion_validator_requires_all_four_claim_rows(self):
        from reviewer_followup.summarize_fusion_uncertainty import validate

        rows = []
        for comparison in ("ft_vs_pt", "ft_vs_unseen"):
            for baseline in ("lora_leak", "proposed_en"):
                rows.append(
                    {"comparison": comparison, "augmented_method": "fusion_2d", "baseline_method": baseline,
                     "delta_auc": 0.02, "ci_low": -0.01, "ci_high": 0.04}
                )
        result = validate(pd.DataFrame(rows))
        self.assertEqual(len(result), 4)
        with self.assertRaises(ValueError):
            validate(pd.DataFrame(rows[:-1]))


class TestRevisionController(unittest.TestCase):
    def test_fidelity_manifest_serializes_only_relative_paths(self):
        from reviewer_followup.write_baseline_fidelity_manifest import BASELINES, main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            required_by_entrypoint = {}
            for spec in BASELINES.values():
                required_by_entrypoint.setdefault(spec["entrypoint"], set()).update(spec["required_tokens"])
            for entrypoint, required_tokens in required_by_entrypoint.items():
                path = root / entrypoint
                tokens = "\n".join(sorted(required_tokens))
                path.write_text(tokens, encoding="utf-8")
            output = root / "baseline_fidelity_manifest.json"
            main(["--data-dir", str(root), "--output-json", str(output)])
            manifest = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(all(not Path(row["path"]).is_absolute() for row in manifest["baselines"].values()))
            self.assertTrue(all(not Path(token).is_absolute() for token in manifest["command"]))

    def test_revision_plan_is_isolated_and_complete(self):
        from reviewer_followup.revision_controller import build_plan

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stages = build_plan(Path("/data"), root / "source", root / "revision")
            self.assertEqual(set(stages), {"prepare", "e13_train", "e13_extract", "e13_evaluate", "e14", "uncertainty"})
            self.assertEqual(len(stages["e13_train"]), 4)
            self.assertEqual(len(stages["e13_extract"]), 4)
            self.assertTrue(all(command.gpu for command in stages["e13_train"] + stages["e13_extract"]))
            self.assertEqual(sum(command.gpu for command in stages["e14"]), 3)
            all_outputs = [Path(path) for commands in stages.values() for command in commands for path in command.expected_outputs]
            self.assertTrue(all(str(path).startswith(str(root / "revision")) for path in all_outputs))

    def test_pythia160m_model_preset(self):
        from hardsplit.models import resolve_model_spec

        spec = resolve_model_spec("pythia160m")
        self.assertEqual(spec.hf_id, "EleutherAI/pythia-160m")
        self.assertEqual(spec.family, "gpt_neox")
        self.assertIn("query_key_value", spec.lora_target_modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
