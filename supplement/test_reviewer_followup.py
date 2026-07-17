#!/usr/bin/env python3
"""CPU-only regression tests for all reviewer-follow-up experiment paths."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


class TestFactorialData(unittest.TestCase):
    def test_e12_duplicate_writer_repair_keeps_one_complete_raw_block(self):
        from reviewer_followup.repair_e12_duplicate_writers import repair_shard

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "all_protocols_shard_0_of_10"
            shard.mkdir()
            sample = pd.DataFrame(
                [
                    {"condition": "dynamic_attention", "sample_id": 10, "score": 1.0},
                    {"condition": "dynamic_attention", "sample_id": 10, "score": 2.0},
                    {"condition": "dynamic_attention", "sample_id": 20, "score": 3.0},
                ]
            )
            sample.to_csv(shard / "sample_level_experiment4.csv", index=False)
            first = pd.DataFrame(
                [
                    {"condition": "dynamic_attention__p", "sample_id": 10, "layer": 0, "head": head, "value": head}
                    for head in range(2)
                ]
            )
            second = first.assign(value=lambda frame: frame["value"] + 10)
            committed = pd.DataFrame(
                [
                    {"condition": "dynamic_attention__p", "sample_id": 20, "layer": 0, "head": head, "value": head}
                    for head in range(2)
                ]
            )
            orphan = pd.DataFrame(
                [{"condition": "dynamic_attention__p", "sample_id": 30, "layer": 0, "head": head, "value": head} for head in range(2)]
            )
            pd.concat([first, second, committed, orphan], ignore_index=True).to_csv(
                shard / "raw_experiment4_attention_shift.csv", index=False
            )

            report = repair_shard(shard, root / "backup", expected_rows_per_key=2)
            repaired_sample = pd.read_csv(shard / "sample_level_experiment4.csv")
            repaired_raw = pd.read_csv(shard / "raw_experiment4_attention_shift.csv")
            self.assertEqual(len(repaired_sample), 2)
            self.assertEqual(repaired_sample.loc[repaired_sample["sample_id"] == 10, "score"].item(), 2.0)
            self.assertEqual(repaired_raw[repaired_raw["sample_id"] == 10]["value"].tolist(), [10, 11])
            self.assertNotIn(30, repaired_raw["sample_id"].tolist())
            self.assertEqual(report["duplicate_sample_rows_removed"], 1)
            self.assertEqual(report["duplicate_raw_rows_removed"], 2)
            self.assertEqual(report["orphan_raw_rows_removed"], 2)

    def test_head_stability_normalizes_head_column_without_attribute_collision(self):
        from reviewer_followup.analyze_head_stability import normalize_effects

        frame = pd.DataFrame(
            [
                {
                    "metric": "top10_shift_mean",
                    "layer": 15,
                    "head": 7,
                    "cliffs_delta_pos_minus_neg": 0.25,
                    "significant_fdr_0.05": True,
                }
            ]
        )
        normalized = normalize_effects(frame, seed=42)
        self.assertEqual(normalized.loc[0, "key"], "top10_shift_mean|L15|H7")
        self.assertTrue(bool(normalized.loc[0, "significant"]))

    def test_resume_sanitizes_protocol_raw_and_orphan_update_rows(self):
        from extract_attention_hardsplit import load_existing_progress

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame(
                [{"condition": "dynamic_attention", "sample_id": 0, "group": "g"}]
            ).to_csv(root / "sample_level_experiment4.csv", index=False)
            pd.DataFrame(
                [
                    {"condition": "dynamic_attention__q0", "sample_id": 0, "head": 0},
                    {"condition": "dynamic_attention__q0", "sample_id": 1, "head": 0},
                ]
            ).to_csv(root / "raw_experiment4_attention_shift.csv", index=False)
            pd.DataFrame(
                [
                    {"condition": "dynamic_attention", "sample_id": 0, "feature": "gradient"},
                    {"condition": "dynamic_attention", "sample_id": 1, "feature": "gradient"},
                ]
            ).to_csv(root / "raw_update_baseline_features.csv", index=False)

            raw_count, sample_count, done = load_existing_progress(root)
            self.assertEqual((raw_count, sample_count), (1, 1))
            self.assertEqual(done, {("dynamic_attention", 0)})
            self.assertEqual(pd.read_csv(root / "raw_experiment4_attention_shift.csv")["sample_id"].tolist(), [0])
            self.assertEqual(pd.read_csv(root / "raw_update_baseline_features.csv")["sample_id"].tolist(), [0])

    def test_generic_target_sampling_preserves_group_on_new_pandas(self):
        from extract_attention_hardsplit import load_all_samples

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "targets.csv"
            pd.DataFrame(
                [{"text": f"{group}-{index}", "group": group, "ft_exposed": index % 2} for group in ("a", "b") for index in range(4)]
            ).to_csv(path, index=False)
            args = argparse.Namespace(targets_csv=str(path), target_groups=[], n_per_group=2, seed=42)
            sampled = load_all_samples(args)
            self.assertEqual(sampled["group"].value_counts().to_dict(), {"a": 2, "b": 2})

    def _write_pool(self, path: Path, prefix: str, n: int) -> None:
        pd.DataFrame({"text": [f"{prefix} document {i}" for i in range(n)]}).to_csv(path, index=False)

    def test_both_factorial_build_modes_are_balanced_and_disjoint(self):
        from reviewer_followup.build_factorial_dataset import build_controlled_exposure, build_mimir_membership
        from reviewer_followup.common import validate_factorial_targets

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            member, first, second = root / "member.csv", root / "first.csv", root / "second.csv"
            self._write_pool(member, "member", 8)
            self._write_pool(first, "nonmember-a", 8)
            self._write_pool(second, "nonmember-b", 8)
            base = dict(
                member_csv=str(member),
                nonmember_csv=str(first),
                nonmember_extra_csv=str(second),
                n_per_cell=2,
                seed=42,
            )
            crossed = build_mimir_membership(argparse.Namespace(**base))
            controlled = build_controlled_exposure(argparse.Namespace(**base))
            for frame in (crossed, controlled):
                frame["sample_id"] = [f"s{i}" for i in range(len(frame))]
                result = validate_factorial_targets(frame)
                self.assertEqual(result["group_counts"], {"p0f0": 2, "p0f1": 2, "p1f0": 2, "p1f1": 2})
                self.assertEqual(frame["text_sha256"].nunique(), 8)

    def test_factorial_ols_recovers_interaction(self):
        from reviewer_followup.evaluate_crossed_2x2 import factorial_ols

        rows = []
        rng = np.random.default_rng(7)
        for pt in (0, 1):
            for ft in (0, 1):
                for _ in range(30):
                    rows.append({"pt_member": pt, "ft_exposed": ft, "attn_x": pt + 2 * ft + 3 * pt * ft + rng.normal(0, 0.05)})
        result = factorial_ols(pd.DataFrame(rows), ["attn_x"])
        interaction = result[result["term"] == "interaction"].iloc[0]
        self.assertAlmostEqual(float(interaction["estimate"]), 3.0, delta=0.1)
        self.assertLess(float(interaction["p"]), 1e-10)


class TinyLM:
    @staticmethod
    def build():
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(11, 5)
                self.projection = torch.nn.Linear(5, 11)

            def forward(self, input_ids, attention_mask=None, labels=None, **_kwargs):
                import torch.nn.functional as functional

                logits = self.projection(self.embedding(input_ids))
                targets = input_ids if labels is None else labels
                loss = functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
                return SimpleNamespace(logits=logits, loss=loss)

        return Model()


class TestGradientAndUpdateBaselines(unittest.TestCase):
    def test_gradient_logit_token_scores(self):
        import torch
        from mimir_hardsplit_attention_common import compute_token_logit_gradient_norms

        model = TinyLM.build()
        batch = {"input_ids": torch.tensor([[1, 2, 3, 4]]), "attention_mask": torch.ones((1, 4), dtype=torch.long)}
        norms, mask = compute_token_logit_gradient_norms(model, batch)
        self.assertEqual(tuple(norms.shape), (3,))
        self.assertEqual(mask.tolist(), [1, 1, 1])
        self.assertTrue(torch.isfinite(norms).all())
        self.assertTrue((norms >= 0).all())

    def test_gradient_curve_and_parameter_delta_are_recorded(self):
        import torch
        from reviewer_followup.update_features import (
            initial_gradient_features,
            overfit_fixed_steps_with_gradient_curve,
            parameter_delta_features,
        )

        model = TinyLM.build()
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "labels": torch.tensor([[1, 2, 3, 4]]),
        }
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        move = lambda values, device: {key: value.to(device) for key, value in values.items()}
        device = lambda current: next(current.parameters()).device
        gradient_rows, gradients = initial_gradient_features(model, batch, model_device_fn=device, move_batch_fn=move)
        losses, _, steps, curve = overfit_fixed_steps_with_gradient_curve(
            model,
            batch,
            steps=2,
            lr=1e-2,
            model_device_fn=device,
            move_batch_fn=move,
            use_amp=False,
        )
        delta_rows = parameter_delta_features(model, before, gradients)
        self.assertGreaterEqual(len(gradient_rows), 1)
        self.assertEqual((len(losses), len(curve), steps), (2, 2, 2))
        self.assertGreater(next(row for row in delta_rows if row["scope"] == "global")["delta_l2"], 0.0)


class TestEvaluationProtocols(unittest.TestCase):
    def test_multi_query_protocol_parser_validates_and_preserves_settings(self):
        from extract_attention_hardsplit import make_query_protocols_from_args

        args = argparse.Namespace(
            query_protocol=[
                ["q0_top_r5", "0", "top_loss", "5"],
                ["q1_grad_r20", "1", "gradient_logit", "20"],
            ]
        )
        self.assertEqual(
            make_query_protocols_from_args(args),
            [("q0_top_r5", 0, "top_loss", 5), ("q1_grad_r20", 1, "gradient_logit", 20)],
        )

    def _attention_raw(self, groups: list[str], n: int = 8) -> pd.DataFrame:
        rows = []
        rng = np.random.default_rng(3)
        for group_index, group in enumerate(groups):
            for index in range(n):
                rows.append(
                    {
                        "condition": "fixed_attention_20",
                        "sample_id": f"{group}::{index}",
                        "group": group,
                        "layer": 0,
                        "head": 0,
                        "l1_mean": group_index + rng.normal(0, 0.1),
                        "l2_rms": group_index * 0.5 + rng.normal(0, 0.1),
                    }
                )
        return pd.DataFrame(rows)

    def test_repeated_cv_saves_fold_level_feature_identity(self):
        from reviewer_followup.evaluation import evaluate_feature_sets, wide_attention

        wide = wide_attention(self._attention_raw(["pos", "neg"]))
        features = [column for column in wide if column.startswith("attn_")]
        repeats, predictions, selections = evaluate_feature_sets(
            wide,
            {"proposed_en": features},
            {"comparison": ("pos", "neg")},
            repeats=2,
            cv_splits=2,
            seed=4,
        )
        self.assertEqual(len(repeats), 2)
        self.assertEqual(predictions["repeat"].nunique(), 2)
        self.assertIn("feature", selections.columns)

    def test_nested_protocol_uses_inner_cv_only(self):
        from reviewer_followup.evaluation import wide_attention
        from reviewer_followup.run_nested_protocol_selection import nested_select

        strong = wide_attention(self._attention_raw(["pos", "neg"], n=10)).set_index("sample_id")
        weak = strong.copy()
        feature_columns = [column for column in weak if column.startswith("attn_")]
        rng = np.random.default_rng(9)
        weak.loc[:, feature_columns] = rng.normal(size=(len(weak), len(feature_columns)))
        predictions, selections, repeats = nested_select(
            {"strong": strong, "weak": weak},
            positive="pos",
            negative="neg",
            repeats=1,
            outer_splits=2,
            inner_splits=2,
            seed=11,
            n_jobs=1,
        )
        self.assertEqual(len(predictions), 20)
        self.assertEqual(len(selections), 2)
        self.assertEqual(len(repeats), 1)
        self.assertTrue(set(selections["selected_protocol"]).issubset({"strong", "weak"}))


class TestController(unittest.TestCase):
    def test_downstream_status_requires_completed_marker(self):
        from reviewer_followup.watch_downstream import status_is_complete

        with tempfile.TemporaryDirectory() as tmp:
            status = Path(tmp) / "run_status.txt"
            self.assertFalse(status_is_complete(status))
            status.write_text("started\n", encoding="utf-8")
            self.assertFalse(status_is_complete(status))
            status.write_text("extraction_completed\n", encoding="utf-8")
            self.assertTrue(status_is_complete(status))

    def test_partial_extraction_is_not_treated_as_complete(self):
        from reviewer_followup.controller import Command, command_is_complete
        import json

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "features" / "raw_experiment4_attention_shift.csv"
            raw.parent.mkdir(parents=True)
            raw.write_text("sample_id,value\n1,0.5\n", encoding="utf-8")
            command = Command("extract_test", ["false"], [str(raw)], gpu=True)
            self.assertFalse(command_is_complete(command, root))
            (raw.parent / "run_status.txt").write_text("extraction_completed_skip_analyze\n", encoding="utf-8")
            self.assertFalse(command_is_complete(command, root))
            marker = root / ".controller_done" / "extract_test.json"
            marker.parent.mkdir()
            marker.write_text(json.dumps({"name": "extract_test", "outputs": [str(raw)]}), encoding="utf-8")
            self.assertTrue(command_is_complete(command, root))

    def test_strict_completion_rejects_invalid_marker_and_header_only_csv(self):
        import json
        from reviewer_followup.controller import Command, command_completion_report

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "summary.csv"
            output.write_text("metric,value\n", encoding="utf-8")
            marker = root / ".controller_done" / "evaluate_test.json"
            marker.parent.mkdir()
            marker.write_text(json.dumps({"name": "wrong", "outputs": [str(output)]}), encoding="utf-8")
            report = command_completion_report(Command("evaluate_test", ["false"], [str(output)]), root)
            self.assertFalse(report["completed"])
            self.assertTrue(any("csv_has_no_data_rows" in reason for reason in report["reasons"]))
            self.assertIn("controller_marker:name_mismatch", report["reasons"])

    def test_reconcile_creates_marker_only_for_supported_verified_legacy_output(self):
        import json
        from reviewer_followup.controller import Command, command_is_complete, reconcile_existing_commands

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "e7" / "data"
            data_dir.mkdir(parents=True)
            output = data_dir / "factorial_targets.csv"
            output.write_text("sample_id,group\n1,p0f0\n", encoding="utf-8")
            (data_dir / "factorial_manifest.json").write_text(
                json.dumps({"status": "completed", "validation": {"group_counts": {"p0f0": 1}}}),
                encoding="utf-8",
            )
            command = Command("build_test", ["false"], [str(output)])
            reconciled, skipped = reconcile_existing_commands({"prepare": [command]}, root)
            self.assertEqual(reconciled, 1)
            self.assertEqual(skipped, [])
            self.assertTrue(command_is_complete(command, root))

    def test_final_audit_writes_incomplete_evidence(self):
        from reviewer_followup.audit_results import main
        from reviewer_followup.controller import main as controller_main

        with tempfile.TemporaryDirectory() as tmp:
            controller_main(["plan", "--output-root", tmp])
            main(["--output-root", tmp, "--allow-incomplete"])
            payload = json.loads((Path(tmp) / "final_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "incomplete")
            self.assertTrue((Path(tmp) / "final_audit_summary.md").exists())

    def test_base_checkpoint_archive_records_pretraining_state(self):
        from types import SimpleNamespace
        from reviewer_followup.train_controlled_pretraining import archive_base_checkpoint

        class FakeModel:
            config = SimpleNamespace(_commit_hash="abc123")

            def save_pretrained(self, path):
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "config.json").write_text('{"model_type":"gpt_neo"}\n', encoding="utf-8")
                (Path(path) / "model.safetensors").write_bytes(b"weights")

        class FakeTokenizer:
            def save_pretrained(self, path):
                (Path(path) / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            payload = archive_base_checkpoint(FakeModel(), FakeTokenizer(), Path(tmp), "example/model")
            self.assertEqual(payload["resolved_model_commit"], "abc123")
            self.assertEqual(payload["status"], "completed")
            self.assertTrue((Path(tmp) / "base_model_before_controlled_pt" / "config.json").exists())
            self.assertTrue((Path(tmp) / "base_checkpoint_manifest.json").exists())

    def test_external_gpu_snapshot_is_freshness_checked_and_compacted(self):
        import datetime as dt
        import json
        from unittest.mock import patch
        from reviewer_followup.controller import snapshot_gpu

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            source.write_text(
                json.dumps(
                    {
                        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "servers": [
                            {
                                "id": "host",
                                "status": "ok",
                                "age_seconds": 1,
                                "gpus": [{"gpu_index": 0, "history": [{"unused": True}], "memory_free_mib": 24000}],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"GPU_STATUS_SNAPSHOT": str(source)}):
                snapshot_gpu(root / "out")
            saved = next((root / "out" / "provenance").glob("gpu_status_*.json"))
            payload = json.loads(saved.read_text(encoding="utf-8"))
            self.assertNotIn("history", payload["servers"][0]["gpus"][0])

    def test_status_does_not_rewrite_the_frozen_plan(self):
        from reviewer_followup.controller import main

        with tempfile.TemporaryDirectory() as tmp:
            main(["plan", "--output-root", tmp])
            plan = Path(tmp) / "experiment_plan.json"
            before = plan.read_bytes()
            main(["status", "--output-root", tmp])
            self.assertEqual(before, plan.read_bytes())

    def test_multiseed_aggregation_names_the_uncertainty_axis(self):
        from reviewer_followup.aggregate_multiseed import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = []
            for seed, auc in ((4242, 0.61), (4343, 0.65)):
                path = root / f"summary_{seed}.csv"
                pd.DataFrame(
                    [{"comparison": "ft_vs_pt", "method": "proposed_en", "auc_mean": auc, "auc_std": 0.01}]
                ).to_csv(path, index=False)
                inputs.extend(["--result", f"{seed}={path}"])
            main([*inputs, "--output-dir", str(root / "out"), "--seed-axis", "sample"])
            result = pd.read_csv(root / "out" / "sample_seed_results.csv")
            self.assertIn("sample_seed", result.columns)
            self.assertNotIn("ft_seed", result.columns)
            summary = pd.read_csv(root / "out" / "sample_seed_summary.csv")
            self.assertIn("auc_sample_seed_bootstrap_ci_low", summary.columns)

    def test_plan_covers_all_experiments_and_keeps_outputs_isolated(self):
        from reviewer_followup.controller import build_plan, parse_args

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            stages = build_plan(HERE, root, "EleutherAI/gpt-neo-125m", 4242)
            self.assertEqual(set(stages), {"prepare", "e7", "e8", "e9", "e10", "e11", "e12"})
            self.assertEqual(sum(command.gpu for command in stages["e12"]), 10)
            self.assertEqual(len(stages["e12"]), 13)
            self.assertEqual(len(stages["e9"]), 26)
            all_outputs = [path for commands in stages.values() for command in commands for path in command.expected_outputs]
            self.assertTrue(all(str(path).startswith(str(root)) for path in all_outputs))
            nested = [command for command in stages["e12"] if command.name.startswith("nested_select_")]
            self.assertEqual(len(nested), 2)
            self.assertTrue(all(command.argv.count("--candidate") == 80 for command in nested))
            extracts = [command for command in stages["e12"] if command.gpu]
            self.assertTrue(all(command.argv.count("--query-protocol") == 20 for command in extracts))
            merge = next(command for command in stages["e12"] if command.name.startswith("merge_"))
            self.assertIn("80", merge.argv)
            parsed = parse_args(
                ["run-sequence", "--command", "train_ft_seed_123", "--command", "extract_ft_seed_123"]
            )
            self.assertEqual(parsed.command, ["train_ft_seed_123", "extract_ft_seed_123"])

    def test_extraction_shard_merge_validates_coverage_and_counts(self):
        from reviewer_followup.merge_extraction_shards import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = []
            for shard_index, ids in enumerate(((0, 2), (1, 3))):
                shard = root / f"shard_{shard_index}"
                shard.mkdir()
                shards.append(shard)
                (shard / "run_status.txt").write_text("extraction_completed\n", encoding="utf-8")
                pd.DataFrame(
                    [{"global_sample_id": sample_id, "text": f"sample-{sample_id}"} for sample_id in ids]
                ).to_csv(shard / "experiment4_target_samples.csv", index=False)
                pd.DataFrame(
                    [
                        {"condition": condition, "sample_id": sample_id, "metric": float(sample_id)}
                        for sample_id in ids
                        for condition in ("c0", "c1")
                    ]
                ).to_csv(shard / "sample_level_experiment4.csv", index=False)
                pd.DataFrame(
                    [
                        {"condition": condition, "sample_id": sample_id, "head": 0, "value": float(sample_id)}
                        for sample_id in ids
                        for condition in ("q0_c0", "q0_c1", "q1_c0", "q1_c1")
                    ]
                ).to_csv(shard / "raw_experiment4_attention_shift.csv", index=False)

            output = root / "merged"
            argv = []
            for shard in shards:
                argv.extend(["--shard-dir", str(shard)])
            main([*argv, "--output-dir", str(output), "--expected-targets", "4", "--expected-conditions", "4"])
            targets = pd.read_csv(output / "experiment4_target_samples.csv")
            samples = pd.read_csv(output / "sample_level_experiment4.csv")
            self.assertEqual(set(targets["global_sample_id"]), {0, 1, 2, 3})
            self.assertEqual(len(samples), 8)
            manifest = json.loads((output / "shard_merge_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["expected_conditions"], 4)
            self.assertEqual(manifest["sample_condition_count"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
