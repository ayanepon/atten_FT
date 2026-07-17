# -*- coding: utf-8 -*-
"""Offline regression tests for the additional experiment implementations."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


class TestStrictPTUnseen(unittest.TestCase):
    def test_comparison_and_label_are_registered(self):
        import run_strict_fixed20_comparison_10runs as strict

        self.assertEqual(
            strict.COMPARISONS["pt_vs_unseen"],
            (strict.GROUP_PT, strict.GROUP_UNSEEN),
        )
        self.assertEqual(strict.COMPARISON_LABELS["pt_vs_unseen"], "PT--Unseen")

    def test_latex_table_emits_pt_unseen_row(self):
        import run_strict_fixed20_comparison_10runs as strict

        frame = pd.DataFrame(
            [
                {
                    "model": "pythia1b",
                    "comparison": "pt_vs_unseen",
                    "method": "proposed_en",
                    "auc_mean": 0.51,
                    "tpr_at_10_fpr_mean": 0.10,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "table.tex"
            strict.latex_table(frame, path)
            self.assertIn("PT--Unseen", path.read_text(encoding="utf-8"))


class TestQuerySelection(unittest.TestCase):
    def test_default_wrapper_matches_paper_condition(self):
        import torch
        import mimir_hardsplit_attention_common as common

        losses = torch.tensor([1.0, 4.0, 2.0, 3.0])
        mask = torch.ones(4, dtype=torch.long)
        self.assertEqual(
            common.select_topk_query_positions(losses, mask, 50),
            common.select_query_positions(
                losses, mask, 50, query_position_offset=1, selection_mode="top_loss"
            ),
        )

    def test_controls_are_deterministic_and_offset_changes_positions(self):
        import torch
        import mimir_hardsplit_attention_common as common

        losses = torch.tensor([1.0, 4.0, 2.0, 3.0])
        mask = torch.ones(4, dtype=torch.long)
        random_a = common.select_query_positions(
            losses, mask, 50, query_position_offset=0, selection_mode="random", random_seed=7
        )
        random_b = common.select_query_positions(
            losses, mask, 50, query_position_offset=0, selection_mode="random", random_seed=7
        )
        self.assertEqual(random_a, random_b)
        self.assertEqual(
            common.select_query_positions(losses, mask, 50, query_position_offset=0),
            [1, 3],
        )
        self.assertEqual(
            common.select_query_positions(losses, mask, 50, query_position_offset=1),
            [2, 4],
        )
        self.assertEqual(
            common.select_query_positions(losses, mask, 50, selection_mode="all_valid"),
            [1, 2, 3, 4],
        )


class TestDataDiagnostics(unittest.TestCase):
    def test_exact_duplicate_and_length_diagnostics(self):
        from run_data_confound_diagnostics import exact_duplicate_diagnostics, length_summary

        frame = pd.DataFrame(
            [
                {"group_key": "ft", "group": "ft", "text": "Same text"},
                {"group_key": "pt", "group": "pt", "text": " same   text "},
                {"group_key": "unseen", "group": "unseen", "text": "Different"},
            ]
        )
        duplicate = exact_duplicate_diagnostics(frame)
        ft_pt = duplicate[duplicate["comparison"] == "ft_vs_pt"].iloc[0]
        self.assertEqual(int(ft_pt["cross_group_exact_duplicate_hashes"]), 1)
        summary = length_summary(frame)
        self.assertEqual(set(summary["group_key"]), {"ft", "pt", "unseen"})

    def test_targets_csv_uses_exact_evaluation_rows(self):
        from run_data_confound_diagnostics import GROUP_NAMES, load_groups

        rows = []
        for key, group in GROUP_NAMES.items():
            rows.append({"group": group, "text": f"{key} target", "original_index": 7})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "targets.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            loaded = load_groups(Path(tmp), n_per_group=1, seed=42, targets_csv=path)
        self.assertEqual(set(loaded["group_key"]), {"ft", "pt", "unseen"})
        self.assertEqual(set(loaded["original_index"]), {7})


class TestPairedRobustness(unittest.TestCase):
    def test_bootstrap_and_holm_are_finite(self):
        from run_paired_robustness import analyze_auc_table, paired_bootstrap_ci

        ci = paired_bootstrap_ci(np.array([0.1, 0.2, 0.3]), n_bootstrap=100, seed=1)
        self.assertLessEqual(ci["ci_low"], ci["mean_diff"])
        self.assertLessEqual(ci["mean_diff"], ci["ci_high"])
        auc = pd.DataFrame(
            [
                {"model": "m", "comparison": "c", "method": "proposed_en", "repeat": i, "auc": 0.7 + i / 100}
                for i in range(5)
            ]
            + [
                {"model": "m", "comparison": "c", "method": "baseline", "repeat": i, "auc": 0.6 + i / 100}
                for i in range(5)
            ]
        )
        result = analyze_auc_table(auc, n_bootstrap=100, seed=1)
        self.assertEqual(len(result), 1)
        self.assertTrue(np.isfinite(result.iloc[0]["holm_p"]))


class TestNestedSelection(unittest.TestCase):
    def test_fit_predict_en_on_synthetic_features(self):
        from run_nested_step_selection import fit_predict_en

        rows = []
        for group, offset in [("pos", 1.0), ("neg", -1.0)]:
            for i in range(10):
                rows.append(
                    {
                        "sample_id": i,
                        "group": group,
                        "attn_l0_h0_l1_mean": offset + i * 0.01,
                        "attn_l0_h0_l2_rms": offset - i * 0.01,
                    }
                )
        frame = pd.DataFrame(rows)
        train = [f"{g}::{i}" for g in ["pos", "neg"] for i in range(0, 8)]
        test = [f"{g}::{i}" for g in ["pos", "neg"] for i in range(8, 10)]
        y, scores, selected = fit_predict_en(
            frame,
            "pos",
            "neg",
            train,
            test,
            selection_c=0.1,
            classifier_c=1.0,
            l1_ratio=0.7,
            max_iter=100,
            tol=5e-4,
            seed=42,
        )
        self.assertEqual(len(y), 4)
        self.assertEqual(len(scores), 4)
        self.assertGreaterEqual(selected, 1)


class TestCrossfitFusion(unittest.TestCase):
    def test_inner_crossfit_covers_each_training_row(self):
        from run_crossfit_fusion_en_lora_leak import crossfit_base_scores

        rng = np.random.default_rng(7)
        y = np.array([0, 1] * 10)
        x = rng.normal(size=(len(y), 4)) + y[:, None] * 0.4
        lora = rng.normal(size=len(y)) + y * 0.2
        args = type(
            "Args",
            (),
            {
                "inner_splits": 5,
                "elasticnet_l1_ratio": 0.7,
                "selection_c": 0.1,
                "elasticnet_tol": 5e-4,
                "elasticnet_max_iter": 100,
                "classifier_c": 1.0,
            },
        )()
        en, ll = crossfit_base_scores(x, y, lora, args=args, seed=42)
        self.assertEqual(en.shape, y.shape)
        self.assertTrue(np.isfinite(en).all())
        self.assertTrue(np.isfinite(ll).all())


class TestDistributedOrchestrator(unittest.TestCase):
    def test_gpu_api_selection_rejects_busy_and_stale_entries(self):
        from run_distributed_paper_experiments import parse_gpu_api_payload, select_gpus
        from unittest.mock import patch

        payload = {
            "servers": [
                {
                    "id": "hosta",
                    "status": "ok",
                    "age_seconds": 2,
                    "gpus": [
                        {"gpu_index": 0, "memory_free_mib": 97000, "utilization_percent": 100, "processes": []},
                        {"gpu_index": 1, "memory_free_mib": 97000, "utilization_percent": 0, "processes": []},
                        {"gpu_index": 2, "memory_free_mib": 97000, "utilization_percent": 0, "processes": [{"pid": 123}]},
                    ],
                }
            ]
        }
        infos = parse_gpu_api_payload(payload, "hosta")
        self.assertEqual([g.index for g in infos], [0, 1, 2])
        self.assertEqual([g.index for g in infos if g.process_count == 0], [0, 1])
        with patch("run_distributed_paper_experiments.gpu_snapshot", return_value=infos):
            self.assertEqual(
                select_gpus("hosta", 16.0, 3, max_util_percent=80),
                [1],
            )
        with self.assertRaisesRegex(RuntimeError, "stale"):
            parse_gpu_api_payload({"servers": [{**payload["servers"][0], "age_seconds": 999}]}, "hosta")


if __name__ == "__main__":
    unittest.main(verbosity=2)
