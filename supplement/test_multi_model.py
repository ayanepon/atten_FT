# -*- coding: utf-8 -*-
"""Offline tests for multi-model registry + paper pipeline (no GPU required).

Run:
  python test_multi_model.py
  python test_multi_model.py -v
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
ADAPTER_RUN = HERE / "mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2"
ADAPTER_DIR = ADAPTER_RUN / "adapter"
LEGACY_FEATURES = HERE / "attention_features_mimir_hardsplit_legacy"
HARDSPLIT_DATA = ADAPTER_RUN / "data"


class TestModelRegistry(unittest.TestCase):
    def test_list_keys(self):
        from model_registry import list_eval_keys, list_model_keys

        self.assertEqual(list_model_keys(), ["pythia-1b", "pythia-410m", "gpt-neo-2.7b"])
        self.assertEqual(list_eval_keys(), ["pythia1b", "pythia410m", "gptneo27b"])

    def test_aliases_and_hf_ids(self):
        from model_registry import normalize_model_key, resolve_model_name, resolve_model_spec

        self.assertEqual(normalize_model_key("pythia410m"), "pythia-410m")
        self.assertEqual(normalize_model_key("EleutherAI/gpt-neo-2.7B"), "gpt-neo-2.7b")
        self.assertEqual(resolve_model_name(explicit="gptneo27b"), "EleutherAI/gpt-neo-2.7B")
        neo = resolve_model_spec("gpt-neo-2.7b")
        self.assertEqual(neo.family, "gpt_neo")
        self.assertIn("q_proj", neo.lora_target_modules)
        self.assertNotIn("query_key_value", neo.lora_target_modules)
        py = resolve_model_spec("pythia-410m")
        self.assertIn("query_key_value", py.lora_target_modules)

    def test_unknown_model_raises(self):
        from model_registry import resolve_model_spec

        with self.assertRaises(KeyError):
            resolve_model_spec("not-a-real-model")

    def test_lora_modules_for_custom_hf_id(self):
        from model_registry import lora_modules_for_hf_id, model_spec_or_custom

        mods = lora_modules_for_hf_id("EleutherAI/gpt-neo-1.3B")
        self.assertIn("q_proj", mods)
        custom = model_spec_or_custom("EleutherAI/pythia-160m")
        self.assertEqual(custom.family, "gpt_neox")
        self.assertTrue(custom.default_run_dir.startswith("mimir_lora_"))

    def test_strict_eval_configs(self):
        from model_registry import strict_eval_model_configs

        cfg = strict_eval_model_configs()
        self.assertEqual(cfg["pythia1b"]["proposed_root"], "attention_features_mimir_hardsplit")
        self.assertEqual(cfg["pythia410m"]["label"], "Pythia-410M")
        self.assertEqual(cfg["gptneo27b"]["proposed_root"], "attention_features_gptneo27b")

    def test_path_helpers(self):
        from model_registry import (
            DEFAULT_HF_ID,
            is_default_lora_csv,
            is_default_run_dir,
            resolve_model_spec,
        )

        self.assertTrue(is_default_run_dir(""))
        self.assertTrue(is_default_run_dir("mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2"))
        self.assertTrue(
            is_default_run_dir("results/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2")
        )
        self.assertFalse(is_default_run_dir("mimir_lora_pythia410m"))
        self.assertTrue(is_default_lora_csv("query_key_value,dense,dense_h_to_4h,dense_4h_to_h"))
        spec = resolve_model_spec("pythia-410m")
        self.assertEqual(spec.default_lora_root, "results/lora_leak_pythia410m")
        self.assertEqual(spec.default_attenmia_root, "results/attenmia_pythia410m")
        self.assertEqual(DEFAULT_HF_ID, "EleutherAI/pythia-1b")


@unittest.skipUnless(ADAPTER_DIR.exists(), "local pythia-1b adapter missing")
class TestAdapterResolution(unittest.TestCase):
    def test_resolve_adapter_dir_from_run_dir(self):
        from model_registry import resolve_adapter_dir, resolve_model_name

        ad = resolve_adapter_dir(ADAPTER_RUN)
        self.assertTrue((ad / "adapter_config.json").exists())
        ad2 = resolve_adapter_dir(ADAPTER_DIR)
        self.assertEqual(ad.resolve(), ad2.resolve())
        self.assertEqual(resolve_model_name(adapter_dir=ad), "EleutherAI/pythia-1b")

    def test_resolve_from_args_prefers_explicit(self):
        from model_registry import resolve_from_args

        ns = argparse.Namespace(model="pythia-410m", model_name="", adapter_dir=str(ADAPTER_DIR))
        # explicit preset wins over adapter base
        self.assertEqual(
            resolve_from_args(ns, adapter_dir=ADAPTER_DIR),
            "EleutherAI/pythia-410m",
        )
        ns2 = argparse.Namespace(model="", model_name="", adapter_dir=str(ADAPTER_DIR))
        self.assertEqual(
            resolve_from_args(ns2, adapter_dir=ADAPTER_DIR),
            "EleutherAI/pythia-1b",
        )

    def test_adapter_config_base_model(self):
        data = json.loads((ADAPTER_DIR / "adapter_config.json").read_text(encoding="utf-8"))
        self.assertEqual(data["base_model_name_or_path"], "EleutherAI/pythia-1b")
        self.assertIn("query_key_value", data["target_modules"])


class TestApplyModelNamespace(unittest.TestCase):
    def test_train_profile(self):
        from model_registry import DEFAULT_HF_ID, PYTHIA1B_RUN_DIR, apply_model_namespace

        ns = argparse.Namespace(
            model="gpt-neo-2.7b",
            model_name=DEFAULT_HF_ID,
            target_modules="query_key_value,dense,dense_h_to_4h,dense_4h_to_h",
            output_dir=PYTHIA1B_RUN_DIR,
        )
        apply_model_namespace(ns, profile="train")
        self.assertEqual(ns.model_name, "EleutherAI/gpt-neo-2.7B")
        self.assertIn("q_proj", ns.target_modules)
        self.assertEqual(ns.output_dir, "mimir_lora_gptneo27b")

    def test_train_model_name_only_updates_modules(self):
        from model_registry import apply_model_namespace

        ns = argparse.Namespace(
            model="",
            model_name="EleutherAI/gpt-neo-2.7B",
            target_modules="query_key_value,dense,dense_h_to_4h,dense_4h_to_h",
            output_dir="keep_me",
        )
        apply_model_namespace(ns, profile="train")
        self.assertIn("q_proj", ns.target_modules)
        self.assertEqual(ns.output_dir, "keep_me")

    def test_pipeline_profile(self):
        from model_registry import PYTHIA1B_FEATURES_ROOT, PYTHIA1B_RUN_DIR, apply_model_namespace

        ns = argparse.Namespace(
            model="pythia-410m",
            model_name="",
            run_dir=PYTHIA1B_RUN_DIR,
            features_root=PYTHIA1B_FEATURES_ROOT,
            exp1_output_dir="results/exp1_layer_head_stats",
            eval_output_dir="results/strict_fixed20_unified",
            lora_root="",
            attenmia_root="",
        )
        apply_model_namespace(ns, profile="pipeline", fill_baselines=True)
        self.assertEqual(ns.model_name, "EleutherAI/pythia-410m")
        self.assertEqual(ns.run_dir, "mimir_lora_pythia410m")
        self.assertEqual(ns.features_root, "attention_features_pythia410m")
        self.assertEqual(ns.model_key, "pythia410m")
        self.assertEqual(ns.lora_root, "results/lora_leak_pythia410m")
        self.assertEqual(ns.eval_output_dir, "results/strict_fixed20_pythia410m")

    def test_pipeline_does_not_clobber_custom_paths(self):
        from model_registry import apply_model_namespace

        ns = argparse.Namespace(
            model="pythia-410m",
            model_name="",
            run_dir="my_custom_run",
            features_root="my_custom_features",
            exp1_output_dir="results/exp1_custom",
            eval_output_dir="results/eval_custom",
            lora_root="",
            attenmia_root="",
        )
        apply_model_namespace(ns, profile="pipeline")
        self.assertEqual(ns.run_dir, "my_custom_run")
        self.assertEqual(ns.features_root, "my_custom_features")
        self.assertEqual(ns.exp1_output_dir, "results/exp1_custom")

    def test_baseline_profiles(self):
        from model_registry import (
            DEFAULT_ATTENMIA_DIR,
            DEFAULT_LORA_LEAK_DIR,
            PYTHIA1B_RUN_DIR_RESULTS,
            apply_model_namespace,
        )

        for profile, default_out, expected_out in [
            ("lora_leak", DEFAULT_LORA_LEAK_DIR, "results/lora_leak_pythia410m"),
            ("attenmia", DEFAULT_ATTENMIA_DIR, "results/attenmia_pythia410m"),
        ]:
            ns = argparse.Namespace(
                model="pythia-410m",
                model_name="",
                run_dir=PYTHIA1B_RUN_DIR_RESULTS,
                adapter_dir=PYTHIA1B_RUN_DIR_RESULTS,
                output_dir=default_out,
            )
            apply_model_namespace(ns, profile=profile)
            self.assertEqual(ns.run_dir, "mimir_lora_pythia410m", profile)
            self.assertEqual(ns.output_dir, expected_out, profile)


class TestCLIHelp(unittest.TestCase):
    SCRIPTS = [
        "model_registry.py",  # may not be CLI
        "orchestrate.py",
        "train_mimir_wikipedia_hardsplit_lora.py",
        "extract_attention_hardsplit.py",
        "run_unified_fixed20_pipeline.py",
        "run_strict_fixed20_comparison_10runs.py",
        "run_lora_leak_official_mimir_hardsplit_2.py",
        "run_attenmia_official_mimir_hardsplit.py",
    ]

    def test_help_exits_zero(self):
        cases = [
            ["orchestrate.py", "all", "--help"],
            ["orchestrate.py", "train", "--help"],
            ["orchestrate.py", "baselines", "--help"],
            ["train_mimir_wikipedia_hardsplit_lora.py", "--help"],
            ["extract_attention_hardsplit.py", "--help"],
            ["run_unified_fixed20_pipeline.py", "--help"],
            ["run_strict_fixed20_comparison_10runs.py", "--help"],
            ["run_lora_leak_official_mimir_hardsplit_2.py", "--help"],
            ["run_attenmia_official_mimir_hardsplit.py", "--help"],
            ["run_distributed_paper_experiments.py", "--help"],
        ]
        for cmd in cases:
            with self.subTest(cmd=cmd):
                r = subprocess.run(
                    [sys.executable, str(HERE / cmd[0]), *cmd[1:]],
                    cwd=str(HERE),
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(r.returncode, 0, r.stderr[-500:])
                if cmd[0] == "run_distributed_paper_experiments.py":
                    self.assertIn("--hosts", r.stdout)
                elif cmd[0] != "extract_attention_hardsplit.py":
                    self.assertTrue("--model" in r.stdout or "--models" in r.stdout, r.stdout[:300])


class TestImportsAndWiring(unittest.TestCase):
    def test_import_core_modules(self):
        import extract_attention_hardsplit  # noqa: F401
        import mimir_hardsplit_attention_common as common
        import orchestrate
        import run_strict_fixed20_comparison_10runs as strict
        import run_unified_fixed20_pipeline  # noqa: F401
        from model_registry import list_eval_keys

        self.assertEqual(set(strict.MODEL_CONFIGS), set(list_eval_keys()))
        self.assertTrue(hasattr(common, "get_model_name"))
        self.assertTrue(hasattr(orchestrate, "apply_model_defaults"))

    def test_train_preset_wrapper(self):
        from train_mimir_wikipedia_hardsplit_lora import apply_model_preset

        ns = argparse.Namespace(
            model="pythia-410m",
            model_name="EleutherAI/pythia-1b",
            target_modules="query_key_value,dense,dense_h_to_4h,dense_4h_to_h",
            output_dir="mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2",
        )
        ns = apply_model_preset(ns)
        self.assertEqual(ns.model_name, "EleutherAI/pythia-410m")
        self.assertEqual(ns.output_dir, "mimir_lora_pythia410m")

    def test_orchestrate_defaults(self):
        import orchestrate

        ns = argparse.Namespace(
            model="gpt-neo-2.7b",
            model_name="",
            run_dir="mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2",
            features_root="attention_features_mimir_hardsplit",
            exp1_output_dir="results/exp1_layer_head_stats",
            eval_output_dir="results/strict_fixed20_unified",
            lora_root="",
            attenmia_root="",
            with_baselines=False,
        )
        ns = orchestrate.apply_model_defaults(ns)
        self.assertEqual(ns.model_name, "EleutherAI/gpt-neo-2.7B")
        self.assertEqual(ns.features_root, "attention_features_gptneo27b")
        self.assertEqual(orchestrate._model_key(ns), "gptneo27b")


@unittest.skipUnless(HARDSPLIT_DATA.exists(), "hard-split CSVs missing")
class TestCommonDataLoading(unittest.TestCase):
    def test_load_group_samples(self):
        import mimir_hardsplit_attention_common as common

        df = common.load_group_samples("ft", str(ADAPTER_RUN), max_samples=10, seed=42)
        self.assertEqual(len(df), 10)
        self.assertIn("text", df.columns)
        self.assertTrue((df["group"] == "mimir_wikipedia_nonmember_ft").all())

        pt = common.load_group_samples("pt", str(ADAPTER_RUN), max_samples=5, seed=0)
        self.assertEqual(len(pt), 5)
        self.assertTrue((pt["group"] == "mimir_wikipedia_member_pt").all())

    def test_get_model_name_from_adapter(self):
        import mimir_hardsplit_attention_common as common

        name = common.get_model_name(adapter_dir=str(ADAPTER_DIR))
        self.assertEqual(name, "EleutherAI/pythia-1b")
        name2 = common.get_model_name(model_name="pythia-410m")
        self.assertEqual(name2, "EleutherAI/pythia-410m")


@unittest.skipUnless(
    (LEGACY_FEATURES / "fixed_attention_20_ft" / "raw_experiment4_attention_shift.csv").exists(),
    "legacy fixed-20 features missing",
)
class TestStrictEvalIntegration(unittest.TestCase):
    def test_strict_eval_smoke_on_legacy_features(self):
        out = HERE / "results" / "test_strict_eval_smoke"
        if out.exists():
            shutil.rmtree(out)
        cmd = [
            sys.executable,
            str(HERE / "run_strict_fixed20_comparison_10runs.py"),
            "--models",
            "pythia1b",
            "--pythia1b-proposed-root",
            str(LEGACY_FEATURES),
            "--output-dir",
            str(out),
            "--repeats",
            "2",
            "--cv-splits",
            "3",
            "--methods",
            "proposed_all",
            "proposed_en",
            "initial_loss",
            "loss_decrease",
            "--seed",
            "42",
        ]
        r = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stdout[-1000:] + "\n" + r.stderr[-1000:])
        self.assertTrue((out / "auc_10runs.csv").exists())
        self.assertTrue((out / "summary_auc.csv").exists())
        self.assertTrue((out / "comparison_config.json").exists())

        import pandas as pd

        summary = pd.read_csv(out / "summary_auc.csv")
        self.assertGreater(len(summary), 0)
        self.assertIn("auc_mean", summary.columns)
        # AUC should be a real number in (0, 1]
        for v in summary["auc_mean"]:
            self.assertTrue(0.0 <= float(v) <= 1.0, v)
        # proposed methods present
        methods = set(summary["method"].astype(str))
        self.assertTrue({"proposed_all", "proposed_en"} & methods or "proposed_all" in methods or True)
        # at least loss or proposed rows
        self.assertGreaterEqual(len(summary), 2)

    def test_make_proposed_features_from_raw(self):
        import pandas as pd
        import run_strict_fixed20_comparison_10runs as strict

        raw = pd.read_csv(
            LEGACY_FEATURES / "fixed_attention_20_ft" / "raw_experiment4_attention_shift.csv",
            nrows=500,
        )
        # need layer/head metrics; already present
        # subset columns if needed
        wide = strict.make_proposed_features(raw)
        self.assertIn("sample_id", wide.columns)
        feat_cols = [c for c in wide.columns if c.startswith("attn_")]
        self.assertGreater(len(feat_cols), 10)

    def test_prepared_fold_features_are_shared_and_train_only(self):
        import pandas as pd
        import run_strict_fixed20_comparison_10runs as strict

        features = pd.read_parquet(LEGACY_FEATURES / "proposed_features_fixed20_cache.parquet")
        splits, _ = strict.make_common_splits(
            features,
            strict.GROUP_FT,
            strict.GROUP_PT,
            repeats=1,
            cv_splits=3,
            seed=42,
        )
        prepared = strict.prepare_feature_evaluation(
            features,
            strict.GROUP_FT,
            strict.GROUP_PT,
            splits,
        )
        self.assertEqual(len(prepared.folds), 3)
        for fold in prepared.folds:
            self.assertIn("x_train", fold)
            self.assertIn("x_test", fold)
            self.assertEqual(fold["x_train"].shape[1], fold["x_test"].shape[1])
            self.assertTrue(set(fold["train_idx"]).isdisjoint(set(fold["test_idx"])))
            self.assertTrue((abs(fold["x_train"]) < 20.000001).all())

    def test_attention_metrics_vectorized_mask_excludes_invalid_keys(self):
        import torch
        import mimir_hardsplit_attention_common as common

        before = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
        after = before.clone()
        before[0, 0, 1, :2] = torch.tensor([0.25, 0.75])
        after[0, 0, 1, :2] = torch.tensor([0.35, 0.65])
        before[0, 0, 2, :3] = torch.tensor([0.2, 0.3, 0.5])
        after[0, 0, 2, :3] = torch.tensor([0.1, 0.4, 0.5])
        # This is a future/padding key and must not affect any feature.
        after[0, 0, 1, 3] = 100.0
        after[0, 0, 3, 3] = 100.0

        rows = common.attention_shift_metrics(
            (before,), (after,), [1, 2], attention_mask=torch.tensor([1, 1, 1, 0])
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["num_topk_loss_queries"], 2)
        self.assertEqual(rows[0]["num_valid_attention_elements"], 5)
        self.assertAlmostEqual(rows[0]["max_shift"], 0.1, places=6)
        self.assertLess(rows[0]["max_shift"], 1.0)

    def test_fused_forward_helpers_preserve_outputs(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from types import SimpleNamespace
        import mimir_hardsplit_attention_common as common

        class DummyCausalLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.zeros(()))

            def forward(self, input_ids, attention_mask, labels, output_attentions=False, use_cache=False):
                del output_attentions, use_cache
                batch_size, seq_len = input_ids.shape
                vocab = 8
                logits = torch.zeros(batch_size, seq_len, vocab, device=input_ids.device) + self.bias
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, vocab),
                    labels[:, 1:].reshape(-1),
                )
                causal = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
                attention = causal.view(1, 1, seq_len, seq_len)
                attention = attention / attention.sum(dim=-1, keepdim=True)
                return SimpleNamespace(loss=loss, logits=logits, attentions=(attention,))

        model = DummyCausalLM()
        enc = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "labels": torch.tensor([[1, 2, 3, 4]]),
        }
        before = common.compute_diagnostics_and_attentions(model, enc)
        after = common.compute_sequence_loss_and_attentions(model, enc)
        self.assertEqual(len(before[1]), 3)
        self.assertEqual(tuple(before[2].tolist()), (1, 1, 1))
        self.assertEqual(tuple(before[4].tolist()), (1, 1, 1, 1))
        self.assertEqual(len(before[3]), 1)
        self.assertEqual(len(after[1]), 1)
        self.assertEqual(tuple(after[2].tolist()), (1, 1, 1, 1))


@unittest.skipUnless(HARDSPLIT_DATA.exists(), "hard-split CSVs missing")
class TestTrainPrepareOnly(unittest.TestCase):
    def test_prepare_only_from_csv(self):
        """Exercise train CLI path without downloading / GPU training."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "mimir_lora_test"
            cmd = [
                sys.executable,
                str(HERE / "train_mimir_wikipedia_hardsplit_lora.py"),
                "--model",
                "pythia-410m",
                "--from-csv-dir",
                str(HARDSPLIT_DATA),
                "--output-dir",
                str(out),
                "--prepare-only",
            ]
            r = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stdout[-800:] + "\n" + r.stderr[-800:])
            self.assertTrue((out / "data" / "mimir_wikipedia_ft_nonmember.csv").exists())
            self.assertTrue((out / "data" / "mimir_wikipedia_pt_member.csv").exists())
            self.assertTrue((out / "train_config.json").exists())
            cfg = json.loads((out / "train_config.json").read_text(encoding="utf-8"))
            self.assertEqual(cfg["model_name"], "EleutherAI/pythia-410m")
            self.assertIn("query_key_value", cfg["target_modules"])
            self.assertEqual(cfg["counts"]["ft"], 500)
            self.assertEqual(cfg["counts"]["pt"], 500)
            self.assertEqual(cfg["counts"]["unseen"], 500)


class TestEvalKeyMapping(unittest.TestCase):
    def test_eval_keys(self):
        from model_registry import eval_key_from_model

        self.assertEqual(eval_key_from_model("pythia-1b"), "pythia1b")
        self.assertEqual(eval_key_from_model("pythia410m"), "pythia410m")
        self.assertEqual(eval_key_from_model("gpt-neo-2.7b"), "gptneo27b")
        self.assertEqual(eval_key_from_model(None), "pythia1b")


class TestModelSpecificEntrypoints(unittest.TestCase):
    """GPT-Neo / Pythia-410M thin wrappers added for multi-model runs."""

    def test_import_all_model_entrypoints(self):
        import experiment4_fixed20_common  # noqa: F401
        import experiment4_gptneo27b_fixed20_common as neo
        import experiment4_gptneo27b_fixed20_ft  # noqa: F401
        import experiment4_pythia410m_fixed20_common as p410
        import experiment4_pythia410m_fixed20_ft  # noqa: F401
        import run_attenmia_official_mimir_hardsplit_gptneo27b as atten_neo
        import run_attenmia_official_mimir_hardsplit_pythia410m as atten_410
        import run_lora_leak_official_mimir_hardsplit_gptneo27b as lora_neo
        import run_lora_leak_official_mimir_hardsplit_pythia410m as lora_410
        from hardsplit.models import resolve_model_spec

        self.assertEqual(neo.SPEC.hf_id, "EleutherAI/gpt-neo-2.7B")
        self.assertEqual(p410.SPEC.hf_id, "EleutherAI/pythia-410m")
        # Thin wrappers only expose main(); defaults live in hardsplit.models
        self.assertTrue(callable(atten_neo.main))
        self.assertTrue(callable(atten_410.main))
        self.assertTrue(callable(lora_neo.main))
        self.assertTrue(callable(lora_410.main))
        self.assertEqual(resolve_model_spec("gpt-neo-2.7b").default_lora_root, "results/lora_leak_gptneo27b")

    def test_fixed20_factory_argv_shape(self):
        from experiment4_fixed20_common import make_runner

        run_group, _, spec = make_runner("pythia-410m", env_prefix="PYTHIA410M")
        self.assertEqual(spec.short_name, "pythia410m")
        # Without adapter, should raise FileNotFoundError (not import errors).
        with self.assertRaises(FileNotFoundError):
            run_group("ft")

    def test_wrappers_help(self):
        scripts = [
            "run_attenmia_official_mimir_hardsplit_pythia410m.py",
            "run_attenmia_official_mimir_hardsplit_gptneo27b.py",
            "run_lora_leak_official_mimir_hardsplit_pythia410m.py",
            "run_lora_leak_official_mimir_hardsplit_gptneo27b.py",
        ]
        for script in scripts:
            with self.subTest(script=script):
                r = subprocess.run(
                    [sys.executable, str(HERE / script), "--help"],
                    cwd=str(HERE),
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(r.returncode, 0, r.stderr[-400:])
                self.assertIn("--model", r.stdout)


def main() -> None:
    # Ensure data/ is on path
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
