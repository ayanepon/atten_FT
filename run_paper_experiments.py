#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orchestrate **all** experiments that appear in the BlackboxNLP / ACL paper.

Paper stages (``acl_latex.tex`` §Experiment):

  0. Data / LoRA FT
     - MIMIR Wikipedia hard split (PT / FT / Unseen × 500)
     - LoRA FT on FT split (lr=1e-4, 5 epochs, r=8, α=16)
     - Models: Pythia-1B (main), Pythia-410M (Exp.2), GPT-Neo-2.7B (Appendix)

  1. Experiment 1 — localization of attention updates
     - Fixed-20 additional training (lr=1e-5)
     - Mann–Whitney + BH-FDR + Cliff's δ (layer–head)

  2. Experiment 2 — vs loss / LoRA-Leak / AttenMIA
     - Proposed (all) + Proposed+EN
     - initial_loss, loss_decrease, LoRA-Leak, AttenMIA
     - 10×5-fold strict eval, FT positive, no test flip
     - Models: Pythia-1B + Pythia-410M (main); GPT-Neo appendix optional

  3. Experiment 3 — effect of additional-training steps
     - Fixed 20 / 50 / 100 + early-stopping (dynamic)
     - Models: Pythia-1B (main), Pythia-410M, GPT-Neo-2.7B — run per model in `models`
     - Proposed (all) + Proposed+EN strict eval per setting

Usage:
  # Full paper pipeline (train if adapters missing; multi-GPU extract)
  ./run_paper_experiments.sh full --gpus auto \\
      --from-csv-dir mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2/data

  # Print planned stages only
  ./run_paper_experiments.sh plan
  ./run_paper_experiments.sh full --dry-run

  # Main fixed-20 only (skip Exp.3 long extract)
  ./run_paper_experiments.sh full --skip-exp3

  # Status / stop
  ./run_paper_experiments.sh status
  ./run_paper_experiments.sh stop
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


HERE = Path(__file__).resolve().parent
DEFAULT_CSV = "mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2/data"
STATE_DIR_NAME = "paper_pipeline"
DYNAMIC_HELPER = "_run_dynamic_extract.py"

# Models ordered as in the paper narrative
PAPER_MODELS_MAIN = ("pythia-1b", "pythia-410m")  # Exp.1–2 tables
PAPER_MODELS_APPENDIX = ("gpt-neo-2.7b",)  # Appendix GPT-Neo
PAPER_MODELS_ALL = PAPER_MODELS_MAIN + PAPER_MODELS_APPENDIX

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_spec(model_key: str):
    from model_registry import resolve_model_spec

    return resolve_model_spec(model_key)


def exp3_settings_for_model(model: str) -> Tuple[Tuple[str, Optional[int], str, str], ...]:
    """(label, fixed_steps | None for dynamic, features_root relative, condition_prefix).

    20-step reuses the model's main (Phase A) features root; 50/100/dynamic get
    their own per-model roots named after the model's short_name.
    """
    spec = load_spec(model)
    sn = spec.short_name
    return (
        ("20-step", 20, spec.default_features_root, "fixed_attention_20"),
        ("50-step", 50, f"attention_features_{sn}_steps50", "fixed_attention_50"),
        ("100-step", 100, f"attention_features_{sn}_steps100", "fixed_attention_100"),
        ("Early", None, f"attention_features_{sn}_dynamic", "dynamic_attention"),
    )


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    name: str
    model: str
    status: str  # pending|running|ok|skipped|failed
    started: str = ""
    finished: str = ""
    detail: str = ""
    cmd: List[str] = field(default_factory=list)


@dataclass
class PipelineState:
    created: str
    updated: str
    stages: List[Dict[str, Any]] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated = datetime.now().isoformat(timespec="seconds")
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")


def state_path(root: Path) -> Path:
    return root / STATE_DIR_NAME / "state.json"


def load_state(root: Path) -> PipelineState:
    p = state_path(root)
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        return PipelineState(**data)
    now = datetime.now().isoformat(timespec="seconds")
    return PipelineState(created=now, updated=now, stages=[])


def record(state: PipelineState, results: List[StageResult]) -> None:
    state.stages = [asdict(r) for r in results]
    state.save(state_path(HERE))


# ---------------------------------------------------------------------------
# Command helpers
# ---------------------------------------------------------------------------


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Path,
    dry_run: bool,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
) -> int:
    pretty = " ".join(str(c) for c in cmd)
    log(f"$ {pretty}")
    if dry_run:
        return 0
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    proc = subprocess.run(list(cmd), cwd=str(cwd), env=full_env)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {pretty}")
    return proc.returncode


def adapter_exists(run_dir: Path) -> bool:
    return (run_dir / "adapter" / "adapter_config.json").exists() or (
        run_dir / "adapter_config.json"
    ).exists()


def _count_csv_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size < 10:
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return max(0, sum(1 for _ in f) - 1)


def features_ready(
    features_root: Path,
    *,
    condition_prefix: str = "fixed_attention_20",
    n_per_group: int = 500,
) -> bool:
    """Heuristic: each group has at least n_per_group sample rows."""
    for g in ("ft", "pt", "unseen"):
        p = features_root / f"{condition_prefix}_{g}" / "sample_level_experiment4.csv"
        if _count_csv_rows(p) < n_per_group:
            return False
    return True


def resolve_csv_dir(from_csv_dir: str) -> str:
    if not from_csv_dir:
        return DEFAULT_CSV
    p = Path(from_csv_dir)
    if p.is_absolute():
        return str(p)
    if (HERE / p).exists():
        return str(HERE / p)
    return from_csv_dir


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def stage_train(
    *,
    python: str,
    model: str,
    from_csv_dir: str,
    force_train: bool,
    dry_run: bool,
    seed: int,
) -> StageResult:
    spec = load_spec(model)
    run_dir = HERE / spec.default_run_dir
    name = f"train:{model}"
    if adapter_exists(run_dir) and not force_train:
        log(f"skip {name} (adapter exists at {run_dir / 'adapter'})")
        return StageResult(name=name, model=model, status="skipped", detail=str(run_dir))

    cmd = [
        python,
        str(HERE / "train_mimir_wikipedia_hardsplit_lora.py"),
        "--model",
        model,
        "--output-dir",
        str(run_dir),
        "--seed",
        str(seed),
    ]
    if from_csv_dir:
        cmd.extend(["--from-csv-dir", from_csv_dir])
    started = datetime.now().isoformat(timespec="seconds")
    run_cmd(cmd, cwd=HERE, dry_run=dry_run)
    return StageResult(
        name=name,
        model=model,
        status="ok",
        started=started,
        finished=datetime.now().isoformat(timespec="seconds"),
        cmd=cmd,
        detail=str(run_dir),
    )


def stage_extract_fixed(
    *,
    python: str,
    model: str,
    fixed_steps: int,
    features_root: Path,
    gpus: Sequence[str],
    min_free_gib: float,
    sample_shards: int,
    fresh: bool,
    n_per_group: int,
    lr: float,
    seed: int,
    dry_run: bool,
    skip_if_ready: bool = True,
) -> StageResult:
    name = f"extract_fixed{fixed_steps}:{model}"
    prefix = f"fixed_attention_{fixed_steps}"
    if skip_if_ready and not fresh and features_ready(
        features_root, condition_prefix=prefix, n_per_group=n_per_group
    ):
        log(f"skip {name} (features ready under {features_root})")
        return StageResult(name=name, model=model, status="skipped", detail=str(features_root))

    cmd = [
        python,
        str(HERE / "orchestrate.py"),
        "extract",
        "--model",
        model,
        "--features-root",
        str(features_root),
        "--fixed-steps",
        str(fixed_steps),
        "--n-per-group",
        str(n_per_group),
        "--lr",
        str(lr),
        "--seed",
        str(seed),
        "--gpus",
        *([str(g) for g in gpus] if gpus else ["auto"]),
        "--min-free-gib",
        str(min_free_gib),
        "--sample-shards",
        str(sample_shards),
        "--wait",
    ]
    if fresh:
        cmd.append("--fresh")
    started = datetime.now().isoformat(timespec="seconds")
    run_cmd(cmd, cwd=HERE, dry_run=dry_run)
    return StageResult(
        name=name,
        model=model,
        status="ok",
        started=started,
        finished=datetime.now().isoformat(timespec="seconds"),
        cmd=cmd,
        detail=str(features_root),
    )


def stage_extract_dynamic(
    *,
    python: str,
    model: str,
    gpus: Sequence[str],
    min_free_gib: float,
    fresh: bool,
    n_per_group: int,
    lr: float,
    seed: int,
    dry_run: bool,
) -> List[StageResult]:
    """Exp.3 early-stopping extract (one process per group, run concurrently across free GPUs).

    The extraction helper resumes per-sample (``--resume`` on by default), so
    launching all pending groups at once and letting each claim its own GPU
    (round-robin over ``free``) is safe even if a prior sequential run left a
    group partially done.
    """
    results: List[StageResult] = []
    spec = load_spec(model)
    feat_dyn = HERE / f"attention_features_{spec.short_name}_dynamic"
    helper = HERE / DYNAMIC_HELPER
    if not helper.exists():
        raise FileNotFoundError(f"Missing {helper}; expected dynamic extract helper.")

    try:
        import orchestrate as orch

        free = (
            orch.free_gpus(min_free_gib)
            if not gpus or list(gpus) == ["auto"]
            else [int(x) for x in gpus]
        )
    except Exception:
        free = [0]
    if not free:
        free = [0]

    pending: List[Tuple[str, Path, str, List[str], Dict[str, str], int]] = []
    for i, group in enumerate(("ft", "pt", "unseen")):
        out = feat_dyn / f"dynamic_attention_{group}"
        out.mkdir(parents=True, exist_ok=True)
        sample_csv = out / "sample_level_experiment4.csv"
        name = f"extract_dynamic:{model}:{group}"
        if not fresh and _count_csv_rows(sample_csv) >= n_per_group:
            log(f"skip {name} (n={_count_csv_rows(sample_csv)})")
            results.append(StageResult(name=name, model=model, status="skipped", detail=str(out)))
            continue

        gpu = free[i % len(free)]
        cmd = [
            python,
            str(helper),
            "--run-dir",
            str(HERE / spec.default_run_dir),
            "--adapter-dir",
            str(HERE / spec.default_run_dir / "adapter"),
            "--model-name",
            spec.hf_id,
            "--output-dir",
            str(out),
            "--groups",
            group,
            "--n-per-group",
            str(n_per_group),
            "--lr",
            str(lr),
            "--seed",
            str(seed),
        ]
        env = {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        pending.append((group, out, name, cmd, env, gpu))

    if dry_run:
        for group, out, name, cmd, env, gpu in pending:
            log(f"$ (dry-run, gpu={gpu}) " + " ".join(str(c) for c in cmd))
            results.append(StageResult(name=name, model=model, status="ok", cmd=cmd, detail=str(out)))
        return results

    running = []
    for group, out, name, cmd, env, gpu in pending:
        full_env = os.environ.copy()
        full_env.update(env)
        log(f"$ (parallel, gpu={gpu}) " + " ".join(str(c) for c in cmd))
        log_path = out / "extract_dynamic_stdout.log"
        log_fh = log_path.open("a")
        started = datetime.now().isoformat(timespec="seconds")
        proc = subprocess.Popen(cmd, cwd=str(HERE), env=full_env, stdout=log_fh, stderr=subprocess.STDOUT)
        running.append((group, out, name, cmd, gpu, proc, started, log_fh))

    OWNERSHIP_CONFLICT_RC = 75  # keep in sync with _run_dynamic_extract.EXIT_OWNERSHIP_CONFLICT

    failed: List[Tuple[str, int]] = []
    for group, out, name, cmd, gpu, proc, started, log_fh in running:
        rc = proc.wait()
        log_fh.close()
        finished = datetime.now().isoformat(timespec="seconds")
        if rc == OWNERSHIP_CONFLICT_RC:
            log(f"skip {name} (output_dir already owned by another host; see {out / 'extract_dynamic_stdout.log'})")
            results.append(
                StageResult(
                    name=name,
                    model=model,
                    status="skipped",
                    started=started,
                    finished=finished,
                    cmd=cmd,
                    detail=f"gpu={gpu} owned by another host, log={out / 'extract_dynamic_stdout.log'}",
                )
            )
        elif rc != 0:
            failed.append((name, rc))
            results.append(
                StageResult(
                    name=name,
                    model=model,
                    status="error",
                    started=started,
                    finished=finished,
                    cmd=cmd,
                    detail=f"gpu={gpu} returncode={rc} log={out / 'extract_dynamic_stdout.log'}",
                )
            )
        else:
            log(f"dynamic extract group={group} gpu={gpu} done")
            results.append(
                StageResult(
                    name=name,
                    model=model,
                    status="ok",
                    started=started,
                    finished=finished,
                    cmd=cmd,
                    detail=str(out),
                )
            )

    if failed:
        raise RuntimeError(f"dynamic extract failed for: {failed}")

    return results


def stage_baselines(
    *,
    python: str,
    model: str,
    n_per_group: int,
    seed: int,
    dry_run: bool,
    which: Sequence[str] = ("lora_leak", "attenmia"),
    lora_fast: bool = False,
) -> List[StageResult]:
    results: List[StageResult] = []
    for kind in which:
        name = f"baseline_{kind}:{model}"
        if kind == "lora_leak" and lora_fast:
            spec = load_spec(model)
            script = HERE / "run_lora_leak_official_mimir_hardsplit.py"
            if not script.exists():
                script = HERE / "run_lora_leak_official_mimir_hardsplit_2.py"
            cmd = [
                python,
                str(script),
                "--model",
                model,
                "--run-dir",
                str(HERE / spec.default_run_dir),
                "--adapter-dir",
                str(HERE / spec.default_run_dir),
                "--output-dir",
                str(HERE / spec.default_lora_root),
                "--data-dir",
                str(HERE / "mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2" / "data"),
                "--n-per-group",
                str(n_per_group),
                "--seed",
                str(seed),
                "--fast",
            ]
        else:
            cmd = [
                python,
                str(HERE / "orchestrate.py"),
                "baselines",
                "--model",
                model,
                "--baseline-which",
                kind,
                "--n-per-group",
                str(n_per_group),
                "--seed",
                str(seed),
            ]
        started = datetime.now().isoformat(timespec="seconds")
        run_cmd(cmd, cwd=HERE, dry_run=dry_run)
        results.append(
            StageResult(
                name=name,
                model=model,
                status="ok",
                started=started,
                finished=datetime.now().isoformat(timespec="seconds"),
                cmd=cmd,
            )
        )
    return results


def stage_exp1(*, python: str, model: str, dry_run: bool) -> StageResult:
    spec = load_spec(model)
    name = f"exp1:{model}"
    out = HERE / "results" / f"exp1_layer_head_stats_{spec.short_name}"
    cmd = [
        python,
        str(HERE / "analyze_exp1_layer_head_significance.py"),
        "--root",
        str(HERE / spec.default_features_root),
        "--output-dir",
        str(out),
    ]
    started = datetime.now().isoformat(timespec="seconds")
    run_cmd(cmd, cwd=HERE, dry_run=dry_run)
    return StageResult(
        name=name,
        model=model,
        status="ok",
        started=started,
        finished=datetime.now().isoformat(timespec="seconds"),
        cmd=cmd,
        detail=str(out),
    )


def stage_eval(
    *,
    python: str,
    model: str,
    repeats: int,
    seed: int,
    n_jobs: int,
    dry_run: bool,
    with_baselines: bool,
    methods: Optional[Sequence[str]] = None,
    proposed_root: Optional[Path] = None,
    condition_prefix: str = "fixed_attention_20",
    output_dir: Optional[Path] = None,
    name: Optional[str] = None,
) -> StageResult:
    spec = load_spec(model)
    name = name or f"eval:{model}"
    out = output_dir or (HERE / "results" / f"strict_fixed20_{spec.short_name}")
    if methods is None:
        methods = [
            "proposed_all",
            "proposed_en",
            "initial_loss",
            "loss_decrease",
        ]
        if with_baselines:
            methods = list(methods) + ["lora_leak", "attenmia"]
    model_key = spec.short_name
    root = proposed_root or (HERE / spec.default_features_root)
    cmd = [
        python,
        str(HERE / "run_strict_fixed20_comparison_10runs.py"),
        "--models",
        model_key,
        f"--{model_key}-proposed-root",
        str(root),
        "--output-dir",
        str(out),
        "--repeats",
        str(repeats),
        "--seed",
        str(seed),
        "--n-jobs",
        str(n_jobs),
        "--condition-prefix",
        condition_prefix,
        "--methods",
        *list(methods),
    ]
    if with_baselines and "lora_leak" in methods:
        cmd.extend([f"--{model_key}-lora-root", str(HERE / spec.default_lora_root)])
    if with_baselines and "attenmia" in methods:
        cmd.extend([f"--{model_key}-attenmia-root", str(HERE / spec.default_attenmia_root)])
    started = datetime.now().isoformat(timespec="seconds")
    run_cmd(cmd, cwd=HERE, dry_run=dry_run)
    return StageResult(
        name=name,
        model=model,
        status="ok",
        started=started,
        finished=datetime.now().isoformat(timespec="seconds"),
        cmd=cmd,
        detail=str(out),
    )


def stage_eval_multimodel(
    *,
    python: str,
    models: Sequence[str],
    repeats: int,
    seed: int,
    n_jobs: int,
    dry_run: bool,
    with_baselines: bool,
) -> StageResult:
    """Single comparison table across all models that have features."""
    name = "eval:multi_model"
    out = HERE / "results" / "strict_fixed20_paper_all_models"
    eval_keys: List[str] = []
    cmd = [
        python,
        str(HERE / "run_strict_fixed20_comparison_10runs.py"),
        "--output-dir",
        str(out),
        "--repeats",
        str(repeats),
        "--seed",
        str(seed),
        "--n-jobs",
        str(n_jobs),
        "--condition-prefix",
        "fixed_attention_20",
    ]
    methods = ["proposed_all", "proposed_en", "initial_loss", "loss_decrease"]
    if with_baselines:
        methods = methods + ["lora_leak", "attenmia"]
    cmd.extend(["--methods", *methods])
    for model in models:
        spec = load_spec(model)
        key = spec.short_name
        eval_keys.append(key)
        cmd.extend([f"--{key}-proposed-root", str(HERE / spec.default_features_root)])
        if with_baselines:
            cmd.extend(
                [
                    f"--{key}-lora-root",
                    str(HERE / spec.default_lora_root),
                    f"--{key}-attenmia-root",
                    str(HERE / spec.default_attenmia_root),
                ]
            )
    cmd.extend(["--models", *eval_keys])
    started = datetime.now().isoformat(timespec="seconds")
    run_cmd(cmd, cwd=HERE, dry_run=dry_run)
    return StageResult(
        name=name,
        model=",".join(models),
        status="ok",
        started=started,
        finished=datetime.now().isoformat(timespec="seconds"),
        cmd=cmd,
        detail=str(out),
    )


def stage_exp3_extract(
    *,
    python: str,
    model: str,
    gpus: Sequence[str],
    min_free_gib: float,
    sample_shards: int,
    fresh: bool,
    n_per_group: int,
    lr: float,
    seed: int,
    dry_run: bool,
    only: Optional[str] = None,
) -> List[StageResult]:
    """Extract fixed 50/100 + dynamic for Exp.3 (20-step reuses main features root).

    `only` restricts to a single condition ("50", "100", or "dynamic") so a
    condition can be delegated to a separate host without racing the other
    conditions on their own writers.
    """
    results: List[StageResult] = []
    for label, steps, feat_rel, _prefix in exp3_settings_for_model(model):
        if steps is None:
            continue  # dynamic handled separately
        if steps == 20:
            # Main path already extracts fixed-20 into default features root
            continue
        if only is not None and only != str(steps):
            continue
        results.append(
            stage_extract_fixed(
                python=python,
                model=model,
                fixed_steps=steps,
                features_root=HERE / feat_rel,
                gpus=gpus,
                min_free_gib=min_free_gib,
                sample_shards=sample_shards,
                fresh=fresh,
                n_per_group=n_per_group,
                lr=lr,
                seed=seed,
                dry_run=dry_run,
                skip_if_ready=not fresh,
            )
        )
    if only is None or only == "dynamic":
        results.extend(
            stage_extract_dynamic(
                python=python,
                model=model,
                gpus=gpus,
                min_free_gib=min_free_gib,
                fresh=fresh,
                n_per_group=n_per_group,
                lr=lr,
                seed=seed,
                dry_run=dry_run,
            )
        )
    return results


EXP3_ONLY_LABELS = {"50": "50-step", "100": "100-step", "dynamic": "Early"}


def stage_exp3_eval(
    *,
    python: str,
    model: str,
    repeats: int,
    seed: int,
    n_jobs: int,
    n_per_group: int,
    dry_run: bool,
    only: Optional[str] = None,
) -> List[StageResult]:
    """Paper Table (ablation_step): Proposed all + EN for 20/50/100/Early on `model`.

    `only` restricts eval to a single condition's label and skips writing the
    combined cross-condition table/LaTeX, since that aggregate file is owned
    by the run that covers all conditions (avoids clobbering it from a
    single-condition delegate run on another host).
    """
    results: List[StageResult] = []
    combined_rows: List[Dict[str, Any]] = []
    spec = load_spec(model)
    settings = exp3_settings_for_model(model)
    if only is not None:
        target_label = EXP3_ONLY_LABELS[only]
        settings = tuple(s for s in settings if s[0] == target_label)
    out_root = HERE / "results" / f"exp3_step_ablation_{spec.short_name}"
    out_root.mkdir(parents=True, exist_ok=True)

    for label, _steps, feat_rel, prefix in settings:
        feat_root = HERE / feat_rel
        out = out_root / label.replace("-", "").replace(" ", "_").lower()
        name = f"exp3_eval:{model}:{label}"
        # Require the FULL n_per_group here (not just >=1) -- a condition can
        # be left partially populated by another host's still-running (or
        # ownership-conflict-skipped) extraction, and evaluating on that
        # partial data would silently corrupt the combined summary table.
        if not dry_run and not features_ready(
            feat_root, condition_prefix=prefix, n_per_group=n_per_group
        ):
            log(f"skip {name} (features missing/incomplete under {feat_root})")
            results.append(
                StageResult(
                    name=name,
                    model=model,
                    status="skipped",
                    detail=f"missing features: {feat_root}",
                )
            )
            continue
        r = stage_eval(
            python=python,
            model=model,
            repeats=repeats,
            seed=seed,
            n_jobs=n_jobs,
            dry_run=dry_run,
            with_baselines=False,
            methods=["proposed_all", "proposed_en"],
            proposed_root=feat_root,
            condition_prefix=prefix,
            output_dir=out,
            name=name,
        )
        results.append(r)
        if dry_run:
            continue
        summary_path = out / "summary_auc.csv"
        if summary_path.exists():
            try:
                import pandas as pd

                sdf = pd.read_csv(summary_path)
                for _, row in sdf.iterrows():
                    combined_rows.append(
                        {
                            "setting": label,
                            "condition_prefix": prefix,
                            "method": row.get("method"),
                            "comparison": row.get("comparison"),
                            "auc_mean": row.get("auc_mean"),
                            "tpr_at_10_fpr_mean": row.get("tpr_at_10_fpr_mean"),
                        }
                    )
            except Exception as exc:
                log(f"warn: could not read {summary_path}: {exc}")

    if combined_rows and not dry_run and only is None:
        import pandas as pd

        cdf = pd.DataFrame(combined_rows)
        cdf.to_csv(out_root / "exp3_combined_summary.csv", index=False)
        # Paper-style wide table (All/EN × setting × FT-PT / FT-Unseen)
        lines = [
            r"\begin{table}[t]",
            r"\caption{Experiment 3: additional-training steps (auto-generated).}",
            r"\label{tab:ablation_step_reproduced}",
            r"\centering",
            r"\small",
            r"\begin{tabular}{llcccc}",
            r"\toprule",
            r"& & \multicolumn{2}{c}{FT--PT} & \multicolumn{2}{c}{FT--Unseen} \\",
            r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
            r"Feat. & Method & AUC & TPR & AUC & TPR \\",
            r"\midrule",
        ]
        method_map = {"proposed_all": "All", "proposed_en": "EN"}
        for feat_key, feat_label in (("proposed_all", "All"), ("proposed_en", "EN")):
            for label, *_rest in settings:
                sub = cdf[(cdf["setting"] == label) & (cdf["method"] == feat_key)]
                if sub.empty:
                    continue
                pt = sub[sub["comparison"] == "ft_vs_pt"]
                uns = sub[sub["comparison"] == "ft_vs_unseen"]

                def _cell(frame: Any, col: str) -> str:
                    if frame is None or len(frame) == 0:
                        return "--"
                    try:
                        return f"{float(frame.iloc[0][col]):.3f}"
                    except Exception:
                        return "--"

                lines.append(
                    f"{feat_label} & {label} & "
                    f"{_cell(pt, 'auc_mean')} & {_cell(pt, 'tpr_at_10_fpr_mean')} & "
                    f"{_cell(uns, 'auc_mean')} & {_cell(uns, 'tpr_at_10_fpr_mean')} \\\\"
                )
            if feat_key == "proposed_all":
                lines.append(r"\midrule")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
        (out_root / "paper_table_exp3.tex").write_text("\n".join(lines), encoding="utf-8")
        log(f"Exp.3 combined table → {out_root / 'exp3_combined_summary.csv'}")
        log(f"Exp.3 LaTeX → {out_root / 'paper_table_exp3.tex'}")

    if only is not None:
        table_status = "skipped"
        table_detail = f"{out_root} (only={only}, combined table owned by full run)"
    else:
        table_status = "ok" if combined_rows or dry_run else "skipped"
        table_detail = str(out_root)
    results.append(
        StageResult(
            name=f"exp3_table:{spec.short_name}",
            model=model,
            status=table_status,
            detail=table_detail,
        )
    )
    return results


# ---------------------------------------------------------------------------
# Plan / full pipeline
# ---------------------------------------------------------------------------


def plan_models(args: argparse.Namespace) -> List[str]:
    if args.models:
        return list(args.models)
    models = list(PAPER_MODELS_MAIN)
    if not getattr(args, "skip_appendix", False):
        models.extend(PAPER_MODELS_APPENDIX)
    return models


def describe_plan(args: argparse.Namespace) -> List[str]:
    """Human-readable stage list (no execution)."""
    models = plan_models(args)
    lines = [
        "Paper experiment plan",
        f"  models: {models}",
        f"  n_per_group={getattr(args, 'n_per_group', 500)} "
        f"lr_add={getattr(args, 'lr', 1e-5)} repeats={getattr(args, 'repeats', 10)}",
        "",
        "Phase A — per model (train → extract-20 → baselines → Exp.1 → Exp.2 eval)",
    ]
    for m in models:
        lines.append(f"  [{m}]")
        if not getattr(args, "skip_train", False):
            lines.append("    - train LoRA FT (lr=1e-4) if adapter missing")
        if not getattr(args, "skip_extract", False):
            lines.append("    - extract fixed-20 attention features (lr=1e-5)")
        if not getattr(args, "skip_baselines", False):
            lines.append(f"    - baselines: {getattr(args, 'baseline_which', ['lora_leak','attenmia'])}")
        if not getattr(args, "skip_exp1", False):
            lines.append("    - Exp.1 layer–head FDR / Cliff's δ")
        if not getattr(args, "skip_eval", False):
            lines.append("    - Exp.2 strict 10×5 eval (Proposed / EN / loss / baselines)")
    if not getattr(args, "skip_eval", False) and len(models) > 1 and not getattr(args, "skip_joint_eval", False):
        lines.append("")
        lines.append("Phase B — joint multi-model Exp.2 table")
        lines.append(f"  models: {models}")
    if not getattr(args, "skip_exp3", False):
        exp3_only = getattr(args, "exp3_only", None)
        lines.append("")
        if exp3_only:
            lines.append(f"Phase C — Exp.3 step ablation (per model, ONLY condition={exp3_only})")
        else:
            lines.append("Phase C — Exp.3 step ablation (per model)")
        for m in models:
            spec = load_spec(m)
            lines.append(f"  [{m}]")
            if exp3_only:
                lines.append(f"    - extract + eval condition '{exp3_only}' only")
            else:
                lines.append("    - extract fixed-50, fixed-100, early-stopping (dynamic)")
                lines.append("    - strict eval proposed_all + proposed_en for 20/50/100/Early")
            lines.append(f"    - write results/exp3_step_ablation_{spec.short_name}/")
    return lines


def cmd_plan(args: argparse.Namespace) -> None:
    for line in describe_plan(args):
        print(line)


def cmd_full(args: argparse.Namespace) -> None:
    models = plan_models(args)
    state = load_state(HERE)
    results: List[StageResult] = []
    gpus = args.gpus
    csv_dir = resolve_csv_dir(args.from_csv_dir or DEFAULT_CSV)

    log("=" * 72)
    log("PAPER EXPERIMENT PIPELINE")
    for line in describe_plan(args):
        log(line)
    log(f"from_csv_dir={csv_dir}")
    log(f"dry_run={args.dry_run}")
    log("=" * 72)

    # ---- Phase A: per-model main path
    for model in models:
        log(f"\n######## MODEL {model} ########")
        if not args.skip_train:
            results.append(
                stage_train(
                    python=args.python,
                    model=model,
                    from_csv_dir=csv_dir,
                    force_train=args.force_train,
                    dry_run=args.dry_run,
                    seed=args.seed,
                )
            )
        if not args.skip_extract:
            spec = load_spec(model)
            results.append(
                stage_extract_fixed(
                    python=args.python,
                    model=model,
                    fixed_steps=20,
                    features_root=HERE / spec.default_features_root,
                    gpus=gpus,
                    min_free_gib=args.min_free_gib,
                    sample_shards=args.sample_shards,
                    fresh=args.fresh,
                    n_per_group=args.n_per_group,
                    lr=args.lr,
                    seed=args.seed,
                    dry_run=args.dry_run,
                    skip_if_ready=not args.fresh,
                )
            )
        if not args.skip_baselines:
            results.extend(
                stage_baselines(
                    python=args.python,
                    model=model,
                    n_per_group=args.n_per_group,
                    seed=args.seed,
                    dry_run=args.dry_run,
                    which=args.baseline_which,
                    lora_fast=args.lora_fast,
                )
            )
        if not args.skip_exp1:
            results.append(stage_exp1(python=args.python, model=model, dry_run=args.dry_run))
        if not args.skip_eval:
            results.append(
                stage_eval(
                    python=args.python,
                    model=model,
                    repeats=args.repeats,
                    seed=args.seed,
                    n_jobs=args.n_jobs,
                    dry_run=args.dry_run,
                    with_baselines=not args.skip_baselines,
                )
            )
        record(state, results)

    # ---- Phase B: multi-model joint table
    if not args.skip_eval and len(models) > 1 and not args.skip_joint_eval:
        log("\n######## JOINT MULTI-MODEL EVAL (Exp.2) ########")
        results.append(
            stage_eval_multimodel(
                python=args.python,
                models=models,
                repeats=args.repeats,
                seed=args.seed,
                n_jobs=args.n_jobs,
                dry_run=args.dry_run,
                with_baselines=not args.skip_baselines,
            )
        )
        record(state, results)

    # ---- Phase C: Exp.3 step ablation (per model in `models`)
    if not args.skip_exp3:
        exp3_only = getattr(args, "exp3_only", None)
        for model in models:
            label = f" [only {exp3_only}]" if exp3_only else ""
            log(f"\n######## EXPERIMENT 3: STEPS ABLATION ({model}){label} ########")
            results.extend(
                stage_exp3_extract(
                    python=args.python,
                    model=model,
                    gpus=gpus,
                    min_free_gib=args.min_free_gib,
                    sample_shards=args.sample_shards,
                    fresh=args.fresh,
                    n_per_group=args.n_per_group,
                    lr=args.lr,
                    seed=args.seed,
                    dry_run=args.dry_run,
                    only=exp3_only,
                )
            )
            results.extend(
                stage_exp3_eval(
                    python=args.python,
                    model=model,
                    repeats=args.repeats,
                    seed=args.seed,
                    n_jobs=args.n_jobs,
                    n_per_group=args.n_per_group,
                    dry_run=args.dry_run,
                    only=exp3_only,
                )
            )
            record(state, results)

    log("\n" + "=" * 72)
    log("PIPELINE FINISHED")
    for r in results:
        log(f"  [{r.status:8s}] {r.name:42s} {r.detail}")
    log(f"State: {state_path(HERE)}")
    log("=" * 72)


def cmd_status(args: argparse.Namespace) -> None:
    st = load_state(HERE)
    print(f"state file: {state_path(HERE)}")
    if not st.stages:
        print("(no stages recorded yet)")
    for s in st.stages:
        print(f"  [{s.get('status', '?'):8s}] {s.get('name')}  {s.get('detail', '')}")

    print("\nReadiness:")
    for model in plan_models(args):
        try:
            spec = load_spec(model)
            adapter = adapter_exists(HERE / spec.default_run_dir)
            ready20 = features_ready(
                HERE / spec.default_features_root,
                condition_prefix="fixed_attention_20",
                n_per_group=args.n_per_group,
            )
            print(
                f"  {model:14s} adapter={'yes' if adapter else 'NO '} "
                f"features20={'yes' if ready20 else 'NO '}"
            )
        except Exception as exc:
            print(f"  {model}: error {exc}")

    print("\nExp.3 feature dirs (per model):")
    for model in plan_models(args):
        print(f"  [{model}]")
        for label, _steps, feat_rel, prefix in exp3_settings_for_model(model):
            root = HERE / feat_rel
            ready = features_ready(root, condition_prefix=prefix, n_per_group=args.n_per_group)
            print(f"    {label:10s} {prefix:22s} ready={'yes' if ready else 'NO '}  ({root.name})")


def cmd_stop(args: argparse.Namespace) -> None:
    for model in plan_models(args):
        cmd = [args.python, str(HERE / "orchestrate.py"), "stop", "--model", model]
        run_cmd(cmd, cwd=HERE, dry_run=args.dry_run, check=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run all paper experiments (train / extract / baselines / exp1 / eval / exp3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_shared(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--python", default=sys.executable)
        sp.add_argument(
            "--models",
            nargs="+",
            default=[],
            help="Subset of models (default: paper set). "
            f"Choices: {', '.join(PAPER_MODELS_ALL)}",
        )
        sp.add_argument("--skip-appendix", action="store_true", help="Skip GPT-Neo appendix model")
        sp.add_argument("--from-csv-dir", default=DEFAULT_CSV)
        sp.add_argument("--force-train", action="store_true", help="Retrain even if adapter exists")
        sp.add_argument("--skip-train", action="store_true")
        sp.add_argument("--skip-extract", action="store_true")
        sp.add_argument("--skip-baselines", action="store_true")
        sp.add_argument("--skip-exp1", action="store_true")
        sp.add_argument("--skip-eval", action="store_true")
        sp.add_argument("--skip-joint-eval", action="store_true")
        sp.add_argument("--skip-exp3", action="store_true", help="Skip step ablation (50/100/dynamic)")
        sp.add_argument(
            "--exp3-only",
            choices=["50", "100", "dynamic"],
            default=None,
            help="Restrict Phase C to a single condition (for delegating one "
            "condition to a separate host without racing the others); "
            "skips writing the cross-condition combined table.",
        )
        sp.add_argument("--fresh", action="store_true", help="Re-extract even if features exist")
        sp.add_argument("--gpus", nargs="+", default=["auto"])
        sp.add_argument("--min-free-gib", type=float, default=8.0)
        sp.add_argument("--sample-shards", type=int, default=1)
        sp.add_argument("--n-per-group", type=int, default=500)
        sp.add_argument("--lr", type=float, default=1e-5, help="Additional-training lr")
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--repeats", type=int, default=10, help="Eval repeats (paper: 10)")
        sp.add_argument("--n-jobs", type=int, default=4, help="Parallel workers for Proposed+EN")
        sp.add_argument(
            "--baseline-which",
            nargs="+",
            default=["lora_leak", "attenmia"],
            choices=["lora_leak", "attenmia"],
        )
        sp.add_argument("--lora-fast", action="store_true", help="LoRA-Leak --fast (no GradNormx)")
        sp.add_argument("--dry-run", action="store_true")

    for name, help_text in (
        ("full", "Run the full paper pipeline"),
        ("plan", "Print planned stages without running"),
        ("status", "Show pipeline state and feature readiness"),
        ("stop", "Stop extract jobs for paper models"),
    ):
        sp = sub.add_parser(name, help=help_text)
        add_shared(sp)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.models:
        from model_registry import normalize_model_key

        args.models = [normalize_model_key(m) for m in args.models]

    if args.command == "full":
        cmd_full(args)
    elif args.command == "plan":
        cmd_plan(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "stop":
        cmd_stop(args)
    else:
        parser.error(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
