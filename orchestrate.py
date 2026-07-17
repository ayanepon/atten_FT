#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orchestrator for BlackboxNLP hard-split attention experiments.

Stages:
  0. train     — LoRA FT on MIMIR hard split (--model preset)
  1. extract   — sample-wise attention features (multi-GPU, resume-safe)
  2. baselines — LoRA-Leak / AttenMIA feature scoring
  3. exp1      — Mann–Whitney + BH-FDR + Cliff's δ
  4. eval      — Proposed / Proposed+EN / loss baselines (+ optional LoRA/AttenMIA)
  5. all       — (optional train) → extract → wait → (optional baselines) → exp1 → eval
  6. status    — show job / sample progress
  7. stop      — kill running extraction jobs for this features root

Models (via --model):
  pythia-1b | pythia-410m | gpt-neo-2.7b

Examples:
  # Multi-GPU re-extract fixed-20 for all groups, then analyze when done
  python orchestrate.py all \\
      --model pythia-1b \\
      --fixed-steps 20 --lr 1e-5 --gpus auto

  # Train + extract + eval for Pythia-410M (reuse 1B split CSVs)
  python orchestrate.py all --model pythia-410m --do-train \\
      --from-csv-dir mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2/data \\
      --gpus auto --fresh

  # GPT-Neo-2.7B
  python orchestrate.py all --model gpt-neo-2.7b --gpus auto --min-free-gib 20

  # Status / eval / stop
  python orchestrate.py status --model pythia-410m
  python orchestrate.py eval --model pythia-410m
  python orchestrate.py stop --model pythia-410m
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence


HERE = Path(__file__).resolve().parent
GROUPS = ("ft", "pt", "unseen")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def resolve_under(base: Path, path_like: str) -> Path:
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p
    cand = (base / p).resolve()
    if cand.exists() or not (Path.cwd() / p).exists():
        return cand
    return (Path.cwd() / p).resolve()


@dataclass
class JobSpec:
    group: str
    gpu: int
    output_dir: str
    pid: Optional[int] = None
    log_file: str = ""
    status: str = "pending"  # pending|running|completed|failed


def free_gpus(min_free_gib: float = 8.0) -> List[int]:
    """Return GPU indices sorted by free memory descending."""
    try:
        import torch
    except ImportError:
        # fallback: assume single GPU 0
        return [0]
    if not torch.cuda.is_available():
        return []
    scored = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        free_gib = free / (1024**3)
        if free_gib >= min_free_gib:
            scored.append((free_gib, i))
    scored.sort(reverse=True)
    return [i for _, i in scored]


def orchestrator_dir(features_root: Path) -> Path:
    d = features_root / "orchestrator"
    d.mkdir(parents=True, exist_ok=True)
    return d


def jobs_path(features_root: Path) -> Path:
    return orchestrator_dir(features_root) / "jobs.json"


def save_jobs(features_root: Path, jobs: List[JobSpec]) -> None:
    path = jobs_path(features_root)
    path.write_text(json.dumps([asdict(j) for j in jobs], indent=2), encoding="utf-8")


def load_jobs(features_root: Path) -> List[JobSpec]:
    path = jobs_path(features_root)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [JobSpec(**item) for item in data]


def pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def sample_count(output_dir: Path) -> int:
    f = output_dir / "sample_level_experiment4.csv"
    if not f.exists() or f.stat().st_size < 2:
        return 0
    # header + rows
    with f.open("r", encoding="utf-8", errors="ignore") as handle:
        n = sum(1 for _ in handle)
    return max(0, n - 1)


def run_status(output_dir: Path) -> str:
    f = output_dir / "run_status.txt"
    if not f.exists():
        return "missing"
    first = f.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    return first[0] if first else "empty"


def refresh_job_status(job: JobSpec, expected_n: int) -> JobSpec:
    out = Path(job.output_dir)
    n = sample_count(out)
    st = run_status(out)
    if n >= expected_n and st.startswith(("completed", "extraction_completed")):
        job.status = "completed"
    elif pid_alive(job.pid):
        job.status = "running"
    elif job.pid and not pid_alive(job.pid):
        # finished process but incomplete samples
        job.status = "completed" if n >= expected_n else "failed"
    elif n > 0:
        job.status = "partial"
    else:
        job.status = job.status if job.status != "running" else "failed"
    return job


def apply_model_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill run-dir / features-root / model-name / output dirs from --model preset."""
    try:
        from model_registry import apply_model_namespace
    except ImportError:
        return args
    return apply_model_namespace(
        args,
        profile="pipeline",
        fill_baselines=bool(getattr(args, "with_baselines", False)),
        log=log,
    )


def cmd_train(args: argparse.Namespace) -> None:
    """Train LoRA FT adapter for the selected model."""
    args = apply_model_defaults(args)
    cmd = [
        args.python,
        str(HERE / "train_mimir_wikipedia_hardsplit_lora.py"),
    ]
    if getattr(args, "model", ""):
        cmd.extend(["--model", args.model])
    elif getattr(args, "model_name", ""):
        cmd.extend(["--model-name", args.model_name])
    if args.run_dir:
        cmd.extend(["--output-dir", args.run_dir])
    if getattr(args, "from_csv_dir", ""):
        cmd.extend(["--from-csv-dir", args.from_csv_dir])
    cmd.extend(["--seed", str(args.seed)])
    log("train LoRA FT: " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(HERE))


def launch_extract_job(
    *,
    python: str,
    group: str,
    gpu: int,
    run_dir: Path,
    features_root: Path,
    fixed_steps: int,
    n_per_group: int,
    lr: float,
    seed: int,
    resume: bool,
    model_name: str = "",
    shard_index: int = 0,
    shard_total: int = 1,
    output_dir: Optional[Path] = None,
    query_position_offset: int = 1,
    query_selection: str = "top_loss",
) -> JobSpec:
    if output_dir is None:
        if shard_total > 1:
            out = features_root / f"fixed_attention_{fixed_steps}_{group}_shard{shard_index}"
        else:
            out = features_root / f"fixed_attention_{fixed_steps}_{group}"
    else:
        out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_dir = orchestrator_dir(features_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    shard_tag = f"_s{shard_index}of{shard_total}" if shard_total > 1 else ""
    log_file = log_dir / f"extract_{group}_steps{fixed_steps}{shard_tag}_gpu{gpu}.log"

    cmd = [
        python,
        str(HERE / "extract_attention_hardsplit.py"),
        "--run-dir",
        str(run_dir),
        "--adapter-dir",
        str(run_dir / "adapter"),
        "--output-dir",
        str(out),
        "--no-run-dynamic",
        "--fixed-steps",
        str(fixed_steps),
        "--groups",
        group,
        "--n-per-group",
        str(n_per_group),
        "--lr",
        str(lr),
        "--seed",
        str(seed),
        "--query-position-offset",
        str(query_position_offset),
        "--query-selection",
        query_selection,
        "--skip-analyze",
        "--resume" if resume else "--no-resume",
        "--flush-every",
        "25",
    ]
    if shard_total > 1:
        cmd.extend(["--shard", f"{shard_index}/{shard_total}"])
    if model_name:
        cmd.extend(["--model-name", model_name])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OVERFIT_LR"] = str(lr)
    env["FLUSH_EVERY"] = env.get("FLUSH_EVERY", "25")
    env["MIMIR_HARDSPLIT_RUN_DIR"] = str(run_dir)
    env["MIMIR_HARDSPLIT_ADAPTER_DIR"] = str(run_dir / "adapter")
    if model_name:
        env["BASE_MODEL_NAME"] = model_name

    log_handle = open(log_file, "a", encoding="utf-8")
    log_handle.write(f"\n==== launch {datetime.now().isoformat()} ====\n")
    log_handle.write("CMD: " + " ".join(cmd) + "\n")
    log_handle.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(HERE),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    job = JobSpec(
        group=group if shard_total <= 1 else f"{group}:s{shard_index}/{shard_total}",
        gpu=gpu,
        output_dir=str(out),
        pid=proc.pid,
        log_file=str(log_file),
        status="running",
    )
    log(f"launched group={group} shard={shard_index}/{shard_total} gpu={gpu} pid={proc.pid} out={out}")
    return job


def merge_extract_shards(
    features_root: Path,
    groups: Sequence[str],
    fixed_steps: int,
    sample_shards: int,
) -> None:
    """Merge per-shard extract outputs into the canonical group directories."""
    if sample_shards <= 1:
        return
    from hardsplit.progress import merge_csv_shards

    for group in groups:
        final = features_root / f"fixed_attention_{fixed_steps}_{group}"
        final.mkdir(parents=True, exist_ok=True)
        shard_dirs = [
            features_root / f"fixed_attention_{fixed_steps}_{group}_shard{i}"
            for i in range(sample_shards)
        ]
        for filename in (
            "raw_experiment4_attention_shift.csv",
            "sample_level_experiment4.csv",
            "experiment4_target_samples.csv",
        ):
            paths = [d / filename for d in shard_dirs]
            n = merge_csv_shards(paths, final / filename)
            log(f"merged {group}/{filename}: {n} rows -> {final / filename}")
        (final / "run_status.txt").write_text(
            "extraction_completed\n"
            f"merged_from_shards={sample_shards}\n",
            encoding="utf-8",
        )


def cmd_extract(args: argparse.Namespace) -> None:
    args = apply_model_defaults(args)
    features_root = resolve_under(HERE, args.features_root)
    run_dir = resolve_under(HERE, args.run_dir)
    features_root.mkdir(parents=True, exist_ok=True)

    if not (run_dir / "adapter" / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"LoRA adapter not found under {run_dir / 'adapter'}. "
            f"Train first, e.g. python train_mimir_wikipedia_hardsplit_lora.py --model {getattr(args, 'model', 'pythia-410m')}"
        )

    gpus = free_gpus(args.min_free_gib) if args.gpus == ["auto"] else [int(x) for x in args.gpus]
    if not gpus:
        raise RuntimeError("No GPUs available (check CUDA / --min-free-gib / --gpus)")

    model_name = getattr(args, "model_name", "") or ""
    sample_shards = max(1, int(getattr(args, "sample_shards", 1) or 1))
    log(f"features_root={features_root}")
    log(f"run_dir={run_dir}")
    log(f"model_name={model_name or '(infer from adapter)'}")
    log(f"gpus={gpus}")
    log(
        f"groups={args.groups} fixed_steps={args.fixed_steps} lr={args.lr} "
        f"sample_shards={sample_shards} query={args.query_selection}/t+{args.query_position_offset}"
    )

    from hardsplit.parallel import plan_group_shards

    plan = plan_group_shards(args.groups, gpus, sample_shards)
    jobs: List[JobSpec] = []
    for item in plan:
        group = item["group"]
        s_idx = int(item["shard_index"])
        s_tot = int(item["shard_total"])
        gpu = int(item["gpu"])
        if args.fresh:
            if s_tot > 1:
                out = features_root / f"fixed_attention_{args.fixed_steps}_{group}_shard{s_idx}"
            else:
                out = features_root / f"fixed_attention_{args.fixed_steps}_{group}"
            for name in [
                "raw_experiment4_attention_shift.csv",
                "sample_level_experiment4.csv",
                "sample_level_experiment4_features.csv",
                "run_status.txt",
            ]:
                p = out / name
                if p.exists():
                    p.unlink()
        job = launch_extract_job(
            python=args.python,
            group=group,
            gpu=gpu,
            run_dir=run_dir,
            features_root=features_root,
            fixed_steps=args.fixed_steps,
            n_per_group=args.n_per_group,
            lr=args.lr,
            seed=args.seed,
            resume=not args.fresh and not args.no_resume,
            model_name=model_name,
            shard_index=s_idx,
            shard_total=s_tot,
            query_position_offset=args.query_position_offset,
            query_selection=args.query_selection,
        )
        jobs.append(job)
        time.sleep(args.stagger_sec)

    save_jobs(features_root, jobs)
    log(f"saved job table: {jobs_path(features_root)}")
    if args.wait:
        cmd_wait(args)
        merge_extract_shards(features_root, args.groups, args.fixed_steps, sample_shards)


def cmd_status(args: argparse.Namespace) -> None:
    args = apply_model_defaults(args)
    features_root = resolve_under(HERE, args.features_root)
    jobs = load_jobs(features_root)
    print(f"features_root: {features_root}")
    print(f"{'group':8} {'gpu':>4} {'status':12} {'samples':>8} {'pid':>8} log")
    if not jobs:
        # still show feature dirs if present
        for group in GROUPS:
            out = features_root / f"fixed_attention_{args.fixed_steps}_{group}"
            n = sample_count(out)
            st = run_status(out)
            print(f"{group:8} {'-':>4} {st:12} {n:8} {'-':>8} -")
        return

    updated = []
    for job in jobs:
        job = refresh_job_status(job, args.n_per_group)
        updated.append(job)
        n = sample_count(Path(job.output_dir))
        print(
            f"{job.group:8} {job.gpu:4d} {job.status:12} {n:8d} "
            f"{(job.pid or 0):8d} {Path(job.log_file).name}"
        )
    save_jobs(features_root, updated)

    # free GPUs snapshot
    gpus = free_gpus(0.0)
    if gpus or True:
        try:
            import torch

            if torch.cuda.is_available():
                print("\nGPU free memory:")
                for i in range(torch.cuda.device_count()):
                    free, total = torch.cuda.mem_get_info(i)
                    print(f"  gpu{i}: free={free/1024**3:.1f}GiB / total={total/1024**3:.1f}GiB")
        except Exception:
            pass


def cmd_wait(args: argparse.Namespace) -> None:
    args = apply_model_defaults(args)
    features_root = resolve_under(HERE, args.features_root)
    log(f"waiting for jobs under {features_root}")
    while True:
        jobs = load_jobs(features_root)
        if not jobs:
            # fall back to sample counts
            done = all(
                sample_count(features_root / f"fixed_attention_{args.fixed_steps}_{g}") >= args.n_per_group
                for g in args.groups
            )
            if done:
                log("all groups reached target sample count")
                break
            time.sleep(args.poll_sec)
            continue

        updated = [refresh_job_status(j, args.n_per_group) for j in jobs]
        save_jobs(features_root, updated)
        running = [j for j in updated if j.status in {"running", "pending", "partial"}]
        failed = [j for j in updated if j.status == "failed"]
        completed = [j for j in updated if j.status == "completed"]
        log(
            f"completed={len(completed)} running={len(running)} failed={len(failed)} "
            + " ".join(f"{j.group}:{sample_count(Path(j.output_dir))}" for j in updated)
        )
        if failed and not running:
            raise RuntimeError(f"extraction failed for: {[j.group for j in failed]}")
        if not running and len(completed) == len(updated):
            log("all extraction jobs completed")
            break
        time.sleep(args.poll_sec)


def cmd_stop(args: argparse.Namespace) -> None:
    args = apply_model_defaults(args)
    features_root = resolve_under(HERE, args.features_root)
    jobs = load_jobs(features_root)
    for job in jobs:
        if job.pid and pid_alive(job.pid):
            log(f"stopping pid={job.pid} group={job.group}")
            try:
                os.killpg(job.pid, signal.SIGTERM)
            except Exception:
                try:
                    os.kill(job.pid, signal.SIGTERM)
                except Exception as exc:
                    log(f"  kill failed: {exc}")
        job.status = "stopped"
    save_jobs(features_root, jobs)
    # also pkill by features root path string for safety
    try:
        subprocess.run(
            ["pkill", "-f", f"extract_attention_hardsplit.py.*{features_root.name}"],
            check=False,
        )
    except Exception:
        pass
    log("stop requested")


def cmd_exp1(args: argparse.Namespace) -> None:
    args = apply_model_defaults(args)
    features_root = resolve_under(HERE, args.features_root)
    out = resolve_under(HERE, args.exp1_output_dir)
    cmd = [
        args.python,
        str(HERE / "analyze_exp1_layer_head_significance.py"),
        "--root",
        str(features_root),
        "--output-dir",
        str(out),
    ]
    log("run Exp.1 analysis")
    subprocess.run(cmd, check=True, cwd=str(HERE))


def _model_key(args: argparse.Namespace) -> str:
    try:
        from model_registry import eval_key_from_model

        return eval_key_from_model(
            getattr(args, "model", "") or getattr(args, "model_name", "") or getattr(args, "model_key", ""),
        )
    except ImportError:
        return getattr(args, "model_key", None) or "pythia1b"


def _baseline_output(args: argparse.Namespace, kind: str) -> str:
    """kind: lora_leak | attenmia"""
    attr = "lora_root" if kind == "lora_leak" else "attenmia_root"
    explicit = getattr(args, attr, "") or ""
    if explicit:
        return explicit
    try:
        from model_registry import resolve_model_spec

        flag = getattr(args, "model", "") or getattr(args, "model_name", "")
        if flag:
            spec = resolve_model_spec(flag)
            return spec.default_lora_root if kind == "lora_leak" else spec.default_attenmia_root
    except Exception:
        pass
    key = _model_key(args)
    return f"results/{kind}_{key}"


def cmd_eval(args: argparse.Namespace) -> None:
    args = apply_model_defaults(args)
    features_root = resolve_under(HERE, args.features_root)
    out = resolve_under(HERE, args.eval_output_dir)
    methods = list(args.methods)
    model_key = _model_key(args)
    if args.lora_root and "lora_leak" not in methods:
        methods.append("lora_leak")
    if args.attenmia_root and "attenmia" not in methods:
        methods.append("attenmia")
    cmd = [
        args.python,
        str(HERE / "run_strict_fixed20_comparison_10runs.py"),
        "--models",
        model_key,
        f"--{model_key}-proposed-root",
        str(features_root),
        "--output-dir",
        str(out),
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
        "--methods",
        *methods,
        "--comparisons",
        *args.comparisons,
    ]
    if args.lora_root:
        cmd.extend([f"--{model_key}-lora-root", args.lora_root])
    if args.attenmia_root:
        cmd.extend([f"--{model_key}-attenmia-root", args.attenmia_root])
    log("run strict evaluation")
    subprocess.run(cmd, check=True, cwd=str(HERE))


def cmd_baselines(args: argparse.Namespace) -> None:
    """Run LoRA-Leak and/or AttenMIA feature extraction for the selected model."""
    args = apply_model_defaults(args)
    model_flag = getattr(args, "model", "") or ""
    run_dir = str(resolve_under(HERE, args.run_dir))
    which = getattr(args, "baseline_which", ["lora_leak", "attenmia"])

    # Prefer canonical name; fall back to historical _2 filename.
    lora_script = "run_lora_leak_official_mimir_hardsplit.py"
    if not (HERE / lora_script).exists():
        lora_script = "run_lora_leak_official_mimir_hardsplit_2.py"
    scripts = {
        "lora_leak": lora_script,
        "attenmia": "run_attenmia_official_mimir_hardsplit.py",
    }
    for kind in which:
        out = _baseline_output(args, kind)
        cmd = [
            args.python,
            str(HERE / scripts[kind]),
            "--run-dir",
            run_dir,
            "--adapter-dir",
            run_dir,
            "--output-dir",
            out,
            "--seed",
            str(args.seed),
            "--n-per-group",
            str(args.n_per_group),
        ]
        if model_flag:
            cmd.extend(["--model", model_flag])
        if getattr(args, "model_name", ""):
            cmd.extend(["--model-name", args.model_name])
        log(f"run {kind} baseline → {out}")
        subprocess.run(cmd, check=True, cwd=str(HERE))
        if kind == "lora_leak":
            args.lora_root = out
        else:
            args.attenmia_root = out


def cmd_all(args: argparse.Namespace) -> None:
    args = apply_model_defaults(args)
    if getattr(args, "do_train", False):
        cmd_train(args)
    if not args.skip_extract:
        # always wait after extract in all mode
        args.wait = True
        cmd_extract(args)
    else:
        log("skip extract")
    if getattr(args, "with_baselines", False):
        cmd_baselines(args)
    if not args.skip_exp1:
        cmd_exp1(args)
    else:
        log("skip exp1")
    if not args.skip_eval:
        cmd_eval(args)
    else:
        log("skip eval")
    log("orchestration finished")


def add_common_args(p: argparse.ArgumentParser) -> None:
    try:
        from model_registry import (
            DEFAULT_EVAL_DIR,
            DEFAULT_EXP1_DIR,
            PYTHIA1B_FEATURES_ROOT,
            PYTHIA1B_RUN_DIR,
            add_model_arguments,
        )
    except ImportError:
        DEFAULT_EVAL_DIR = "results/strict_fixed20_unified"
        DEFAULT_EXP1_DIR = "results/exp1_layer_head_stats"
        PYTHIA1B_FEATURES_ROOT = "attention_features_mimir_hardsplit"
        PYTHIA1B_RUN_DIR = "mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2"
        add_model_arguments = None

    p.add_argument("--python", default=sys.executable)
    if add_model_arguments is not None:
        add_model_arguments(p)
    else:
        p.add_argument("--model", default="")
        p.add_argument("--model-name", default="")
    p.add_argument("--run-dir", default=PYTHIA1B_RUN_DIR)
    p.add_argument("--features-root", default=PYTHIA1B_FEATURES_ROOT)
    p.add_argument("--groups", nargs="+", default=list(GROUPS), choices=list(GROUPS))
    p.add_argument("--fixed-steps", type=int, default=20)
    p.add_argument(
        "--query-position-offset",
        type=int,
        choices=[0, 1],
        default=1,
        help="Attention query mapping offset; 1 is canonical t+1, 0 is predictor-state ablation.",
    )
    p.add_argument(
        "--query-selection",
        choices=["top_loss", "low_loss", "random", "all_valid"],
        default="top_loss",
        help="Query selection strategy; top_loss is the canonical paper condition.",
    )
    p.add_argument("--n-per-group", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-5, help="Additional-training lr (correct: 1e-5)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpus", nargs="+", default=["auto"], help="'auto' or GPU ids e.g. 0 2")
    p.add_argument("--min-free-gib", type=float, default=8.0)
    p.add_argument("--stagger-sec", type=float, default=2.0)
    p.add_argument("--poll-sec", type=float, default=60.0)
    p.add_argument("--fresh", action="store_true", help="Delete partial outputs before extract")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--wait", action="store_true", help="Wait for extract jobs (extract command)")
    p.add_argument(
        "--sample-shards",
        type=int,
        default=1,
        help="Split each group across N GPU jobs (sample i goes to shard i%%N). "
        "Use with --wait to auto-merge shard CSVs into fixed_attention_*_{group}/.",
    )
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-exp1", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--exp1-output-dir", default=DEFAULT_EXP1_DIR)
    p.add_argument("--eval-output-dir", default=DEFAULT_EVAL_DIR)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument(
        "--methods",
        nargs="+",
        default=["proposed_all", "proposed_en", "initial_loss", "loss_decrease"],
    )
    p.add_argument(
        "--comparisons",
        nargs="+",
        choices=["ft_vs_pt", "ft_vs_unseen", "pt_vs_unseen"],
        default=["ft_vs_pt", "ft_vs_unseen"],
        help="Binary comparisons for strict evaluation; add pt_vs_unseen for the control experiment.",
    )
    p.add_argument("--lora-root", default="")
    p.add_argument("--attenmia-root", default="")
    p.add_argument(
        "--from-csv-dir",
        default="",
        help="For train: reuse existing hard-split CSVs (e.g. pythia-1b data/)",
    )
    p.add_argument(
        "--do-train",
        action="store_true",
        help="In 'all': also train LoRA FT before extract",
    )
    p.add_argument(
        "--with-baselines",
        action="store_true",
        help="In 'all': also run LoRA-Leak + AttenMIA before eval",
    )
    p.add_argument(
        "--baseline-which",
        nargs="+",
        default=["lora_leak", "attenmia"],
        choices=["lora_leak", "attenmia"],
        help="Which baselines to run (baselines / all --with-baselines)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BlackboxNLP hard-split experiment orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ["train", "extract", "status", "wait", "stop", "exp1", "eval", "baselines", "all"]:
        p = sub.add_parser(name, help=f"{name} stage")
        add_common_args(p)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cmd = args.command
    if cmd == "train":
        cmd_train(args)
    elif cmd == "extract":
        cmd_extract(args)
    elif cmd == "status":
        cmd_status(args)
    elif cmd == "wait":
        cmd_wait(args)
    elif cmd == "stop":
        cmd_stop(args)
    elif cmd == "exp1":
        cmd_exp1(args)
    elif cmd == "eval":
        cmd_eval(args)
    elif cmd == "baselines":
        cmd_baselines(args)
    elif cmd == "all":
        cmd_all(args)
    else:
        parser.error(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
