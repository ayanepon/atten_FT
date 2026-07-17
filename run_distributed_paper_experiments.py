#!/usr/bin/env python3
"""Launch the paper pipeline across hosta, hostb, and hostc.

This controller is intended to run on ``hosta``.  It deliberately uses only
SSH, ``nvidia-smi``, and the existing paper orchestrator on each host:

* hosta: Pythia-1B (main result)
* hostb: Pythia-410M (size analysis)
* hostc: GPT-Neo-2.7B (appendix)

The controller never kills existing processes.  It selects GPUs whose current
free memory exceeds the requested threshold, records the decision, launches a
single-model ``run_paper_experiments.py full --skip-exp3`` job per host, and
polls both process state and GPU availability until all jobs finish.
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
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


DEFAULT_ROOT = "/remote/homes/user/anonymous_experiments"
DEFAULT_PYTHON = "/remote/homes/user/implementation/.venv_hosta/bin/python"
DEFAULT_GPU_API_URL = "https://www.gpu-status.example.edu/api/gpu/status"
MODEL_HOSTS = {
    "hosta": "pythia-1b",
    "hostb": "pythia-410m",
    "hostc": "gpt-neo-2.7b",
}


@dataclass(frozen=True)
class GPUInfo:
    index: int
    free_mib: int
    utilization: int
    process_count: int = 0


@dataclass
class Job:
    host: str
    model: str
    gpus: List[int]
    pid: int
    log_path: str


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def ssh(host: str, command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command],
        check=check,
        text=True,
        capture_output=True,
    )


def parse_gpu_api_payload(
    payload: Dict[str, object],
    host: str,
    *,
    max_age_sec: float = 120.0,
    allow_stale: bool = False,
) -> List[GPUInfo]:
    """Extract one host's current GPU state from the lab status API."""
    servers = payload.get("servers", [])
    if not isinstance(servers, list):
        raise RuntimeError("GPU API response has no server list")
    server = next(
        (
            item
            for item in servers
            if isinstance(item, dict)
            and (
                item.get("id") == host
                or str(item.get("hostname", "")).split(".")[0] == host
            )
        ),
        None,
    )
    if not isinstance(server, dict):
        raise RuntimeError(f"GPU API has no entry for {host}")
    if server.get("status") != "ok":
        raise RuntimeError(f"GPU API reports {host} status={server.get('status')}")
    age = float(server.get("age_seconds", 0.0))
    if age > max_age_sec and not allow_stale:
        raise RuntimeError(
            f"GPU API response for {host} is stale ({age:.1f}s > {max_age_sec:.1f}s)"
        )
    raw_gpus = server.get("gpus", [])
    if not isinstance(raw_gpus, list):
        raise RuntimeError(f"GPU API has no GPU list for {host}")
    rows: List[GPUInfo] = []
    for gpu in raw_gpus:
        if not isinstance(gpu, dict):
            continue
        processes = gpu.get("processes", [])
        rows.append(
            GPUInfo(
                index=int(gpu["gpu_index"]),
                free_mib=int(gpu["memory_free_mib"]),
                utilization=int(round(float(gpu.get("utilization_percent", 0.0)))),
                process_count=len(processes) if isinstance(processes, list) else 0,
            )
        )
    if not rows:
        raise RuntimeError(f"GPU API returned no GPUs for {host}")
    return rows


def gpu_snapshot(
    host: str,
    *,
    api_url: str = DEFAULT_GPU_API_URL,
    max_age_sec: float = 120.0,
    allow_stale: bool = False,
    allow_nvidia_fallback: bool = False,
) -> List[GPUInfo]:
    """Read live GPU state from the lab API; never silently fall back to stale SSH data."""
    request = urllib.request.Request(
        api_url,
        headers={"Accept": "application/json", "User-Agent": "blackboxnlp-orchestrator/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        if not allow_nvidia_fallback:
            raise RuntimeError(f"Could not read GPU status API {api_url}: {exc}") from exc
        log(f"GPU API unavailable for {host} ({exc}); using explicit nvidia-smi fallback")
        return nvidia_smi_snapshot(host)
    if not isinstance(payload, dict):
        raise RuntimeError("GPU API response is not an object")
    try:
        return parse_gpu_api_payload(
            payload,
            host,
            max_age_sec=max_age_sec,
            allow_stale=allow_stale,
        )
    except RuntimeError:
        if not allow_nvidia_fallback:
            raise
        log(f"GPU API data invalid for {host}; using explicit nvidia-smi fallback")
        return nvidia_smi_snapshot(host)


def nvidia_smi_snapshot(host: str) -> List[GPUInfo]:
    """Fallback only for hosts that cannot reach the API (must be explicitly enabled)."""
    query = (
        "nvidia-smi --query-gpu=index,memory.free,utilization.gpu "
        "--format=csv,noheader,nounits"
    )
    result = ssh(host, query)
    rows: List[GPUInfo] = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) < 3:
            continue
        rows.append(GPUInfo(int(row[0]), int(row[1]), int(row[2])))
    if not rows:
        raise RuntimeError(f"No GPU information returned by {host}: {result.stderr}")
    return rows


def format_gpus(gpus: Iterable[GPUInfo]) -> str:
    return ", ".join(
        f"{g.index}:{g.free_mib / 1024:.1f}GiB/{g.utilization}%/{g.process_count}proc"
        for g in gpus
    )


def select_gpus(
    host: str,
    min_free_gib: float,
    max_gpus: int,
    *,
    api_url: str = DEFAULT_GPU_API_URL,
    api_max_age_sec: float = 120.0,
    allow_stale_api: bool = False,
    allow_nvidia_fallback: bool = False,
    max_util_percent: int = 80,
) -> List[int]:
    infos = gpu_snapshot(
        host,
        api_url=api_url,
        max_age_sec=api_max_age_sec,
        allow_stale=allow_stale_api,
        allow_nvidia_fallback=allow_nvidia_fallback,
    )
    log(f"GPU check {host}: {format_gpus(infos)}")
    eligible = [
        g
        for g in infos
        if g.free_mib >= min_free_gib * 1024
        and g.utilization <= max_util_percent
        and g.process_count == 0
    ]
    eligible.sort(key=lambda g: (g.utilization, -g.free_mib, g.index))
    return [g.index for g in eligible[:max_gpus]]


def wait_for_gpus(
    host: str,
    *,
    min_free_gib: float,
    max_gpus: int,
    poll_sec: int,
    wait: bool,
    api_url: str,
    api_max_age_sec: float,
    allow_stale_api: bool,
    allow_nvidia_fallback: bool,
    max_util_percent: int,
) -> List[int]:
    while True:
        gpus = select_gpus(
            host,
            min_free_gib,
            max_gpus,
            api_url=api_url,
            api_max_age_sec=api_max_age_sec,
            allow_stale_api=allow_stale_api,
            allow_nvidia_fallback=allow_nvidia_fallback,
            max_util_percent=max_util_percent,
        )
        if gpus:
            return gpus
        if not wait:
            raise RuntimeError(
                f"{host} has no GPU with >= {min_free_gib:.1f} GiB free; "
                "rerun with --wait-for-gpu"
            )
        log(f"{host}: no eligible GPU; retrying in {poll_sec}s")
        time.sleep(poll_sec)


def existing_pipeline(host: str, root: str) -> str:
    """Return existing pipeline processes for this root, excluding the probe itself."""
    pattern = "(run_paper_experiments|orchestrate|extract_attention_hardsplit|train_mimir_wikipedia_hardsplit_lora)\\.py"
    command = (
        f"pgrep -af '{pattern}' | grep -F {shlex.quote(root)} "
        "| grep -v 'pgrep -af' || true"
    )
    result = ssh(host, command, check=False)
    return result.stdout.strip()


def paper_runner_path(root: str) -> str:
    """Support both the local ``data/`` tree and the flattened shared server tree."""
    candidates = [
        Path(root) / "data" / "run_paper_experiments.py",
        Path(root) / "run_paper_experiments.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    # Keep a useful path in dry-run/error messages when the shared mount is unavailable.
    return str(candidates[0])


def launch_job(
    *,
    host: str,
    model: str,
    root: str,
    python: str,
    gpus: Sequence[int],
    min_free_gib: float,
    n_jobs: int,
    repeats: int,
    seed: int,
) -> Job:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"{root}/results/distributed_logs/{timestamp}_{model.replace('.', '_')}.log"
    gpu_text = ",".join(str(i) for i in gpus)
    runner = paper_runner_path(root)
    command = [
        python,
        runner,
        "full",
        "--models",
        model,
        "--skip-exp3",
        "--gpus",
        "auto",
        "--min-free-gib",
        str(min_free_gib),
        "--repeats",
        str(repeats),
        "--n-jobs",
        str(n_jobs),
        "--seed",
        str(seed),
    ]
    quoted = " ".join(shlex.quote(x) for x in command)
    remote = (
        f"mkdir -p {shlex.quote(os.path.dirname(log_path))} && "
        f"cd {shlex.quote(root)} && "
        f"nohup env CUDA_VISIBLE_DEVICES={shlex.quote(gpu_text)} "
        f"{quoted} > {shlex.quote(log_path)} 2>&1 < /dev/null & "
        "echo $!"
    )
    result = ssh(host, remote)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines or not lines[-1].isdigit():
        raise RuntimeError(f"Could not obtain PID from {host}: {result.stdout} {result.stderr}")
    pid = int(lines[-1])
    log(f"launched {model} on {host}: pid={pid}, CUDA_VISIBLE_DEVICES={gpu_text}")
    return Job(host=host, model=model, gpus=list(gpus), pid=pid, log_path=log_path)


def job_alive(job: Job) -> bool:
    result = ssh(job.host, f"kill -0 {job.pid} >/dev/null 2>&1", check=False)
    return result.returncode == 0


def tail_log(job: Job, lines: int = 3) -> str:
    result = ssh(job.host, f"tail -n {lines} {shlex.quote(job.log_path)}", check=False)
    return result.stdout.strip().replace("\n", " | ")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distribute the paper pipeline across the three GPU hosts."
    )
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Shared project root on all hosts")
    parser.add_argument("--python", default=DEFAULT_PYTHON, help="Python executable on all hosts")
    parser.add_argument("--min-free-gib", type=float, default=16.0)
    parser.add_argument("--max-util-percent", type=int, default=80)
    parser.add_argument("--gpu-api-url", default=DEFAULT_GPU_API_URL)
    parser.add_argument("--gpu-api-max-age-sec", type=float, default=120.0)
    parser.add_argument(
        "--allow-stale-gpu-api",
        action="store_true",
        help="Allow API data older than --gpu-api-max-age-sec (unsafe; normally reject).",
    )
    parser.add_argument(
        "--allow-nvidia-smi-fallback",
        action="store_true",
        help="Use host nvidia-smi only when the API is unreachable; API remains the default.",
    )
    parser.add_argument("--max-gpus-per-host", type=int, default=2)
    parser.add_argument("--poll-sec", type=int, default=60)
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--no-wait", dest="wait_for_gpu", action="store_false")
    parser.set_defaults(wait_for_gpu=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hosts", nargs="+", default=list(MODEL_HOSTS))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    unknown = [host for host in args.hosts if host not in MODEL_HOSTS]
    if unknown:
        raise SystemExit(f"Unknown host(s): {', '.join(unknown)}")
    if os.uname().nodename != "hosta" and not args.dry_run:
        raise SystemExit(
            "This controller must run on hosta; use --dry-run for local validation."
        )

    preflight: List[tuple[str, str, List[int], str]] = []
    for host in args.hosts:
        model = MODEL_HOSTS[host]
        existing = existing_pipeline(host, args.root)
        if existing and not args.dry_run:
            preflight.append((host, model, [], existing))
            continue
        gpus = wait_for_gpus(
            host,
            min_free_gib=args.min_free_gib,
            max_gpus=args.max_gpus_per_host,
            poll_sec=args.poll_sec,
            wait=args.wait_for_gpu,
            api_url=args.gpu_api_url,
            api_max_age_sec=args.gpu_api_max_age_sec,
            allow_stale_api=args.allow_stale_gpu_api,
            allow_nvidia_fallback=args.allow_nvidia_smi_fallback,
            max_util_percent=args.max_util_percent,
        )
        if existing:
            log(f"existing pipeline on {host}: {existing}")
        preflight.append((host, model, gpus, existing))

    conflicts = [(host, existing) for host, _, _, existing in preflight if existing]
    if conflicts and not args.dry_run:
        details = "; ".join(f"{host}: {proc}" for host, proc in conflicts)
        raise SystemExit(
            "Refusing to launch duplicate pipeline jobs. Existing processes: " + details
        )

    jobs: List[Job] = []
    for host, model, gpus, existing in preflight:
        if args.dry_run:
            log(f"dry-run: would launch {model} on {host} with GPUs {gpus}")
            continue
        jobs.append(
            launch_job(
                host=host,
                model=model,
                root=args.root,
                python=args.python,
                gpus=gpus,
                min_free_gib=args.min_free_gib,
                n_jobs=args.n_jobs,
                repeats=args.repeats,
                seed=args.seed,
            )
        )

    if args.dry_run:
        return 0

    while jobs:
        remaining: List[Job] = []
        for job in jobs:
            if job_alive(job):
                remaining.append(job)
                log(f"running {job.model} on {job.host}: {tail_log(job)}")
            else:
                log(f"finished {job.model} on {job.host}: {tail_log(job, lines=8)}")
        jobs = remaining
        if jobs:
            time.sleep(args.poll_sec)
    log("all distributed paper jobs finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
