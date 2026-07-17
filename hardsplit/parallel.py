# -*- coding: utf-8 -*-
"""Parallelism helpers: sample shards and optional joblib eval repeats."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, TypeVar

T = TypeVar("T")


def map_repeats(
    fn: Callable[[int], T],
    repeats: int,
    *,
    n_jobs: int = 1,
    prefer: str = "processes",
) -> List[T]:
    """Run ``fn(repeat)`` for repeat=1..repeats, optionally in parallel."""
    if n_jobs is None or n_jobs == 0:
        n_jobs = 1
    if n_jobs == 1 or repeats <= 1:
        return [fn(r) for r in range(1, repeats + 1)]
    try:
        from joblib import Parallel, delayed
    except ImportError:
        return [fn(r) for r in range(1, repeats + 1)]
    return Parallel(n_jobs=n_jobs, prefer=prefer)(delayed(fn)(r) for r in range(1, repeats + 1))


def plan_group_shards(
    groups: Sequence[str],
    gpus: Sequence[int],
    sample_shards: int,
) -> List[dict]:
    """Assign (group, shard_index, shard_total, gpu) jobs.

    Strategy:
      - sample_shards <= 1: one job per group (round-robin GPUs)
      - else: for each group, launch sample_shards jobs (round-robin GPUs)
    """
    jobs: List[dict] = []
    if not gpus:
        return jobs
    shards = max(1, int(sample_shards))
    gi = 0
    for group in groups:
        for s in range(shards):
            jobs.append(
                {
                    "group": group,
                    "shard_index": s,
                    "shard_total": shards,
                    "gpu": int(gpus[gi % len(gpus)]),
                }
            )
            gi += 1
    return jobs
