# -*- coding: utf-8 -*-
"""Incremental progress CSV writers for long-running extract jobs.

Avoids rewriting multi‑MB raw tables on every sample.
"""

from __future__ import annotations

import csv
import os
import socket
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


class ExtractOwnershipError(RuntimeError):
    """Raised when another host already owns extraction into this output_dir."""


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        # e.g. PermissionError: process exists but owned by someone else
        return True
    return True


class IncrementalCSVWriter:
    """Append-only CSV writer with stable header from first row."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames: Optional[List[str]] = None
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    self.fieldnames = list(header)

    def append_rows(self, rows: Sequence[Dict]) -> None:
        if not rows:
            return
        if self.fieldnames is None:
            # Stable column order: prefer first row keys, then union of extras
            keys: List[str] = list(rows[0].keys())
            extra = []
            for r in rows[1:]:
                for k in r.keys():
                    if k not in keys and k not in extra:
                        extra.append(k)
            self.fieldnames = keys + extra
            write_header = True
            mode = "w"
        else:
            write_header = False
            mode = "a"
            # extend header only if new keys appear (rare)
            for r in rows:
                for k in r.keys():
                    if k not in self.fieldnames:
                        # cannot safely extend mid-file; drop unknown keys
                        pass

        with self.path.open(mode, encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in self.fieldnames})


class ExtractProgressStore:
    """Manages raw + sample progress files for extract_attention_hardsplit."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.output_dir / ".extract_owner.lock"
        self._acquire_owner_lock()
        self.raw_path = self.output_dir / "raw_experiment4_attention_shift.csv"
        self.sample_path = self.output_dir / "sample_level_experiment4.csv"
        self.raw_writer = IncrementalCSVWriter(self.raw_path)
        self.sample_writer = IncrementalCSVWriter(self.sample_path)
        self._pending_raw = 0
        self._pending_samples = 0

    def _acquire_owner_lock(self) -> None:
        """Claim output_dir for this (host, pid) so a second host can't write here concurrently.

        NFS multi-host writers race on IncrementalCSVWriter's unsynchronized
        append; claiming the dir up front (with a same-host stale-pid override
        so a crashed/resumed run on the SAME host isn't blocked) turns that
        race into a fail-fast error instead of silently corrupted/duplicated rows.
        """
        host = socket.gethostname()
        pid = os.getpid()
        claim = f"{host}:{pid}:{time.time()}\n"
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, claim.encode("utf-8"))
            os.close(fd)
            return
        except FileExistsError:
            pass
        existing = self.lock_path.read_text(encoding="utf-8").strip()
        parts = existing.split(":")
        prev_host = parts[0] if parts else ""
        prev_pid = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1
        if prev_host == host and prev_pid > 0 and not _pid_alive(prev_pid):
            fd = os.open(self.lock_path, os.O_WRONLY | os.O_TRUNC)
            os.write(fd, claim.encode("utf-8"))
            os.close(fd)
            return
        raise ExtractOwnershipError(
            f"{self.output_dir} is already claimed by another extraction process ({existing!r}). "
            "Refusing to start a second writer against the same output directory -- "
            "remove .extract_owner.lock manually once you've confirmed that process is finished."
        )

    def append_sample(self, metric_rows: List[Dict], sample_row: Dict) -> None:
        self.raw_writer.append_rows(metric_rows)
        self.sample_writer.append_rows([sample_row])
        self._pending_raw += len(metric_rows)
        self._pending_samples += 1

    def flush_status(self, message: str) -> None:
        status = self.output_dir / "run_status.txt"
        status.write_text(message.rstrip() + "\n", encoding="utf-8")


def parse_shard_spec(spec: str) -> tuple[int, int]:
    """Parse ``K/N`` or ``K:N`` into (index, total) with 0 <= K < N."""
    raw = (spec or "").strip()
    if not raw:
        return 0, 1
    if "/" in raw:
        a, b = raw.split("/", 1)
    elif ":" in raw:
        a, b = raw.split(":", 1)
    else:
        raise ValueError(f"Invalid shard spec '{spec}'. Use K/N e.g. 0/4")
    k, n = int(a), int(b)
    if n <= 0 or k < 0 or k >= n:
        raise ValueError(f"Invalid shard spec '{spec}'. Need 0 <= K < N")
    return k, n


def filter_shard_indices(n_items: int, shard_index: int, shard_total: int) -> List[int]:
    """Return indices i where i % shard_total == shard_index."""
    if shard_total <= 1:
        return list(range(n_items))
    return [i for i in range(n_items) if i % shard_total == shard_index]


def merge_csv_shards(shard_paths: Iterable[Path], out_path: Path) -> int:
    """Concatenate shard CSVs with one header. Returns data row count."""
    shard_paths = [Path(p) for p in shard_paths if Path(p).exists() and Path(p).stat().st_size > 0]
    if not shard_paths:
        return 0
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = None
    n_rows = 0
    with out_path.open("w", encoding="utf-8", newline="") as out_f:
        writer = None
        for path in shard_paths:
            with path.open("r", encoding="utf-8", newline="") as in_f:
                reader = csv.DictReader(in_f)
                if reader.fieldnames is None:
                    continue
                if header is None:
                    header = list(reader.fieldnames)
                    writer = csv.DictWriter(out_f, fieldnames=header, extrasaction="ignore")
                    writer.writeheader()
                assert writer is not None
                for row in reader:
                    writer.writerow(row)
                    n_rows += 1
    return n_rows
