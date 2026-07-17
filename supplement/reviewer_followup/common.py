"""Shared, dependency-light utilities for reviewer follow-up experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


GROUP_FACTORS: Dict[str, tuple[int, int]] = {
    "p0f0": (0, 0),
    "p0f1": (0, 1),
    "p1f0": (1, 0),
    "p1f1": (1, 1),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        tmp = Path(handle.name)
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        tmp = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(tmp, path)


def append_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows))
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def load_text_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "text" not in frame.columns:
        raise ValueError(f"{path} must contain a text column")
    frame = frame.copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame[frame["text"].str.len() > 0].reset_index(drop=True)
    frame["text_sha256"] = frame["text"].map(text_sha256)
    return frame


def validate_factorial_targets(frame: pd.DataFrame, *, require_balanced: bool = True) -> Dict[str, Any]:
    required = {"sample_id", "group", "pt_member", "ft_exposed", "text", "text_sha256"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Factorial targets missing columns: {missing}")
    unknown = sorted(set(frame["group"]) - set(GROUP_FACTORS))
    if unknown:
        raise ValueError(f"Unknown factorial groups: {unknown}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("sample_id values are not unique")
    if frame["text_sha256"].duplicated().any():
        duplicates = frame.loc[frame["text_sha256"].duplicated(False), ["group", "sample_id"]]
        raise ValueError(f"Exact text duplicates cross cells: {duplicates.to_dict(orient='records')[:8]}")
    for group, (pt_member, ft_exposed) in GROUP_FACTORS.items():
        sub = frame[frame["group"] == group]
        if sub.empty:
            raise ValueError(f"Missing factorial cell: {group}")
        if set(pd.to_numeric(sub["pt_member"]).astype(int)) != {pt_member}:
            raise ValueError(f"pt_member does not match group {group}")
        if set(pd.to_numeric(sub["ft_exposed"]).astype(int)) != {ft_exposed}:
            raise ValueError(f"ft_exposed does not match group {group}")
    counts = frame["group"].value_counts().sort_index().to_dict()
    if require_balanced and len(set(counts.values())) != 1:
        raise ValueError(f"Factorial cells are not balanced: {counts}")
    return {
        "n_rows": int(len(frame)),
        "group_counts": {str(k): int(v) for k, v in counts.items()},
        "exact_duplicate_count": int(frame["text_sha256"].duplicated().sum()),
    }


def bh_fdr(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    if values.size == 0:
        return values
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def t_interval(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float]:
    from scipy.stats import t

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return (float("nan"), float("nan"))
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / math.sqrt(len(arr)))
    radius = float(t.ppf((1.0 + confidence) / 2.0, len(arr) - 1) * sem)
    return mean - radius, mean + radius


def base_manifest(*, experiment: str, command: Sequence[str] | None = None) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": experiment,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": list(command or []),
        "status": "prepared",
        "canonical_outputs_mutated": False,
    }
