# -*- coding: utf-8 -*-
"""Diagnose group confounds and surface-only separability for MIMIR splits.

This script is intentionally model-free.  It checks exact/near duplicate
overlap, basic length statistics, and a train-fold-only character n-gram
baseline for FT/PT, FT/Unseen, and PT/Unseen.  It does not modify canonical
results and writes a self-contained manifest beside its outputs.

Example::

    python data/run_data_confound_diagnostics.py \
      --run-dir data/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2 \
      --output-dir data/results/additional_20260715/data_confound_diagnostics
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors


GROUP_FILES = {
    "ft": "mimir_wikipedia_ft_nonmember.csv",
    "pt": "mimir_wikipedia_pt_member.csv",
    "unseen": "mimir_wikipedia_unseen_nonmember.csv",
}
GROUP_NAMES = {
    "ft": "mimir_wikipedia_nonmember_ft",
    "pt": "mimir_wikipedia_member_pt",
    "unseen": "mimir_wikipedia_nonmember_unseen",
}
COMPARISONS = {
    "ft_vs_pt": ("ft", "pt"),
    "ft_vs_unseen": ("ft", "unseen"),
    "pt_vs_unseen": ("pt", "unseen"),
}


def normalize_text(text: str) -> str:
    """Normalize whitespace and case for exact-duplicate diagnostics."""
    return re.sub(r"\s+", " ", str(text).strip()).casefold()


def text_column(frame: pd.DataFrame) -> str:
    for candidate in ("text", "content", "document"):
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"Could not find text column in {list(frame.columns)}")


def load_groups(
    run_dir: Path,
    n_per_group: int,
    seed: int,
    targets_csv: Path | None = None,
) -> pd.DataFrame:
    """Load analysis groups, preferring the exact evaluated targets when given."""
    if targets_csv is not None:
        if not targets_csv.exists():
            raise FileNotFoundError(targets_csv)
        targets = pd.read_csv(targets_csv)
        required = {"group", "text"}
        missing = required - set(targets.columns)
        if missing:
            raise ValueError(f"Target CSV is missing columns: {sorted(missing)}")
        reverse_names = {value: key for key, value in GROUP_NAMES.items()}
        targets = targets[targets["group"].isin(reverse_names)].copy()
        targets["text"] = targets["text"].astype(str)
        targets = targets.dropna(subset=["text"]).reset_index(drop=True)
        targets["group_key"] = targets["group"].map(reverse_names)
        targets["sample_id"] = targets.groupby("group_key", sort=False).cumcount()
        targets["uid"] = targets["group"] + "::" + targets["sample_id"].astype(str)
        counts = targets.groupby("group_key").size().to_dict()
        missing_groups = set(GROUP_FILES) - set(counts)
        if missing_groups:
            raise ValueError(f"Target CSV has no rows for groups: {sorted(missing_groups)}")
        if n_per_group > 0 and any(counts[key] != n_per_group for key in GROUP_FILES):
            raise ValueError(
                f"Expected exactly {n_per_group} targets per group, got {counts}. "
                "Pass --n-per-group 0 only when a non-canonical target set is intentional."
            )
        keep = ["uid", "sample_id", "group_key", "group", "text"]
        if "original_index" in targets.columns:
            keep.append("original_index")
        return targets[keep].copy()

    parts: List[pd.DataFrame] = []
    for key, filename in GROUP_FILES.items():
        path = run_dir / "data" / filename
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        col = text_column(frame)
        frame = frame.rename(columns={col: "text"})
        frame = frame.dropna(subset=["text"]).copy()
        frame["text"] = frame["text"].astype(str)
        frame["group_key"] = key
        frame["group"] = GROUP_NAMES[key]
        if n_per_group > 0 and len(frame) > n_per_group:
            frame = frame.sample(n=n_per_group, random_state=seed).copy()
        frame = frame.reset_index(drop=True)
        frame["sample_id"] = np.arange(len(frame), dtype=int)
        frame["uid"] = frame["group"] + "::" + frame["sample_id"].astype(str)
        parts.append(frame[["uid", "sample_id", "group_key", "group", "text"]])
    return pd.concat(parts, ignore_index=True)


def length_summary(groups: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, frame in groups.groupby("group_key", sort=True):
        chars = frame["text"].str.len().astype(float)
        tokens = frame["text"].str.split().str.len().astype(float)
        rows.append(
            {
                "group_key": key,
                "group": frame["group"].iloc[0],
                "n": int(len(frame)),
                "char_mean": float(chars.mean()),
                "char_std": float(chars.std(ddof=1)) if len(chars) > 1 else 0.0,
                "token_mean": float(tokens.mean()),
                "token_std": float(tokens.std(ddof=1)) if len(tokens) > 1 else 0.0,
                "unique_normalized_texts": int(frame["text"].map(normalize_text).nunique()),
            }
        )
    return pd.DataFrame(rows)


def exact_duplicate_diagnostics(groups: pd.DataFrame) -> pd.DataFrame:
    normalized = groups["text"].map(normalize_text)
    groups = groups.assign(_norm=normalized, _hash=normalized.map(lambda x: hashlib.sha1(x.encode("utf-8")).hexdigest()))
    rows = []
    for left, right in (("ft", "pt"), ("ft", "unseen"), ("pt", "unseen")):
        a = set(groups.loc[groups["group_key"] == left, "_hash"])
        b = set(groups.loc[groups["group_key"] == right, "_hash"])
        overlap = a & b
        rows.append(
            {
                "comparison": f"{left}_vs_{right}",
                "left_group": left,
                "right_group": right,
                "left_unique_normalized": len(a),
                "right_unique_normalized": len(b),
                "cross_group_exact_duplicate_hashes": len(overlap),
                "cross_group_exact_duplicate_rate_left": len(overlap) / max(len(a), 1),
                "cross_group_exact_duplicate_rate_right": len(overlap) / max(len(b), 1),
            }
        )
    return pd.DataFrame(rows)


def nearest_char_tfidf(groups: pd.DataFrame, ngram_range: Tuple[int, int] = (3, 5)) -> pd.DataFrame:
    """Return nearest cross-group character n-gram cosine similarity."""
    rows = []
    for comparison, (left, right) in COMPARISONS.items():
        left_text = groups.loc[groups["group_key"] == left, "text"].tolist()
        right_text = groups.loc[groups["group_key"] == right, "text"].tolist()
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            min_df=2,
            max_features=30000,
            dtype=np.float32,
        )
        matrix = vectorizer.fit_transform(left_text + right_text)
        left_matrix = matrix[: len(left_text)]
        right_matrix = matrix[len(left_text) :]
        nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
        nn.fit(right_matrix)
        distances, _ = nn.kneighbors(left_matrix)
        similarities = 1.0 - distances[:, 0]
        rows.append(
            {
                "comparison": comparison,
                "left_group": left,
                "right_group": right,
                "n_left": len(left_text),
                "n_right": len(right_text),
                "nearest_char_tfidf_mean": float(np.mean(similarities)),
                "nearest_char_tfidf_median": float(np.median(similarities)),
                "nearest_char_tfidf_max": float(np.max(similarities)),
                "vectorizer_features": int(len(vectorizer.vocabulary_)),
            }
        )
    return pd.DataFrame(rows)


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, max_fpr: float = 0.10) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = fpr <= max_fpr
    return float(np.max(tpr[valid])) if np.any(valid) else 0.0


def surface_baseline(groups: pd.DataFrame, repeats: int, cv_splits: int, seed: int) -> pd.DataFrame:
    rows = []
    for comparison, (positive_key, negative_key) in COMPARISONS.items():
        sub = groups[groups["group_key"].isin([positive_key, negative_key])].reset_index(drop=True)
        y = (sub["group_key"].to_numpy() == positive_key).astype(int)
        texts = sub["text"].tolist()
        for repeat in range(1, repeats + 1):
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed + repeat - 1)
            fold_metrics = []
            for fold, (train_idx, test_idx) in enumerate(cv.split(texts, y), start=1):
                vectorizer = TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=30000,
                    dtype=np.float32,
                )
                x_train = vectorizer.fit_transform([texts[i] for i in train_idx])
                x_test = vectorizer.transform([texts[i] for i in test_idx])
                clf = LogisticRegression(
                    solver="liblinear",
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=seed + repeat * 100 + fold,
                )
                clf.fit(x_train, y[train_idx])
                scores = clf.predict_proba(x_test)[:, 1]
                fold_metrics.append(
                    {
                        "auc": float(roc_auc_score(y[test_idx], scores)),
                        "auprc": float(average_precision_score(y[test_idx], scores)),
                        "tpr_at_10_fpr": tpr_at_fpr(y[test_idx], scores),
                    }
                )
            rows.append(
                {
                    "comparison": comparison,
                    "method": "char_tfidf_logreg",
                    "repeat": repeat,
                    "auc": float(np.mean([x["auc"] for x in fold_metrics])),
                    "auprc": float(np.mean([x["auprc"] for x in fold_metrics])),
                    "tpr_at_10_fpr": float(np.mean([x["tpr_at_10_fpr"] for x in fold_metrics])),
                    "n_positive": int(y.sum()),
                    "n_negative": int((1 - y).sum()),
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="data/mimir_wikipedia_hardsplit_lora_ft_lr1e-4_epoch5_2")
    parser.add_argument("--output-dir", default="data/results/additional_20260715/data_confound_diagnostics")
    parser.add_argument(
        "--targets-csv",
        default=None,
        help="Exact evaluated target CSV. When set, diagnostics use these rows rather than a fresh sample.",
    )
    parser.add_argument("--n-per-group", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    targets_csv = Path(args.targets_csv).expanduser() if args.targets_csv else None
    groups = load_groups(run_dir, args.n_per_group, args.seed, targets_csv=targets_csv)
    length_summary(groups).to_csv(output_dir / "group_length_summary.csv", index=False)
    exact_duplicate_diagnostics(groups).to_csv(output_dir / "exact_duplicate_diagnostics.csv", index=False)
    nearest_char_tfidf(groups).to_csv(output_dir / "near_duplicate_diagnostics.csv", index=False)
    surface_baseline(groups, args.repeats, args.cv_splits, args.seed).to_csv(
        output_dir / "surface_baseline_auc.csv", index=False
    )
    manifest = {
        "script": Path(__file__).name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_dir": str(run_dir.resolve()),
        "targets_csv": str(targets_csv.resolve()) if targets_csv else None,
        "output_dir": str(output_dir.resolve()),
        "groups": GROUP_NAMES,
        "comparisons": COMPARISONS,
        "n_per_group": args.n_per_group,
        "repeats": args.repeats,
        "cv_splits": args.cv_splits,
        "seed": args.seed,
        "diagnostics": ["length", "exact_duplicate", "near_duplicate_char_tfidf", "surface_baseline"],
    }
    (output_dir / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
