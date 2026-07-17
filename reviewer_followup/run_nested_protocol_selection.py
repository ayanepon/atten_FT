#!/usr/bin/env python3
"""Nested selection across query, token-selection, rho, and step settings."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from reviewer_followup.common import atomic_write_csv, atomic_write_json, base_manifest
from reviewer_followup.evaluation import _train_only_transform, elastic_net_penalty_kwargs, wide_attention


def parse_candidates(values: list[str]) -> dict[str, tuple[Path, str]]:
    result = {}
    for value in values:
        if "=" not in value or "@" not in value:
            raise ValueError(f"candidate must be NAME=RAW_CSV@CONDITION: {value}")
        name, location = value.split("=", 1)
        csv_path, condition = location.rsplit("@", 1)
        result[name] = (Path(csv_path), condition)
    if len(result) < 2:
        raise ValueError("At least two candidate protocols are required")
    return result


def load_candidates(specs: dict[str, tuple[Path, str]]) -> dict[str, pd.DataFrame]:
    raw_by_path = {path: pd.read_csv(path) for path in dict.fromkeys(path for path, _ in specs.values())}
    result = {}
    for name, (path, condition) in specs.items():
        raw = raw_by_path[path]
        if "condition" in raw.columns:
            raw = raw[raw["condition"] == condition].copy()
        result[name] = wide_attention(raw).drop_duplicates("sample_id").set_index("sample_id")
    common_ids = set.intersection(*(set(frame.index) for frame in result.values()))
    if not common_ids:
        raise ValueError("Candidate protocols share no sample IDs")
    return {name: frame.loc[sorted(common_ids)].copy() for name, frame in result.items()}


def fit_predict(
    frame: pd.DataFrame,
    positive: str,
    train_ids: list,
    test_ids: list,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    columns = [column for column in frame.columns if column.startswith("attn_")]
    x_train_raw = frame.loc[train_ids, columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    x_test_raw = frame.loc[test_ids, columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y_train = (frame.loc[train_ids, "group"].to_numpy() == positive).astype(int)
    y_test = (frame.loc[test_ids, "group"].to_numpy() == positive).astype(int)
    x_train, x_test, _ = _train_only_transform(x_train_raw, x_test_raw)
    selector = LogisticRegression(
        **elastic_net_penalty_kwargs(),
        solver="saga",
        l1_ratio=0.7,
        C=0.1,
        tol=5e-4,
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
    )
    selector.fit(x_train, y_train)
    selected = np.flatnonzero(np.abs(selector.coef_[0]) > 1e-10)
    if len(selected) == 0:
        selected = np.arange(x_train.shape[1])
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    classifier.fit(x_train[:, selected], y_train)
    return y_test, classifier.predict_proba(x_test[:, selected])[:, 1], int(len(selected))


def nested_select(
    frames: dict[str, pd.DataFrame],
    *,
    positive: str,
    negative: str,
    repeats: int,
    outer_splits: int,
    inner_splits: int,
    seed: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    names = list(frames)
    base = frames[names[0]]
    base = base[base["group"].isin([positive, negative])].copy()
    identifiers = base.index.to_numpy()
    y = (base["group"].to_numpy() == positive).astype(int)
    predictions = []
    selections = []
    repeat_metrics = []
    for repeat in range(1, repeats + 1):
        outer = StratifiedKFold(outer_splits, shuffle=True, random_state=seed + repeat - 1)
        repeat_truth = []
        repeat_scores = []
        for fold, (outer_train, outer_test) in enumerate(outer.split(identifiers, y), start=1):
            train_ids = identifiers[outer_train]
            test_ids = identifiers[outer_test]
            inner = StratifiedKFold(inner_splits, shuffle=True, random_state=seed + repeat * 100 + fold)
            inner_auc = {name: [] for name in names}
            for inner_fold, (inner_train, inner_test) in enumerate(inner.split(train_ids, y[outer_train]), start=1):
                fit_seed = seed + repeat * 1000 + fold * 10 + inner_fold
                inner_train_ids = train_ids[inner_train].tolist()
                inner_test_ids = train_ids[inner_test].tolist()

                def evaluate(name: str) -> tuple[str, float]:
                    truth, score, _ = fit_predict(
                        frames[name], positive, inner_train_ids, inner_test_ids, seed=fit_seed
                    )
                    return name, float(roc_auc_score(truth, score))

                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    results = list(executor.map(evaluate, names))
                for name, auc in results:
                    inner_auc[name].append(auc)
            means = {name: float(np.mean(values)) for name, values in inner_auc.items()}
            selected = max(names, key=lambda name: (means[name], -names.index(name)))
            truth, score, n_selected = fit_predict(
                frames[selected], positive, train_ids.tolist(), test_ids.tolist(), seed=seed + repeat * 1000 + fold
            )
            repeat_truth.extend(truth.tolist())
            repeat_scores.extend(score.tolist())
            selections.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "selected_protocol": selected,
                    "n_selected_features": n_selected,
                    **{f"inner_auc_{name}": value for name, value in means.items()},
                }
            )
            for sample_id, target, prediction in zip(test_ids, truth, score):
                predictions.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "sample_id": sample_id,
                        "selected_protocol": selected,
                        "y_true": int(target),
                        "score": float(prediction),
                    }
                )
        repeat_metrics.append({"repeat": repeat, "auc": float(roc_auc_score(repeat_truth, repeat_scores))})
    return pd.DataFrame(predictions), pd.DataFrame(selections), pd.DataFrame(repeat_metrics)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True, help="NAME=RAW_CSV@CONDITION")
    parser.add_argument("--positive-group", required=True)
    parser.add_argument("--negative-group", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    specs = parse_candidates(args.candidate)
    frames = load_candidates(specs)
    predictions, selections, repeats = nested_select(
        frames,
        positive=args.positive_group,
        negative=args.negative_group,
        repeats=args.repeats,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output / "nested_protocol_outer_predictions.csv", predictions)
    atomic_write_csv(output / "nested_protocol_selection.csv", selections)
    atomic_write_csv(output / "nested_protocol_repeat_auc.csv", repeats)
    counts = selections["selected_protocol"].value_counts().rename_axis("protocol").reset_index(name="n_outer_folds")
    atomic_write_csv(output / "nested_protocol_selection_counts.csv", counts)
    manifest = base_manifest(experiment="e12_nested_protocol", command=sys.argv)
    manifest.update({"status": "completed", "candidates": {name: [str(path), condition] for name, (path, condition) in specs.items()}})
    atomic_write_json(output / "nested_protocol_manifest.json", manifest)
    print(repeats.to_string(index=False))
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
