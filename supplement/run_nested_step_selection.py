# -*- coding: utf-8 -*-
"""Nested CV selection of additional-training step conditions.

Each ``--condition`` points to a proposed-feature root and is evaluated with
the same strict fold-local Elastic-Net + L2 classifier used by the paper.  The
inner CV selects among step conditions; the outer test fold is used exactly
once for the selected condition.  This prevents selecting 20 steps from the
same test results later reported as the main estimate.

Example::

    python data/run_nested_step_selection.py \
      --condition 20=attention_features_mimir_hardsplit \
      --condition 50=attention_features_mimir_hardsplit \
      --condition 100=attention_features_mimir_hardsplit \
      --condition early=attention_features_mimir_hardsplit \
      --output-dir data/results/additional_20260715/nested_step_selection_pythia1b

The default prefixes assume a common root containing ``fixed_attention_20`` /
``fixed_attention_50`` / ``fixed_attention_100`` / ``dynamic_attention``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import run_strict_fixed20_comparison_10runs as strict


GROUPS = {
    "ft_vs_pt": (strict.GROUP_FT, strict.GROUP_PT),
    "ft_vs_unseen": (strict.GROUP_FT, strict.GROUP_UNSEEN),
    "pt_vs_unseen": (strict.GROUP_PT, strict.GROUP_UNSEEN),
}


def condition_prefix(condition: str) -> str:
    return "dynamic_attention" if condition == "early" else f"fixed_attention_{condition}"


def parse_condition_values(values: Sequence[str]) -> Dict[str, Path]:
    parsed: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Condition must be NAME=ROOT, got {value!r}")
        name, root = value.split("=", 1)
        if not name or not root:
            raise ValueError(f"Condition must be NAME=ROOT, got {value!r}")
        if name in parsed:
            raise ValueError(f"Duplicate condition: {name}")
        parsed[name] = Path(root).expanduser()
    if len(parsed) < 2:
        raise ValueError("At least two step conditions are required")
    return parsed


def load_features(conditions: Mapping[str, Path]) -> Dict[str, pd.DataFrame]:
    loaded: Dict[str, pd.DataFrame] = {}
    for condition, root in conditions.items():
        raw = strict.read_group_files(
            strict.resolve_path(str(root)),
            "raw_experiment4_attention_shift.csv",
            condition_prefix=condition_prefix(condition),
        )
        loaded[condition] = strict.ensure_uid(
            strict.load_or_build_proposed_features(
                strict.resolve_path(str(root)), raw, condition_prefix=condition_prefix(condition)
            )
        )
    return loaded


def fit_predict_en(
    features: pd.DataFrame,
    positive_group: str,
    negative_group: str,
    train_uids: Sequence[str],
    test_uids: Sequence[str],
    *,
    selection_c: float,
    classifier_c: float,
    l1_ratio: float,
    max_iter: int,
    tol: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    df = strict.ensure_uid(features)
    df = df[df["group"].isin([positive_group, negative_group])].drop_duplicates("uid").set_index("uid")
    train_uids = [uid for uid in train_uids if uid in df.index]
    test_uids = [uid for uid in test_uids if uid in df.index]
    feature_cols = [c for c in df.columns if c.startswith("attn_l")]
    if not feature_cols:
        raise ValueError("No attention features found")
    x_train_raw = df.loc[train_uids, feature_cols].to_numpy(dtype=float)
    x_test_raw = df.loc[test_uids, feature_cols].to_numpy(dtype=float)
    y_train = (df.loc[train_uids, "group"].to_numpy() == positive_group).astype(int)
    y_test = (df.loc[test_uids, "group"].to_numpy() == positive_group).astype(int)
    x_train, x_test = strict.fit_transform_train_only(x_train_raw, x_test_raw)
    selector_kwargs = strict.elastic_net_selector_kwargs()
    selector = LogisticRegression(
        solver="saga",
        l1_ratio=l1_ratio,
        C=selection_c,
        tol=tol,
        max_iter=max_iter,
        class_weight="balanced",
        random_state=seed,
        **selector_kwargs,
    )
    selector.fit(x_train, y_train)
    selected = np.flatnonzero(np.abs(selector.coef_[0]) > 1e-10)
    if len(selected) == 0:
        selected = np.arange(x_train.shape[1])
    clf = LogisticRegression(
        solver="lbfgs",
        C=classifier_c,
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(x_train[:, selected], y_train)
    scores = clf.predict_proba(x_test[:, selected])[:, 1]
    return y_test, scores, int(len(selected))


def nested_select(
    features_by_condition: Mapping[str, pd.DataFrame],
    comparison: str,
    *,
    repeats: int,
    outer_splits: int,
    inner_splits: int,
    seed: int,
    selection_c: float,
    classifier_c: float,
    l1_ratio: float,
    max_iter: int,
    tol: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    positive_group, negative_group = GROUPS[comparison]
    base = next(iter(features_by_condition.values()))
    outer, _ = strict.make_common_splits(
        base,
        positive_group,
        negative_group,
        repeats=repeats,
        cv_splits=outer_splits,
        seed=seed,
    )
    prediction_rows: List[Dict[str, object]] = []
    selection_rows: List[Dict[str, object]] = []
    ordered_conditions = list(features_by_condition)
    for split in outer:
        train_uids = sorted(split["train_uids"])
        test_uids = sorted(split["test_uids"])
        train_df = base[base["uid"].isin(train_uids)].drop_duplicates("uid").reset_index(drop=True)
        y_train = (train_df["group"].to_numpy() == positive_group).astype(int)
        inner = StratifiedKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=seed + split["repeat"] * 100 + split["fold"],
        )
        inner_scores: Dict[str, float] = {}
        for condition in ordered_conditions:
            scores = []
            for inner_fold, (inner_train_idx, inner_test_idx) in enumerate(
                inner.split(np.zeros(len(y_train)), y_train), start=1
            ):
                inner_train = train_df.iloc[inner_train_idx]["uid"].tolist()
                inner_test = train_df.iloc[inner_test_idx]["uid"].tolist()
                y_true, pred, _ = fit_predict_en(
                    features_by_condition[condition],
                    positive_group,
                    negative_group,
                    inner_train,
                    inner_test,
                    selection_c=selection_c,
                    classifier_c=classifier_c,
                    l1_ratio=l1_ratio,
                    max_iter=max_iter,
                    tol=tol,
                    seed=seed + split["repeat"] * 1000 + split["fold"] * 10 + inner_fold,
                )
                scores.append(float(roc_auc_score(y_true, pred)))
            inner_scores[condition] = float(np.mean(scores))
        selected_condition = max(
            ordered_conditions,
            key=lambda condition: (inner_scores[condition], -ordered_conditions.index(condition)),
        )
        y_true, pred, n_selected = fit_predict_en(
            features_by_condition[selected_condition],
            positive_group,
            negative_group,
            train_uids,
            test_uids,
            selection_c=selection_c,
            classifier_c=classifier_c,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tol=tol,
            seed=seed + split["repeat"] * 1000 + split["fold"],
        )
        for idx, (truth, score) in enumerate(zip(y_true, pred)):
            prediction_rows.append(
                {
                    "comparison": comparison,
                    "repeat": split["repeat"],
                    "fold": split["fold"],
                    "selected_condition": selected_condition,
                    "test_index": idx,
                    "y_true": int(truth),
                    "score": float(score),
                    "n_selected_features": n_selected,
                }
            )
        selection_rows.append(
            {
                "comparison": comparison,
                "repeat": split["repeat"],
                "fold": split["fold"],
                "selected_condition": selected_condition,
                **{f"inner_auc_{key}": value for key, value in inner_scores.items()},
            }
        )
    return pd.DataFrame(prediction_rows), pd.DataFrame(selection_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", action="append", required=True, help="NAME=PROPOSED_FEATURE_ROOT")
    parser.add_argument("--comparison", choices=list(GROUPS), default="ft_vs_pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-c", type=float, default=0.1)
    parser.add_argument("--classifier-c", type=float, default=1.0)
    parser.add_argument("--l1-ratio", type=float, default=0.7)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=5e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conditions = parse_condition_values(args.condition)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    features = load_features(conditions)
    predictions, selections = nested_select(
        features,
        args.comparison,
        repeats=args.repeats,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        seed=args.seed,
        selection_c=args.selection_c,
        classifier_c=args.classifier_c,
        l1_ratio=args.l1_ratio,
        max_iter=args.max_iter,
        tol=args.tol,
    )
    predictions.to_csv(output_dir / "nested_outer_predictions.csv", index=False)
    selections.to_csv(output_dir / "nested_condition_selection.csv", index=False)
    summary = (
        predictions.groupby("selected_condition", as_index=False)
        .agg(
            n_predictions=("score", "count"),
            auc=("score", lambda s: float(roc_auc_score(predictions.loc[s.index, "y_true"], s))),
        )
        .sort_values("auc", ascending=False)
    )
    summary.to_csv(output_dir / "nested_summary.csv", index=False)
    manifest = vars(args).copy()
    manifest.update(
        {
            "script": Path(__file__).name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "conditions": {key: str(value) for key, value in conditions.items()},
            "comparison_groups": GROUPS[args.comparison],
            "selection_protocol": "inner CV only; outer test never used for condition selection",
        }
    )
    (output_dir / "nested_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
