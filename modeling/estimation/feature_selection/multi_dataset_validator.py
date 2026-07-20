#!/usr/bin/env python3
"""Evaluate candidate feature sets across multiple independent datasets.

This script intentionally does *not* align timestamps between parquet files. Each
file is treated as a separate run: interval preparation and chronological splits
happen inside each dataset, and leave-one-dataset-out validation concatenates
interval rows from complete datasets.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cvxpy as cp
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

try:  # direct execution: python multi_dataset_validator.py
    from selection_pipeline import (
        ESTIMATOR_DEFAULT_FEATURES,
        TARGET_COLUMN,
        TIME_COLUMN,
        TIME_ROUNDING,
        EvalResult,
        fit_eval_cvxpy,
        load_interval_data,
        split_interval_data,
    )
except ImportError:  # module execution: python -m ...multi_dataset_validator
    from .selection_pipeline import (
        ESTIMATOR_DEFAULT_FEATURES,
        TARGET_COLUMN,
        TIME_COLUMN,
        TIME_ROUNDING,
        EvalResult,
        fit_eval_cvxpy,
        load_interval_data,
        split_interval_data,
    )


@dataclass(frozen=True)
class FeatureSet:
    name: str
    features: list[str]
    source: str


@dataclass
class DatasetBundle:
    name: str
    path: Path
    x: pd.DataFrame
    y: pd.Series
    missing_features: list[str]
    removed_constant: list[str]


@dataclass
class ScoreRow:
    mode: str
    feature_set: str
    dataset: str
    train_datasets: str
    n_features: int
    n_train: int
    n_test: int
    r2: float
    mae: float
    mae_pct: float
    static_energy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate candidate feature sets across independent datasets."
    )
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="Input parquet files. Each file is treated as one independent run.",
    )
    parser.add_argument(
        "--feature-set",
        action="append",
        default=[],
        metavar="NAME=feat1,feat2",
        help="Candidate feature set to evaluate. Can be provided multiple times.",
    )
    parser.add_argument(
        "--feature-file",
        type=Path,
        action="append",
        default=[],
        help="CSV/text file containing one feature set, e.g. selected_features.csv.",
    )
    parser.add_argument(
        "--selection-output-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "Output directory from selection_pipeline.py. Reads selected_features.csv "
            "and the best rows from candidate_sets.csv if present."
        ),
    )
    parser.add_argument(
        "--candidate-sets-per-output",
        type=int,
        default=5,
        help="How many rows to import from each candidate_sets.csv file.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("multi_validation_output")
    )
    parser.add_argument("--filter-active", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--l1-penalty", type=float, default=0.1)
    parser.add_argument(
        "--mode",
        choices=["chronological", "leave-one-out", "both"],
        default="both",
        help=(
            "chronological: split inside each dataset; leave-one-out: train on all "
            "other datasets and test on the held-out dataset."
        ),
    )
    parser.add_argument(
        "--rank-mode",
        choices=["chronological", "leave-one-out"],
        default="leave-one-out",
        help="Validation mode used to choose the final feature set.",
    )
    return parser.parse_args()


def _clean_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-._") or "unnamed"


def dataset_name(path: Path, existing: set[str]) -> str:
    base = path.parent.name if path.parent.name else path.stem
    if base in {"collections", "data"}:
        base = path.stem
    name = _clean_name(base)
    if name not in existing:
        return name
    i = 2
    while f"{name}-{i}" in existing:
        i += 1
    return f"{name}-{i}"


def parse_feature_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def read_feature_file(path: Path) -> list[str]:
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return []
    except UnicodeDecodeError:
        return []

    if "feature" in df.columns:
        return [str(v).strip() for v in df["feature"].dropna() if str(v).strip()]
    if "features" in df.columns and not df.empty:
        return parse_feature_list(str(df.iloc[0]["features"]))

    text = path.read_text().strip()
    if not text:
        return []
    if "," in text:
        return parse_feature_list(text)
    return [line.strip() for line in text.splitlines() if line.strip()]


def load_feature_sets(args: argparse.Namespace) -> list[FeatureSet]:
    feature_sets: list[FeatureSet] = [
        FeatureSet(
            name="estimator_default_set",
            features=list(ESTIMATOR_DEFAULT_FEATURES),
            source="built_in",
        )
    ]

    for item in args.feature_set:
        if "=" not in item:
            raise ValueError(
                f"Invalid --feature-set '{item}'. Expected NAME=feat1,feat2."
            )
        name, raw_features = item.split("=", 1)
        features = parse_feature_list(raw_features)
        if features:
            feature_sets.append(
                FeatureSet(name=_clean_name(name), features=features, source="cli")
            )

    for path in args.feature_file:
        features = read_feature_file(path)
        if features:
            feature_sets.append(
                FeatureSet(
                    name=_clean_name(path.stem),
                    features=features,
                    source=str(path),
                )
            )

    for output_dir in args.selection_output_dir:
        selected_path = output_dir / "selected_features.csv"
        if selected_path.exists():
            features = read_feature_file(selected_path)
            if features:
                feature_sets.append(
                    FeatureSet(
                        name=_clean_name(f"{output_dir.name}_selected"),
                        features=features,
                        source=str(selected_path),
                    )
                )

        candidates_path = output_dir / "candidate_sets.csv"
        if candidates_path.exists():
            candidates = pd.read_csv(candidates_path)
            if "features" in candidates.columns:
                if "mae_pct" in candidates.columns:
                    candidates = candidates.sort_values("mae_pct", ascending=True)
                candidates = candidates.head(args.candidate_sets_per_output)
                for idx, row in candidates.iterrows():
                    features = parse_feature_list(str(row["features"]))
                    if not features:
                        continue
                    source = str(row.get("source", "candidate"))
                    name = _clean_name(f"{output_dir.name}_{source}_{idx}")
                    feature_sets.append(
                        FeatureSet(
                            name=name,
                            features=features,
                            source=str(candidates_path),
                        )
                    )

    unique: list[FeatureSet] = []
    seen: set[tuple[str, ...]] = set()
    for feature_set in feature_sets:
        key = tuple(feature_set.features)
        if key in seen:
            continue
        seen.add(key)
        unique.append(feature_set)
    return unique


def load_datasets(
    paths: list[Path], features: list[str], filter_active: bool
) -> list[DatasetBundle]:
    datasets: list[DatasetBundle] = []
    names: set[str] = set()
    for path in paths:
        name = dataset_name(path, names)
        names.add(name)
        interval_data = load_interval_data(path, features, filter_active)
        if interval_data.x.empty:
            raise ValueError(f"No usable intervals in dataset {path}")
        datasets.append(
            DatasetBundle(
                name=name,
                path=path,
                x=interval_data.x,
                y=interval_data.y,
                missing_features=interval_data.missing_features,
                removed_constant=interval_data.removed_constant,
            )
        )
    return datasets


def available_feature_sets(
    feature_sets: list[FeatureSet], datasets: list[DatasetBundle]
) -> tuple[list[FeatureSet], list[dict]]:
    valid: list[FeatureSet] = []
    rejected: list[dict] = []
    for feature_set in feature_sets:
        missing_by_dataset = {
            dataset.name: sorted(set(feature_set.features) - set(dataset.x.columns))
            for dataset in datasets
        }
        missing_by_dataset = {
            name: missing for name, missing in missing_by_dataset.items() if missing
        }
        if missing_by_dataset:
            rejected.append(
                {
                    "feature_set": feature_set.name,
                    "features": feature_set.features,
                    "reason": "missing_or_constant_features",
                    "missing_by_dataset": missing_by_dataset,
                }
            )
            continue
        valid.append(feature_set)
    return valid, rejected


def concat_interval_rows(
    datasets: list[DatasetBundle], features: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    x_parts = [
        dataset.x.loc[:, features].reset_index(drop=True) for dataset in datasets
    ]
    y_parts = [dataset.y.reset_index(drop=True) for dataset in datasets]
    x_concat = pd.concat(x_parts, ignore_index=True)
    y_concat = pd.concat(y_parts, ignore_index=True)
    return cast(pd.DataFrame, x_concat), cast(pd.Series, y_concat)


def evaluate_chronological(
    feature_set: FeatureSet,
    datasets: list[DatasetBundle],
    test_size: float,
    l1_penalty: float,
) -> list[ScoreRow]:
    rows: list[ScoreRow] = []
    for dataset in datasets:
        x_train, x_test, y_train, y_test = split_interval_data(
            dataset.x, dataset.y, test_size
        )
        result = fit_eval_cvxpy(
            feature_set.features,
            x_train,
            x_test,
            y_train,
            y_test,
            l1_penalty,
            source="multi_chronological",
        )
        rows.append(
            score_from_result(
                result=result,
                mode="chronological",
                dataset=dataset.name,
                train_datasets=dataset.name,
                n_train=len(y_train),
                n_test=len(y_test),
                feature_set_name=feature_set.name,
            )
        )
    return rows


def evaluate_leave_one_out(
    feature_set: FeatureSet,
    datasets: list[DatasetBundle],
    l1_penalty: float,
) -> list[ScoreRow]:
    if len(datasets) < 2:
        return []

    rows: list[ScoreRow] = []
    for heldout in datasets:
        train_datasets = [
            dataset for dataset in datasets if dataset.name != heldout.name
        ]
        x_train, y_train = concat_interval_rows(train_datasets, feature_set.features)
        x_test = cast(pd.DataFrame, heldout.x.loc[:, feature_set.features])
        y_test = heldout.y
        result = fit_eval_cvxpy(
            feature_set.features,
            x_train,
            x_test,
            y_train,
            y_test,
            l1_penalty,
            source="multi_leave_one_out",
        )
        rows.append(
            score_from_result(
                result=result,
                mode="leave-one-out",
                dataset=heldout.name,
                train_datasets=",".join(dataset.name for dataset in train_datasets),
                n_train=len(y_train),
                n_test=len(y_test),
                feature_set_name=feature_set.name,
            )
        )
    return rows


def score_from_result(
    result: EvalResult,
    mode: str,
    dataset: str,
    train_datasets: str,
    n_train: int,
    n_test: int,
    feature_set_name: str,
) -> ScoreRow:
    return ScoreRow(
        mode=mode,
        feature_set=feature_set_name,
        dataset=dataset,
        train_datasets=train_datasets,
        n_features=len(result.features),
        n_train=n_train,
        n_test=n_test,
        r2=result.r2,
        mae=result.mae,
        mae_pct=result.mae_pct,
        static_energy=result.static_energy,
    )


def build_ranking(scores: pd.DataFrame, feature_sets: list[FeatureSet]) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()

    feature_counts = {
        feature_set.name: len(feature_set.features) for feature_set in feature_sets
    }
    ranking = (
        scores.groupby(["mode", "feature_set"], as_index=False)
        .agg(
            avg_mae_pct=("mae_pct", "mean"),
            median_mae_pct=("mae_pct", "median"),
            worst_mae_pct=("mae_pct", "max"),
            avg_r2=("r2", "mean"),
            worst_r2=("r2", "min"),
            valid_splits=("dataset", "count"),
        )
        .assign(n_features=lambda df: df["feature_set"].map(feature_counts))
        .sort_values(
            ["mode", "avg_mae_pct", "worst_mae_pct", "n_features", "avg_r2"],
            ascending=[True, True, True, True, False],
        )
        .reset_index(drop=True)
    )
    return ranking


def choose_best_feature_set(
    ranking: pd.DataFrame, feature_sets: list[FeatureSet], rank_mode: str
) -> FeatureSet:
    if ranking.empty:
        raise ValueError("No validation scores available")

    mode_ranking = ranking[ranking["mode"] == rank_mode]
    if mode_ranking.empty:
        mode_ranking = ranking
    best_name = str(mode_ranking.iloc[0]["feature_set"])
    for feature_set in feature_sets:
        if feature_set.name == best_name:
            return feature_set
    raise ValueError(f"Could not resolve selected feature set {best_name}")


def fit_final_model(
    datasets: list[DatasetBundle], features: list[str], l1_penalty: float
) -> dict:
    x_all, y_all = concat_interval_rows(datasets, features)
    scaler = MaxAbsScaler()
    x_matrix = scaler.fit_transform(x_all[features].values)
    y_values = y_all.values

    weights = cp.Variable(len(features))
    static_energy = cp.Variable()
    predictions = x_matrix @ weights + static_energy
    loss = cp.sum_squares(predictions - y_values)
    reg = l1_penalty * cp.norm1(weights)
    problem = cp.Problem(
        cp.Minimize(loss + reg), constraints=[weights >= 0, static_energy >= 0]
    )
    problem.solve()

    if weights.value is None or static_energy.value is None:
        raise RuntimeError(f"Final CVXPY solve failed with status {problem.status}")

    return {
        "weights": weights.value,
        "static_energy": float(static_energy.value),
        "scaler": scaler,
        "features": features,
        "target": TARGET_COLUMN,
        "time_column": TIME_COLUMN,
        "time_rounding": TIME_ROUNDING,
        "l1_penalty": l1_penalty,
        "static_penalty": 0.0,
        "static_model": "scalar",
        "solver_status": problem.status,
    }


def write_outputs(
    args: argparse.Namespace,
    datasets: list[DatasetBundle],
    feature_sets: list[FeatureSet],
    rejected_sets: list[dict],
    scores: pd.DataFrame,
    ranking: pd.DataFrame,
    selected: FeatureSet,
    final_model: dict,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scores.to_csv(args.output_dir / "dataset_scores.csv", index=False)
    ranking.to_csv(args.output_dir / "feature_set_ranking.csv", index=False)
    pd.DataFrame({"feature": selected.features}).to_csv(
        args.output_dir / "best_features.csv", index=False
    )

    env_text = (
        f"EST_FEATURES={','.join(selected.features)}\n"
        f"EST_L1_PENALTY={args.l1_penalty}\n"
        "EST_STATIC_PENALTY=0.0\n"
        "EST_STATIC_MODEL=scalar\n"
    )
    (args.output_dir / "best_estimator.env").write_text(env_text)

    model_payload = dict(final_model)
    model_payload.update(
        {
            "source_data": [str(dataset.path) for dataset in datasets],
            "selection_method": "multi_dataset_validator",
            "selected_feature_set": selected.name,
            "validation_rank_mode": args.rank_mode,
        }
    )
    with (args.output_dir / "best_model.pkl").open("wb") as handle:
        pickle.dump(model_payload, handle)

    diagnostics = {
        "datasets": [
            {
                "name": dataset.name,
                "path": str(dataset.path),
                "n_intervals": int(len(dataset.y)),
                "missing_features": dataset.missing_features,
                "constant_features": dataset.removed_constant,
            }
            for dataset in datasets
        ],
        "feature_sets": [
            {
                "name": feature_set.name,
                "features": feature_set.features,
                "source": feature_set.source,
            }
            for feature_set in feature_sets
        ],
        "rejected_feature_sets": rejected_sets,
        "mode": args.mode,
        "rank_mode": args.rank_mode,
        "filter_active": bool(args.filter_active),
        "test_size": args.test_size,
        "l1_penalty": args.l1_penalty,
        "selected_feature_set": selected.name,
        "selected_features": selected.features,
    }
    (args.output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    selected_ranking = ranking[ranking["feature_set"] == selected.name]
    summary = [
        "# Multi-Dataset Feature Validation Summary",
        "",
        "Each parquet file was treated as an independent run. Timestamps were not aligned across datasets.",
        "",
        f"Selected feature set: `{selected.name}`",
        f"Selected features: `{', '.join(selected.features)}`",
        f"Ranking mode: `{args.rank_mode}`",
        "",
        "## Datasets",
        "",
    ]
    for dataset in datasets:
        summary.append(
            f"- `{dataset.name}`: `{dataset.path}` ({len(dataset.y)} intervals)"
        )
    summary.extend(["", "## Selected ranking rows", ""])
    if selected_ranking.empty:
        summary.append("No ranking row found for the selected set.")
    else:
        for _, row in selected_ranking.iterrows():
            summary.append(
                "- "
                f"mode `{row['mode']}`: avg MAE% `{row['avg_mae_pct']:.2f}`, "
                f"worst MAE% `{row['worst_mae_pct']:.2f}`, avg R² `{row['avg_r2']:.4f}`"
            )
    summary.extend(
        [
            "",
            "## Outputs",
            "",
            "- `dataset_scores.csv`",
            "- `feature_set_ranking.csv`",
            "- `best_features.csv`",
            "- `best_estimator.env`",
            "- `best_model.pkl`",
            "- `diagnostics.json`",
        ]
    )
    (args.output_dir / "validation_summary.md").write_text("\n".join(summary) + "\n")


def main() -> None:
    args = parse_args()
    feature_sets = load_feature_sets(args)
    if not feature_sets:
        raise ValueError("No feature sets to evaluate")

    feature_union = sorted({feature for fs in feature_sets for feature in fs.features})
    datasets = load_datasets(args.data, feature_union, args.filter_active)
    feature_sets, rejected_sets = available_feature_sets(feature_sets, datasets)
    if not feature_sets:
        raise ValueError("No feature sets are available in all datasets")

    score_rows: list[ScoreRow] = []
    for feature_set in feature_sets:
        if args.mode in {"chronological", "both"}:
            score_rows.extend(
                evaluate_chronological(
                    feature_set,
                    datasets,
                    test_size=args.test_size,
                    l1_penalty=args.l1_penalty,
                )
            )
        if args.mode in {"leave-one-out", "both"}:
            score_rows.extend(
                evaluate_leave_one_out(
                    feature_set,
                    datasets,
                    l1_penalty=args.l1_penalty,
                )
            )

    scores = pd.DataFrame([row.__dict__ for row in score_rows])
    ranking = build_ranking(scores, feature_sets)
    selected = choose_best_feature_set(ranking, feature_sets, args.rank_mode)
    final_model = fit_final_model(datasets, selected.features, args.l1_penalty)

    write_outputs(
        args=args,
        datasets=datasets,
        feature_sets=feature_sets,
        rejected_sets=rejected_sets,
        scores=scores,
        ranking=ranking,
        selected=selected,
        final_model=final_model,
    )

    print("\n=== Multi-dataset validation ===")
    print(f"Datasets: {[dataset.name for dataset in datasets]}")
    print(f"Selected feature set: {selected.name}")
    print(selected.features)
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
