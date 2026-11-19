
"""Train nine ML models on historical AAPL option candidates.

Training/validation use only outcomes matured by their cutoffs.
The 2024 test is exploratory because those outcomes were examined
in earlier experiments.

This experiment covers the 21-session candidate-level dataset.
It does not modify the existing strategy-level ML pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from xgboost import XGBRegressor


NUMERIC = [
    "Long Put Strike Pct",
    "Short Call Strike Pct",
    "Short Put Strike Pct",
    "Net Premium Pct",
    "Days to Expiry",
    "Option Legs",
    "Long Put Ask Pct",
    "Short Call Bid Pct",
    "Short Put Bid Pct",
]

FEATURES = NUMERIC + ["Strategy"]

GROUP = [
    "Underlying RIC",
    "Decision Date",
    "Hedge Horizon Days",
]

KEY = GROUP + ["Candidate ID"]

TARGET = "Training Target"


def load_candidates(
    path: Path,
    drawdown_weight: float,
) -> pd.DataFrame:

    df = pd.read_parquet(path)

    expected = set(
        KEY + [
            "Strategy",
            "Outcome Date",
            "Final Return",
            "Unhedged Final Return",
            "Hedge Improvement",
            "Maximum Drawdown",
            "Unhedged Maximum Drawdown",
            "Long Put Strike Pct",
            "Short Call Strike Pct",
            "Short Put Strike Pct",
            "Net Premium Pct",
            "Days to Expiry",
            "Option Legs",
            "Spot Price",
            "Long Put Ask",
            "Short Call Bid",
            "Short Put Bid",
        ]
    )

    missing = sorted(expected - set(df.columns))

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df.empty:
        raise ValueError("Empty candidate dataset")

    df = df.copy()

    for column in ("Decision Date", "Outcome Date"):
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
        ).dt.normalize()

        if df[column].isna().any():
            raise ValueError(
                f"Invalid {column}"
            )

    if (
        df["Outcome Date"] <= df["Decision Date"]
    ).any():
        raise ValueError(
            "Outcome Date must follow Decision Date"
        )

    if (
        df[KEY].isna().any().any()
        or df["Strategy"].isna().any()
    ):
        raise ValueError(
            "Missing candidate or strategy identifier"
        )

    if df.duplicated(KEY).any():
        duplicates = int(
            df.duplicated(KEY).sum()
        )

        raise ValueError(
            f"Duplicate candidate key: {duplicates} rows. "
            "Inspect source archives; do not "
            "automatically deduplicate."
        )

    if (
        df["Hedge Horizon Days"].nunique() != 1
        or int(df["Hedge Horizon Days"].iloc[0]) != 21
    ):
        raise ValueError(
            "This candidate-level experiment "
            "requires a 21-session horizon"
        )

    legal_strategies = {
        "Protective Put",
        "Collar",
        "Put Spread",
    }

    unknown = (
        set(df["Strategy"]) - legal_strategies
    )

    if unknown:
        raise ValueError(
            f"Unknown strategy families: {unknown}"
        )

    if df.groupby(
        GROUP,
        dropna=False,
    )["Outcome Date"].nunique().gt(1).any():
        raise ValueError(
            "Inconsistent outcome dates "
            "within a decision group"
        )

    required_numeric = [
        "Final Return",
        "Unhedged Final Return",
        "Hedge Improvement",
        "Maximum Drawdown",
        "Unhedged Maximum Drawdown",
        "Long Put Strike Pct",
        "Net Premium Pct",
        "Days to Expiry",
        "Option Legs",
        "Spot Price",
        "Long Put Ask",
    ]

    for column in required_numeric:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if not np.isfinite(
            df[column].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"Nonfinite required field: {column}"
            )

    if (
        (df["Spot Price"] <= 0).any()
        or (df["Long Put Strike Pct"] <= 0).any()
    ):
        raise ValueError(
            "Nonpositive spot or long put strike"
        )

    if (df["Days to Expiry"] <= 0).any():
        raise ValueError(
            "Invalid Days to Expiry"
        )

    # Final Return is already net of modeled
    # transaction costs. Do not subtract costs again.
    if not np.allclose(
        df["Hedge Improvement"],
        (
            df["Final Return"]
            - df["Unhedged Final Return"]
        ),
        rtol=0,
        atol=1e-8,
    ):
        raise ValueError(
            "Net realized hedge improvement "
            "does not reconcile"
        )

    unhedged_ranges = df.groupby(
        GROUP
    )["Unhedged Final Return"].agg(
        lambda values: (
            values.max() - values.min()
        )
    )

    if (unhedged_ranges > 1e-8).any():
        raise ValueError(
            "Unhedged return differs within "
            "the same decision group"
        )

    optional_leg_columns = [
        "Short Call Strike Pct",
        "Short Put Strike Pct",
        "Short Call Bid",
        "Short Put Bid",
    ]

    for column in optional_leg_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    needs_call = df["Strategy"].eq("Collar")
    needs_put = df["Strategy"].eq("Put Spread")

    for mask, columns in (
        (
            needs_call,
            ("Short Call Strike Pct", "Short Call Bid"),
        ),
        (
            needs_put,
            ("Short Put Strike Pct", "Short Put Bid"),
        ),
    ):
        for column in columns:
            invalid = (
                df.loc[mask, column].isna()
                | (df.loc[mask, column] <= 0)
            )

            if invalid.any():
                raise ValueError(
                    f"Missing active leg field: {column}"
                )

    if (df["Long Put Ask"] <= 0).any():
        raise ValueError(
            "Invalid long put entry ask"
        )

    # Zero represents a leg that does not exist.
    for column in optional_leg_columns:
        df[column] = df[column].fillna(0.0)

    # Normalize option prices by stock price.
    for column in (
        "Long Put Ask",
        "Short Call Bid",
        "Short Put Bid",
    ):
        df[column + " Pct"] = (
            df[column] / df["Spot Price"]
        )

    if (
        not np.isfinite(drawdown_weight)
        or drawdown_weight < 0
    ):
        raise ValueError(
            "drawdown_weight must be nonnegative"
        )

    # Signed drawdown convention:
    # -10% is better than -20%.
    if (
        (df["Maximum Drawdown"] > 1e-8).any()
        or (
            df["Unhedged Maximum Drawdown"] > 1e-8
        ).any()
    ):
        raise ValueError(
            "Drawdowns must be signed "
            "nonpositive returns"
        )

    df[TARGET] = (
        df["Hedge Improvement"]
        + drawdown_weight
        * (
            df["Maximum Drawdown"]
            - df["Unhedged Maximum Drawdown"]
        )
    )

    if not np.isfinite(
        df[TARGET].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Invalid training target"
        )

    return (
        df.sort_values(
            GROUP + ["Candidate ID"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def model_factory(seed: int) -> dict:

    def prep():
        return ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline([
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "standardize",
                            StandardScaler(),
                        ),
                    ]),
                    NUMERIC,
                ),
                (
                    "strategy",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                    ["Strategy"],
                ),
            ],
            remainder="drop",
            sparse_threshold=0,
        )

    return {
        "OLS": Pipeline([
            ("features", prep()),
            ("model", LinearRegression()),
        ]),

        "Ridge": Pipeline([
            ("features", prep()),
            ("model", Ridge(alpha=10.0)),
        ]),

        "Lasso": Pipeline([
            ("features", prep()),
            (
                "model",
                Lasso(
                    alpha=0.0005,
                    max_iter=20000,
                ),
            ),
        ]),

        "Elastic Net": Pipeline([
            ("features", prep()),
            (
                "model",
                ElasticNet(
                    alpha=0.0005,
                    l1_ratio=0.5,
                    max_iter=20000,
                ),
            ),
        ]),

        "PCR": Pipeline([
            ("features", prep()),
            (
                "pca",
                PCA(n_components=5),
            ),
            ("model", LinearRegression()),
        ]),

        "PLS": Pipeline([
            ("features", prep()),
            (
                "model",
                PLSRegression(
                    n_components=4,
                    scale=False,
                ),
            ),
        ]),

        "Random Forest": Pipeline([
            ("features", prep()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=150,
                    max_depth=7,
                    min_samples_leaf=20,
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]),

        "Gradient Boosting": Pipeline([
            ("features", prep()),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=150,
                    learning_rate=0.045,
                    max_depth=3,
                    min_samples_leaf=20,
                    random_state=seed,
                ),
            ),
        ]),

        "XGBoost": Pipeline([
            ("features", prep()),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=160,
                    max_depth=3,
                    learning_rate=0.045,
                    subsample=0.85,
                    colsample_bytree=0.90,
                    reg_lambda=8,
                    tree_method="hist",
                    n_jobs=2,
                    random_state=seed,
                ),
            ),
        ]),
    }


def ex_ante_collar(
    pool: pd.DataFrame,
) -> pd.Series | None:
    """Fixed 90% put / 110% call benchmark.

    Selection uses known entry-time features only,
    never the realized target.
    """

    collars = pool.loc[
        pool["Strategy"].eq("Collar")
    ].copy()

    if collars.empty:
        return None

    collars["Moneyness Distance"] = (
        (
            collars["Long Put Strike Pct"] - 0.90
        ).abs()
        + (
            collars["Short Call Strike Pct"] - 1.10
        ).abs()
    )

    return collars.sort_values(
        [
            "Moneyness Distance",
            "Net Premium Pct",
            "Candidate ID",
        ],
        kind="stable",
    ).iloc[0]


def date_choices(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    model_name: str,
    phase: str,
) -> pd.DataFrame:

    if (
        len(frame) != len(predictions)
        or not np.isfinite(predictions).all()
    ):
        raise ValueError(
            "Prediction shape or values invalid"
        )

    scored = frame[
        GROUP + [
            "Candidate ID",
            "Strategy",
            TARGET,
        ]
    ].copy()

    scored["Predicted Improvement"] = predictions

    rows = []

    for key, group in scored.groupby(
        GROUP,
        sort=True,
    ):
        # Choose using predicted performance only.
        top = group.sort_values(
            [
                "Predicted Improvement",
                "Candidate ID",
            ],
            ascending=[False, True],
            kind="stable",
        ).iloc[0]

        # No Hedge is a valid alternative.
        selected = (
            top
            if top["Predicted Improvement"] > 0
            else None
        )

        # Hindsight oracle is a diagnostic,
        # not an achievable benchmark.
        best_actual = max(
            0.0,
            float(group[TARGET].max()),
        )

        realized = (
            0.0
            if selected is None
            else float(selected[TARGET])
        )

        rows.append({
            "Underlying RIC": key[0],
            "Decision Date": key[1],
            "Hedge Horizon Days": key[2],
            "Phase": phase,
            "Model": model_name,
            "Candidates": len(group),
            "Selected Candidate ID": (
                "NO_HEDGE"
                if selected is None
                else str(selected["Candidate ID"])
            ),
            "Selected Strategy": (
                "No Hedge"
                if selected is None
                else str(selected["Strategy"])
            ),
            "Predicted Improvement": (
                0.0
                if selected is None
                else float(
                    selected["Predicted Improvement"]
                )
            ),
            "Realized Improvement": realized,
            "Hindsight Oracle": best_actual,
            "Regret": best_actual - realized,
        })

    return pd.DataFrame(rows)


def policy_choices(
    frame: pd.DataFrame,
    phase: str,
) -> pd.DataFrame:

    rows = []

    for key, group in frame.groupby(
        GROUP,
        sort=True,
    ):
        collar = ex_ante_collar(group)

        oracle = max(
            0.0,
            float(group[TARGET].max()),
        )

        policies = (
            ("No Hedge", None),
            ("Always Collar 90/110", collar),
        )

        for model, selected in policies:

            actual = (
                0.0
                if selected is None
                else float(selected[TARGET])
            )

            rows.append({
                "Underlying RIC": key[0],
                "Decision Date": key[1],
                "Hedge Horizon Days": key[2],
                "Phase": phase,
                "Model": model,
                "Candidates": len(group),
                "Selected Candidate ID": (
                    "NO_HEDGE"
                    if selected is None
                    else str(selected["Candidate ID"])
                ),
                "Selected Strategy": (
                    "No Hedge"
                    if selected is None
                    else str(selected["Strategy"])
                ),
                "Predicted Improvement": np.nan,
                "Realized Improvement": actual,
                "Hindsight Oracle": oracle,
                "Regret": oracle - actual,
            })

    return pd.DataFrame(rows)


def create_splits(
    df: pd.DataFrame,
    validation_start: str,
    test_start: str,
    test_end: str,
):

    v, t, e = (
        pd.Timestamp(value).normalize()
        for value in (
            validation_start,
            test_start,
            test_end,
        )
    )

    if not v < t < e:
        raise ValueError(
            "Require validation_start "
            "< test_start < test_end"
        )

    # All training outcomes must have matured
    # before the relevant model-fitting cutoff.
    train_v = df.loc[
        (df["Decision Date"] < v)
        & (df["Outcome Date"] < v)
    ].copy()

    valid = df.loc[
        (df["Decision Date"] >= v)
        & (df["Decision Date"] < t)
        & (df["Outcome Date"] < t)
    ].copy()

    train_t = df.loc[
        (df["Decision Date"] < t)
        & (df["Outcome Date"] < t)
    ].copy()

    test = df.loc[
        (df["Decision Date"] >= t)
        & (df["Decision Date"] < e)
    ].copy()

    if any(
        part.empty
        for part in (
            train_v,
            valid,
            train_t,
            test,
        )
    ):
        raise ValueError(
            "Empty training, validation or test partition"
        )

    if (
        train_v["Outcome Date"].max() >= v
        or train_t["Outcome Date"].max() >= t
        or valid["Outcome Date"].max() >= t
    ):
        raise ValueError(
            "Outcome-maturity restriction violated"
        )

    return train_v, valid, train_t, test


def fit_and_forecast(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    seed: int,
    phase: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    forecasts = []
    decisions = [
        policy_choices(evaluation, phase)
    ]

    for name, estimator in model_factory(
        seed
    ).items():

        print(
            f"Fitting {name} ({phase})...",
            flush=True,
        )

        estimator.fit(
            train[FEATURES],
            train[TARGET],
        )

        predicted = np.asarray(
            estimator.predict(
                evaluation[FEATURES]
            ),
            dtype=float,
        ).reshape(-1)

        if (
            len(predicted) != len(evaluation)
            or not np.isfinite(predicted).all()
        ):
            raise ValueError(
                f"Invalid forecasts from {name}"
            )

        predictions = evaluation[
            KEY + [
                "Strategy",
                TARGET,
                "Outcome Date",
            ]
        ].copy()

        predictions["Model"] = name
        predictions["Phase"] = phase
        predictions["Predicted Improvement"] = predicted

        forecasts.append(predictions)

        decisions.append(
            date_choices(
                evaluation,
                predicted,
                name,
                phase,
            )
        )

    return (
        pd.concat(
            forecasts,
            ignore_index=True,
        ),
        pd.concat(
            decisions,
            ignore_index=True,
        ),
    )


def summarize(
    decisions: pd.DataFrame,
) -> pd.DataFrame:

    summary = decisions.groupby(
        ["Phase", "Model"],
        as_index=False,
    ).agg(
        Dates=(
            "Decision Date",
            "nunique",
        ),
        Mean_Realized_Improvement=(
            "Realized Improvement",
            "mean",
        ),
        Median_Realized_Improvement=(
            "Realized Improvement",
            "median",
        ),
        Mean_Regret=(
            "Regret",
            "mean",
        ),
        Hedge_Decisions=(
            "Selected Strategy",
            lambda values: int(
                values.ne("No Hedge").sum()
            ),
        ),
    )

    return summary.sort_values(
        [
            "Phase",
            "Mean_Realized_Improvement",
        ],
        ascending=[True, False],
    )


def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--dataset",
        default=(
            "data/processed/ml_datasets/"
            "aapl_h21/candidate_dataset.parquet"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "data/processed/ml_experiments/"
            "AAPL_candidate_nine_v1"
        ),
    )

    parser.add_argument(
        "--validation-start",
        default="2023-01-01",
    )

    parser.add_argument(
        "--test-start",
        default="2024-01-01",
    )

    parser.add_argument(
        "--test-end",
        default="2025-01-01",
    )

    parser.add_argument(
        "--drawdown-weight",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    if not dataset_path.is_file():
        parser.error(
            f"Dataset not found: {dataset_path}"
        )

    df = load_candidates(
        dataset_path,
        args.drawdown_weight,
    )

    train_v, valid, train_t, test = create_splits(
        df,
        args.validation_start,
        args.test_start,
        args.test_end,
    )

    print(
        "\n=== NINE-MODEL CANDIDATE TRAINING ==="
    )

    sections = (
        ("Train before validation", train_v),
        ("Mature validation", valid),
        ("Train before test", train_t),
        ("Exploratory test", test),
    )

    for label, frame in sections:
        opportunities = len(
            frame[GROUP].drop_duplicates()
        )

        print(
            f"{label}: {len(frame):,} candidates, "
            f"{opportunities} opportunities, "
            f"{frame['Decision Date'].nunique()} dates"
        )

    print("\nFeatures:", FEATURES)

    print(
        "Target: net hedge improvement + "
        "lambda * signed drawdown improvement"
    )

    print(
        "No future outcome fields used as features."
    )

    pred_v, choice_v = fit_and_forecast(
        train_v,
        valid,
        args.seed,
        "Validation",
    )

    pred_t, choice_t = fit_and_forecast(
        train_t,
        test,
        args.seed,
        "Test",
    )

    predictions = pd.concat(
        [pred_v, pred_t],
        ignore_index=True,
    )

    decisions = pd.concat(
        [choice_v, choice_t],
        ignore_index=True,
    )

    summary = summarize(decisions)

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_parquet(
        output_dir / "candidate_predictions.parquet",
        index=False,
    )

    decisions.to_parquet(
        output_dir / "date_decisions.parquet",
        index=False,
    )

    summary.to_csv(
        output_dir / "model_summary.csv",
        index=False,
    )

    experiment = {
        "dataset": str(dataset_path),
        "dataset_sha256": hashlib.sha256(
            dataset_path.read_bytes()
        ).hexdigest(),
        "rows": len(df),
        "unique_dates": df["Decision Date"].nunique(),
        "features": FEATURES,
        "drawdown_weight": args.drawdown_weight,
        "train_cutoff_validation": args.validation_start,
        "frozen_test_cutoff": args.test_start,
        "test_end_exclusive": args.test_end,
        "seed": args.seed,
        "tested_model_count": 9,
        "test_status": (
            "Exploratory: 2024 outcomes "
            "previously examined"
        ),
        "candidate_keys": KEY,
    }

    (
        output_dir / "experiment.json"
    ).write_text(
        json.dumps(
            experiment,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== DECISION-LEVEL RESULTS ==="
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        )
    )

    print("\nOutput directory:", output_dir)

    print(
        "Caution: 85 decision dates, not "
        "8,906 independent investments."
    )

    print(
        "Caution: 2024 results are exploratory."
    )

    print(
        "Caution: quote and corporate-action "
        "audits are still required."
    )


if __name__ == "__main__":
    main()
