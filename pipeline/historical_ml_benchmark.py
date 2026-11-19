
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_datasets"
    / "AAPL_O_20220101_20241231_H21"
    / "completed_candidates.parquet"
)

NUMERIC_FEATURES = [
    "Spot Price",
    "Long Put Strike Pct",
    "Short Call Strike Pct",
    "Short Put Strike Pct",
    "Long Put Ask",
    "Short Call Bid",
    "Short Put Bid",
    "Net Premium Pct",
    "Days to Expiry",
    "Days After Exit",
]

CATEGORICAL_FEATURES = [
    "Strategy",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "Hedge Improvement"


def make_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value=0.0,
            add_indicator=True,
        )),
        ("scaler", StandardScaler()),
    ])

    categorical = Pipeline([
        ("imputer", SimpleImputer(
            strategy="most_frequent",
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        (
            "categorical",
            categorical,
            CATEGORICAL_FEATURES,
        ),
    ])


def make_models():
    estimators = {
        "Dummy": DummyRegressor(
            strategy="mean",
        ),
        "Ridge": Ridge(
            alpha=10.0,
        ),
        "Elastic Net": ElasticNet(
            alpha=0.001,
            l1_ratio=0.5,
            max_iter=20000,
            random_state=42,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        ),
    }

    return {
        name: Pipeline([
            ("preprocessor", make_preprocessor()),
            ("regressor", estimator),
        ])
        for name, estimator in estimators.items()
    }


def load_dataset():
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(DATASET_PATH)

    df = pd.read_parquet(DATASET_PATH)

    required = (
        FEATURES
        + [
            TARGET,
            "Candidate Key",
            "Decision Date",
            "Outcome Date",
            "Drawdown Improvement",
            "Maximum Drawdown",
        ]
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    for column in [
        "Decision Date",
        "Outcome Date",
    ]:
        df[column] = pd.to_datetime(
            df[column]
        ).astype("datetime64[ns]")

    if not df["Candidate Key"].is_unique:
        raise ValueError(
            "Duplicate candidate keys."
        )

    for column in NUMERIC_FEATURES + [
        TARGET,
        "Drawdown Improvement",
        "Maximum Drawdown",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[
        [
            TARGET,
            "Drawdown Improvement",
            "Maximum Drawdown",
        ]
    ].isna().any().any():
        raise ValueError(
            "Missing realized outcomes."
        )

    return df.sort_values(
        ["Decision Date", "Candidate Key"]
    ).reset_index(drop=True)


def split_dataset(df):
    validation_start = pd.Timestamp("2023-01-01")
    test_start = pd.Timestamp("2024-01-01")
    test_end = pd.Timestamp("2025-01-01")

    validation = df.loc[
        (df["Decision Date"] >= validation_start)
        & (df["Decision Date"] < test_start)
    ].copy()

    test = df.loc[
        (df["Decision Date"] >= test_start)
        & (df["Decision Date"] < test_end)
    ].copy()

    if validation.empty or test.empty:
        raise ValueError(
            "Validation or test period is empty."
        )

    # Training outcomes must be known before
    # the first validation decision.
    first_validation_date = (
        validation["Decision Date"].min()
    )

    train = df.loc[
        (df["Decision Date"] < first_validation_date)
        & (df["Outcome Date"] < first_validation_date)
    ].copy()

    # Refit after model selection, but only on
    # outcomes known before the first test decision.
    first_test_date = test["Decision Date"].min()

    final_train = df.loc[
        (df["Decision Date"] < first_test_date)
        & (df["Outcome Date"] < first_test_date)
    ].copy()

    if train.empty or final_train.empty:
        raise ValueError(
            "Insufficient training observations."
        )

    assert (
        train["Outcome Date"].max()
        < first_validation_date
    )

    assert (
        final_train["Outcome Date"].max()
        < first_test_date
    )

    return train, validation, final_train, test


def evaluate_recommendations(df, predictions):
    evaluated = df.copy()

    evaluated["Predicted Improvement"] = predictions

    if not np.isfinite(predictions).all():
        raise ValueError(
            "Non-finite model predictions."
        )

    # Identify the highest-scoring candidate
    # separately for each decision date.
    selected_indices = (
        evaluated.groupby("Decision Date")[
            "Predicted Improvement"
        ].idxmax()
    )

    selected = evaluated.loc[
        selected_indices
    ].sort_values("Decision Date").copy()

    # If every predicted hedge is worse than
    # remaining unhedged, select no hedge.
    selected["Choose Hedge"] = (
        selected["Predicted Improvement"] > 0
    )

    selected["Realized Improvement"] = np.where(
        selected["Choose Hedge"],
        selected["Hedge Improvement"],
        0.0,
    )

    selected["Realized Drawdown Improvement"] = np.where(
        selected["Choose Hedge"],
        selected["Drawdown Improvement"],
        0.0,
    )

    oracle = evaluated.groupby(
        "Decision Date"
    )["Hedge Improvement"].max().clip(lower=0.0)

    realized = selected.set_index(
        "Decision Date"
    )["Realized Improvement"]

    regret = (
        oracle.reindex(realized.index)
        - realized
    )

    return {
        "Decision Dates": len(selected),
        "Hedges Chosen": int(
            selected["Choose Hedge"].sum()
        ),
        "Mean Improvement (pp)": (
            realized.mean() * 100
        ),
        "Mean Drawdown Benefit (pp)": (
            selected[
                "Realized Drawdown Improvement"
            ].mean() * 100
        ),
        "Mean Regret (pp)": (
            regret.mean() * 100
        ),
        "Mean Absolute Error (pp)": (
            np.abs(
                evaluated[TARGET].to_numpy()
                - predictions
            ).mean() * 100
        ),
    }


def compare_models(train, validation):
    scores = []

    for name, model in make_models().items():
        model.fit(
            train[FEATURES],
            train[TARGET],
        )

        predictions = model.predict(
            validation[FEATURES]
        )

        metrics = evaluate_recommendations(
            validation,
            predictions,
        )

        scores.append({
            "Model": name,
            **metrics,
        })

    table = pd.DataFrame(scores).sort_values(
        [
            "Mean Regret (pp)",
            "Mean Absolute Error (pp)",
            "Model",
        ],
        ascending=[True, True, True],
    )

    return table


def main():
    df = load_dataset()

    train, validation, final_train, test = (
        split_dataset(df)
    )

    print("\n--- DATASET ---")
    print("Total candidates:", len(df))
    print(
        "Decision dates:",
        df["Decision Date"].nunique(),
    )

    for name, frame in [
        ("Training", train),
        ("Validation", validation),
        ("Final training", final_train),
        ("Test", test),
    ]:
        print(
            f"{name}: {len(frame)} candidates, "
            f"{frame['Decision Date'].nunique()} "
            "decision dates"
        )

    print("\n--- 2023 MODEL VALIDATION ---")

    comparison = compare_models(
        train,
        validation,
    )

    print(
        comparison.round(4).to_string(
            index=False
        )
    )

    best_name = comparison.iloc[0]["Model"]

    print("\nSelected model:", best_name)

    # Model selection uses 2023 only.
    # 2024 is not used until this point.
    final_model = make_models()[best_name]

    final_model.fit(
        final_train[FEATURES],
        final_train[TARGET],
    )

    final_predictions = final_model.predict(
        test[FEATURES]
    )

    test_metrics = evaluate_recommendations(
        test,
        final_predictions,
    )

    print("\n--- 2024 OUT-OF-SAMPLE TEST ---")
    print("Selected model:", best_name)

    for name, value in test_metrics.items():
        if isinstance(value, (float, np.floating)):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")


    # Audit the final model's 2024 recommendations.
    # This is diagnostic only; no model is retrained.

    audit = test.copy()

    audit["Predicted Improvement"] = (
        final_predictions
    )

    selected_indices = (
        audit.groupby("Decision Date")[
            "Predicted Improvement"
        ].idxmax()
    )

    selected = audit.loc[
        selected_indices
    ].sort_values("Decision Date").copy()

    selected["Choose Hedge"] = (
        selected["Predicted Improvement"] > 0
    )

    selected["Decision"] = np.where(
        selected["Choose Hedge"],
        selected["Strategy"],
        "No Hedge",
    )

    selected["Realized Improvement (pp)"] = (
        np.where(
            selected["Choose Hedge"],
            selected["Hedge Improvement"],
            0.0,
        ) * 100
    )

    selected["Drawdown Benefit (pp)"] = (
        np.where(
            selected["Choose Hedge"],
            selected["Drawdown Improvement"],
            0.0,
        ) * 100
    )

    selected["Predicted Improvement (pp)"] = (
        selected["Predicted Improvement"] * 100
    )

    selected["Expiry"] = np.where(
        selected["Choose Hedge"],
        selected["Expiry"].astype(str),
        "N/A",
    )

    columns = [
        "Decision Date",
        "Decision",
        "Expiry",
        "Predicted Improvement (pp)",
        "Realized Improvement (pp)",
        "Drawdown Benefit (pp)",
    ]

    print("\n--- 2024 MONTHLY RECOMMENDATIONS ---")

    print(
        selected[columns]
        .round(3)
        .to_string(index=False)
    )

    print("\n--- STRATEGY SELECTION COUNTS ---")

    print(
        selected["Decision"]
        .value_counts()
        .to_string()
    )


    print("\nBenchmark finished.")
    print("No historical files were modified.")


if __name__ == "__main__":
    main()
