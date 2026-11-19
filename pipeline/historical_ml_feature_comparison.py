
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from pipeline.historical_ml_benchmark import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    evaluate_recommendations,
    split_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_feature_datasets"
    / "AAPL_O_20220101_20241231_H21"
    / "completed_candidates_with_market_features.parquet"
)

MARKET_FEATURES = [
    "Momentum_21D",
    "Momentum_63D",
    "Realized_Volatility_21D",
    "Realized_Volatility_63D",
    "Historical_Drawdown_63D",
    "Distance_From_High_63D",
]


def make_model(include_market_features):
    numeric_features = NUMERIC_FEATURES.copy()

    if include_market_features:
        numeric_features += MARKET_FEATURES

    preprocessor = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value=0.0,
                        add_indicator=True,
                    ),
                ),
                ("scaler", StandardScaler()),
            ]),
            numeric_features,
        ),
        (
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]),
            CATEGORICAL_FEATURES,
        ),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    features = (
        numeric_features
        + CATEGORICAL_FEATURES
    )

    return model, features


def evaluate_model(
    name,
    train,
    validation,
    final_train,
    test,
    include_market_features,
):
    model, features = make_model(
        include_market_features
    )

    model.fit(
        train[features],
        train[TARGET],
    )

    validation_predictions = model.predict(
        validation[features]
    )

    validation_metrics = evaluate_recommendations(
        validation,
        validation_predictions,
    )

    # Refit on observations whose outcomes were
    # known before the first 2024 decision.
    model, features = make_model(
        include_market_features
    )

    model.fit(
        final_train[features],
        final_train[TARGET],
    )

    test_predictions = model.predict(
        test[features]
    )

    test_metrics = evaluate_recommendations(
        test,
        test_predictions,
    )

    return [
        {
            "Model": name,
            "Period": "2023 validation",
            **validation_metrics,
        },
        {
            "Model": name,
            "Period": "2024 exploratory",
            **test_metrics,
        },
    ]


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_parquet(DATA_PATH)

    if len(df) != 4424:
        raise ValueError(
            f"Unexpected candidate count: {len(df)}"
        )

    if (
        df["Candidate Key"].isna().any()
        or not df["Candidate Key"].is_unique
    ):
        raise ValueError(
            "Invalid candidate keys."
        )

    required_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + MARKET_FEATURES
        + [
            TARGET,
            "Decision Date",
            "Outcome Date",
            "Candidate Key",
            "Drawdown Improvement",
            "Maximum Drawdown",
        ]
    )

    missing = [
        col for col in required_columns
        if col not in df.columns
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

    df = df.sort_values(
        ["Decision Date", "Candidate Key"]
    ).reset_index(drop=True)

    train, validation, final_train, test = (
        split_dataset(df)
    )

    print("\n--- DATASET ---")
    print("Candidates:", len(df))
    print("Decision dates:", df["Decision Date"].nunique())
    print("Training:", len(train))
    print("Validation:", len(validation))
    print("Final training:", len(final_train))
    print("2024 test observations:", len(test))

    results = []

    for name, include_market in [
        ("Original Random Forest", False),
        ("Market-Aware Random Forest", True),
    ]:
        results.extend(
            evaluate_model(
                name,
                train,
                validation,
                final_train,
                test,
                include_market,
            )
        )

    comparison = pd.DataFrame(results)

    columns = [
        "Period",
        "Model",
        "Hedges Chosen",
        "Mean Improvement (pp)",
        "Mean Drawdown Benefit (pp)",
        "Mean Regret (pp)",
        "Mean Absolute Error (pp)",
    ]

    print("\n--- FEATURE COMPARISON ---")

    print(
        comparison[columns]
        .round(4)
        .to_string(index=False)
    )

    print(
        "\n2024 results are exploratory because "
        "the period has already been examined."
    )

    print("\nNo files modified.")


if __name__ == "__main__":
    main()
