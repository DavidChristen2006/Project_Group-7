
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from ml.ml_dataset_preparation import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMNS,
    load_ml_dataset,
    temporal_train_test_split,
)


def build_elastic_net():
    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        ),
    ])

    preprocessing = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        ),
    ])

    model = Pipeline([
        (
            "preprocessing",
            preprocessing
        ),
        (
            "regression",
            ElasticNet(
                alpha=0.001,
                l1_ratio=0.5,
                max_iter=100000,
                tol=1e-6
            )
        ),
    ])

    return model


def historical_mean_baseline(
    train,
    test,
    target
):
    group_columns = [
        "Hedge Horizon Days",
        "Strategy"
    ]

    group_means = (
        train.groupby(group_columns)[target]
        .mean()
    )

    global_mean = train[target].mean()

    keys = pd.MultiIndex.from_frame(
        test[group_columns]
    )

    predictions = (
        group_means
        .reindex(keys)
        .to_numpy(dtype=float)
    )

    return np.where(
        np.isnan(predictions),
        global_mean,
        predictions
    )


def evaluate_model(
    train,
    test,
    target
):
    model = build_elastic_net()

    X_train = train[FEATURE_COLUMNS]
    X_test = test[FEATURE_COLUMNS]

    y_train = train[target].to_numpy(
        dtype=float
    )

    y_test = test[target].to_numpy(
        dtype=float
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    baseline_predictions = (
        historical_mean_baseline(
            train,
            test,
            target
        )
    )

    model_mae = mean_absolute_error(
        y_test,
        predictions
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions
    )

    result = test[
        [
            "Decision Date",
            "Hedge Horizon Days",
            "Strategy",
        ]
    ].copy()

    result["Actual"] = y_test
    result["Elastic Net"] = predictions
    result["Historical Mean"] = (
        baseline_predictions
    )

    print(f"\n--- {target.upper()} ---")

    print(f"Elastic Net MAE: {model_mae:.4%}")
    print(f"Historical mean MAE: {baseline_mae:.4%}")

    improvement = (
        baseline_mae - model_mae
    )

    print(
        "MAE improvement vs baseline: "
        f"{improvement:.4%}"
    )

    print("\nPredictions:")
    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.4%}"
        )
    )

    return model, result


def main():
    dataset = load_ml_dataset()

    train, test, excluded = (
        temporal_train_test_split(dataset)
    )

    print("\n--- ELASTIC NET TRAINING ---")

    print(f"Training observations: {len(train)}")
    print(f"Test observations: {len(test)}")
    print(
        f"Excluded overlapping outcomes: "
        f"{len(excluded)}"
    )

    for target in TARGET_COLUMNS:
        evaluate_model(
            train,
            test,
            target
        )

    print("\nElastic Net evaluation completed.")


if __name__ == "__main__":
    main()
