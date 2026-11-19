
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml.ml_dataset_preparation import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    load_ml_dataset,
    temporal_train_test_split,
)

from ml.train_elastic_net import (
    build_elastic_net,
    historical_mean_baseline,
)


def build_gradient_boosting():
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        (
            "regression",
            HistGradientBoostingRegressor(
                max_iter=100,
                learning_rate=0.05,
                max_leaf_nodes=7,
                min_samples_leaf=10,
                l2_regularization=1.0,
                random_state=42,
            ),
        ),
    ])


def main():
    dataset = load_ml_dataset()

    train, test, excluded = temporal_train_test_split(
        dataset
    )

    print("\n--- MODEL COMPARISON ---")
    print(f"Training observations: {len(train)}")
    print(f"Test observations: {len(test)}")
    print(f"Excluded outcomes: {len(excluded)}")

    X_train = train[FEATURE_COLUMNS]
    X_test = test[FEATURE_COLUMNS]

    results = []

    for target in TARGET_COLUMNS:
        y_train = train[target].to_numpy(dtype=float)
        y_test = test[target].to_numpy(dtype=float)

        models = {
            "Elastic Net": build_elastic_net(),
            "Gradient Boosting": build_gradient_boosting(),
        }

        predictions = {
            "Historical Mean": historical_mean_baseline(
                train,
                test,
                target,
            )
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions[name] = model.predict(X_test)

        for name, predicted in predictions.items():
            results.append({
                "Target": target,
                "Model": name,
                "MAE": mean_absolute_error(
                    y_test,
                    predicted,
                ),
            })

        print(f"\n--- {target.upper()} ---")

        for name, predicted in predictions.items():
            mae = mean_absolute_error(
                y_test,
                predicted,
            )
            print(f"{name}: MAE {mae:.4%}")

    results_df = pd.DataFrame(results)

    print("\n--- COMPARISON SUMMARY ---")
    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4%}",
        )
    )

    print("\nModel comparison completed.")


if __name__ == "__main__":
    main()
