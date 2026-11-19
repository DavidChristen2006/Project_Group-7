
from pathlib import Path

import numpy as np
import pandas as pd
import argparse

from sklearn.metrics import mean_absolute_error

from ml.ml_dataset_preparation import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    load_ml_dataset,
)

from ml.train_elastic_net import (
    build_elastic_net,
    historical_mean_baseline,
)

from ml.train_gradient_boosting import (
    build_gradient_boosting,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_evaluations"
    / "aapl_2023_jul_dec"
)

TEST_START = pd.Timestamp("2023-07-03")

MIN_TRAINING_DATES = 6

MODEL_NAMES = [
    "Historical Mean",
    "Elastic Net",
    "Gradient Boosting",
]


def build_models():
    return {
        "Elastic Net": build_elastic_net(),
        "Gradient Boosting": build_gradient_boosting(),
    }


def get_walk_forward_split(df, test_date):
    test_date = pd.Timestamp(test_date).normalize()

    # Only completed historical outcomes are
    # eligible for training.
    train = df.loc[
        (df["Decision Date"] < test_date)
        & (df["Outcome Date"] < test_date)
    ].copy()

    # All completed strategy records initiated
    # on the current decision date.
    test = df.loc[
        df["Decision Date"] == test_date
    ].copy()

    # Earlier decisions with outcomes that
    # were not yet available.
    excluded = df.loc[
        (df["Decision Date"] < test_date)
        & (df["Outcome Date"] >= test_date)
    ].copy()

    if not train.empty:
        assert (
            train["Outcome Date"] < test_date
        ).all()

    assert (
        test["Decision Date"] == test_date
    ).all()

    return train, test, excluded


def predict_one_target(train, test, target):
    X_train = train[FEATURE_COLUMNS]
    X_test = test[FEATURE_COLUMNS]

    y_train = train[target].to_numpy(dtype=float)

    predictions = {
        "Historical Mean": historical_mean_baseline(
            train,
            test,
            target,
        )
    }

    for name, model in build_models().items():
        model.fit(X_train, y_train)

        predictions[name] = model.predict(X_test)

    return predictions


def make_prediction_records(
    test,
    target,
    predictions,
):
    records = []

    actual_values = test[target].to_numpy(dtype=float)

    for model_name, predicted_values in predictions.items():
        for position, (_, row) in enumerate(test.iterrows()):
            records.append({
                "Decision Date": row["Decision Date"],
                "Outcome Date": row["Outcome Date"],
                "Hedge Horizon Days": int(
                    row["Hedge Horizon Days"]
                ),
                "Strategy": row["Strategy"],
                "Target": target,
                "Model": model_name,
                "Actual": actual_values[position],
                "Predicted": float(
                    predicted_values[position]
                ),
            })

    return records


def evaluate_recommendations(predictions_df):
    records = []

    group_columns = [
        "Decision Date",
        "Hedge Horizon Days",
        "Target",
        "Model",
    ]

    for keys, group in predictions_df.groupby(
        group_columns,
        sort=True,
    ):
        (
            decision_date,
            horizon,
            target,
            model_name,
        ) = keys

        # Strategies are sorted to provide
        # deterministic tie-breaking.
        group = group.sort_values(
            "Strategy"
        ).reset_index(drop=True)

        # Higher Hedge Improvement is better.
        # Higher Maximum Drawdown is also
        # better because it is less negative.
        selected_position = int(
            group["Predicted"].to_numpy().argmax()
        )

        best_position = int(
            group["Actual"].to_numpy().argmax()
        )

        selected = group.iloc[selected_position]
        best = group.iloc[best_position]

        realized_value = float(
            selected["Actual"]
        )

        hindsight_best = float(
            best["Actual"]
        )

        regret = max(
            0.0,
            hindsight_best - realized_value,
        )

        records.append({
            "Decision Date": decision_date,
            "Hedge Horizon Days": int(horizon),
            "Target": target,
            "Model": model_name,
            "Selected Strategy": selected["Strategy"],
            "Best Strategy": best["Strategy"],
            "Selected Actual": realized_value,
            "Best Actual": hindsight_best,
            "Regret": regret,
            "Correct Selection": bool(
                np.isclose(
                    realized_value,
                    hindsight_best,
                    rtol=0.0,
                    atol=1e-10,
                )
            ),
            "Available Strategies": len(group),
        })

    return pd.DataFrame(records)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to consolidated ML dataset",
    )

    parser.add_argument(
        "--test-start",
        default="2023-07-03",
        help="First walk-forward evaluation date",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for evaluation results",
    )

    args = parser.parse_args()

    dataset = load_ml_dataset(args.dataset)

    test_start = pd.Timestamp(
        args.test_start
    ).normalize()

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "ml_evaluations"
            / f"walk_forward_{test_start:%Y%m%d}"
        )


    test_dates = sorted(
        dataset.loc[
            dataset["Decision Date"] >= test_start,
            "Decision Date"
        ].unique()
    )

    prediction_records = []
    fold_records = []

    print("\n--- WALK-FORWARD VALIDATION ---")

    for test_date in test_dates:
        test_date = pd.Timestamp(test_date)

        train, test, excluded = get_walk_forward_split(
            dataset,
            test_date,
        )

        n_dates = train["Decision Date"].nunique()

        if (
            test.empty
            or n_dates < MIN_TRAINING_DATES
        ):
            print(
                f"Skipping {test_date.date()}: "
                f"insufficient training history."
            )
            continue

        print(
            f"\nTest: {test_date.date()} | "
            f"Train: {len(train)} | "
            f"Test: {len(test)} | "
            f"Unfinished: {len(excluded)}"
        )

        fold_records.append({
            "Test Date": test_date,
            "Training Observations": len(train),
            "Training Decision Dates": n_dates,
            "Test Observations": len(test),
            "Unfinished Outcomes": len(excluded),
        })

        for target in TARGET_COLUMNS:
            predictions = predict_one_target(
                train,
                test,
                target,
            )

            prediction_records.extend(
                make_prediction_records(
                    test,
                    target,
                    predictions,
                )
            )

    if not prediction_records:
        raise ValueError(
            "No eligible walk-forward test folds."
        )

    predictions_df = pd.DataFrame(
        prediction_records
    )

    recommendations_df = evaluate_recommendations(
        predictions_df
    )

    folds_df = pd.DataFrame(fold_records)

    predictions_df["Absolute Error"] = (
        predictions_df["Actual"]
        - predictions_df["Predicted"]
    ).abs()

    print("\n--- PREDICTION ACCURACY ---")

    accuracy = (
        predictions_df.groupby(
            ["Target", "Model"]
        )["Absolute Error"]
        .agg(
            MAE="mean",
            Observations="count",
        )
        .reset_index()
    )

    accuracy_display = accuracy.copy()

    accuracy_display["MAE"] = (
        accuracy_display["MAE"] * 100
    )

    print(
        accuracy_display.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nMAE values above are percentage points.")

    print("\n--- STRATEGY RECOMMENDATION QUALITY ---")

    recommendation_summary = (
        recommendations_df.groupby(
            ["Target", "Model"]
        ).agg(
            Mean_Regret=("Regret", "mean"),
            Correct_Selection_Rate=(
                "Correct Selection",
                "mean",
            ),
            Decisions=("Regret", "count"),
        )
        .reset_index()
    )

    recommendation_display = (
        recommendation_summary.copy()
    )

    recommendation_display["Mean_Regret"] *= 100

    recommendation_display[
        "Correct_Selection_Rate"
    ] *= 100

    print(
        recommendation_display.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\nMean Regret: percentage points; "
        "Correct Selection Rate: percent."
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_df.to_parquet(
        output_dir
        / "walk_forward_predictions.parquet",
        index=False,
    )

    recommendations_df.to_parquet(
        output_dir
        / "walk_forward_recommendations.parquet",
        index=False,
    )

    folds_df.to_parquet(
        output_dir
        / "walk_forward_folds.parquet",
        index=False,
    )

    print("\n--- EVALUATION SUMMARY ---")

    print(f"Test dates: {len(folds_df)}")

    print(
        "Predicted strategy observations: "
        f"{len(predictions_df) // (
            len(TARGET_COLUMNS) * len(MODEL_NAMES)
        )}"
    )

    print(
        "Recommendation comparisons: "
        f"{len(recommendations_df)}"
    )

    print(f"Saved to: {output_dir}")

    print("\nWalk-forward validation completed.")


if __name__ == "__main__":
    main()
