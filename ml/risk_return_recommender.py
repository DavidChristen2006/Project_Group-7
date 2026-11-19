
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TARGET_RETURN = "Hedge Improvement"
TARGET_RISK = "Maximum Drawdown"

KEY_COLUMNS = [
    "Decision Date",
    "Hedge Horizon Days",
]


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_data(evaluation_dir, dataset_path):
    predictions = pd.read_parquet(
        evaluation_dir / "walk_forward_predictions.parquet"
    ).copy()

    dataset = pd.read_parquet(dataset_path).copy()

    required_predictions = {
        *KEY_COLUMNS,
        "Strategy",
        "Model",
        "Target",
        "Predicted",
        "Actual",
    }

    required_dataset = {
        *KEY_COLUMNS,
        "Outcome Date",
        "Unhedged Maximum Drawdown",
    }

    if not required_predictions.issubset(predictions.columns):
        raise ValueError("Missing prediction columns.")

    if not required_dataset.issubset(dataset.columns):
        raise ValueError("Missing historical dataset columns.")

    for df in (predictions, dataset):
        df["Decision Date"] = pd.to_datetime(
            df["Decision Date"]
        ).dt.normalize()

        df["Hedge Horizon Days"] = pd.to_numeric(
            df["Hedge Horizon Days"]
        ).astype(int)

    dataset["Outcome Date"] = pd.to_datetime(
        dataset["Outcome Date"]
    ).dt.normalize()

    for column in ["Predicted", "Actual"]:
        predictions[column] = pd.to_numeric(
            predictions[column],
            errors="raise",
        )

    dataset["Unhedged Maximum Drawdown"] = pd.to_numeric(
        dataset["Unhedged Maximum Drawdown"],
        errors="raise",
    )

    if predictions.empty:
        raise ValueError("No walk-forward predictions found.")

    if not np.isfinite(
        predictions[["Predicted", "Actual"]].to_numpy()
    ).all():
        raise ValueError("Non-finite prediction values.")

    if predictions.duplicated(
        KEY_COLUMNS + ["Model", "Strategy", "Target"]
    ).any():
        raise ValueError("Duplicate predictions found.")

    # All strategies for the same decision and horizon
    # must share the same unhedged outcome and exit date.
    unhedged = dataset[
        KEY_COLUMNS
        + ["Outcome Date", "Unhedged Maximum Drawdown"]
    ].drop_duplicates()

    if unhedged.duplicated(KEY_COLUMNS).any():
        raise ValueError(
            "Conflicting unhedged outcomes in historical data."
        )

    if not np.isfinite(
        unhedged["Unhedged Maximum Drawdown"].to_numpy()
    ).all():
        raise ValueError("Invalid unhedged drawdowns.")

    if (
        unhedged["Outcome Date"]
        <= unhedged["Decision Date"]
    ).any():
        raise ValueError("Invalid historical outcome dates.")

    return predictions, unhedged


def historical_no_hedge_forecast(
    unhedged,
    decision_date,
    horizon,
):
    history = unhedged.loc[
        (unhedged["Hedge Horizon Days"] == horizon)
        & (unhedged["Outcome Date"] < decision_date)
    ]

    if history.empty:
        raise ValueError(
            f"No completed No Hedge history before "
            f"{decision_date.date()}, horizon {horizon}."
        )

    return float(
        history["Unhedged Maximum Drawdown"].mean()
    )


def evaluate_policy(
    predictions,
    unhedged,
    maximum_drawdown,
):
    records = []

    groups = predictions.groupby(
        KEY_COLUMNS + ["Model"],
        sort=True,
    )

    for (date, horizon, model), group in groups:
        horizon = int(horizon)

        # Each strategy needs both predictions.
        predicted = group.pivot(
            index="Strategy",
            columns="Target",
            values="Predicted",
        )

        actual = group.pivot(
            index="Strategy",
            columns="Target",
            values="Actual",
        )

        required_targets = [
            TARGET_RETURN,
            TARGET_RISK,
        ]

        if not set(required_targets).issubset(
            predicted.columns
        ):
            raise ValueError(
                f"Missing prediction targets: "
                f"{date.date()}, {horizon}, {model}"
            )

        if predicted[required_targets].isna().any().any():
            raise ValueError(
                "Incomplete strategy predictions."
            )

        if actual[required_targets].isna().any().any():
            raise ValueError(
                "Incomplete realized outcomes."
            )

        if not predicted.index.equals(actual.index):
            raise ValueError(
                "Prediction and outcome strategies differ."
            )

        # Historical No Hedge forecast uses strictly
        # completed outcomes from earlier decisions.
        no_hedge_predicted_mdd = (
            historical_no_hedge_forecast(
                unhedged,
                date,
                horizon,
            )
        )

        current_unhedged = unhedged.loc[
            (unhedged["Decision Date"] == date)
            & (unhedged["Hedge Horizon Days"] == horizon)
        ]

        if len(current_unhedged) != 1:
            raise ValueError(
                "Current unhedged outcome unavailable."
            )

        # This future value is used ONLY for evaluation,
        # never for choosing the recommendation.
        no_hedge_actual_mdd = float(
            current_unhedged.iloc[0][
                "Unhedged Maximum Drawdown"
            ]
        )

        candidates = []

        # No Hedge has zero return improvement.
        candidates.append({
            "Strategy": "No Hedge",
            "Predicted Improvement": 0.0,
            "Predicted Drawdown": no_hedge_predicted_mdd,
            "Actual Improvement": 0.0,
            "Actual Drawdown": no_hedge_actual_mdd,
        })

        for strategy in predicted.index:
            candidates.append({
                "Strategy": strategy,
                "Predicted Improvement": float(
                    predicted.loc[
                        strategy, TARGET_RETURN
                    ]
                ),
                "Predicted Drawdown": float(
                    predicted.loc[
                        strategy, TARGET_RISK
                    ]
                ),
                "Actual Improvement": float(
                    actual.loc[
                        strategy, TARGET_RETURN
                    ]
                ),
                "Actual Drawdown": float(
                    actual.loc[
                        strategy, TARGET_RISK
                    ]
                ),
            })

        # Maximum drawdown is negative.
        # Example: -0.07 satisfies a 10% risk limit.
        feasible = [
            candidate
            for candidate in candidates
            if candidate["Predicted Drawdown"]
            >= -maximum_drawdown
        ]

        if feasible:
            selected = max(
                feasible,
                key=lambda candidate: (
                    candidate["Predicted Improvement"],
                    candidate["Predicted Drawdown"],
                ),
            )

            status = "Feasible"

        else:
            # Do not pretend the risk constraint
            # can be satisfied when it cannot.
            selected = None
            status = "No Feasible Strategy"

        if selected is None:
            records.append({
                "Decision Date": date,
                "Horizon": horizon,
                "Model": model,
                "Risk Limit": maximum_drawdown,
                "Status": status,
                "Selected Strategy": None,
                "Predicted Improvement": np.nan,
                "Predicted Drawdown": np.nan,
                "Actual Improvement": np.nan,
                "Actual Drawdown": np.nan,
                "Realized Risk Violation": np.nan,
                "Feasible Candidates": 0,
            })

            continue

        records.append({
            "Decision Date": date,
            "Horizon": horizon,
            "Model": model,
            "Risk Limit": maximum_drawdown,
            "Status": status,
            "Selected Strategy": selected["Strategy"],
            "Predicted Improvement": (
                selected["Predicted Improvement"]
            ),
            "Predicted Drawdown": (
                selected["Predicted Drawdown"]
            ),
            "Actual Improvement": (
                selected["Actual Improvement"]
            ),
            "Actual Drawdown": (
                selected["Actual Drawdown"]
            ),
            "Realized Risk Violation": (
                selected["Actual Drawdown"]
                < -maximum_drawdown
            ),
            "Feasible Candidates": len(feasible),
        })

    return pd.DataFrame(records)


def summarize_results(results):
    summary = (
        results.groupby("Model")
        .agg(
            Decisions=("Status", "size"),
            Feasible_Decisions=(
                "Status",
                lambda values: (values == "Feasible").sum(),
            ),
            Mean_Improvement=(
                "Actual Improvement",
                "mean",
            ),
            Mean_Drawdown=(
                "Actual Drawdown",
                "mean",
            ),
            Realized_Violation_Rate=(
                "Realized Risk Violation",
                "mean",
            ),
        )
        .reset_index()
    )

    for column in [
        "Mean_Improvement",
        "Mean_Drawdown",
        "Realized_Violation_Rate",
    ]:
        summary[column] *= 100

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--evaluation-dir",
        required=True,
    )

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=0.10,
        help="Maximum predicted drawdown magnitude, e.g. 0.10",
    )

    args = parser.parse_args()

    if not 0 < args.max_drawdown <= 1:
        raise ValueError(
            "--max-drawdown must be between 0 and 1."
        )

    evaluation_dir = resolve_path(
        args.evaluation_dir
    )

    dataset_path = resolve_path(
        args.dataset
    )

    predictions, unhedged = load_data(
        evaluation_dir,
        dataset_path,
    )

    results = evaluate_policy(
        predictions,
        unhedged,
        maximum_drawdown=args.max_drawdown,
    )

    summary = summarize_results(results)

    print("\n--- RISK-RETURN RECOMMENDATIONS ---")

    display_columns = [
        "Decision Date",
        "Horizon",
        "Model",
        "Selected Strategy",
        "Status",
        "Actual Improvement",
        "Actual Drawdown",
    ]

    display_df = results[display_columns].copy()

    for column in [
        "Actual Improvement",
        "Actual Drawdown",
    ]:
        display_df[column] *= 100

    print(
        display_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\n--- RISK-RETURN SUMMARY ---")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\nImprovement and Drawdown: percentage-point "
        "values after multiplying by 100."
    )

    print(
        "\n--- STRATEGIES SELECTED ---"
    )

    print(
        pd.crosstab(
            results["Model"],
            results["Selected Strategy"],
        ).to_string()
    )

    evaluation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    risk_tag = f"{args.max_drawdown:.4f}".replace(
        ".", "p"
    )

    output_path = (
        evaluation_dir
        / f"risk_return_recommendations_{risk_tag}.parquet"
    )

    summary_path = (
        evaluation_dir
        / f"risk_return_summary_{risk_tag}.csv"
    )

    results.to_parquet(
        output_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(f"\nSaved to: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
