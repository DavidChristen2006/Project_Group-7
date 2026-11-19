
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

STRATEGY_PRIORITY = [
    "Collar",
    "Protective Put",
    "Put Spread",
]

GROUP_COLUMNS = [
    "Decision Date",
    "Hedge Horizon Days",
    "Target",
]

MODEL_COLUMNS = GROUP_COLUMNS + ["Model"]

KEY_COLUMNS = [
    "Decision Date",
    "Hedge Horizon Days",
]

TARGETS = [
    "Hedge Improvement",
    "Maximum Drawdown",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ML hedging recommendations "
            "against simple decision rules."
        )
    )

    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        required=True,
        help="Directory containing walk-forward predictions",
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Consolidated historical ML dataset",
    )

    return parser.parse_args()


def resolve_path(path):
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_inputs(evaluation_dir, dataset_path):
    predictions_path = (
        evaluation_dir
        / "walk_forward_predictions.parquet"
    )

    predictions = pd.read_parquet(
        predictions_path
    ).copy()

    dataset = pd.read_parquet(
        dataset_path
    ).copy()

    required_predictions = set(
        MODEL_COLUMNS
        + ["Strategy", "Predicted", "Actual"]
    )

    required_dataset = set(
        KEY_COLUMNS
        + [
            "Strategy",
            "Unhedged Maximum Drawdown",
        ]
    )

    if not required_predictions.issubset(
        predictions.columns
    ):
        raise ValueError(
            "Prediction file is missing required columns: "
            f"{sorted(required_predictions - set(predictions.columns))}"
        )

    if not required_dataset.issubset(
        dataset.columns
    ):
        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(required_dataset - set(dataset.columns))}"
        )

    for df in (predictions, dataset):
        df["Decision Date"] = pd.to_datetime(
            df["Decision Date"]
        ).dt.normalize()

        df["Hedge Horizon Days"] = pd.to_numeric(
            df["Hedge Horizon Days"]
        ).astype(int)

    if predictions.empty:
        raise ValueError(
            "Prediction file contains no observations."
        )

    if not predictions["Target"].isin(TARGETS).all():
        raise ValueError(
            "Unexpected prediction target."
        )

    for column in ["Actual", "Predicted"]:
        predictions[column] = pd.to_numeric(
            predictions[column],
            errors="raise",
        )

    if not np.isfinite(
        predictions[["Actual", "Predicted"]].to_numpy()
    ).all():
        raise ValueError(
            "Predictions contain non-finite values."
        )

    if predictions.duplicated(
        MODEL_COLUMNS + ["Strategy"]
    ).any():
        raise ValueError(
            "Duplicate prediction rows detected."
        )

    dataset["Unhedged Maximum Drawdown"] = (
        pd.to_numeric(
            dataset["Unhedged Maximum Drawdown"],
            errors="raise",
        )
    )

    # Each decision date and horizon has one
    # unhedged drawdown, repeated across strategies.
    unhedged = dataset[
        KEY_COLUMNS + ["Unhedged Maximum Drawdown"]
    ].drop_duplicates()

    if unhedged.duplicated(KEY_COLUMNS).any():
        raise ValueError(
            "Conflicting unhedged drawdowns "
            "for the same decision and horizon."
        )

    if not np.isfinite(
        unhedged["Unhedged Maximum Drawdown"].to_numpy()
    ).all():
        raise ValueError(
            "Unhedged drawdowns contain invalid values."
        )

    unhedged = unhedged.set_index(
        KEY_COLUMNS
    )["Unhedged Maximum Drawdown"]

    return predictions, unhedged


def rule_based_selection(group):
    for strategy in STRATEGY_PRIORITY:
        available = group.loc[
            group["Strategy"] == strategy
        ]

        if not available.empty:
            return available.iloc[0]

    raise ValueError(
        "No available option strategy."
    )


def make_record(
    date,
    horizon,
    target,
    approach,
    strategy,
    realized,
    oracle,
):
    return {
        "Decision Date": date,
        "Horizon": int(horizon),
        "Target": target,
        "Approach": approach,
        "Selected Strategy": strategy,
        "Realized Outcome": float(realized),
        "Regret": max(
            0.0,
            float(oracle - realized),
        ),
    }


def evaluate_benchmarks(predictions, unhedged):
    records = []

    for keys, group in predictions.groupby(
        GROUP_COLUMNS,
        sort=True,
    ):
        date, horizon, target = keys

        key = (date, int(horizon))

        if key not in unhedged.index:
            raise ValueError(
                f"Missing No Hedge drawdown for {key}."
            )

        if target == "Hedge Improvement":
            no_hedge_actual = 0.0

        elif target == "Maximum Drawdown":
            no_hedge_actual = float(
                unhedged.loc[key]
            )

        else:
            raise ValueError(
                f"Unexpected target: {target}"
            )

        # Validate identical realized outcomes
        # across all model prediction rows.
        actual_by_strategy = (
            group.groupby("Strategy")["Actual"]
            .agg(["min", "max"])
        )

        if not np.allclose(
            actual_by_strategy["min"],
            actual_by_strategy["max"],
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(
                f"Actual outcomes differ across models: {key}"
            )

        actual_options = (
            group.drop_duplicates("Strategy")
            .sort_values("Strategy")
            .reset_index(drop=True)
        )

        oracle = max(
            float(actual_options["Actual"].max()),
            no_hedge_actual,
        )

        # One Always Collar decision per
        # date, horizon and target.
        rule_best = rule_based_selection(
            actual_options
        )

        records.append(
            make_record(
                date,
                horizon,
                target,
                "Always Collar Rule",
                rule_best["Strategy"],
                rule_best["Actual"],
                oracle,
            )
        )

        # One No Hedge decision per group.
        records.append(
            make_record(
                date,
                horizon,
                target,
                "No Hedge",
                "No Hedge",
                no_hedge_actual,
                oracle,
            )
        )

        # One recommendation per ML model.
        for model, model_group in group.groupby(
            "Model",
            sort=True,
        ):
            if set(model_group["Strategy"]) != set(
                actual_options["Strategy"]
            ):
                raise ValueError(
                    f"Inconsistent strategy availability: "
                    f"{key}, {model}"
                )

            predicted_best = model_group.loc[
                model_group["Predicted"].idxmax()
            ]

            selected_strategy = (
                predicted_best["Strategy"]
            )

            realized = float(
                predicted_best["Actual"]
            )

            # For Hedge Improvement, No Hedge
            # has a known predicted value of zero.
            # Do not use future actual values here.
            if (
                target == "Hedge Improvement"
                and float(predicted_best["Predicted"]) <= 0
            ):
                selected_strategy = "No Hedge"
                realized = 0.0

            records.append(
                make_record(
                    date,
                    horizon,
                    target,
                    model,
                    selected_strategy,
                    realized,
                    oracle,
                )
            )

    return pd.DataFrame(records)


def summarize_results(results):
    summary = (
        results.groupby(
            ["Target", "Approach"]
        )
        .agg(
            Decisions=("Regret", "size"),
            Mean_Regret=("Regret", "mean"),
            Mean_Realized_Outcome=(
                "Realized Outcome",
                "mean",
            ),
            Zero_Regret_Rate=(
                "Regret",
                lambda values: np.mean(
                    np.isclose(
                        values,
                        0.0,
                        atol=1e-12,
                    )
                ),
            ),
        )
        .reset_index()
    )

    # Returns and drawdowns are stored as
    # fractions; display in percentage points.
    for column in [
        "Mean_Regret",
        "Mean_Realized_Outcome",
        "Zero_Regret_Rate",
    ]:
        summary[column] *= 100

    return summary


def main():
    args = parse_arguments()

    evaluation_dir = resolve_path(
        args.evaluation_dir
    )

    dataset_path = resolve_path(
        args.dataset
    )

    predictions, unhedged = load_inputs(
        evaluation_dir,
        dataset_path,
    )

    results = evaluate_benchmarks(
        predictions,
        unhedged,
    )

    summary = summarize_results(
        results
    )

    print("\n--- RULE-BASED BENCHMARK ---")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\nMean regret and outcome: percentage points; "
        "Zero Regret Rate: percent."
    )

    print("\n--- STRATEGIES SELECTED ---")

    print(
        pd.crosstab(
            [
                results["Target"],
                results["Approach"],
            ],
            results["Selected Strategy"],
        ).to_string()
    )

    output_path = (
        evaluation_dir
        / "rule_based_benchmark.parquet"
    )

    summary_path = (
        evaluation_dir
        / "rule_based_benchmark_summary.csv"
    )

    results.to_parquet(
        output_path,
        index=False
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print(f"\nSaved details to: {output_path}")
    print(f"Saved summary to: {summary_path}")
    print("\nRule-based benchmark completed.")


if __name__ == "__main__":
    main()
