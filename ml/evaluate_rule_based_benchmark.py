"""Evaluate ML hedging recommendations against simple decision rules.

For every decision date, hedge horizon and target, the script compares
the strategy recommended by each ML model with two simple approaches and
with the best possible choice in hindsight (oracle):

    Always Collar Rule : first available strategy in STRATEGY_PRIORITY
                         (Collar, otherwise Protective Put, otherwise
                         Put Spread), independent of any prediction
    No Hedge           : no hedge at all
    ML model           : strategy with the highest predicted value
    Oracle             : best actual outcome of all options and of
                         No Hedge; used only to measure regret

Regret is the oracle outcome minus the realized outcome and is never
negative. The two targets are "Hedge Improvement" (higher is better) and
"Maximum Drawdown" (higher, i.e. a smaller drawdown, is better).

Returns and drawdowns are stored as fractions; the printed summary shows
them in percentage points.

Run from project root:
    py -m ml.evaluate_rule_based_benchmark

Reads:  walk_forward_predictions.parquet in the evaluation folder and
        the consolidated historical ML dataset (--dataset).
Writes: rule_based_benchmark.parquet and
        rule_based_benchmark_summary.csv in the evaluation folder.

GenAI usage: Claude (Anthropic) was used to explain the code and to
draft, review and correct comments and docstrings. The comments were
checked and adapted by the author(s).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# two folders above this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# order of preference of the "Always Collar Rule": the first strategy
# in this list that is available is chosen
STRATEGY_PRIORITY = [
    "Collar",
    "Protective Put",
    "Put Spread",
]

# one decision is made per decision date, horizon and target
GROUP_COLUMNS = [
    "Decision Date",
    "Hedge Horizon Days",
    "Target",
]

# a model recommendation additionally depends on the model
MODEL_COLUMNS = GROUP_COLUMNS + ["Model"]

# decision date and horizon identify one unhedged drawdown
KEY_COLUMNS = [
    "Decision Date",
    "Hedge Horizon Days",
]

# targets that the models predict
TARGETS = [
    "Hedge Improvement",
    "Maximum Drawdown",
]


def parse_arguments():
    """Read the command-line arguments.

    Command-line arguments:
        --evaluation-dir: Folder with walk_forward_predictions.parquet
            (required).
        --dataset: Consolidated historical ML dataset (required).

    Returns:
        argparse.Namespace with the attributes evaluation_dir and
        dataset (both Path).
    """

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
    """Turn a relative path into a path inside the project.

    Args:
        path: Absolute or relative Path.

    Returns:
        The path itself if it is absolute, otherwise the path relative
        to PROJECT_ROOT.
    """

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_inputs(evaluation_dir, dataset_path):
    """Load and validate the predictions and the unhedged drawdowns.

    Args:
        evaluation_dir: Folder with walk_forward_predictions.parquet.
        dataset_path: Path to the consolidated historical ML dataset.

    Returns:
        Tuple (predictions, unhedged). predictions is a DataFrame with
        one row per decision date, horizon, target, model and strategy.
        unhedged is a Series with the unhedged maximum drawdown, indexed
        by decision date and horizon.

    Raises:
        ValueError: if required columns are missing, the predictions are
            empty, have an unexpected target, non-finite values or
            duplicates, or if the unhedged drawdowns conflict or are not
            finite.
    """

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

    # use the same date and horizon format in both tables so that they
    # can be matched
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

    # each model must predict each strategy only once per decision
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
    # After drop_duplicates, a key that still occurs twice has two
    # different values, i.e. a conflict.
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
    """Choose a strategy by the fixed priority order.

    Args:
        group: DataFrame with one row per available strategy and a
            "Strategy" column.

    Returns:
        The row of the first strategy in STRATEGY_PRIORITY that is
        available in the group.

    Raises:
        ValueError: if none of the priority strategies is available.
    """

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
    """Create the result record of one decision.

    Args:
        date: Decision date.
        horizon: Hedge horizon in days.
        target: "Hedge Improvement" or "Maximum Drawdown".
        approach: Name of the approach, e.g. "No Hedge" or a model.
        strategy: Strategy that the approach selected.
        realized: Actual outcome of the selected strategy.
        oracle: Best possible outcome in hindsight.

    Returns:
        Dict with the decision data, the realized outcome and the
        regret (oracle minus realized, at least 0).
    """

    return {
        "Decision Date": date,
        "Horizon": int(horizon),
        "Target": target,
        "Approach": approach,
        "Selected Strategy": strategy,
        "Realized Outcome": float(realized),
        # max(0, ...) protects against tiny negative values caused by
        # floating-point rounding
        "Regret": max(
            0.0,
            float(oracle - realized),
        ),
    }


def evaluate_benchmarks(predictions, unhedged):
    """Evaluate the rule, "No Hedge" and all models on every decision.

    For each decision date, horizon and target, the oracle is the best
    actual outcome of all strategies and of "No Hedge". Then the Always
    Collar rule, "No Hedge" and each model's recommendation are scored
    against it. For "Hedge Improvement", "No Hedge" has a known value of
    0, and a model chooses "No Hedge" if its best predicted value is 0
    or lower. For "Maximum Drawdown", the models choose only between the
    option strategies.

    Args:
        predictions: DataFrame as returned by load_inputs().
        unhedged: Series of unhedged maximum drawdowns, indexed by
            decision date and horizon.

    Returns:
        DataFrame with one row per decision and approach, created with
        make_record().

    Raises:
        ValueError: if an unhedged drawdown is missing, the target is
            unexpected, actual outcomes differ between models or a
            model does not cover the same strategies.
    """

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

        # actual outcome of not hedging: no improvement (0) or the
        # drawdown of the unhedged position
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
        # The actual outcome of a strategy does not depend on the
        # model, so min and max must be equal.
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

        # one row per strategy with its actual outcome
        actual_options = (
            group.drop_duplicates("Strategy")
            .sort_values("Strategy")
            .reset_index(drop=True)
        )

        # best outcome in hindsight, including not hedging
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
            # every model must have predicted the same strategies
            if set(model_group["Strategy"]) != set(
                actual_options["Strategy"]
            ):
                raise ValueError(
                    f"Inconsistent strategy availability: "
                    f"{key}, {model}"
                )

            # the strategy with the highest predicted value
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
    """Summarize regret and outcome per target and approach.

    Args:
        results: DataFrame as returned by evaluate_benchmarks().

    Returns:
        DataFrame with one row per target and approach: the number of
        decisions, the mean regret and mean realized outcome (in
        percentage points) and the share of decisions with zero regret
        (in percent).
    """

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
            # share of decisions where the approach chose the best
            # option (regret of 0, allowing for rounding)
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
    """Run the rule-based benchmark and save the results.

    Reads the predictions and the dataset, evaluates all approaches,
    prints a summary and the chosen strategies and saves the detailed
    results and the summary.

    Raises:
        ValueError: if the inputs are inconsistent (raised by the
            called functions).

    Side effects:
        Prints the result tables and writes
        rule_based_benchmark.parquet and
        rule_based_benchmark_summary.csv to the evaluation folder.
    """

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

    # how often each approach chose each strategy
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