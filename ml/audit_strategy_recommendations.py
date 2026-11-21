"""
This script audits how strategy recommendations perform using 
walk-forward model predictions. It produces a summary and saves 
detailed audit data as a Parquet file.

Reads:  walk_forward_predictions.parquet, the hedging dataset
Writes: strategy_recommendation_audit.parquet (one row per
        decision date, horizon, target and model)
Prints: selection counts, hindsight winners, mean regret summary

GenAI usage: The explanations and comments in this file were
reviewed by Claude (Anthropic)
"""

from pathlib import Path

import numpy as np
import pandas as pd

# calculates the project root two folders above this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# EVALUATION_DIR is where model predictions are stored 
EVALUATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_evaluations"
)

# DATA_PATH is a hedge dataset from Apple 
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_datasets"
    / "aapl_historical_hedging_2022_2023.parquet"
)


def main():
    """
    This function evaluates which strategy is selected by the model, 
    what would have been the hindsight winner, how much was lost by
    following the model and how often each strategy was selected.
    The result of the audit is then stored in the file 
    'strategy_recommendation_audit.parquet' 
    """
    # predictions contains model predictions for individual strategies 
    predictions = pd.read_parquet(
        EVALUATION_DIR
        / "walk_forward_predictions.parquet"
    )

    dataset = pd.read_parquet(DATA_PATH)

    # remove any time components to ensure that all dates have the 
    # same format 
    predictions["Decision Date"] = pd.to_datetime(
        predictions["Decision Date"]
    ).dt.normalize()

    dataset["Decision Date"] = pd.to_datetime(
        dataset["Decision Date"]
    ).dt.normalize()

    # base grouping columns shared by all groups 
    keys = [
        "Decision Date",
        "Hedge Horizon Days",
    ]

    records = []

    # splits the predictions into separate groups based on
    # decision date, hedge horizon days, target and model 
    groups = predictions.groupby(
        keys + ["Target", "Model"]
    )

    # iterates through groups and reads the date, horizon, 
    # target and model
    for group_key, group in groups:
        date, horizon, target, model = group_key

        # sorts the strategy alphabetically that the first entry is chosen
        # if the result is the same 
        group = group.sort_values(
            "Strategy"
        ).reset_index(drop=True)

        # selects the model's strategy with the highest predicted value
        selected = group.loc[
            group["Predicted"].idxmax()
        ]

        # selects the hindsight winner with the strategy that actually
        # performed best for this group 
        best = group.loc[
            group["Actual"].idxmax()
        ]

        # selects the Collar row of this group 
        collar_rows = group.loc[
            group["Strategy"] == "Collar"
        ]

        collar = (
            collar_rows.iloc[0]
            if not collar_rows.empty
            else None
        )

        # Zero is the known hedge-improvement
        # benchmark for the No Hedge alternative.
        # It is NOT a forecast of stock return.
        if target == "Hedge Improvement":
            model_selects_no_hedge = (
                float(selected["Predicted"]) <= 0
            )

            # sets the actual result, which means No Hedge has a benchmark
            # actual improvement of 0.0
            selected_actual = (
                0.0
                if model_selects_no_hedge
                else float(selected["Actual"])
            )

            # recommendation becomes "No Hedge" if even the
            # best predicted strategy has an improvement 
            # equal to zero or negative 
            selected_strategy = (
                "No Hedge"
                if model_selects_no_hedge
                else selected["Strategy"]
            )

            # calculate the hindsight winner, which is either the 
            # hedging strategy if the best actual improvement is 
            # positive or No Hedge if the best actual improvement
            # is below zero  
            hindsight_best = max(
                0.0,
                float(best["Actual"])
            )

            hindsight_strategy = (
                "No Hedge"
                if float(best["Actual"]) < 0
                else best["Strategy"]
            )

            # reads the Collar result if available
            # 0.0 = no Hedge/ NaN = not comparable 
            collar_rule_actual = (
                float(collar["Actual"])
                if collar is not None
                else 0.0
            )

        else:
            # Drawdown predictions are compared
            # between available option strategies only.
            selected_actual = float(
                selected["Actual"]
            )

            selected_strategy = selected["Strategy"]
            hindsight_best = float(best["Actual"])
            hindsight_strategy = best["Strategy"]

            collar_rule_actual = (
                float(collar["Actual"])
                if collar is not None
                else np.nan
            )

        # create audit records as a dictionary 
        # regret is the difference between the hindsight best result
        # and the selected result and max(0.0, ...) guards against 
        # tiny negative values caused by floating-point rounding
        records.append({
            "Decision Date": date,
            "Horizon": int(horizon),
            "Target": target,
            "Model": model,
            "Selected Strategy": selected_strategy,
            "Hindsight Best": hindsight_strategy,
            "Selected Actual": selected_actual,
            "Best Actual": hindsight_best,
            "Regret": max(
                0.0,
                hindsight_best - selected_actual
            ),
            "Collar Rule Actual": collar_rule_actual,
        })

    # converts records into a DataFrame
    audit = pd.DataFrame(records)

    print("\n--- SELECTED STRATEGIES ---")

    # shows how often every strategy was selected by each target/ model 
    # combination 
    print(
        pd.crosstab(
            [
                audit["Target"],
                audit["Model"],
            ],
            audit["Selected Strategy"],
        ).to_string()
    )

    print("\n--- HINDSIGHT BEST STRATEGIES ---")

    # shows the number of times each strategy would have been the 
    # hindsight winner counted one per model 
    print(
        pd.crosstab(
            audit["Target"],
            audit["Hindsight Best"],
        ).to_string()
    )

    print("\n--- RECOMMENDATION AUDIT ---")

    # groups the audit by Target and Model and reports
    # the number of audited decisions, the average regret and
    # the average actual result from model-selected strategies 
    summary = audit.groupby(
        ["Target", "Model"]
    ).agg(
        Decisions=("Regret", "size"),
        Mean_Regret=("Regret", "mean"),
        Mean_Selected_Actual=(
            "Selected Actual",
            "mean"
        ),
    ).reset_index()

    # converts the printed result into percentage points while 
    # the saved file keeps the decimal numbers   
    summary["Mean_Regret"] *= 100
    summary["Mean_Selected_Actual"] *= 100

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # save the detailed audit in the file 
    # "strategy_recommendation_audit.parquet"
    output_path = (
        EVALUATION_DIR
        / "strategy_recommendation_audit.parquet"
    )

    audit.to_parquet(
        output_path,
        index=False
    )

    print(
        "\nAudit values are in percentage points."
    )
    print(f"Saved to: {output_path}")

    print("\nRecommendation audit completed.")


if __name__ == "__main__":
    main()
