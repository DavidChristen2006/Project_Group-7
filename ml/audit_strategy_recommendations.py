
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVALUATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_evaluations"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_datasets"
    / "aapl_historical_hedging_2022_2023.parquet"
)


def main():
    predictions = pd.read_parquet(
        EVALUATION_DIR
        / "walk_forward_predictions.parquet"
    )

    dataset = pd.read_parquet(DATA_PATH)

    predictions["Decision Date"] = pd.to_datetime(
        predictions["Decision Date"]
    ).dt.normalize()

    dataset["Decision Date"] = pd.to_datetime(
        dataset["Decision Date"]
    ).dt.normalize()

    keys = [
        "Decision Date",
        "Hedge Horizon Days",
    ]

    records = []

    groups = predictions.groupby(
        keys + ["Target", "Model"]
    )

    for group_key, group in groups:
        date, horizon, target, model = group_key

        group = group.sort_values(
            "Strategy"
        ).reset_index(drop=True)

        selected = group.loc[
            group["Predicted"].idxmax()
        ]

        best = group.loc[
            group["Actual"].idxmax()
        ]

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

            selected_actual = (
                0.0
                if model_selects_no_hedge
                else float(selected["Actual"])
            )

            selected_strategy = (
                "No Hedge"
                if model_selects_no_hedge
                else selected["Strategy"]
            )

            hindsight_best = max(
                0.0,
                float(best["Actual"])
            )

            hindsight_strategy = (
                "No Hedge"
                if float(best["Actual"]) < 0
                else best["Strategy"]
            )

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

    audit = pd.DataFrame(records)

    print("\n--- SELECTED STRATEGIES ---")

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

    print(
        pd.crosstab(
            audit["Target"],
            audit["Hindsight Best"],
        ).to_string()
    )

    print("\n--- RECOMMENDATION AUDIT ---")

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

    summary["Mean_Regret"] *= 100
    summary["Mean_Selected_Actual"] *= 100

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

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
