
import numpy as np
import pandas as pd

from ml.candidate_uncertainty_audit import (
    collect_decisions,
    MODELS,
    CALIBRATION_START,
    CALIBRATION_END,
    TEST_START,
    TEST_END,
)

from ml.candidate_walk_forward_comparison import (
    load_dataset,
)

from ml.candidate_decision_audit import (
    RISK_WEIGHT,
)


def analyse_period(df, model, period, start, end):
    decisions = collect_decisions(
        df, model, start, end
    )

    hedges = decisions.loc[
        decisions["Decision"].ne("No Hedge")
    ].copy()

    if hedges.empty:
        raise ValueError(
            f"No selected hedges: {model}, {period}"
        )

    hedges["Return Optimism (pp)"] = (
        hedges["Predicted Return (pp)"]
        - hedges["Actual Return (pp)"]
    )

    hedges["Drawdown Optimism (pp)"] = (
        hedges["Predicted Risk (pp)"]
        - hedges["Actual Risk (pp)"]
    )

    hedges["Weighted Drawdown Optimism (pp)"] = (
        RISK_WEIGHT
        * hedges["Drawdown Optimism (pp)"]
    )

    hedges["Utility Optimism (pp)"] = (
        hedges["Predicted Utility (pp)"]
        - hedges["Actual Utility (pp)"]
    )

    reconstructed = (
        hedges["Return Optimism (pp)"]
        + hedges["Weighted Drawdown Optimism (pp)"]
    )

    if not np.allclose(
        reconstructed,
        hedges["Utility Optimism (pp)"],
        atol=1e-8,
    ):
        raise ValueError(
            "Utility error decomposition failed."
        )

    hedges["Call OTM (%)"] = (
        hedges["Call Strike %"] - 100
    )

    hedges["Negative Utility"] = (
        hedges["Actual Utility (pp)"] < 0
    )

    hedges["Model"] = model
    hedges["Period"] = period

    return hedges


def main():
    df = load_dataset()

    periods = [
        (
            "2023 validation",
            CALIBRATION_START,
            CALIBRATION_END,
        ),
        (
            "2024 exploratory",
            TEST_START,
            TEST_END,
        ),
    ]

    frames = []

    for period, start, end in periods:
        for model in MODELS:
            print(
                f"Evaluating {model}: {period}"
            )

            frames.append(
                analyse_period(
                    df, model, period, start, end
                )
            )

    results = pd.concat(
        frames, ignore_index=True
    )

    summary = (
        results.groupby(["Period", "Model"])
        .agg(
            Hedges=("Decision", "size"),
            Return_Optimism_pp=(
                "Return Optimism (pp)", "mean"
            ),
            Drawdown_Optimism_pp=(
                "Drawdown Optimism (pp)", "mean"
            ),
            Weighted_Drawdown_Optimism_pp=(
                "Weighted Drawdown Optimism (pp)",
                "mean",
            ),
            Utility_Optimism_pp=(
                "Utility Optimism (pp)", "mean"
            ),
            Negative_Utility_Rate=(
                "Negative Utility", "mean"
            ),
            Mean_Call_OTM_pct=(
                "Call OTM (%)", "mean"
            ),
        )
        .reset_index()
    )

    summary["Negative_Utility_Rate"] *= 100

    print("\n--- ERROR DECOMPOSITION SUMMARY ---")

    print(
        summary.round(4).to_string(
            index=False
        )
    )

    print("\n--- MONTHLY ERROR DECOMPOSITION ---")

    columns = [
        "Period",
        "Decision Date",
        "Model",
        "Decision",
        "Call OTM (%)",
        "Return Optimism (pp)",
        "Drawdown Optimism (pp)",
        "Utility Optimism (pp)",
        "Actual Utility (pp)",
    ]

    print(
        results[columns]
        .round(3)
        .sort_values(
            ["Period", "Decision Date", "Model"]
        )
        .to_string(index=False)
    )

    print("\n--- WORST UTILITY FORECAST ERRORS ---")

    worst = results.nlargest(
        10, "Utility Optimism (pp)"
    )

    print(
        worst[columns]
        .round(3)
        .to_string(index=False)
    )

    print("\nNo historical files modified.")
    print(
        "All analyses use selected hedges only."
    )
    print(
        "2024 remains exploratory."
    )


if __name__ == "__main__":
    main()
