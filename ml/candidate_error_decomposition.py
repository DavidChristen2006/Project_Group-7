"""Decompose the forecast errors of the selected hedges of each ML model.

For every model and period, the script collects the monthly hedge
decisions and keeps only the months in which a hedge was selected
("No Hedge" months are ignored). For each selected hedge it measures how
much the model was too optimistic (predicted minus actual) in three
parts:

    Return optimism    = predicted return  - actual return
    Drawdown optimism  = predicted risk    - actual risk
    Utility optimism   = predicted utility - actual utility
                       = return optimism + RISK_WEIGHT * drawdown optimism

The last identity is verified numerically. Positive values mean the hedge
performed worse than forecast, negative values mean it performed better.
The script also reports how often the realized utility was negative and
how far out of the money the selected calls were.

Run from project root:
    py -m ml.candidate_error_decomposition

Reads:  decisions from collect_decisions() and the dataset from
        load_dataset(); no command-line arguments.
Writes: nothing; results are printed only.

GenAI usage: Claude (Anthropic) was used to explain the code and to
draft, review and correct comments and docstrings. The comments were
checked and adapted by the author(s).
"""

import numpy as np
import pandas as pd

# periods, models and decision logic are shared with the uncertainty audit
# so that all audits evaluate exactly the same decisions
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

# weight of the risk benefit in the utility (same value as in the
# decision audit, so the error decomposition matches the decisions)
from ml.candidate_decision_audit import (
    RISK_WEIGHT,
)


def analyse_period(df, model, period, start, end):
    """Decompose the forecast errors of one model's hedges in one period.

    Collects the model's monthly decisions between start and end, keeps
    only months with a selected hedge and computes, for each of them, how
    much the prediction exceeded the actual outcome ("optimism") for
    return, drawdown and utility. All values are in percentage points.
    Also verifies that utility optimism equals return optimism plus
    RISK_WEIGHT times drawdown optimism.

    Args:
        df: Dataset as returned by load_dataset().
        model: Name of the model, e.g. "Random Forest".
        period: Label of the period, used in the output and in error
            messages.
        start: First decision date of the period.
        end: Last decision date of the period.

    Returns:
        DataFrame with one row per selected hedge: the decision columns
        from collect_decisions() plus "Return Optimism (pp)",
        "Drawdown Optimism (pp)", "Weighted Drawdown Optimism (pp)",
        "Utility Optimism (pp)", "Call OTM (%)", "Negative Utility",
        "Model" and "Period".

    Raises:
        ValueError: if the model selected no hedge in the period, or if
            the utility error decomposition does not add up.
    """

    # decisions of this model for every decision date in the period
    decisions = collect_decisions(
        df, model, start, end
    )

    # only months with a real hedge can have a forecast error;
    # "No Hedge" months have predicted and actual values of 0
    hedges = decisions.loc[
        decisions["Decision"].ne("No Hedge")
    ].copy()

    if hedges.empty:
        raise ValueError(
            f"No selected hedges: {model}, {period}"
        )

    # optimism = predicted minus actual:
    # positive = the model expected more than the hedge delivered
    hedges["Return Optimism (pp)"] = (
        hedges["Predicted Return (pp)"]
        - hedges["Actual Return (pp)"]
    )

    hedges["Drawdown Optimism (pp)"] = (
        hedges["Predicted Risk (pp)"]
        - hedges["Actual Risk (pp)"]
    )

    # the drawdown part enters the utility with the weight RISK_WEIGHT
    hedges["Weighted Drawdown Optimism (pp)"] = (
        RISK_WEIGHT
        * hedges["Drawdown Optimism (pp)"]
    )

    hedges["Utility Optimism (pp)"] = (
        hedges["Predicted Utility (pp)"]
        - hedges["Actual Utility (pp)"]
    )

    # consistency check: utility = return + weight * drawdown, therefore
    # utility optimism must equal return optimism plus weighted
    # drawdown optimism (atol allows for floating-point rounding)
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

    # call strike is given in % of the current price, so 105 means
    # 5% out of the money; strategies without a call have NaN here
    # and are ignored by the mean in the summary
    hedges["Call OTM (%)"] = (
        hedges["Call Strike %"] - 100
    )

    # True if the hedge turned out to be worse than no hedge
    hedges["Negative Utility"] = (
        hedges["Actual Utility (pp)"] < 0
    )

    # labels so that results of all models and periods can be combined
    hedges["Model"] = model
    hedges["Period"] = period

    return hedges


def main():
    """Analyse the selected hedges of all models in both periods.

    Evaluates every model in MODELS for the calibration period (labeled
    "2023 validation") and the test period (labeled "2024 exploratory"),
    then prints a summary per period and model, the monthly error
    decomposition and the ten most overoptimistic utility forecasts.

    Raises:
        ValueError: if a model selected no hedge in a period or the error
            decomposition does not add up (raised by analyse_period).

    Side effects:
        Prints progress and all result tables; no files are written.
    """

    df = load_dataset()

    # each period is a tuple of (label, first date, last date)
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

    # evaluate every model in every period
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

    # one table with all selected hedges of all models and periods
    results = pd.concat(
        frames, ignore_index=True
    )

    # averages per period and model:
    # Hedges                 = number of selected hedges
    # *_Optimism_pp          = mean predicted-minus-actual error
    #                          (positive = too optimistic)
    # Negative_Utility_Rate  = share of hedges with negative actual
    #                          utility (the mean of a True/False column)
    # Mean_Call_OTM_pct      = average call distance from the price
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

    # convert the share (0 to 1) into percent
    summary["Negative_Utility_Rate"] *= 100

    print("\n--- ERROR DECOMPOSITION SUMMARY ---")

    print(
        summary.round(4).to_string(
            index=False
        )
    )

    print("\n--- MONTHLY ERROR DECOMPOSITION ---")

    # columns shown for single decisions
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

    # the ten decisions with the largest overoptimism, i.e. where the
    # hedge fell furthest short of the forecast
    worst = results.nlargest(
        10, "Utility Optimism (pp)"
    )

    print(
        worst[columns]
        .round(3)
        .to_string(index=False)
    )

    # closing notes that help to interpret the output
    print("\nNo historical files modified.")
    print(
        "All analyses use selected hedges only."
    )
    print(
        "2024 remains exploratory."
    )


if __name__ == "__main__":
    main()