"""Test whether an uncertainty margin makes the hedge decisions more cautious.

The ML models tend to be too optimistic: the utility they predict for a
selected hedge is on average higher than the utility the hedge actually
delivers. This script estimates that optimism on a calibration period
and then checks on a test period whether a more conservative decision
rule would have done better.

Procedure for each model:
    1. Collect the walk-forward decisions of the calibration period
       (2023) and measure the optimism of the selected hedges
       (predicted utility minus actual utility).
    2. Set the uncertainty margin to the OPTIMISM_QUANTILE of that
       optimism (at least 0).
    3. Collect the decisions of the test period (2024). Original rule:
       hedge if the predicted utility is above 0. Conservative rule:
       hedge only if the predicted utility is above the margin.
    4. Compare both rules by their average realized utility.

All utilities, returns and margins are in percentage points (pp).

The constants and collect_decisions() are also imported by other audit
scripts, so importing this module has no side effects.

Run from project root:
    py -m ml.candidate_uncertainty_audit

Reads:  dataset from load_dataset(); no command-line arguments.
Writes: nothing; results are printed only.

GenAI usage: Claude (Anthropic) was used to explain the code and to
draft, review and correct comments and docstrings. The comments were
checked and adapted by the author(s).
"""

import numpy as np
import pandas as pd

from ml.walk_forward_validation import (
    get_walk_forward_split,
)

from ml.candidate_walk_forward_comparison import (
    load_dataset,
    fit_predict,
)

# evaluate_decision() makes the hedge decision for one date;
# RISK_WEIGHT is the weight of the risk benefit in the utility
from ml.candidate_decision_audit import (
    evaluate_decision,
    RISK_WEIGHT,
)

from pipeline.historical_ml_benchmark import (
    NUMERIC_FEATURES,
)


# models that are audited
MODELS = ["Random Forest", "XGBoost"]

# calibration period: decision dates in 2023 (end date exclusive)
CALIBRATION_START = pd.Timestamp("2023-01-01")
CALIBRATION_END = pd.Timestamp("2024-01-01")

# test period: decision dates in 2024 (end date exclusive)
TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2025-01-01")

# Prespecified exploratory safety margin:
# 80th percentile of historical optimism.
OPTIMISM_QUANTILE = 0.80


def collect_decisions(df, model, start, end):
    """Collect one walk-forward hedge decision per decision date.

    For every decision date in [start, end), the model is trained only
    on outcomes that were already known before that date, predicts the
    candidates of that date and makes a decision with evaluate_decision().

    Args:
        df: Dataset as returned by load_dataset().
        model: Name of the model, e.g. "Random Forest".
        start: First decision date (inclusive).
        end: End of the period (exclusive).

    Returns:
        DataFrame with one row per decision date: the decision record
        from evaluate_decision() plus "Decision Date", "Model" and
        "Excluded Training Rows" (number of rows excluded by the
        walk-forward split).

    Raises:
        ValueError: if fewer than 6 distinct decision dates are
            available for training, or if a training row has an outcome
            on or after the decision date (look-ahead).
    """

    # unique decision dates of the period
    dates = sorted(
        df.loc[
            (df["Decision Date"] >= start)
            & (df["Decision Date"] < end),
            "Decision Date",
        ].unique()
    )

    records = []

    for date in dates:
        train, test, excluded = (
            get_walk_forward_split(df, date)
        )

        # a model trained on very few dates would be unreliable;
        # 6 is the minimum used in this project
        if train["Decision Date"].nunique() < 6:
            raise ValueError(
                f"Insufficient training dates: {date}"
            )

        # no training outcome may lie in the future of the decision date
        if not (
            train["Outcome Date"] < pd.Timestamp(date)
        ).all():
            raise ValueError(
                "Future outcome detected in training."
            )

        # NUMERIC_FEATURES is copied so that fit_predict cannot change
        # the shared feature list
        predictions = fit_predict(
            model,
            train,
            test,
            NUMERIC_FEATURES.copy(),
        )

        result = evaluate_decision(
            test,
            predictions,
            RISK_WEIGHT,
        )

        result["Decision Date"] = pd.Timestamp(date)
        result["Model"] = model
        result["Excluded Training Rows"] = len(excluded)

        records.append(result)

    return pd.DataFrame(records)


def estimate_margin(calibration):
    """Estimate the uncertainty margin from the calibration decisions.

    The optimism of a selected hedge is its predicted utility minus its
    actual utility. The margin is the OPTIMISM_QUANTILE of this optimism
    over all selected hedges, but never below 0.

    Args:
        calibration: DataFrame of decisions as returned by
            collect_decisions() for the calibration period.

    Returns:
        Tuple (margin, number of selected hedges). The margin is in
        percentage points.

    Raises:
        ValueError: if fewer than 5 hedges were selected, which is too
            few to estimate a quantile.
    """

    # only months with a real hedge have a forecast error;
    # "No Hedge" months are ignored
    hedges = calibration.loc[
        calibration["Decision"] != "No Hedge"
    ].copy()

    if len(hedges) < 5:
        raise ValueError(
            "Too few selected hedges for calibration."
        )

    # positive = the model expected more than the hedge delivered
    optimism = (
        hedges["Predicted Utility (pp)"]
        - hedges["Actual Utility (pp)"]
    )

    # a negative margin would make the rule less cautious than the
    # original one, so it is floored at 0
    margin = max(
        0.0,
        float(
            optimism.quantile(
                OPTIMISM_QUANTILE
            )
        ),
    )

    return margin, len(hedges)


def evaluate_threshold(test, margin):
    """Compare the original and the conservative decision rule.

    The original rule hedges if the predicted utility is above 0 (this
    is already contained in the decisions). The conservative rule keeps
    a recommended hedge only if its predicted utility exceeds the
    margin; otherwise it chooses "No Hedge", with return, drawdown and
    utility of 0.

    Args:
        test: DataFrame of decisions as returned by collect_decisions()
            for the test period.
        margin: Uncertainty margin in percentage points, as returned by
            estimate_margin().

    Returns:
        Copy of test with the additional columns "Original Utility
        (pp)", "Conservative Decision", "Conservative Utility (pp)",
        "Conservative Return (pp)", "Conservative Drawdown (pp)" and
        "Accepted" (True if the hedge passes the margin).
    """

    result = test.copy()

    # Original policy uses a zero threshold.
    result["Original Utility (pp)"] = (
        result["Actual Utility (pp)"]
    )

    # Conservative policy:
    # reject the recommended hedge unless
    # its predicted utility exceeds margin.
    # "No Hedge" decisions stay "No Hedge" and are never accepted.
    accepted = (
        result["Decision"].ne("No Hedge")
        & (
            result["Predicted Utility (pp)"]
            > margin
        )
    )

    # np.where(condition, value if True, value if False):
    # accepted hedges keep their actual results, rejected ones get
    # "No Hedge" with a result of 0
    result["Conservative Decision"] = np.where(
        accepted,
        result["Decision"],
        "No Hedge",
    )

    result["Conservative Utility (pp)"] = np.where(
        accepted,
        result["Actual Utility (pp)"],
        0.0,
    )

    result["Conservative Return (pp)"] = np.where(
        accepted,
        result["Actual Return (pp)"],
        0.0,
    )

    result["Conservative Drawdown (pp)"] = np.where(
        accepted,
        result["Actual Risk (pp)"],
        0.0,
    )

    result["Accepted"] = accepted

    return result


def main():
    """Calibrate the margin in 2023 and test both policies in 2024.

    For each model in MODELS, estimates the uncertainty margin on the
    calibration period, applies the original and the conservative rule
    to the test period and prints the monthly decisions. Finally prints
    a comparison table of both policies for all models.

    Raises:
        ValueError: if a model has too little training data or too few
            selected hedges in the calibration period, or if a training
            outcome lies in the future (raised by the called functions).

    Side effects:
        Prints progress and result tables; no files are written.
    """

    df = load_dataset()

    print("\n--- UNCERTAINTY CALIBRATION ---")
    print(
        "Calibration: 2023 walk-forward"
    )
    print(
        "Evaluation: 2024 exploratory"
    )
    print(
        f"Optimism quantile: {OPTIMISM_QUANTILE:.0%}"
    )

    # one summary row per model
    summaries = []

    for model in MODELS:
        print(f"\n--- {model} ---")

        # step 1: decisions of the calibration period
        calibration = collect_decisions(
            df,
            model,
            CALIBRATION_START,
            CALIBRATION_END,
        )

        # step 2: margin from the optimism of the selected hedges
        margin, n_hedges = estimate_margin(
            calibration
        )

        print(
            "Calibration dates:",
            len(calibration)
        )

        print(
            "Calibration hedge decisions:",
            n_hedges
        )

        print(
            f"Uncertainty margin: {margin:.4f} pp"
        )

        # step 3: decisions of the test period, which the margin
        # has not seen
        test = collect_decisions(
            df,
            model,
            TEST_START,
            TEST_END,
        )

        comparison = evaluate_threshold(
            test,
            margin,
        )

        # step 4: key figures of both policies for this model
        # (utilities and returns are averages over all test dates,
        # No Hedge months count as 0)
        summaries.append({
            "Model": model,
            "Margin (pp)": margin,
            "Dates": len(comparison),
            "Original Hedges": int(
                comparison["Decision"]
                .ne("No Hedge")
                .sum()
            ),
            "Conservative Hedges": int(
                comparison["Accepted"].sum()
            ),
            "Original Utility (pp)": (
                comparison[
                    "Original Utility (pp)"
                ].mean()
            ),
            "Conservative Utility (pp)": (
                comparison[
                    "Conservative Utility (pp)"
                ].mean()
            ),
            "Conservative Return (pp)": (
                comparison[
                    "Conservative Return (pp)"
                ].mean()
            ),
            "Conservative Drawdown (pp)": (
                comparison[
                    "Conservative Drawdown (pp)"
                ].mean()
            ),
        })

        print("\nMONTHLY DECISIONS")

        # columns shown for each decision date
        columns = [
            "Decision Date",
            "Decision",
            "Predicted Utility (pp)",
            "Original Utility (pp)",
            "Conservative Decision",
            "Conservative Utility (pp)",
        ]

        print(
            comparison[columns]
            .round(3)
            .to_string(index=False)
        )

    summary = pd.DataFrame(summaries)

    print("\n--- POLICY COMPARISON ---")
    print(
        summary.round(4).to_string(index=False)
    )

    # closing notes that help to interpret the output
    print("\nNo files modified.")
    print(
        "2024 evaluation remains exploratory."
    )
    print(
        "The margin is not a guaranteed "
        "risk bound."
    )


if __name__ == "__main__":
    main()