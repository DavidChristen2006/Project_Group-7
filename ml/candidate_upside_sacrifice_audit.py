"""Audit how much upside the selected hedges give up when the stock rises.

For the XGBoost model, the script takes every month in which a hedge was
selected and compares the stock return without hedge with the return
with hedge. The difference is the "return sacrifice": the return that
the hedge costs in that month. The audit shows this for the validation
period (2023) and the exploratory period (2024) and checks in
particular the months with a rising stock, where a hedge with a capped
upside (e.g. a collar) gives up the most.

Procedure:
    1. Collect the walk-forward decisions of each period and keep the
       selected hedges ("No Hedge" months are ignored).
    2. Attach the backtest results of the chosen candidates (stock
       return, hedged return, call strike, premium, dates).
    3. Compute return sacrifice, call distance, net debit and, for
       collars, the stock gain beyond the call strike.
    4. Print the monthly table, a summary per period, the months with
       a stock return above 5 pp and the correlation between stock
       return and return sacrifice.

All returns, sacrifices and utilities are in percentage points (pp).

Run from project root:
    py -m ml.<module_name>

Reads:  dataset from load_dataset() and decisions from
        collect_decisions(); no command-line arguments.
Writes: nothing; results are printed only.

GenAI usage: Claude (Anthropic) was used to explain the code and to
draft, review and correct comments and docstrings. The comments were
checked and adapted by the author(s).
"""

import numpy as np
import pandas as pd

from ml.candidate_walk_forward_comparison import (
    load_dataset,
)

# collect_decisions() returns one walk-forward hedge decision per date
from ml.candidate_uncertainty_audit import (
    collect_decisions,
)


# model that is audited
MODEL = "XGBoost"

# (label, first decision date, end of period); the end date is exclusive
PERIODS = [
    ("2023 validation", "2023-01-01", "2024-01-01"),
    ("2024 exploratory", "2024-01-01", "2025-01-01"),
]


def analyze_period(df, period, start, end):
    """Analyze the selected hedges of MODEL in one period.

    Collects the model's decisions for the period, keeps the months with
    a selected hedge and attaches the backtest data of the chosen
    candidates. Then computes the return sacrifice, the call distance
    from the price, the net debit and, for collars, the stock gain
    beyond the call strike.

    Args:
        df: Dataset as returned by load_dataset().
        period: Label of the period, e.g. "2024 exploratory".
        start: First decision date of the period (inclusive).
        end: End of the period (exclusive).

    Returns:
        DataFrame with one row per selected hedge: the decision columns,
        the backtest columns and "Period", "Stock Return (pp)",
        "Hedged Return (pp)", "Return Sacrifice (pp)", "Call OTM (%)",
        "Net Debit (%)", "Call Above Entry (%)", "Stock Gain Beyond
        Call (%)" and "Days After Exit".

    Raises:
        ValueError: if no hedge was selected in the period, if required
            dataset columns are missing, if candidate keys are not
            unique, or if a selected candidate is not found in the
            dataset.
    """

    # decisions of the model for every decision date in the period
    decisions = collect_decisions(
        df,
        MODEL,
        pd.Timestamp(start),
        pd.Timestamp(end),
    )

    # only months with a real hedge can have a return sacrifice
    selected = decisions.loc[
        decisions["Decision"] != "No Hedge"
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No selected hedges for {period}"
        )

    # columns of the backtest data that this analysis needs
    required = [
        "Candidate Key",
        "Unhedged Final Return",
        "Final Return",
        "Hedge Improvement",
        "Strategy",
        "Spot Price",
        "Short Call Strike Pct",
        "Net Premium Pct",
        "Expiry",
        "Decision Date",
        "Outcome Date",
    ]

    # fail early with a clear message if the dataset lacks a column
    missing = [
        name for name in required
        if name not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    historical = df[required].copy()

    # a unique key is needed so that every selected hedge is matched
    # with exactly one backtest row
    if not historical["Candidate Key"].is_unique:
        raise ValueError(
            "Historical candidate keys not unique."
        )

    # attach the backtest data to the selected hedges; columns that
    # exist in both tables (e.g. "Expiry", "Decision Date") keep the
    # decision version, the dataset version gets the suffix "_Backtest";
    # indicator=True adds "_merge" to verify that every key was found
    selected = selected.merge(
        historical,
        on="Candidate Key",
        how="left",
        validate="one_to_one",
        suffixes=("", "_Backtest"),
        indicator=True,
    )

    if not selected["_merge"].eq("both").all():
        raise ValueError(
            "Some selected candidates were not found."
        )

    selected["Period"] = period

    # return of the stock itself without any hedge, in pp
    selected["Stock Return (pp)"] = (
        selected["Unhedged Final Return"] * 100
    )

    # return of the position including the hedge, in pp
    selected["Hedged Return (pp)"] = (
        selected["Final Return"] * 100
    )

    # return given up because of the hedge: positive = the hedge cost
    # return, negative = the hedge added return
    selected["Return Sacrifice (pp)"] = (
        -selected["Hedge Improvement"] * 100
    )

    # the call strike is a fraction of the entry price, so 1.05 means
    # 5% out of the money; strategies without a call have NaN here
    selected["Call OTM (%)"] = (
        selected["Short Call Strike Pct"] - 1
    ) * 100

    # premium paid for the hedge in % of the position (a negative
    # value would be a net credit)
    selected["Net Debit (%)"] = (
        selected["Net Premium Pct"] * 100
    )

    # same value as "Call OTM (%)", named from the point of view of
    # the upside: how far the stock can rise before the call cuts in
    selected["Call Above Entry (%)"] = (
        selected["Call OTM (%)"]
    )

    # Collars only: stock gain beyond the call strike, i.e. the part of
    # the rise that the short call gives away. The maximum with 0 means
    # no gain is counted if the stock stays below the strike. Other
    # strategies have no short call and get NaN.
    selected["Stock Gain Beyond Call (%)"] = np.where(
        selected["Strategy"].eq("Collar"),
        np.maximum(
            selected["Stock Return (pp)"]
            - selected["Call Above Entry (%)"],
            0.0,
        ),
        np.nan,
    )

    # make sure all dates are real timestamps before subtracting
    selected["Expiry"] = pd.to_datetime(
        selected["Expiry"]
    )

    selected["Decision Date"] = pd.to_datetime(
        selected["Decision Date"]
    )

    selected["Outcome Date"] = pd.to_datetime(
        selected["Outcome Date"]
    )

    # calendar days between the outcome date and the option expiry;
    # positive = the outcome was measured before the expiry
    selected["Days After Exit"] = (
        selected["Expiry"]
        - selected["Outcome Date"]
    ).dt.days

    return selected


def main():
    """Analyze the selected hedges in all periods and print the results.

    Runs analyze_period() for the validation and the exploratory period
    and prints the monthly table, a summary per period, the months with
    a stock return above 5 pp and the correlation between stock return
    and return sacrifice.

    Raises:
        ValueError: if a period has no selected hedges or the dataset is
            inconsistent (raised by analyze_period).

    Side effects:
        Prints progress and all result tables; no files are written.
    """

    df = load_dataset()

    frames = []

    for period, start, end in PERIODS:
        print(f"Evaluating: {period}")

        frames.append(
            analyze_period(
                df,
                period,
                start,
                end,
            )
        )

    # one table with the selected hedges of all periods
    results = pd.concat(
        frames,
        ignore_index=True,
    )

    print("\n--- SELECTED HEDGE UPSIDE AUDIT ---")

    # columns shown for single decisions
    columns = [
        "Period",
        "Decision Date",
        "Decision",
        "Stock Return (pp)",
        "Hedged Return (pp)",
        "Return Sacrifice (pp)",
        "Call OTM (%)",
        "Stock Gain Beyond Call (%)",
        "Net Debit (%)",
        "Days After Exit",
        "Actual Utility (pp)",
    ]

    print(
        results[columns]
        .round(3)
        .to_string(index=False)
    )

    print("\n--- PERIOD SUMMARY ---")

    # averages per period over all selected hedges; the mean of the
    # call distance ignores strategies without a call (NaN)
    summary = results.groupby("Period").agg(
        Hedges=("Decision", "size"),
        Mean_Stock_Return_pp=(
            "Stock Return (pp)", "mean"
        ),
        Mean_Hedged_Return_pp=(
            "Hedged Return (pp)", "mean"
        ),
        Mean_Return_Sacrifice_pp=(
            "Return Sacrifice (pp)", "mean"
        ),
        Mean_Call_OTM_pct=(
            "Call OTM (%)", "mean"
        ),
        Mean_Utility_pp=(
            "Actual Utility (pp)", "mean"
        ),
    )

    print(
        summary.round(4).to_string()
    )

    print("\n--- RISING STOCK MONTHS ---")

    # months in which the stock rose by more than 5 pp; this is where
    # a capped upside costs the most
    rising = results.loc[
        results["Stock Return (pp)"] > 5
    ]

    print(
        rising[columns]
        .round(3)
        .to_string(index=False)
    )

    print("\n--- NEGATIVE RETURN RELATIONSHIP ---")

    # correlation needs at least 3 complete pairs to be meaningful;
    # a positive value means the hedge costs more return the more the
    # stock rises
    valid = results[
        [
            "Stock Return (pp)",
            "Return Sacrifice (pp)",
        ]
    ].dropna()

    if len(valid) >= 3:
        print(
            "Correlation between stock return "
            "and hedge return sacrifice:",
            round(
                valid["Stock Return (pp)"].corr(
                    valid["Return Sacrifice (pp)"]
                ),
                4,
            ),
        )

    # closing notes that help to interpret the output
    print("\nNo historical files modified.")
    print(
        "Stock gain beyond the call strike is "
        "a descriptive proxy, not the actual "
        "short-call P&L."
    )


if __name__ == "__main__":
    main()