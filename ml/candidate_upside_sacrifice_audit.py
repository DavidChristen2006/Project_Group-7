
import numpy as np
import pandas as pd

from ml.candidate_walk_forward_comparison import (
    load_dataset,
)

from ml.candidate_uncertainty_audit import (
    collect_decisions,
)


MODEL = "XGBoost"

PERIODS = [
    ("2023 validation", "2023-01-01", "2024-01-01"),
    ("2024 exploratory", "2024-01-01", "2025-01-01"),
]


def analyze_period(df, period, start, end):
    decisions = collect_decisions(
        df,
        MODEL,
        pd.Timestamp(start),
        pd.Timestamp(end),
    )

    selected = decisions.loc[
        decisions["Decision"] != "No Hedge"
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No selected hedges for {period}"
        )

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

    missing = [
        name for name in required
        if name not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    historical = df[required].copy()

    if not historical["Candidate Key"].is_unique:
        raise ValueError(
            "Historical candidate keys not unique."
        )

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

    selected["Stock Return (pp)"] = (
        selected["Unhedged Final Return"] * 100
    )

    selected["Hedged Return (pp)"] = (
        selected["Final Return"] * 100
    )

    selected["Return Sacrifice (pp)"] = (
        -selected["Hedge Improvement"] * 100
    )

    selected["Call OTM (%)"] = (
        selected["Short Call Strike Pct"] - 1
    ) * 100

    selected["Net Debit (%)"] = (
        selected["Net Premium Pct"] * 100
    )

    selected["Call Above Entry (%)"] = (
        selected["Call OTM (%)"]
    )

    selected["Stock Gain Beyond Call (%)"] = np.where(
        selected["Strategy"].eq("Collar"),
        np.maximum(
            selected["Stock Return (pp)"]
            - selected["Call Above Entry (%)"],
            0.0,
        ),
        np.nan,
    )

    selected["Expiry"] = pd.to_datetime(
        selected["Expiry"]
    )

    selected["Decision Date"] = pd.to_datetime(
        selected["Decision Date"]
    )

    selected["Outcome Date"] = pd.to_datetime(
        selected["Outcome Date"]
    )

    selected["Days After Exit"] = (
        selected["Expiry"]
        - selected["Outcome Date"]
    ).dt.days

    return selected


def main():
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

    results = pd.concat(
        frames,
        ignore_index=True,
    )

    print("\n--- SELECTED HEDGE UPSIDE AUDIT ---")

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

    rising = results.loc[
        results["Stock Return (pp)"] > 5
    ]

    print(
        rising[columns]
        .round(3)
        .to_string(index=False)
    )

    print("\n--- NEGATIVE RETURN RELATIONSHIP ---")

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

    print("\nNo historical files modified.")
    print(
        "Stock gain beyond the call strike is "
        "a descriptive proxy, not the actual "
        "short-call P&L."
    )


if __name__ == "__main__":
    main()

