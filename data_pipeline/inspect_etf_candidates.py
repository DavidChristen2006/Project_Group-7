
"""Inspect ambiguous ETF RIC candidates from saved LSEG results.

No LSEG session required.
No existing files modified.
"""

import argparse
from pathlib import Path

import pandas as pd


SOURCE = Path(
    "data/processed/etf_universe/"
    "ric_verification/etf_ric_candidates.csv"
)


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--tickers",
        nargs="+",
        help="ETF tickers to inspect",
    )

    group.add_argument(
        "--all",
        action="store_true",
        help="Show all ETF candidates",
    )

    args = parser.parse_args()

    df = pd.read_csv(
        SOURCE,
        dtype=str,
        keep_default_na=False,
    )

    if "Verification Eligible" not in df.columns:
        raise ValueError(
            "Missing Verification Eligible column."
        )

    df["Eligible"] = (
        df["Verification Eligible"]
        .str.strip()
        .str.lower()
        .eq("true")
    )

    eligible = df.loc[df["Eligible"]].copy()

    if args.tickers:
        tickers = {
            ticker.upper()
            for ticker in args.tickers
        }

        eligible = eligible.loc[
            eligible["Ticker"].str.upper().isin(tickers)
        ]

    summary = (
        eligible
        .groupby("Ticker")["Candidate RIC"]
        .nunique()
        .sort_index()
    )

    print("\n--- ELIGIBLE RIC COUNTS ---")
    print(summary.to_string())

    print("\n--- EXCHANGE-SPECIFIC CANDIDATES ---")

    for ticker, group in eligible.groupby(
        "Ticker",
        sort=True,
    ):
        print(f"\n=== {ticker} ===")

        columns = [
            "Candidate RIC",
            "Exchange Name",
            "Exchange Code",
            "Quote Currency",
            "Asset State",
        ]

        available = [
            column
            for column in columns
            if column in group.columns
        ]

        display = (
            group[available]
            .drop_duplicates()
            .sort_values("Candidate RIC")
        )

        print(
            display.to_string(
                index=False,
                max_colwidth=45,
            )
        )

    print("\n--- SUMMARY ---")
    print("Eligible candidate rows:", len(eligible))
    print(
        "Distinct eligible RICs:",
        eligible["Candidate RIC"].nunique(),
    )


if __name__ == "__main__":
    main()
