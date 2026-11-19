
"""Inspect historical S&P 500 membership anomalies.

Run:
    py -m data_pipeline.inspect_membership_anomalies

Uses saved Parquet files only.
Does not modify any files.
Does not open an LSEG session.
"""

from pathlib import Path

import pandas as pd


ROOT = Path("data/processed/historical_universe")

BASE_RICS = (
    "KVUE.N",
    "AGN.N",
    "CB.N",
    "JCI.N",
)


def base_ric(value):
    return str(value).split("^", 1)[0]


def main():
    universe = pd.read_parquet(
        ROOT / "sp500_historical_universe.parquet"
    )

    events = pd.read_parquet(
        ROOT / "sp500_membership_events_raw.parquet"
    )

    universe["Decision Date"] = pd.to_datetime(
        universe["Decision Date"]
    ).dt.normalize()

    events["Date"] = pd.to_datetime(
        events["Date"]
    ).dt.normalize()

    universe["Base RIC"] = (
        universe["Underlying RIC"].map(base_ric)
    )

    events["Base RIC"] = (
        events["Constituent RIC"].map(base_ric)
    )

    print("\n--- HISTORICAL MEMBERSHIP INSPECTION ---")

    for base in BASE_RICS:
        print("\n" + "=" * 65)
        print("BASE RIC:", base)
        print("=" * 65)

        history = universe.loc[
            universe["Base RIC"].eq(base)
        ].copy()

        changes = events.loc[
            events["Base RIC"].eq(base)
        ].copy()

        changes = changes.sort_values(
            ["Date", "Constituent RIC", "Change"]
        )

        print("\nMembership change events:")

        if changes.empty:
            print("No events found.")
        else:
            print(
                changes[
                    ["Date", "Constituent RIC", "Change"]
                ].to_string(index=False)
            )

        print("\nHistorical constituent identifiers:")

        if history.empty:
            print("No membership observations.")
        else:
            identifiers = (
                history.groupby("Underlying RIC")
                ["Decision Date"]
                .agg(["min", "max", "count"])
                .reset_index()
            )

            print(
                identifiers.to_string(index=False)
            )

        print("\nMembership around event dates:")

        decision_dates = sorted(
            universe["Decision Date"].unique()
        )

        for _, event in changes.iterrows():
            event_date = event["Date"]

            earlier = [
                date
                for date in decision_dates
                if date < event_date
            ]

            later = [
                date
                for date in decision_dates
                if date >= event_date
            ]

            if not earlier or not later:
                continue

            before_date = earlier[-1]
            after_date = later[0]

            before = sorted(
                history.loc[
                    history["Decision Date"].eq(
                        before_date
                    ),
                    "Underlying RIC",
                ].unique()
            )

            after = sorted(
                history.loc[
                    history["Decision Date"].eq(
                        after_date
                    ),
                    "Underlying RIC",
                ].unique()
            )

            print(
                f"\nEvent: {event_date.date()} "
                f"{event['Change']} "
                f"{event['Constituent RIC']}"
            )

            print(
                f"Before ({before_date.date()}):",
                before,
            )

            print(
                f"After  ({after_date.date()}):",
                after,
            )

    print("\nInspection complete.")
    print("No files modified.")


if __name__ == "__main__":
    main()
