
"""Build an ever-member S&P 500 research universe.

Includes every RIC observed in:
    1. The current S&P 500 constituent snapshot.
    2. Cached historical joiner/leaver events.
    3. The reconstructed historical membership dataset.

Every included RIC is assigned every research decision date,
regardless of its actual S&P 500 membership dates.

This creates a DATA COLLECTION PLAN, not a verified list of
tradable securities on every historical date.

Run from project root:
    py -m data_pipeline.build_ever_member_universe

LSEG uses the default configured Platform session.
"""

from pathlib import Path
import json

import lseg.data as ld
import pandas as pd


ROOT = Path("data/processed")
SOURCE = ROOT / "historical_universe"
OUTPUT = ROOT / "ever_member_universe"

PRICE_FILE = (
    ROOT
    / "ml_datasets/aapl_h21/historical_market_prices.parquet"
)

START_DATE = "2014-12-01"
END_DATE = "2024-12-31"

INDEX_RIC = ".SPX"
CURRENT_CHAIN = "0#.SPX"


def clean_ric(value):
    if pd.isna(value):
        return None

    ric = str(value).strip()

    if not ric or ric.lower() in {
        "nan",
        "none",
        "null",
        "<na>",
    }:
        return None

    return ric


def get_current_members():
    print("\n--- CURRENT S&P 500 MEMBERS ---")

    ld.open_session()

    try:
        data = ld.get_data(
            universe=[CURRENT_CHAIN],
            fields=["TR.RIC"],
        )

    finally:
        ld.close_session()

    if data is None or data.empty:
        raise ValueError(
            "No current S&P 500 constituents returned."
        )

    if "RIC" not in data.columns:
        raise ValueError(
            f"Unexpected columns: {data.columns.tolist()}"
        )

    members = {
        ric
        for value in data["RIC"]
        if (ric := clean_ric(value)) is not None
    }

    if not 450 <= len(members) <= 550:
        raise ValueError(
            "Unexpected current constituent count: "
            f"{len(members)}"
        )

    print("Current constituent RICs:", len(members))

    return members


def get_historical_event_rics():
    print("\n--- HISTORICAL MEMBERSHIP EVENTS ---")

    path = (
        SOURCE
        / "sp500_membership_events_raw.parquet"
    )

    events = pd.read_parquet(path)

    required = {
        "Date",
        "Constituent RIC",
        "Change",
    }

    if not required.issubset(events.columns):
        raise ValueError(
            "Unexpected historical event columns: "
            f"{events.columns.tolist()}"
        )

    changes = (
        events["Change"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    if not changes.isin(
        ["joiner", "leaver"]
    ).all():
        raise ValueError(
            "Unknown event directions found."
        )

    rics = {
        ric
        for value in events["Constituent RIC"]
        if (ric := clean_ric(value)) is not None
    }

    print("Historical event rows:", len(events))
    print("Unique event RICs:", len(rics))

    return rics, events


def get_reconstructed_rics():
    print("\n--- RECONSTRUCTED HISTORICAL MEMBERS ---")

    path = (
        SOURCE
        / "sp500_historical_universe.parquet"
    )

    universe = pd.read_parquet(path)

    rics = {
        ric
        for value in universe["Underlying RIC"]
        if (ric := clean_ric(value)) is not None
    }

    print(
        "Historical constituent-date records:",
        len(universe),
    )

    print(
        "Unique reconstructed RICs:",
        len(rics),
    )

    return rics


def get_monthly_decision_dates():
    print("\n--- FULL RESEARCH CALENDAR ---")

    prices = pd.read_parquet(PRICE_FILE)

    dates = pd.DatetimeIndex(
        pd.to_datetime(prices.index)
    ).normalize()

    dates = dates.unique().sort_values()

    dates = dates[
        (dates >= pd.Timestamp(START_DATE))
        & (dates <= pd.Timestamp(END_DATE))
    ]

    if len(dates) == 0:
        raise ValueError(
            "No historical trading dates found."
        )

    # Select the first available historical trading
    # session in each calendar month.
    months = dates.to_period("M")

    first_by_month = (
        pd.Series(dates, index=months)
        .groupby(level=0)
        .min()
    )

    expected_months = pd.period_range(
        start=START_DATE,
        end=END_DATE,
        freq="M",
    )

    missing_months = expected_months.difference(
        first_by_month.index
    )

    if len(missing_months):
        raise ValueError(
            "Missing trading calendar months: "
            f"{missing_months.tolist()}"
        )

    selected = first_by_month.reindex(
        expected_months
    )

    decision_dates = pd.DatetimeIndex(
        selected.to_numpy()
    )

    print("Research months:", len(decision_dates))

    print(
        "First decision date:",
        decision_dates.min().date(),
    )

    print(
        "Last decision date:",
        decision_dates.max().date(),
    )

    return decision_dates


def build_master_instruments(
    current,
    event_rics,
    reconstructed,
):
    print("\n--- EVER-MEMBER INSTRUMENT UNIVERSE ---")

    # A RIC is included if it appears in any source.
    # No requirement to be a member on the test date.
    all_rics = sorted(
        current | event_rics | reconstructed
    )

    records = []

    for ric in all_rics:
        records.append({
            "Underlying RIC": ric,
            "Base RIC": ric.split("^", 1)[0],
            "Ticker Symbol": (
                ric.split("^", 1)[0].split(".", 1)[0]
            ),
            "Current Member Source": (
                ric in current
            ),
            "Historical Event Source": (
                ric in event_rics
            ),
            "Reconstructed Member Source": (
                ric in reconstructed
            ),
            "Has Historical RIC Suffix": (
                "^" in ric
            ),
            "Security ID Verified": False,
            "Price Coverage Checked": False,
            "Option Coverage Checked": False,
        })

    master = pd.DataFrame(records)

    master["Base RIC Count"] = (
        master.groupby("Base RIC")[
            "Underlying RIC"
        ].transform("nunique")
    )
    
    master["Symbol RIC Count"] = (
        master.groupby("Ticker Symbol")[
            "Underlying RIC"
        ].transform("nunique")
    )

    master["Identifier Review Required"] = (
        (master["Base RIC Count"] > 1)
        | (master["Symbol RIC Count"] > 1)
    )

    print(
        "Ticker symbols with multiple RICs:",
        master.loc[
            master["Symbol RIC Count"] > 1,
            "Ticker Symbol",
        ].nunique(),
    )


    print("Unique exact RICs:", len(master))

    print(
        "Unique base RICs:",
        master["Base RIC"].nunique(),
    )

    print(
        "RICs requiring identifier review:",
        int(
            master[
                "Identifier Review Required"
            ].sum()
        ),
    )

    return master


def build_collection_grid(
    master,
    decision_dates,
):
    print("\n--- FULL-PERIOD COLLECTION GRID ---")

    # Every security is requested for every month.
    # Actual eligibility will be determined later
    # from its observed price and option history.
    grid = pd.MultiIndex.from_product(
        [
            decision_dates,
            master["Underlying RIC"].tolist(),
        ],
        names=[
            "Decision Date",
            "Underlying RIC",
        ],
    ).to_frame(index=False)

    grid["Price Available"] = pd.NA
    grid["Options Available"] = pd.NA
    grid["Executable Hedge Available"] = pd.NA

    print(
        "Planned ticker-date observations:",
        len(grid),
    )

    if grid.duplicated(
        ["Decision Date", "Underlying RIC"]
    ).any():
        raise ValueError(
            "Duplicate collection-grid records."
        )

    return grid


def main():
    print("\n--- RESEARCH CONFIGURATION ---")

    print("Index:", INDEX_RIC)
    print("Universe method: Ever-member union")
    print("Study start:", START_DATE)
    print("Study end:", END_DATE)

    current = get_current_members()

    event_rics, events = (
        get_historical_event_rics()
    )

    reconstructed = get_reconstructed_rics()

    decision_dates = get_monthly_decision_dates()

    master = build_master_instruments(
        current,
        event_rics,
        reconstructed,
    )

    grid = build_collection_grid(
        master,
        decision_dates,
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    master.to_parquet(
        OUTPUT / "ever_member_instruments.parquet",
        index=False,
    )

    master.to_csv(
        OUTPUT / "ever_member_instruments.csv",
        index=False,
    )

    grid.to_parquet(
        OUTPUT / "ever_member_collection_grid.parquet",
        index=False,
    )

    pd.DataFrame({
        "Decision Date": decision_dates,
    }).to_csv(
        OUTPUT / "research_decision_dates.csv",
        index=False,
    )

    manifest = {
        "index": INDEX_RIC,
        "universe_type": "ever_member",
        "start_date": START_DATE,
        "end_date": END_DATE,
        "unique_rics": len(master),
        "decision_dates": len(decision_dates),
        "planned_observations": len(grid),
        "sources": [
            "Current SPX constituent chain",
            "Historical joiner/leaver event RICs",
            "Reconstructed historical constituents",
        ],
        "selection_note": (
            "Future S&P 500 membership can influence "
            "inclusion in this retrospective universe."
        ),
        "coverage_note": (
            "Historical events currently cover only "
            "the cached membership-history window."
        ),
        "identifier_note": (
            "RICs have not yet been resolved to "
            "permanent security identifiers."
        ),
    }

    with (
        OUTPUT / "ever_member_manifest.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    print("\n--- FINAL SUMMARY ---")

    print("Unique instruments:", len(master))
    print("Decision dates:", len(decision_dates))
    print("Collection-grid rows:", len(grid))

    print("\nFiles saved to:", OUTPUT)
    print("\nEver-member universe preparation complete.")


if __name__ == "__main__":
    main()
