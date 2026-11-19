
"""Build a research universe of popular equity ETFs.

Run from project root:
    py -m data_pipeline.build_etf_universe

Uses the existing 121-month research calendar.

No LSEG requests are made here. ETF RICs and historical
option availability will be verified separately.

The existing S&P 500 universe is not modified.
"""

from pathlib import Path

import pandas as pd


ROOT = Path("data/processed/ever_member_universe")

DATE_FILE = ROOT / "research_decision_dates.csv"

OUTPUT = Path("data/processed/etf_universe")


def etf(
    ticker,
    venue,
    currency,
    benchmark,
    category,
    group=None,
    isin=None,
):
    """Create one ETF exchange-listing record."""

    return {
        "Ticker": ticker,
        "Venue": venue,
        "Currency": currency,
        "Benchmark": benchmark,
        "Category": category,
        "Fund Group": group or ticker,
        "ISIN": isin,
        "Asset Type": "ETF",
        "LSEG RIC": None,
        "RIC Verified": False,
        "Price Coverage": "NOT_CHECKED",
        "Option Coverage": "NOT_CHECKED",
    }


ETFS = [
    # US broad market
    etf("SPY", "US", "USD", "S&P 500", "US Large Cap"),
    etf("VOO", "US", "USD", "S&P 500", "US Large Cap"),
    etf("IVV", "US", "USD", "S&P 500", "US Large Cap"),
    etf("VTI", "US", "USD", "US Total Market", "US Total Market"),
    etf("ITOT", "US", "USD", "US Total Market", "US Total Market"),

    # US major indices
    etf("QQQ", "US", "USD", "Nasdaq-100", "US Growth"),
    etf("QQQM", "US", "USD", "Nasdaq-100", "US Growth"),
    etf("IWM", "US", "USD", "Russell 2000", "US Small Cap"),
    etf("DIA", "US", "USD", "Dow Jones Industrial Average", "US Large Cap"),

    # Global equity markets
    etf("VT", "US", "USD", "FTSE Global All Cap", "Global"),
    etf("ACWI", "US", "USD", "MSCI ACWI", "Global"),

    # International equity markets
    etf("VXUS", "US", "USD", "Global ex-US", "International"),
    etf("VEA", "US", "USD", "Developed ex-US", "International"),
    etf("VWO", "US", "USD", "Emerging Markets", "Emerging Markets"),
    etf("EFA", "US", "USD", "MSCI EAFE", "International"),
    etf("EEM", "US", "USD", "MSCI Emerging Markets", "Emerging Markets"),

    # US sectors
    etf("XLK", "US", "USD", "US Technology Sector", "Sector"),
    etf("XLF", "US", "USD", "US Financial Sector", "Sector"),
    etf("XLE", "US", "USD", "US Energy Sector", "Sector"),

    # FTSE All-World UCITS
    etf(
        "VWRL",
        "SIX",
        "CHF",
        "FTSE All-World",
        "Global UCITS",
        group="VANGUARD_ALL_WORLD_DIST",
        isin="IE00B3RBWM25",
    ),
    etf(
        "VWRA",
        "LSE",
        "USD",
        "FTSE All-World",
        "Global UCITS",
        group="VANGUARD_ALL_WORLD_ACC",
        isin="IE00BK5BQT80",
    ),
    etf(
        "VWCE",
        "XETRA",
        "EUR",
        "FTSE All-World",
        "Global UCITS",
        group="VANGUARD_ALL_WORLD_ACC",
        isin="IE00BK5BQT80",
    ),

    # Other popular UCITS ETFs
    etf(
        "IWDA",
        "EURONEXT_AMSTERDAM",
        "EUR",
        "MSCI World",
        "Developed Markets UCITS",
    ),
    etf(
        "CSPX",
        "LSE",
        "USD",
        "S&P 500",
        "US Large Cap UCITS",
    ),
    etf(
        "EIMI",
        "LSE",
        "USD",
        "MSCI Emerging Markets IMI",
        "Emerging Markets UCITS",
    ),
]


def load_decision_dates():
    dates = pd.read_csv(
        DATE_FILE,
        parse_dates=["Decision Date"],
    )

    if dates["Decision Date"].isna().any():
        raise ValueError(
            "Invalid research decision dates."
        )

    index = pd.DatetimeIndex(
        dates["Decision Date"]
    ).normalize()

    if index.has_duplicates:
        raise ValueError(
            "Duplicate research decision dates."
        )

    return index.sort_values()


def build_etf_master():
    master = pd.DataFrame(ETFS)

    master["Listing ID"] = (
        master["Ticker"]
        + "|"
        + master["Venue"]
        + "|"
        + master["Currency"]
    )

    if master["Listing ID"].duplicated().any():
        raise ValueError(
            "Duplicate ETF listing identifiers."
        )

    # Multiple listings may represent the same fund
    # or share class. They must not be treated as
    # independent assets without validation.
    master["Listings Per Fund Group"] = (
        master.groupby("Fund Group")[
            "Listing ID"
        ].transform("nunique")
    )

    master["Shared Fund Group"] = (
        master["Listings Per Fund Group"] > 1
    )

    return master.sort_values(
        ["Category", "Ticker", "Venue"]
    ).reset_index(drop=True)


def build_collection_grid(master, dates):
    grid = pd.MultiIndex.from_product(
        [
            dates,
            master["Listing ID"].tolist(),
        ],
        names=[
            "Decision Date",
            "Listing ID",
        ],
    ).to_frame(index=False)

    grid = grid.merge(
        master[
            [
                "Listing ID",
                "Ticker",
                "Venue",
                "Currency",
                "Fund Group",
                "Category",
            ]
        ],
        on="Listing ID",
        how="left",
        validate="many_to_one",
    )

    grid["Tradable On Date"] = pd.NA
    grid["Historical Price Available"] = pd.NA
    grid["Historical Options Available"] = pd.NA
    grid["Executable Hedge Available"] = pd.NA

    if grid.duplicated(
        ["Decision Date", "Listing ID"]
    ).any():
        raise ValueError(
            "Duplicate ETF-date observations."
        )

    return grid


def main():
    print("\n--- ETF RESEARCH UNIVERSE ---")

    master = build_etf_master()
    dates = load_decision_dates()

    grid = build_collection_grid(
        master,
        dates,
    )

    print("ETF listings:", len(master))
    print("Distinct fund groups:", master["Fund Group"].nunique())
    print("Research dates:", len(dates))
    print("Planned ETF-date rows:", len(grid))

    print("\n--- ETF CATEGORIES ---")

    print(
        master["Category"]
        .value_counts()
        .to_string()
    )

    print("\n--- MULTIPLE LISTINGS ---")

    print(
        master.loc[
            master["Shared Fund Group"],
            [
                "Ticker",
                "Venue",
                "Currency",
                "Fund Group",
                "ISIN",
            ],
        ].to_string(index=False)
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    master.to_parquet(
        OUTPUT / "etf_instruments.parquet",
        index=False,
    )

    master.to_csv(
        OUTPUT / "etf_instruments.csv",
        index=False,
    )

    grid.to_parquet(
        OUTPUT / "etf_collection_grid.parquet",
        index=False,
    )

    print("\n--- FILES SAVED ---")

    print(OUTPUT / "etf_instruments.parquet")
    print(OUTPUT / "etf_instruments.csv")
    print(OUTPUT / "etf_collection_grid.parquet")

    print("\nETF universe preparation complete.")
    print("All ETF LSEG RICs still require verification.")


if __name__ == "__main__":
    main()
