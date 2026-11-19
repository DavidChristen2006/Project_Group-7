
"""Inspect ETF quality, research dates and equity membership schema.


No LSEG session required.
No existing data files are modified.
"""

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("data/processed")
HISTORY = ROOT / "market_history_v2" / "series"

ETF_FILE = (
    ROOT / "etf_universe/ric_verification/primary_listing"
    / "etf_instruments_verified.parquet"
)

GRID_FILE = (
    ROOT / "etf_universe/etf_collection_grid.parquet"
)

EQUITY_FILE = (
    ROOT / "ever_member_universe/ever_member_instruments.parquet"
)

START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2026-10-30")
SPLIT_DATE = pd.Timestamp("2025-12-05")


def price_file(ric, mode):
    safe = re.sub(
        r"[^A-Za-z0-9_.-]",
        "_",
        ric,
    )

    digest = hashlib.sha256(
        ric.encode("utf-8")
    ).hexdigest()[:8]

    return (
        HISTORY
        / f"{safe}_{digest}"
        / mode
        / f"{START.date()}_{END.date()}.parquet"
    )


def load_prices(ric, mode="adjusted"):
    path = price_file(ric, mode)

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing historical series: {path}"
        )

    frame = pd.read_parquet(path)

    required = {
        "Date",
        "Close",
        "RIC",
        "Adjustment Mode",
    }

    if not required.issubset(frame.columns):
        raise ValueError(
            f"Missing columns for {ric}: "
            f"{required - set(frame.columns)}"
        )

    if (
        not frame["RIC"].eq(ric).all()
        or not frame["Adjustment Mode"].eq(mode).all()
    ):
        raise ValueError(
            f"Historical series mislabeled: {ric} / {mode}"
        )

    frame = frame.copy()

    frame["Date"] = pd.to_datetime(
        frame["Date"],
        errors="raise",
    ).dt.normalize()

    frame["Close"] = pd.to_numeric(
        frame["Close"],
        errors="coerce",
    )

    if frame["Date"].duplicated().any():
        raise ValueError(
            f"Duplicate dates for {ric} / {mode}"
        )

    return (
        frame
        .sort_values("Date")
        .reset_index(drop=True)
    )


def examine_grid():
    print("\n--- ETF RESEARCH GRID ---")

    grid = pd.read_parquet(GRID_FILE)

    print("Rows:", len(grid))
    print("Columns:", grid.columns.tolist())

    date_candidates = [
        col
        for col in grid.columns
        if re.sub(
            r"[^a-z0-9]",
            "",
            str(col).lower(),
        ) == "decisiondate"
    ]

    if len(date_candidates) != 1:
        raise ValueError(
            "Expected one 'Decision Date' column, "
            f"found {date_candidates}"
        )

    date_col = date_candidates[0]

    dates = pd.to_datetime(
        grid[date_col],
        errors="coerce",
    ).dt.normalize()

    if dates.isna().any():
        raise ValueError(
            f"Invalid Decision Dates: "
            f"{int(dates.isna().sum())}"
        )

    print("Distinct decision dates:", dates.nunique())
    print("First decision date:", dates.min().date())
    print("Last decision date:", dates.max().date())

    print(
        "Rows before collected history:",
        int((dates < START).sum()),
    )

    print(
        "Rows after collected history:",
        int((dates > END).sum()),
    )

    if "Listing ID" in grid.columns:
        print(
            "Listings:",
            grid["Listing ID"].nunique(),
        )

        keys = pd.DataFrame({
            "Date": dates,
            "Listing ID": grid["Listing ID"],
        })

        print(
            "Duplicate listing-date rows:",
            int(keys.duplicated().sum()),
        )

    flag_columns = [
        "Tradable On Date",
        "Historical Price Available",
        "Historical Options Available",
        "Executable Hedge Available",
    ]

    for column in flag_columns:
        if column in grid.columns:
            print(
                f"{column}:",
                grid[column]
                .value_counts(dropna=False)
                .head(5)
                .to_dict(),
            )

    print(
        "Grid flags are inspected, not updated."
    )


def examine_london(etfs):
    print("\n--- LONDON VENUE DATE GAPS ---")

    london = etfs.loc[
        etfs["Venue"].astype(str).eq("LSE")
    ]

    calendars = {}

    for _, listing in london.iterrows():
        ric = str(listing["LSEG RIC"])

        calendars[ric] = set(
            load_prices(ric)["Date"]
        )

    union = (
        set().union(*calendars.values())
        if calendars
        else set()
    )

    for ric, dates in calendars.items():
        missing = sorted(union - dates)

        print(
            f"\n{ric}: {len(dates)} records; "
            f"{len(missing)} absent vs peer union"
        )

        for day in missing:
            peers = [
                other
                for other, peer_dates in calendars.items()
                if (
                    other != ric
                    and day in peer_dates
                )
            ]

            print(
                f"  {day.date()}: "
                f"peers trading = {', '.join(peers)}"
            )

    print(
        "\nPeer-date differences are review flags, "
        "not necessarily data errors."
    )


def examine_splits():
    print("\n--- XLE / XLK ADJUSTMENT RATIO CHECK ---")

    for ric in ("XLE.P", "XLK.P"):
        raw = load_prices(
            ric,
            "unadjusted",
        )[["Date", "Close"]]

        adj = load_prices(
            ric,
            "adjusted",
        )[["Date", "Close"]]

        both = raw.merge(
            adj,
            on="Date",
            how="inner",
            validate="one_to_one",
            suffixes=("_raw", "_adj"),
        )

        both = both.loc[
            both["Close_raw"].gt(0)
            & both["Close_adj"].gt(0)
        ].copy()

        both["ratio"] = (
            both["Close_adj"]
            / both["Close_raw"]
        )

        pre = both.loc[
            both["Date"] < SPLIT_DATE,
            "ratio",
        ]

        post = both.loc[
            both["Date"] >= SPLIT_DATE,
            "ratio",
        ]

        changed = ~np.isclose(
            both["ratio"].to_numpy(),
            1.0,
            atol=1e-8,
            rtol=1e-8,
        )

        print(f"\n{ric}")

        print(
            "Adjusted-price differences:",
            int(changed.sum()),
        )

        if not pre.empty:
            print(
                "Median pre-split ratio:",
                round(float(pre.median()), 6),
            )

        if not post.empty:
            print(
                "Median post-split ratio:",
                round(float(post.median()), 6),
            )

        vicinity = both.loc[
            both["Date"].between(
                "2025-12-03",
                "2025-12-08",
            )
        ]

        print("\nPrices around split:")
        print(vicinity.to_string(index=False))

    print(
        "\nExpected approximate ratios for a "
        "clean 2:1 split: 0.5 before and 1.0 after."
    )


def examine_equity_schema():
    print(
        "\n--- EQUITY MASTER SCHEMA "
        "(FOR MEMBERSHIP MASKING) ---"
    )

    master = pd.read_parquet(EQUITY_FILE)

    print("Rows:", len(master))
    print("Columns:", master.columns.tolist())

    likely = [
        col
        for col in master.columns
        if any(
            word in str(col).lower()
            for word in (
                "ric",
                "member",
                "start",
                "end",
                "entry",
                "exit",
                "from",
                "to",
                "date",
            )
        )
    ]

    if likely:
        print(
            "\nPotential membership-related columns:",
            likely,
        )

        print(
            master[likely]
            .head(3)
            .to_string(index=False)
        )

    print(
        "\nDo not assume identifier mapping records "
        "also contain point-in-time membership windows."
    )


def main():
    etfs = pd.read_parquet(ETF_FILE)

    examine_grid()
    examine_london(etfs)
    examine_splits()
    examine_equity_schema()

    print(
        "\nRead-only inspection complete. "
        "No LSEG requests were made."
    )


if __name__ == "__main__":
    main()
