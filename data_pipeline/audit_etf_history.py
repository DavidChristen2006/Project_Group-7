
"""Audit ETF daily prices and inspect the research-date grid.

No LSEG session required.
No existing source files are modified.
"""

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("data/processed")

UNIVERSE_FILE = (
    ROOT / "etf_universe/ric_verification/primary_listing"
    / "etf_instruments_verified.parquet"
)

GRID_FILE = (
    ROOT / "etf_universe/etf_collection_grid.parquet"
)

HISTORY_ROOT = ROOT / "market_history_v2"
REPORT_DIR = HISTORY_ROOT / "reports"

START = "2024-01-01"
END = "2026-10-30"

LARGE_RETURN_THRESHOLD = 0.20


def history_path(ric, mode):
    safe = re.sub(
        r"[^A-Za-z0-9_.-]",
        "_",
        ric,
    )

    digest = hashlib.sha256(
        ric.encode("utf-8")
    ).hexdigest()[:8]

    return (
        HISTORY_ROOT
        / "series"
        / f"{safe}_{digest}"
        / mode
        / f"{START}_{END}.parquet"
    )


def load_history(ric, mode):
    path = history_path(ric, mode)

    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)

    required = {
        "Date",
        "RIC",
        "Adjustment Mode",
        "Close",
        "Close Field",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path}: missing columns {sorted(missing)}"
        )

    df = df.copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    if df["Date"].isna().any():
        raise ValueError(
            f"{ric} {mode}: invalid dates."
        )

    if df["Date"].duplicated().any():
        raise ValueError(
            f"{ric} {mode}: duplicate dates."
        )

    if not df["RIC"].eq(ric).all():
        raise ValueError(
            f"{ric} {mode}: incorrect RIC."
        )

    if not df["Adjustment Mode"].eq(mode).all():
        raise ValueError(
            f"{ric}: incorrect adjustment mode."
        )

    if not df["Date"].between(
        pd.Timestamp(START),
        pd.Timestamp(END),
    ).all():
        raise ValueError(
            f"{ric}: observations outside date range."
        )

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce",
    )

    return (
        df.sort_values("Date")
        .reset_index(drop=True)
    )


def audit_research_grid():
    print("\n--- RESEARCH GRID INSPECTION ---")

    if not GRID_FILE.is_file():
        print("Research grid not found:", GRID_FILE)
        return

    grid = pd.read_parquet(GRID_FILE)

    print("Grid rows:", len(grid))
    print("Columns:", grid.columns.tolist())

    date_names = {
        "date",
        "researchdate",
        "rebalancedate",
        "asofdate",
        "observationdate",
        "signaldate",
    }

    date_columns = [
        column
        for column in grid.columns
        if re.sub(
            r"[^a-z0-9]",
            "",
            str(column).lower(),
        ) in date_names
    ]

    if len(date_columns) != 1:
        print(
            "Research date column not uniquely identified:",
            date_columns,
        )
        return

    date_column = date_columns[0]

    dates = pd.to_datetime(
        grid[date_column],
        errors="coerce",
    )

    if dates.isna().any():
        print(
            "Invalid research dates:",
            int(dates.isna().sum()),
        )

    dates = dates.dropna().dt.normalize()

    if dates.empty:
        print("No usable research dates.")
        return

    unique_dates = (
        dates.drop_duplicates()
        .sort_values()
    )

    outside = unique_dates.loc[
        ~unique_dates.between(
            pd.Timestamp(START),
            pd.Timestamp(END),
        )
    ]

    print("Research-date column:", date_column)
    print("Distinct dates:", len(unique_dates))
    print("First date:", unique_dates.iloc[0].date())
    print("Last date:", unique_dates.iloc[-1].date())

    print(
        "Dates outside collected history:",
        len(outside),
    )

    if not outside.empty:
        print(
            "WARNING: Extend historical collection "
            "before constructing the full ML panel."
        )


def main():
    universe = pd.read_parquet(
        UNIVERSE_FILE
    )

    required = {
        "Ticker",
        "Venue",
        "LSEG RIC",
        "RIC Verified",
    }

    if not required.issubset(universe.columns):
        raise ValueError(
            f"Missing columns: "
            f"{sorted(required - set(universe.columns))}"
        )

    if not universe["RIC Verified"].fillna(
        False
    ).astype(bool).all():
        raise ValueError(
            "Unverified ETF RIC in universe."
        )

    if universe["LSEG RIC"].duplicated().any():
        raise ValueError(
            "Duplicate RIC in ETF universe."
        )

    results = []
    venue_dates = {}
    failures = []

    for _, etf in universe.iterrows():
        ric = str(etf["LSEG RIC"])
        ticker = str(etf["Ticker"])
        venue = str(etf["Venue"])

        try:
            raw = load_history(
                ric, "unadjusted"
            )

            adj = load_history(
                ric, "adjusted"
            )

        except (
            FileNotFoundError,
            ValueError,
            KeyError,
        ) as error:
            failures.append({
                "Ticker": ticker,
                "RIC": ric,
                "Error": str(error),
            })

            results.append({
                "Ticker": ticker,
                "RIC": ric,
                "Venue": venue,
                "Status": "FAILED_LOAD",
            })

            continue

        raw_dates = set(raw["Date"])
        adj_dates = set(adj["Date"])

        venue_dates.setdefault(
            venue, {}
        )[ric] = raw_dates

        comparison = raw[
            ["Date", "Close"]
        ].merge(
            adj[["Date", "Close"]],
            on="Date",
            how="inner",
            validate="one_to_one",
            suffixes=(
                "_Unadjusted",
                "_Adjusted",
            ),
        )

        same_price = np.isclose(
            comparison[
                "Close_Unadjusted"
            ].to_numpy(dtype=float),
            comparison[
                "Close_Adjusted"
            ].to_numpy(dtype=float),
            rtol=1e-8,
            atol=1e-8,
            equal_nan=True,
        )

        returns = adj["Close"].pct_change(
            fill_method=None
        )

        large_jumps = returns.abs().gt(
            LARGE_RETURN_THRESHOLD
        )

        close_fields = sorted(
            adj["Close Field"]
            .dropna()
            .astype(str)
            .unique()
        )

        invalid_closes = int(
            raw["Close"].le(0).sum()
            + raw["Close"].isna().sum()
            + adj["Close"].le(0).sum()
            + adj["Close"].isna().sum()
        )

        only_raw = len(
            raw_dates - adj_dates
        )

        only_adjusted = len(
            adj_dates - raw_dates
        )

        status = (
            "REVIEW"
            if (
                invalid_closes
                or only_raw
                or only_adjusted
                or large_jumps.any()
            )
            else "PASS_BASIC"
        )

        results.append({
            "Ticker": ticker,
            "RIC": ric,
            "Venue": venue,
            "Rows Unadjusted": len(raw),
            "Rows Adjusted": len(adj),
            "Close Fields Adjusted": ", ".join(
                close_fields
            ),
            "Invalid Close Values": invalid_closes,
            "Dates Only Unadjusted": only_raw,
            "Dates Only Adjusted": only_adjusted,
            "Adjusted Prices Differ (Days)": int(
                (~same_price).sum()
            ),
            "Adjusted Returns >20% (Days)": int(
                large_jumps.sum()
            ),
            "Max Abs Adjusted Daily Return": round(
                float(returns.abs().max()),
                6,
            ),
            "Missing Dates vs Venue Union": 0,
            "Status": status,
        })

    report = pd.DataFrame(results)

    # Compare date availability among ETFs trading
    # on the same venue, without filling missing dates.
    for venue, groups in venue_dates.items():
        if len(groups) < 2:
            continue

        all_dates = set().union(
            *groups.values()
        )

        for ric, dates in groups.items():
            missing_count = len(
                all_dates - dates
            )

            mask = report["RIC"].eq(ric)

            report.loc[
                mask,
                "Missing Dates vs Venue Union",
            ] = missing_count

            if missing_count:
                report.loc[
                    mask & report["Status"].eq(
                        "PASS_BASIC"
                    ),
                    "Status",
                ] = "REVIEW_VENUE_GAPS"

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_file = (
        REPORT_DIR
        / f"etf_quality_{START}_{END}.csv"
    )

    report.to_csv(
        report_file,
        index=False,
    )

    pd.DataFrame(
        failures,
        columns=["Ticker", "RIC", "Error"],
    ).to_csv(
        REPORT_DIR
        / f"etf_quality_errors_{START}_{END}.csv",
        index=False,
    )

    print("\n--- ETF HISTORY QUALITY AUDIT ---")

    print(
        report.to_string(index=False)
    )

    print("\nStatus summary:")

    print(
        report["Status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nSaved:", report_file)

    print(
        "Venue gaps and >20% daily returns "
        "require review; they are not necessarily errors."
    )

    print(
        "Corporate-action-adjusted prices are "
        "not dividend-reinvested total returns."
    )

    audit_research_grid()


if __name__ == "__main__":
    main()
