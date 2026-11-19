
"""Audit S&P 500 membership reconstruction and ETF price consistency.

Uses saved Parquet data. No LSEG session required.
Original datasets are not modified.
"""

from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd


ROOT = Path("data/processed")

HIST_FILE = (
    ROOT / "historical_universe"
    / "sp500_historical_universe.parquet"
)

GRID_FILE = (
    ROOT / "ever_member_universe"
    / "ever_member_collection_grid.parquet"
)

EVENTS_FILE = (
    ROOT / "historical_universe"
    / "sp500_membership_events.parquet"
)

ETF_FILE = (
    ROOT / "etf_universe/ric_verification/primary_listing"
    / "etf_instruments_verified.parquet"
)

PRICE_ROOT = ROOT / "market_history_v2/series"
REPORT_DIR = ROOT / "research_audit"

LONG_RANGE = "2013-09-01_2025-04-30"
SHORT_RANGE = "2024-01-01_2026-10-30"


def as_dates(frame, column):
    result = frame.copy()

    result[column] = pd.to_datetime(
        result[column],
        errors="raise",
    ).dt.normalize()

    return result


def audit_membership():
    hist = as_dates(
        pd.read_parquet(HIST_FILE),
        "Decision Date",
    )

    grid = as_dates(
        pd.read_parquet(GRID_FILE),
        "Decision Date",
    )

    events = as_dates(
        pd.read_parquet(EVENTS_FILE),
        "Change Date",
    )

    hist = hist.loc[
        hist["Index"].eq(".SPX")
    ].copy()

    flags = (
        hist["Index Member"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    if not flags.isin(
        ["true", "false", "1", "0"]
    ).all():
        raise ValueError(
            "Unexpected or missing Index Member values."
        )

    hist = hist.loc[
        flags.isin(["true", "1"])
    ].copy()

    if hist.duplicated(
        ["Decision Date", "Underlying RIC"]
    ).any():
        raise ValueError(
            "Duplicate membership date/RIC keys."
        )

    events["Change"] = (
        events["Change"]
        .astype("string")
        .str.strip()
    )

    if not events["Change"].isin(
        ["Joiner", "Leaver"]
    ).all():
        raise ValueError(
            "Unexpected membership event types."
        )

    events["RIC"] = (
        events["RIC"]
        .astype("string")
        .str.strip()
    )

    if (
        events["RIC"].isna().any()
        or events["RIC"].eq("").any()
    ):
        raise ValueError(
            "Missing event RIC."
        )

    events = events.sort_values(
        "Change Date",
        kind="stable",
    )

    dates = [
        pd.Timestamp(d)
        for d in sorted(
            grid["Decision Date"].unique()
        )
    ]

    grouped = {
        date: set(
            chunk["Underlying RIC"].astype(str)
        )
        for date, chunk in hist.groupby(
            "Decision Date"
        )
    }

    missing = [
        date
        for date in dates
        if date not in grouped
    ]

    print(
        "\n--- MISSING MEMBERSHIP DECISION DATES ---"
    )

    print("Grid decision dates:", len(dates))
    print("Dates with membership:", len(grouped))
    print("Dates missing:", len(missing))

    print(
        "Missing by year:",
        pd.Series(
            [d.year for d in missing],
            dtype="int64",
        ).value_counts().sort_index().to_dict(),
    )

    for date in missing:
        print(" ", date.date())

    # Group consecutive missing monthly decisions.
    runs = []
    start_idx = None

    for i in range(len(dates) + 1):
        absent = (
            i < len(dates)
            and dates[i] not in grouped
        )

        if absent and start_idx is None:
            start_idx = i

        elif not absent and start_idx is not None:
            end_idx = i - 1

            runs.append({
                "First missing": dates[start_idx],
                "Last missing": dates[end_idx],
                "Missing months": (
                    end_idx - start_idx + 1
                ),
                "Prior observed date": (
                    dates[start_idx - 1]
                    if start_idx
                    else pd.NaT
                ),
                "Next observed date": (
                    dates[i]
                    if i < len(dates)
                    else pd.NaT
                ),
            })

            start_idx = None

    runs_df = pd.DataFrame(runs)

    print("\n--- CONSECUTIVE MISSING RUNS ---")

    if runs_df.empty:
        print("None")
    else:
        print(runs_df.to_string(index=False))

    # Validate whether historical membership events
    # reproduce the next available monthly snapshot.
    # This is a consistency test, not a reconstruction.
    observed = sorted(grouped)
    checks = []

    for previous, current in zip(
        observed,
        observed[1:],
    ):
        expected = set(grouped[previous])

        changes = events.loc[
            events["Change Date"].gt(previous)
            & events["Change Date"].le(current)
        ]

        redundant = 0

        for event in changes.itertuples(index=False):
            ric = str(event.RIC)

            if event.Change == "Joiner":
                redundant += int(ric in expected)
                expected.add(ric)

            else:
                redundant += int(ric not in expected)
                expected.discard(ric)

        actual = grouped[current]

        checks.append({
            "Previous date": previous,
            "Current date": current,
            "Elapsed grid dates": sum(
                previous < date <= current
                for date in dates
            ),
            "Events": len(changes),
            "Redundant event operations": redundant,
            "Expected count": len(expected),
            "Observed count": len(actual),
            "Unexpected in observed": len(
                actual - expected
            ),
            "Missing from observed": len(
                expected - actual
            ),
            "Exact match": expected == actual,
        })

    checks_df = pd.DataFrame(checks)

    print(
        "\n--- EVENT-TO-SNAPSHOT RECONCILIATION ---"
    )

    print(
        "Observed intervals tested:",
        len(checks_df),
    )

    if not checks_df.empty:
        bad = checks_df.loc[
            ~checks_df["Exact match"]
        ]

        print(
            "Exact matches:",
            int(checks_df["Exact match"].sum()),
        )

        print("Mismatches:", len(bad))

        if not bad.empty:
            print(
                bad.head(12).to_string(index=False)
            )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs_df.to_csv(
        REPORT_DIR / "missing_membership_runs.csv",
        index=False,
    )

    checks_df.to_csv(
        REPORT_DIR / "membership_event_reconciliation.csv",
        index=False,
    )

    print(
        "Source membership and event datasets unchanged."
    )


def price_path(ric, period):
    safe = re.sub(
        r"[^A-Za-z0-9_.-]",
        "_",
        ric,
    )

    digest = hashlib.sha256(
        ric.encode("utf-8")
    ).hexdigest()[:8]

    return (
        PRICE_ROOT
        / f"{safe}_{digest}"
        / "adjusted"
        / f"{period}.parquet"
    )


def load_prices(path, ric):
    frame = pd.read_parquet(path)

    required = {
        "Date",
        "Close",
        "Close Field",
        "RIC",
        "Adjustment Mode",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"{path}: missing columns {missing}"
        )

    if (
        not frame["RIC"].eq(ric).all()
        or not frame["Adjustment Mode"]
        .eq("adjusted").all()
    ):
        raise ValueError(
            f"{path}: inconsistent metadata"
        )

    frame = as_dates(
        frame,
        "Date",
    )

    frame["Close"] = pd.to_numeric(
        frame["Close"],
        errors="coerce",
    )

    if frame["Date"].duplicated().any():
        raise ValueError(
            f"{path}: duplicate dates"
        )

    return (
        frame
        .sort_values("Date")
        .reset_index(drop=True)
    )


def audit_etf_overlap():
    etfs = pd.read_parquet(ETF_FILE)

    records = []

    print(
        "\n--- LONG VS SHORT ETF PRICE OVERLAP ---"
    )

    for ric in (
        etfs["LSEG RIC"]
        .dropna()
        .astype(str)
    ):
        p_long = price_path(
            ric,
            LONG_RANGE,
        )

        p_short = price_path(
            ric,
            SHORT_RANGE,
        )

        if (
            not p_long.exists()
            or not p_short.exists()
        ):
            print(
                f"{ric}: missing one or both price files"
            )

            records.append({
                "RIC": ric,
                "Status": "MISSING_FILE",
            })

            continue

        long = load_prices(
            p_long,
            ric,
        )

        short = load_prices(
            p_short,
            ric,
        )

        both = long[
            ["Date", "Close", "Close Field"]
        ].merge(
            short[
                ["Date", "Close", "Close Field"]
            ],
            on="Date",
            how="inner",
            validate="one_to_one",
            suffixes=("_Long", "_Short"),
        )

        if both.empty:
            records.append({
                "RIC": ric,
                "Status": "NO_OVERLAP",
            })

            continue

        a = both["Close_Long"].to_numpy(
            dtype=float
        )

        b = both["Close_Short"].to_numpy(
            dtype=float
        )

        valid = (
            np.isfinite(a)
            & np.isfinite(b)
            & (a > 0)
            & (b > 0)
        )

        differing = np.zeros(
            len(both),
            dtype=bool,
        )

        differing[valid] = ~np.isclose(
            a[valid],
            b[valid],
            rtol=1e-6,
            atol=1e-6,
        )

        relative = np.abs(
            a[valid] / b[valid] - 1
        )

        sources = long[
            "Close Field"
        ].astype("string")

        switches = sources.ne(
            sources.shift()
        ).fillna(False)

        if len(switches):
            switches.iloc[0] = False

        record = {
            "RIC": ric,
            "Overlap dates": len(both),
            "Comparable positive closes": int(
                valid.sum()
            ),
            "Different overlap closes": int(
                differing.sum()
            ),
            "Max relative difference": (
                float(relative.max())
                if len(relative)
                else float("nan")
            ),
            "Close source changes (long series)": int(
                switches.sum()
            ),
            "Long close fields": ", ".join(
                sorted(
                    long["Close Field"]
                    .dropna()
                    .astype(str)
                    .unique()
                )
            ),
            "Short close fields": ", ".join(
                sorted(
                    short["Close Field"]
                    .dropna()
                    .astype(str)
                    .unique()
                )
            ),
            "Status": (
                "REVIEW"
                if (
                    differing.any()
                    or switches.any()
                    or not valid.all()
                )
                else "CONSISTENT"
            ),
        }

        records.append(record)

        if record["Status"] != "CONSISTENT":
            print(
                f"{ric}: {record['Status']}, "
                f"{record['Different overlap closes']} "
                "different closes, "
                f"{record['Close source changes (long series)']} "
                "source switches"
            )

            if switches.any():
                print(
                    "  Source-switch dates:",
                    [
                        str(d.date())
                        for d in long.loc[
                            switches,
                            "Date",
                        ].head(10)
                    ],
                )

            if differing.any():
                print(
                    "  First differing dates:",
                    [
                        str(d.date())
                        for d in both.loc[
                            differing,
                            "Date",
                        ].head(10)
                    ],
                )

    report = pd.DataFrame(records)

    print("\nOverlap status counts:")

    print(
        report["Status"]
        .value_counts(dropna=False)
        .to_dict()
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        REPORT_DIR
        / "etf_overlap_close_consistency.csv",
        index=False,
    )

    print("Audit files saved in:", REPORT_DIR)


def main():
    audit_membership()
    audit_etf_overlap()


if __name__ == "__main__":
    main()
