
"""Inspect historical membership source files and LSE holiday bars.

Read-only. No LSEG API requests or file modifications.
"""

from pathlib import Path
import hashlib
import re

import pandas as pd


DATA_ROOT = Path("data")

HISTORY_ROOT = (
    DATA_ROOT
    / "processed"
    / "market_history_v2"
    / "series"
)

START = "2024-01-01"
END = "2026-10-30"

CANDIDATE_WORDS = (
    "member",
    "constituen",
    "change",
    "event",
    "rebalance",
    "composition",
    "sp500",
    "s_p_500",
    "index",
    "universe",
    "inclusion",
    "exclusion",
)

SKIP_FOLDERS = {
    "cache",
    "market_history_v1",
    "market_history_v2",
    "instrument_mapping",
    "ric_verification",
    "isin_bootstrap",
    "__pycache__",
}

HOLIDAY_DATES = [
    "2024-05-06",
    "2024-05-27",
    "2024-08-26",
    "2024-12-26",
    "2025-05-26",
    "2026-05-25",
    "2026-08-31",
]


def history_file(ric):
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
        / f"{safe}_{digest}"
        / "unadjusted"
        / f"{START}_{END}.parquet"
    )


def inspect_london_bars():
    print("\n--- LSE HOLIDAY RECORDS ---")

    for ric in (
        "EIMI.L",
        "VWRA.L",
        "CSPX.L",
    ):
        path = history_file(ric)

        if not path.is_file():
            print(
                f"{ric}: File not found: {path}"
            )
            continue

        df = pd.read_parquet(path)

        df["Date"] = pd.to_datetime(
            df["Date"]
        ).dt.normalize()

        mask = df["Date"].isin(
            pd.to_datetime(HOLIDAY_DATES)
        )

        bars = df.loc[mask]

        print(
            f"\n{ric}: {len(bars)} records "
            "on selected UK holidays"
        )

        columns = [
            column
            for column in (
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Close Field",
            )
            if column in bars.columns
        ]

        if not bars.empty:
            print(
                bars[columns].to_string(
                    index=False
                )
            )

        if ric == "CSPX.L":
            nearby = df.loc[
                df["Date"].between(
                    "2024-12-23",
                    "2024-12-30",
                )
            ]

            print("\nCSPX around Boxing Day:")

            print(
                nearby[columns].to_string(
                    index=False
                )
            )


def inspect_membership_sources():
    print(
        "\n--- POTENTIAL POINT-IN-TIME "
        "MEMBERSHIP SOURCES ---"
    )

    if not DATA_ROOT.exists():
        print("No data directory found.")
        return

    paths = []

    for path in DATA_ROOT.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".parquet",
            ".csv",
        }:
            continue

        relative = path.relative_to(
            DATA_ROOT
        )

        if any(
            part.lower() in SKIP_FOLDERS
            for part in relative.parts[:-1]
        ):
            continue

        if not any(
            word in path.name.lower()
            for word in CANDIDATE_WORDS
        ):
            continue

        paths.append(path)

    print("Candidate datasets:", len(paths))

    for path in sorted(paths):
        try:
            if path.suffix.lower() == ".parquet":
                import pyarrow.parquet as pq

                info = pq.ParquetFile(path)

                names = (
                    info.schema_arrow.names
                )

                count = (
                    info.metadata.num_rows
                )

            else:
                names = pd.read_csv(
                    path,
                    nrows=0,
                    encoding="utf-8-sig",
                ).columns.tolist()

                count = "not counted"

            date_columns = [
                column
                for column in names
                if any(
                    key in str(column).lower()
                    for key in (
                        "date",
                        "effective",
                        "added",
                        "removed",
                        "start",
                        "end",
                        "entry",
                        "exit",
                        "from",
                        "until",
                    )
                )
            ]

            print(
                f"\n{path} | rows: {count}"
            )

            print("Columns:", names)

            print(
                "Potential effective-date columns:",
                date_columns or "NONE",
            )

        except Exception as error:
            print(
                f"\n{path} | Could not inspect: "
                f"{type(error).__name__}: {error}"
            )

    if not paths:
        print(
            "No candidate files found under data/. "
            "The source records may be elsewhere."
        )

    print(
        "\nSource flags alone do not establish "
        "historical index membership intervals."
    )


def main():
    inspect_london_bars()

    inspect_membership_sources()

    print(
        "\nRead-only inspection complete."
    )


if __name__ == "__main__":
    main()
