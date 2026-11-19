
"""Verify S&P 500 historical constituent snapshots using LSEG.

Run from project root:
    py -m data_pipeline.verify_historical_constituents

Uses the configured default LSEG Platform session.

Does not modify existing files.
"""

from pathlib import Path

import lseg.data as ld
import pandas as pd


ROOT = Path("data/processed/historical_universe")

INDEX_RIC = ".SPX"

CHECK_DATES = (
    "2014-12-01",
    "2023-09-01",
    "2024-12-02",
)

FOCUS_RICS = (
    "ABNB.OQ",
    "AGN.N",
    "CB.N",
    "JCI.N",
    "KVUE.N",
)


def base_ric(ric):
    return str(ric).split("^", 1)[0]


def get_snapshot(date):
    print(f"\n--- LSEG HISTORICAL SNAPSHOT: {date} ---")

    result = ld.get_data(
        universe=[INDEX_RIC],
        fields=[
            "TR.IndexConstituentRIC",
        ],
        parameters={
            "SDate": date,
            "EDate": date,
        },
    )

    if result is None or result.empty:
        raise ValueError(
            f"No constituent data returned for {date}."
        )

    print("Columns:", result.columns.tolist())
    print("Rows returned:", len(result))

    print("\nFirst five rows:")
    print(result.head(5).to_string(index=False))

    possible_columns = (
        "Constituent RIC",
        "Index Constituent RIC",
        "TR.IndexConstituentRIC",
    )

    ric_column = next(
        (
            column
            for column in possible_columns
            if column in result.columns
        ),
        None,
    )

    if ric_column is None:
        raise ValueError(
            "Unable to identify constituent RIC column. "
            f"Returned: {result.columns.tolist()}"
        )

    rics = (
        result[ric_column]
        .astype("string")
        .str.strip()
        .dropna()
    )

    rics = rics.loc[
        rics.ne("")
        & rics.str.lower().ne("nan")
    ]

    constituents = set(rics.tolist())

    if not 450 <= len(constituents) <= 550:
        raise ValueError(
            "Unexpected number of historical constituents: "
            f"{len(constituents)}"
        )

    return constituents


def compare(universe, date, snapshot):
    date_value = pd.Timestamp(date)

    reconstructed = set(
        universe.loc[
            universe["Decision Date"].eq(date_value),
            "Underlying RIC",
        ]
    )

    missing = snapshot - reconstructed
    extra = reconstructed - snapshot

    print("\n--- COMPARISON ---")

    print("Date:", date)
    print("Historical snapshot:", len(snapshot))
    print("Reconstructed:", len(reconstructed))

    print(
        "Snapshot-only RICs:",
        len(missing),
    )

    print(
        "Reconstruction-only RICs:",
        len(extra),
    )

    print("\nSnapshot-only sample:")
    print(sorted(missing)[:20])

    print("\nReconstruction-only sample:")
    print(sorted(extra)[:20])

    print("\n--- FOCUS INSTRUMENTS ---")

    for base in FOCUS_RICS:
        historical_matches = sorted(
            ric
            for ric in snapshot
            if base_ric(ric) == base
        )

        reconstructed_matches = sorted(
            ric
            for ric in reconstructed
            if base_ric(ric) == base
        )

        print(
            f"{base}: "
            f"historical={historical_matches}, "
            f"reconstructed={reconstructed_matches}"
        )

    return {
        "Date": date,
        "Historical Count": len(snapshot),
        "Reconstructed Count": len(reconstructed),
        "Snapshot Only": len(missing),
        "Reconstruction Only": len(extra),
        "Exact Match": snapshot == reconstructed,
        "Historical RICs": snapshot,
    }


def main():
    universe = pd.read_parquet(
        ROOT / "sp500_historical_universe.parquet"
    )

    universe["Decision Date"] = pd.to_datetime(
        universe["Decision Date"]
    ).dt.normalize()

    print("\n--- HISTORICAL CONSTITUENT VERIFICATION ---")

    print("Index:", INDEX_RIC)
    print("Dates to verify:", len(CHECK_DATES))

    results = []

    ld.open_session()

    try:
        for date in CHECK_DATES:
            try:
                snapshot = get_snapshot(date)

                result = compare(
                    universe,
                    date,
                    snapshot,
                )

                results.append(result)

            except Exception as error:
                print(
                    f"\nVerification failed on {date}: "
                    f"{type(error).__name__}: {error}"
                )

    finally:
        ld.close_session()

    print("\n--- SUMMARY ---")

    for result in results:
        print(
            result["Date"],
            "| Snapshot:",
            result["Historical Count"],
            "| Reconstructed:",
            result["Reconstructed Count"],
            "| Snapshot only:",
            result["Snapshot Only"],
            "| Reconstruction only:",
            result["Reconstruction Only"],
        )

    if len(results) >= 2:
        print("\n--- HISTORICAL VARIATION CHECK ---")

        for i in range(1, len(results)):
            previous = results[i - 1]
            current = results[i]

            added = (
                current["Historical RICs"]
                - previous["Historical RICs"]
            )

            removed = (
                previous["Historical RICs"]
                - current["Historical RICs"]
            )

            print(
                f"{previous['Date']} to {current['Date']}: "
                f"{len(added)} additions, "
                f"{len(removed)} removals"
            )

        first = results[0]["Historical RICs"]
        last = results[-1]["Historical RICs"]

        if first == last:
            print(
                "\nWARNING: Historical snapshots are identical. "
                "Check whether LSEG is ignoring the date parameters."
            )
        else:
            print(
                "\nHistorical snapshots differ across dates."
            )

    print("\nNo files modified.")


if __name__ == "__main__":
    main()
