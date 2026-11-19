
"""Verify reconstructed S&P 500 membership against LSEG snapshots.

Run:
    py -m data_pipeline.verify_historical_snapshots

Uses the configured default LSEG session.
No dataset files are modified.
"""

from pathlib import Path

import lseg.data as ld
import pandas as pd


ROOT = Path("data/processed/historical_universe")

UNIVERSE_FILE = (
    ROOT / "sp500_historical_universe.parquet"
)

CHECK_DATES = (
    "2014-12-01",
    "2015-03-02",
    "2016-08-01",
    "2023-09-01",
    "2024-12-02",
)

FOCUS_RICS = (
    "AGN.N",
    "CB.N",
    "JCI.N",
    "KVUE.N",
)


def base_ric(ric):
    return str(ric).split("^", 1)[0]


def get_historical_snapshot(date):
    date_string = pd.Timestamp(
        date
    ).strftime("%Y%m%d")

    chain = f"0#.SPX({date_string})"

    print("\nRequesting historical chain:", chain)

    result = ld.get_data(
        universe=[chain],
        fields=["TR.RIC"],
    )

    if result is None or result.empty:
        raise ValueError(
            f"No data returned for {chain}"
        )

    print(
        "Columns:",
        result.columns.tolist(),
    )

    if "RIC" not in result.columns:
        raise ValueError(
            "Historical chain response has no RIC column."
        )

    values = (
        result["RIC"]
        .astype("string")
        .str.strip()
        .dropna()
    )

    values = values.loc[
        values.ne("")
        & values.str.lower().ne("nan")
    ]

    snapshot = set(values.tolist())

    if not 450 <= len(snapshot) <= 550:
        raise ValueError(
            "Unexpected historical constituent count: "
            f"{len(snapshot)}"
        )

    return snapshot


def compare_snapshot(universe, date, snapshot):
    timestamp = pd.Timestamp(date)

    reconstructed = set(
        universe.loc[
            universe["Decision Date"].eq(timestamp),
            "Underlying RIC",
        ]
    )

    print("\nDecision date:", date)

    print(
        "LSEG historical snapshot:",
        len(snapshot),
    )

    print(
        "Reconstructed membership:",
        len(reconstructed),
    )

    missing = sorted(
        snapshot - reconstructed
    )

    extra = sorted(
        reconstructed - snapshot
    )

    print(
        "In LSEG snapshot but not reconstruction:",
        len(missing),
    )

    print(
        "In reconstruction but not LSEG snapshot:",
        len(extra),
    )

    print("\nMissing RICs:")
    print(missing[:30])

    print("\nExtra RICs:")
    print(extra[:30])

    print("\n--- FOCUS SECURITIES ---")

    for base in FOCUS_RICS:
        snapshot_matches = sorted(
            ric
            for ric in snapshot
            if base_ric(ric) == base
        )

        reconstructed_matches = sorted(
            ric
            for ric in reconstructed
            if base_ric(ric) == base
        )

        print(f"\n{base}")

        print(
            "Historical snapshot:",
            snapshot_matches,
        )

        print(
            "Reconstructed:",
            reconstructed_matches,
        )

    return {
        "Date": date,
        "Snapshot Count": len(snapshot),
        "Reconstructed Count": len(reconstructed),
        "Missing From Reconstruction": len(missing),
        "Extra In Reconstruction": len(extra),
        "Exact Match": snapshot == reconstructed,
        "KVUE In Snapshot": any(
            base_ric(ric) == "KVUE.N"
            for ric in snapshot
        ),
    }


def main():
    universe = pd.read_parquet(
        UNIVERSE_FILE
    )

    universe["Decision Date"] = pd.to_datetime(
        universe["Decision Date"]
    ).dt.normalize()

    print("\n--- HISTORICAL SNAPSHOT VERIFICATION ---")

    print(
        "Reconstructed records:",
        len(universe),
    )

    rows = []

    ld.open_session()

    try:
        for date in CHECK_DATES:
            try:
                snapshot = get_historical_snapshot(
                    date
                )

                comparison = compare_snapshot(
                    universe,
                    date,
                    snapshot,
                )

                rows.append(comparison)

            except Exception as error:
                print(
                    "\nSnapshot verification failed:",
                    date,
                )

                print(
                    type(error).__name__,
                    str(error),
                )

                rows.append({
                    "Date": date,
                    "Error": str(error),
                })

    finally:
        ld.close_session()

    print("\n--- VERIFICATION SUMMARY ---")

    summary = pd.DataFrame(rows)

    print(
        summary.to_string(index=False)
    )

    print("\nNo dataset files were modified.")


if __name__ == "__main__":
    main()
