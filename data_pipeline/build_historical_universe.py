
"""Reconstruct point-in-time S&P 500 membership from LSEG.

The configured default LSEG Platform session is used.
No explicit desktop session is opened.

Initial decision calendar:
    Existing 85 AAPL historical decision dates.

Output:
    data/processed/historical_universe/
"""

from datetime import date
from pathlib import Path
import json
import re

import lseg.data as ld
import pandas as pd


INDEX_RIC = ".SPX"

ROOT = Path("data/processed")
AAPL_DATA = ROOT / "ml_datasets/aapl_h21"
OUTPUT = ROOT / "historical_universe"

MIN_CONSTITUENTS = 450
MAX_CONSTITUENTS = 550


def clean_ric(value):
    if pd.isna(value):
        return None

    result = str(value).strip()

    if not result or result.lower() in {
        "nan", "none", "null", "<na>"
    }:
        return None

    return result


def normalized_column(column):
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(column).lower(),
    )


def identify_column(frame, candidates, label):
    lookup = {
        normalized_column(col): col
        for col in frame.columns
    }

    for candidate in candidates:
        key = normalized_column(candidate)

        if key in lookup:
            return lookup[key]

    raise ValueError(
        f"Cannot identify {label} column.\n"
        f"Actual columns: {frame.columns.tolist()}"
    )


def get_decision_dates():
    path = AAPL_DATA / "metadata.parquet"

    metadata = pd.read_parquet(path)

    if "Decision Date" not in metadata.columns:
        raise ValueError(
            "Decision Date missing from metadata."
        )

    dates = pd.to_datetime(
        metadata["Decision Date"],
        errors="raise",
    ).dt.normalize()

    if dates.isna().any():
        raise ValueError(
            "Invalid decision dates."
        )

    dates = pd.DatetimeIndex(
        dates.unique()
    ).sort_values()

    if len(dates) == 0:
        raise ValueError(
            "No historical decision dates found."
        )

    return dates


def get_current_constituents():
    print("\n--- CURRENT INDEX CONSTITUENTS ---")

    response = ld.get_data(
        universe=["0#.SPX"],
        fields=["TR.RIC"],
    )

    if response is None or response.empty:
        raise ValueError(
            "LSEG returned no current constituents."
        )

    print(
        "Returned columns:",
        response.columns.tolist(),
    )

    ric_column = identify_column(
        response,
        ["RIC", "TR.RIC"],
        "current constituent RIC",
    )

    constituents = {
        ric
        for value in response[ric_column]
        if (ric := clean_ric(value)) is not None
    }

    print(
        "Unique current constituents:",
        len(constituents),
    )

    if not (
        MIN_CONSTITUENTS
        <= len(constituents)
        <= MAX_CONSTITUENTS
    ):
        raise ValueError(
            "Unexpected current constituent count. "
            "Check the LSEG response for truncation."
        )

    print(
        "Current constituent sample:",
        sorted(constituents)[:10],
    )

    return constituents


def get_constituent_changes(start_date, end_date):
    """Retrieve joiner/leaver events in yearly chunks."""

    print("\n--- INDEX MEMBERSHIP CHANGES ---")

    chunks = []

    for year in range(
        start_date.year,
        end_date.year + 1,
    ):
        chunk_start = max(
            start_date,
            date(year, 1, 1),
        )

        chunk_end = min(
            end_date,
            date(year, 12, 31),
        )

        print(
            f"Downloading {chunk_start} to {chunk_end}"
        )

        response = ld.get_data(
            universe=[INDEX_RIC],
            fields=[
                "TR.IndexJLConstituentChangeDate",
                "TR.IndexJLConstituentRIC",
                "TR.IndexJLConstituentRIC.change",
            ],
            parameters={
                "SDate": chunk_start.strftime("%Y%m%d"),
                "EDate": chunk_end.strftime("%Y%m%d"),
                "IC": "B",
            },
        )

        if response is None or response.empty:
            raise ValueError(
                f"Empty membership response for {year}. "
                "Verify availability before continuing."
            )

        print(
            "Columns:",
            response.columns.tolist(),
        )

        print(
            "Rows:",
            len(response),
        )

        chunks.append(response)

    return pd.concat(
        chunks,
        ignore_index=True,
    )


def normalize_changes(raw_changes):
    date_column = identify_column(
        raw_changes,
        [
            "Date",
            "Constituent Change Date",
            "Index JL Constituent Change Date",
            "TR.IndexJLConstituentChangeDate",
        ],
        "membership change date",
    )

    ric_column = identify_column(
        raw_changes,
        [
            "Constituent RIC",
            "Index JL Constituent RIC",
            "TR.IndexJLConstituentRIC",
        ],
        "membership change RIC",
    )

    change_column = identify_column(
        raw_changes,
        [
            "Change",
            "Constituent Change",
            "Constituent RIC Change",
            "Index JL Constituent RIC Change",
            "TR.IndexJLConstituentRIC.change",
        ],
        "joiner/leaver direction",
    )

    events = pd.DataFrame({
        "Change Date": pd.to_datetime(
            raw_changes[date_column],
            errors="coerce",
        ).dt.normalize(),
        "RIC": raw_changes[ric_column].map(
            clean_ric
        ),
        "Raw Change": raw_changes[
            change_column
        ].astype("string").str.strip(),
    })

    if events[
        ["Change Date", "RIC", "Raw Change"]
    ].isna().any().any():
        raise ValueError(
            "Missing date, RIC or membership direction. "
            "Inspect the raw LSEG response."
        )

    direction_map = {
        "j": "Joiner",
        "joiner": "Joiner",
        "joined": "Joiner",
        "added": "Joiner",
        "addition": "Joiner",
        "l": "Leaver",
        "leaver": "Leaver",
        "left": "Leaver",
        "deleted": "Leaver",
        "removed": "Leaver",
        "deletion": "Leaver",
    }

    events["Change"] = (
        events["Raw Change"]
        .str.lower()
        .map(direction_map)
    )

    if events["Change"].isna().any():
        unknown = events.loc[
            events["Change"].isna(),
            "Raw Change",
        ].unique()

        raise ValueError(
            "Unknown joiner/leaver descriptions: "
            f"{unknown.tolist()}"
        )

    events = events[
        ["Change Date", "RIC", "Change"]
    ].drop_duplicates()

    events = events.sort_values(
        ["Change Date", "RIC", "Change"]
    ).reset_index(drop=True)

    # Detect same-day joiner/leaver pairs.
    # A balanced pair has zero net end-of-day membership
    # effect, but must be reported for auditing.

    conflicts = (
        events.groupby(
            ["Change Date", "RIC"]
        )["Change"].nunique()
    )

    conflict_keys = conflicts[
        conflicts > 1
    ].index

    if len(conflict_keys) > 0:
        print("\n--- SAME-DAY MEMBERSHIP CONFLICTS ---")

        print(
            "Conflicting date/RIC groups:",
            len(conflict_keys),
        )

        conflict_rows = (
            events.set_index(
                ["Change Date", "RIC"]
            )
            .loc[conflict_keys]
            .reset_index()
        )

        print(
            conflict_rows.to_string(index=False)
        )

        balanced_keys = []

        for change_date, ric in conflict_keys:
            group = events.loc[
                events["Change Date"].eq(change_date)
                & events["RIC"].eq(ric)
            ]

            joiners = int(
                group["Change"].eq("Joiner").sum()
            )

            leavers = int(
                group["Change"].eq("Leaver").sum()
            )

            if joiners != leavers:
                raise ValueError(
                    "Unbalanced membership conflict: "
                    f"{ric} on {change_date.date()}. "
                    f"Joiners={joiners}, Leavers={leavers}"
                )

            balanced_keys.append(
                (change_date, ric)
            )

        # These balanced pairs cancel for an
        # end-of-day membership reconstruction.
        conflict_index = pd.MultiIndex.from_tuples(
            balanced_keys,
            names=["Change Date", "RIC"],
        )

        event_index = pd.MultiIndex.from_frame(
            events[["Change Date", "RIC"]]
        )

        cancelled = event_index.isin(
            conflict_index
        )

        print(
            "\nBalanced same-day event rows:",
            int(cancelled.sum()),
        )

        print(
            "Balanced date/RIC groups:",
            len(balanced_keys),
        )

        events = events.loc[
            ~cancelled
        ].reset_index(drop=True)

        print(
            "Remaining membership events:",
            len(events),
        )

    print("\n--- NORMALIZED CHANGES ---")

    print(
        events["Change"].value_counts()
    )

    print("\nSample events:")

    print(
        events.head(10).to_string(
            index=False
        )
    )

    return events


def reconstruct_membership(
    current_constituents,
    events,
    decision_dates,
    anchor_date,
):
    """Reverse changes after each date from current membership.

    Change events dated on the decision date are treated
    as already effective on that date.
    """

    print("\n--- RECONSTRUCTING MEMBERSHIP ---")

    membership = set(current_constituents)

    events = events.sort_values(
        ["Change Date", "RIC"]
    )

    event_groups = {
        timestamp: group
        for timestamp, group in events.groupby(
            "Change Date"
        )
    }

    dates_desc = sorted(
        decision_dates,
        reverse=True,
    )

    event_dates_desc = sorted(
        event_groups,
        reverse=True,
    )

    event_cursor = 0
    records = []
    counts = []

    for decision_date in dates_desc:
        if decision_date.date() > anchor_date:
            raise ValueError(
                "Decision date is later than "
                "the current membership snapshot."
            )

        while (
            event_cursor < len(event_dates_desc)
            and event_dates_desc[event_cursor]
            > decision_date
        ):
            change_date = event_dates_desc[
                event_cursor
            ]

            group = event_groups[change_date]

            for row in group.itertuples(
                index=False
            ):
                ric = row[1]
                change = row[2]

                if change == "Joiner":
                    # Reverse a historical addition.
                    membership.discard(ric)

                elif change == "Leaver":
                    # Reverse a historical removal.
                    membership.add(ric)

            event_cursor += 1

        count = len(membership)

        print(
            f"{decision_date.date()}: "
            f"{count} constituent RICs"
        )

        if not (
            MIN_CONSTITUENTS
            <= count
            <= MAX_CONSTITUENTS
        ):
            raise ValueError(
                "Unexpected reconstructed constituent "
                f"count on {decision_date.date()}: "
                f"{count}. Check for missing change "
                "events or historical RIC changes."
            )

        counts.append({
            "Decision Date": decision_date,
            "Index": INDEX_RIC,
            "Constituent Count": count,
        })

        for ric in sorted(membership):
            records.append({
                "Decision Date": decision_date,
                "Index": INDEX_RIC,
                "Underlying RIC": ric,
                "Index Member": True,
            })

    universe = pd.DataFrame(records)

    universe = universe.sort_values(
        ["Decision Date", "Underlying RIC"]
    ).reset_index(drop=True)

    count_frame = pd.DataFrame(counts)

    count_frame = count_frame.sort_values(
        "Decision Date"
    ).reset_index(drop=True)

    if universe.duplicated(
        ["Decision Date", "Underlying RIC"]
    ).any():
        raise ValueError(
            "Duplicate ticker-date observations."
        )

    return universe, count_frame


def main():
    dates = get_decision_dates()

    print("\n--- RESEARCH CONFIGURATION ---")

    print("Index:", INDEX_RIC)

    print("Decision dates:", len(dates))

    print(
        "First decision date:",
        dates.min().date(),
    )

    print(
        "Last decision date:",
        dates.max().date(),
    )

    anchor_date = date.today()

    # Retrieve changes from the earliest decision
    # date through the current snapshot date.
    start_date = dates.min().date()

    ld.open_session()

    try:
        current = get_current_constituents()

        raw_changes = get_constituent_changes(
            start_date,
            anchor_date,
        )

    finally:
        ld.close_session()

    print("\n--- RAW LSEG CHANGE VALUES ---")

    print(
        raw_changes["Change"]
        .astype("string")
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n--- RAW CHANGE SAMPLE ---")

    print(
        raw_changes[
            ["Date", "Constituent RIC", "Change"]
        ].head(15).to_string(index=False)
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_changes.to_parquet(
        OUTPUT / "sp500_membership_events_raw.parquet",
        index=False,
    )

    print(
        "\nRaw membership events cached:",
        len(raw_changes),
    )

    events = normalize_changes(
        raw_changes
    )

    universe, counts = reconstruct_membership(
        current,
        events,
        dates,
        anchor_date,
    )

    print("\n--- UNIVERSE SUMMARY ---")

    print(
        "Ticker-date records:",
        len(universe),
    )

    print(
        "Unique historical RICs:",
        universe["Underlying RIC"].nunique(),
    )

    print(
        "Minimum members:",
        counts["Constituent Count"].min(),
    )

    print(
        "Maximum members:",
        counts["Constituent Count"].max(),
    )

    print(
        "Average members:",
        f"{counts['Constituent Count'].mean():.1f}",
    )

    print("\n--- SAMPLE HISTORICAL UNIVERSE ---")

    print(
        universe.head(20).to_string(
            index=False
        )
    )

    # Save only after all validation checks pass.
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    universe.to_parquet(
        OUTPUT / "sp500_historical_universe.parquet",
        index=False,
    )

    events.to_parquet(
        OUTPUT / "sp500_membership_events.parquet",
        index=False,
    )

    counts.to_csv(
        OUTPUT / "sp500_membership_counts.csv",
        index=False,
    )

    with (
        OUTPUT / "universe_manifest.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            {
                "index": INDEX_RIC,
                "anchor_date": str(anchor_date),
                "decision_dates": len(dates),
                "ticker_date_records": len(universe),
                "unique_historical_rics": int(
                    universe[
                        "Underlying RIC"
                    ].nunique()
                ),
                "membership_method": (
                    "Current constituents plus "
                    "reverse joiner/leaver events"
                ),
                "session": (
                    "Configured default LSEG Platform"
                ),
                "security_identifier_status": (
                    "RIC only; permanent identifier "
                    "mapping pending"
                ),
            },
            file,
            indent=2,
        )

    print("\nHistorical universe saved to:", OUTPUT)
    print("Reconstruction complete.")


if __name__ == "__main__":
    main()
