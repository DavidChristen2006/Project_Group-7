
"""Audit reconstructed S&P 500 point-in-time membership.

No LSEG calls. Original files are not modified.
"""

from pathlib import Path

import pandas as pd


ROOT = Path("data/processed")

HISTORICAL = (
    ROOT / "historical_universe"
    / "sp500_historical_universe.parquet"
)

COUNTS = (
    ROOT / "historical_universe"
    / "sp500_membership_counts.csv"
)

EVENTS = (
    ROOT / "historical_universe"
    / "sp500_membership_events.parquet"
)

GRID = (
    ROOT / "ever_member_universe"
    / "ever_member_collection_grid.parquet"
)

REPORTS = ROOT / "historical_universe" / "audit"

# Diagnostic threshold, not an exact index-count requirement.
LOW_COUNT_WARNING = 450


def parse_dates(df, column, label):
    df = df.copy()

    df[column] = pd.to_datetime(
        df[column],
        errors="coerce",
    ).dt.normalize()

    if df[column].isna().any():
        raise ValueError(
            f"{label}: invalid values in {column}"
        )

    return df


def parse_member_flags(values):
    text = (
        values.astype("string")
        .str.strip()
        .str.lower()
    )

    known = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
    }

    parsed = text.map(known).astype("boolean")

    invalid = values.notna() & parsed.isna()

    if invalid.any():
        examples = (
            values.loc[invalid]
            .astype(str)
            .unique()[:5]
            .tolist()
        )

        raise ValueError(
            f"Unrecognized Index Member values: {examples}"
        )

    return parsed


def main():
    hist = pd.read_parquet(HISTORICAL)
    grid = pd.read_parquet(GRID)

    needed_hist = {
        "Decision Date",
        "Index",
        "Underlying RIC",
        "Index Member",
    }

    needed_grid = {
        "Decision Date",
        "Underlying RIC",
    }

    if needed_hist - set(hist.columns):
        raise ValueError(
            "Historical columns missing: "
            f"{needed_hist - set(hist.columns)}"
        )

    if needed_grid - set(grid.columns):
        raise ValueError(
            "Grid columns missing: "
            f"{needed_grid - set(grid.columns)}"
        )

    hist = parse_dates(
        hist, "Decision Date", "historical membership"
    )

    grid = parse_dates(
        grid, "Decision Date", "collection grid"
    )

    hist["Index Member"] = parse_member_flags(
        hist["Index Member"]
    )

    for label, df in (
        ("historical", hist),
        ("grid", grid),
    ):
        if df["Underlying RIC"].isna().any():
            raise ValueError(
                f"{label}: missing Underlying RIC"
            )

        df["Underlying RIC"] = (
            df["Underlying RIC"]
            .astype("string")
            .str.strip()
        )

        if df["Underlying RIC"].eq("").any():
            raise ValueError(
                f"{label}: blank Underlying RIC"
            )

    if hist["Index"].isna().any():
        raise ValueError(
            "Historical membership has blank Index values."
        )

    hist_key = [
        "Decision Date",
        "Index",
        "Underlying RIC",
    ]

    grid_key = [
        "Decision Date",
        "Underlying RIC",
    ]

    hist_dupes = int(
        hist.duplicated(hist_key).sum()
    )

    grid_dupes = int(
        grid.duplicated(grid_key).sum()
    )

    dates = pd.DatetimeIndex(
        sorted(grid["Decision Date"].unique())
    )

    index_names = sorted(
        hist["Index"]
        .astype(str)
        .unique()
        .tolist()
    )

    hist_dates = set(hist["Decision Date"])
    grid_dates = set(dates)

    # Count members separately for each index and date.
    grouped = (
        hist.groupby(
            ["Decision Date", "Index"],
            dropna=False,
        )
        .agg(
            Rows=("Underlying RIC", "size"),
            Member_Count=(
                "Index Member",
                lambda x: int(
                    x.fillna(False).sum()
                ),
            ),
            Unknown_Flags=(
                "Index Member",
                lambda x: int(
                    x.isna().sum()
                ),
            ),
        )
    )

    full_index = pd.MultiIndex.from_product(
        [dates, index_names],
        names=["Decision Date", "Index"],
    )

    summary = (
        grouped
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    summary["Below_450_Heuristic"] = (
        summary["Member_Count"] < LOW_COUNT_WARNING
    )

    print("\n--- MEMBERSHIP SOURCE INTEGRITY ---")

    print("Historical rows:", len(hist))
    print("Grid rows:", len(grid))
    print("Indices:", index_names)
    print("Grid decision dates:", len(dates))
    print("Historical decision dates:", len(hist_dates))

    print(
        "Dates missing from historical:",
        len(grid_dates - hist_dates),
    )

    print(
        "Historical dates outside grid:",
        len(hist_dates - grid_dates),
    )

    print(
        "Duplicate historical date-index-RIC keys:",
        hist_dupes,
    )

    print(
        "Duplicate grid date-RIC keys:",
        grid_dupes,
    )

    print(
        "Unknown membership flags:",
        int(hist["Index Member"].isna().sum()),
    )

    print("\n--- CONSTITUENTS PER DECISION DATE ---")

    for index, part in summary.groupby("Index"):
        counts = part["Member_Count"]

        print(
            f"\n{index}: "
            f"min={counts.min()}, "
            f"median={counts.median():.0f}, "
            f"max={counts.max()}, "
            f"mean={counts.mean():.1f}"
        )

        print(
            "Dates with fewer than 450 members:",
            int((counts < 450).sum()),
        )

        print("\nEarliest six dates:")

        print(
            part[
                ["Decision Date", "Rows", "Member_Count"]
            ]
            .head(6)
            .to_string(index=False)
        )

        print("\nLatest six dates:")

        print(
            part[
                ["Decision Date", "Rows", "Member_Count"]
            ]
            .tail(6)
            .to_string(index=False)
        )

    # Compare with the pre-existing membership count file.
    if COUNTS.is_file():
        published = pd.read_csv(COUNTS)

        needed = {
            "Decision Date",
            "Index",
            "Constituent Count",
        }

        if needed.issubset(published.columns):
            published = parse_dates(
                published,
                "Decision Date",
                "count file",
            )

            published["Constituent Count"] = pd.to_numeric(
                published["Constituent Count"],
                errors="coerce",
            )

            if published.duplicated(
                ["Decision Date", "Index"]
            ).any():
                raise ValueError(
                    "Duplicate date/index keys in count CSV."
                )

            summary = summary.merge(
                published[
                    [
                        "Decision Date",
                        "Index",
                        "Constituent Count",
                    ]
                ],
                on=["Decision Date", "Index"],
                how="left",
                validate="one_to_one",
            )

            summary["Count_Difference"] = (
                summary["Member_Count"]
                - summary["Constituent Count"]
            )

            print(
                "\n--- INTERNAL COUNT RECONCILIATION ---"
            )

            print(
                "Count rows matched:",
                int(
                    summary[
                        "Constituent Count"
                    ].notna().sum()
                ),
            )

            print(
                "Nonzero discrepancies:",
                int(
                    summary[
                        "Count_Difference"
                    ].dropna().ne(0).sum()
                ),
            )

            print(
                "Matching reconstructed counts is "
                "not independent proof of completeness."
            )

        else:
            print(
                "\nCount-file schema differs:",
                published.columns.tolist(),
            )

    # Compare positive historical membership records
    # with the existing 787-RIC collection grid.
    membership_keys = (
        hist.loc[
            hist["Index Member"].eq(True).fillna(False),
            grid_key,
        ]
        .drop_duplicates()
        .assign(Is_Member=True)
    )

    matched = grid[grid_key].merge(
        membership_keys,
        on=grid_key,
        how="left",
        validate="one_to_one",
    )

    print("\n--- EVER-MEMBER GRID MATCH ---")

    print(
        "Member grid rows:",
        int(
            matched["Is_Member"]
            .fillna(False)
            .sum()
        ),
    )

    print(
        "Grid rows without positive membership:",
        int(matched["Is_Member"].isna().sum()),
    )

    unmatched_members = membership_keys.merge(
        grid[grid_key],
        on=grid_key,
        how="left",
        indicator=True,
    )

    outside = unmatched_members.loc[
        unmatched_members["_merge"].eq("left_only")
    ]

    print(
        "Positive membership keys absent from "
        "ever-member grid:",
        len(outside),
    )

    print("Sample absent keys:")

    print(
        outside[grid_key]
        .head(10)
        .to_string(index=False)
    )

    if EVENTS.is_file():
        events = pd.read_parquet(EVENTS)

        print("\n--- HISTORICAL EVENTS ---")

        print("Event rows:", len(events))
        print("Columns:", events.columns.tolist())

        if "Change Date" in events:
            event_dates = pd.to_datetime(
                events["Change Date"],
                errors="coerce",
            )

            print(
                "Earliest/last event:",
                event_dates.min(),
                event_dates.max(),
            )

        if "Change" in events:
            print(
                "Change values:",
                events["Change"]
                .value_counts(dropna=False)
                .to_dict(),
            )

    # Save audit reports, not a new authoritative
    # membership panel.
    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_file = (
        REPORTS / "membership_counts_audit.csv"
    )

    summary.to_csv(
        report_file,
        index=False,
    )

    outside[grid_key].to_csv(
        REPORTS / "membership_keys_outside_grid.csv",
        index=False,
    )

    print("\nSaved audit:", report_file)

    print(
        "Historical membership data has not been modified."
    )


if __name__ == "__main__":
    main()
