
"""Prepare an equity history manifest from reconciled membership.

No LSEG requests.
Does not modify original membership or price data.
"""

from pathlib import Path

import pandas as pd


ROOT = (Path(__file__).resolve().parents[1] / "data/processed")

CANDIDATE_FILE = (
    ROOT
    / "historical_universe/reconciliation_candidate"
    / "sp500_membership_reconciled_candidate.parquet"
)

GRID_FILE = (
    ROOT
    / "ever_member_universe"
    / "ever_member_collection_grid.parquet"
)

MASTER_FILE = (
    ROOT
    / "ever_member_universe"
    / "ever_member_instruments.parquet"
)

MAPPING_FILE = (
    ROOT
    / "ever_member_universe/instrument_mapping"
    / "instrument_mapping.parquet"
)

EVENTS_FILE = (
    ROOT
    / "historical_universe"
    / "sp500_membership_events.parquet"
)

OUTPUT = (
    ROOT
    / "historical_universe"
    / "equity_collection_candidate"
)

KEY = ["Decision Date", "Underlying RIC"]

VALID_SOURCES = {
    "OBSERVED_SNAPSHOT_MATCH",
    "PROVISIONAL_EVENT_REPLAY",
    "OFFICIAL_EVENT_CORRECTION_CANDIDATE",
}


def prepare_dates(frame, column):
    """Perform the prepare dates operation for this research workflow.

    Args:
        frame: Input to the prepare_dates operation.
        column: Input to the prepare_dates operation.

    Returns:
        Computed result or status for the caller.
    """
    frame = frame.copy()

    frame[column] = pd.to_datetime(
        frame[column],
        errors="raise",
    ).dt.normalize()

    if frame[column].isna().any():
        raise ValueError(
            f"Missing dates in {column}."
        )

    return frame


def prepare_rics(frame, column):
    """Perform the prepare rics operation for this research workflow.

    Args:
        frame: Input to the prepare_rics operation.
        column: Input to the prepare_rics operation.

    Returns:
        Computed result or status for the caller.
    """
    frame = frame.copy()

    frame[column] = (
        frame[column]
        .astype("string")
        .str.strip()
    )

    if (
        frame[column].isna().any()
        or frame[column].eq("").any()
    ):
        raise ValueError(
            f"Missing RIC in {column}."
        )

    return frame


def main():
    """Run the module entry point and its workflow.
    """
    candidate = prepare_dates(
        pd.read_parquet(CANDIDATE_FILE),
        "Decision Date",
    )

    candidate = prepare_rics(
        candidate,
        "Underlying RIC",
    )

    grid = prepare_dates(
        pd.read_parquet(GRID_FILE),
        "Decision Date",
    )

    grid = prepare_rics(
        grid,
        "Underlying RIC",
    )

    master = prepare_rics(
        pd.read_parquet(MASTER_FILE),
        "Underlying RIC",
    )

    mapping = prepare_rics(
        pd.read_parquet(MAPPING_FILE),
        "Underlying RIC",
    )

    events = prepare_dates(
        pd.read_parquet(EVENTS_FILE),
        "Change Date",
    )

    events = prepare_rics(
        events,
        "RIC",
    )

    if not candidate["Index"].eq(".SPX").all():
        raise ValueError(
            "Candidate includes an unexpected index."
        )

    candidate_flags = (
        candidate["Index Member"]
        .astype("string")
        .str.lower()
    )

    if not candidate_flags.eq("true").all():
        raise ValueError(
            "Candidate contains non-member rows."
        )

    if not candidate["Membership Source"].isin(
        VALID_SOURCES
    ).all():
        raise ValueError(
            "Unknown membership provenance."
        )

    for name, frame in (
        ("candidate", candidate),
        ("grid", grid),
    ):
        if frame.duplicated(KEY).any():
            raise ValueError(
                f"Duplicate keys in {name}."
            )

    if master["Underlying RIC"].duplicated().any():
        raise ValueError(
            "Duplicate RIC in equity master."
        )

    if mapping["Underlying RIC"].duplicated().any():
        raise ValueError(
            "Duplicate RIC in identifier mapping."
        )

    decision_dates = set(
        grid["Decision Date"]
    )

    if set(candidate["Decision Date"]) != decision_dates:
        raise ValueError(
            "Candidate and grid decision dates differ."
        )

    # Verify every member/date pair appears in the
    # original full ever-member collection grid.
    missing_keys = candidate[KEY].merge(
        grid[KEY],
        on=KEY,
        how="left",
        indicator=True,
        validate="one_to_one",
    )

    missing_keys = missing_keys.loc[
        missing_keys["_merge"].eq("left_only")
    ]

    if not missing_keys.empty:
        raise ValueError(
            f"{len(missing_keys)} member-date keys "
            "absent from collection grid."
        )

    allowed = set(
        master["Underlying RIC"]
    )

    required = set(
        candidate["Underlying RIC"]
    )

    unknown = sorted(
        required - allowed
    )

    if unknown:
        raise ValueError(
            f"Member RICs absent from equity master: "
            f"{unknown[:10]}"
        )

    mapping_lookup = mapping.set_index(
        "Underlying RIC"
    )

    unmapped = sorted(
        ric
        for ric in required
        if (
            ric not in mapping_lookup.index
            or mapping_lookup.at[
                ric, "Mapping Status"
            ] != "IDENTIFIER_MAPPED"
        )
    )

    if unmapped:
        raise ValueError(
            f"Member RICs without verified identifiers: "
            f"{unmapped[:10]}"
        )

    # Attach candidate membership without changing
    # existing price/option availability columns.
    member_source = candidate[
        KEY + ["Membership Source"]
    ]

    panel = grid.merge(
        member_source,
        on=KEY,
        how="left",
        validate="one_to_one",
    )

    panel["Candidate Index Member"] = (
        panel["Membership Source"].notna()
    )

    panel["Membership Source"] = (
        panel["Membership Source"]
        .fillna("NOT_MEMBER_BY_REPLAY")
    )

    panel["Provisional Membership"] = (
        panel["Candidate Index Member"]
        & panel["Membership Source"].ne(
            "OBSERVED_SNAPSHOT_MATCH"
        )
    )

    # IMPORTANT:
    # The previous replay script marked every member
    # on an event date with a date-level boolean.
    # Here we flag the actual affected security only.
    same_day = (
        events.loc[
            events["Change Date"].isin(
                decision_dates
            ),
            ["Change Date", "RIC", "Change"],
        ]
        .rename(
            columns={
                "Change Date": "Decision Date",
                "RIC": "Underlying RIC",
                "Change": "Same-Day Event Action",
            }
        )
    )

    if same_day.duplicated(KEY).any():
        raise ValueError(
            "Multiple events for one RIC on a decision date."
        )

    event_check = same_day.merge(
        grid[KEY],
        on=KEY,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if event_check["_merge"].eq(
        "left_only"
    ).any():
        raise ValueError(
            "Same-day event RIC missing from the grid."
        )

    panel = panel.merge(
        same_day,
        on=KEY,
        how="left",
        validate="one_to_one",
    )

    panel["Same-Day Event Timing Unverified"] = (
        panel["Same-Day Event Action"].notna()
    )

    panel["Same-Day Event Action"] = (
        panel["Same-Day Event Action"].fillna("")
    )

    panel = panel.sort_values(
        KEY
    ).reset_index(drop=True)

    members = panel.loc[
        panel["Candidate Index Member"]
    ].copy()

    counts = members.groupby(
        "Decision Date"
    ).size()

    if (
        counts.lt(450).any()
        or counts.gt(550).any()
    ):
        raise ValueError(
            "Implausible constituent counts."
        )

    manifest = (
        members
        .groupby("Underlying RIC", sort=True)
        .agg(
            First_Member_Date=(
                "Decision Date", "min"
            ),
            Last_Member_Date=(
                "Decision Date", "max"
            ),
            Member_Decision_Dates=(
                "Decision Date", "nunique"
            ),
            Provisional_Decision_Dates=(
                "Provisional Membership", "sum"
            ),
            Same_Day_Member_Events=(
                "Same-Day Event Timing Unverified",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            ["First_Member_Date", "Underlying RIC"]
        )
        .reset_index(drop=True)
    )

    if manifest["Underlying RIC"].duplicated().any():
        raise ValueError(
            "Duplicate RIC in collection manifest."
        )

    rics = sorted(
        manifest["Underlying RIC"].astype(str)
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    panel.to_parquet(
        OUTPUT / "equity_membership_grid_candidate.parquet",
        index=False,
    )

    manifest.to_parquet(
        OUTPUT / "equity_collection_manifest.parquet",
        index=False,
    )

    manifest.to_csv(
        OUTPUT / "equity_collection_manifest.csv",
        index=False,
    )

    (OUTPUT / "equity_member_rics.txt").write_text(
        "\n".join(rics) + "\n",
        encoding="utf-8",
    )

    print(
        "\n--- EQUITY HISTORY COLLECTION PREPARATION ---"
    )

    print(
        "Decision dates:",
        len(decision_dates),
    )

    print(
        "Total ever-member grid rows:",
        len(panel),
    )

    print(
        "Candidate member-date rows:",
        len(members),
    )

    print(
        "Unique member RICs to collect:",
        len(rics),
    )

    print(
        "RICs in master not selected:",
        len(allowed - required),
    )

    print(
        "Minimum members per date:",
        int(counts.min()),
    )

    print(
        "Maximum members per date:",
        int(counts.max()),
    )

    print(
        "Provisional member-date rows:",
        int(
            panel["Provisional Membership"].sum()
        ),
    )

    print(
        "Same-day event records:",
        int(
            panel[
                "Same-Day Event Timing Unverified"
            ].sum()
        ),
    )

    print(
        "\nMembership source distribution:"
    )

    print(
        members["Membership Source"]
        .value_counts()
        .to_string()
    )

    print("\nFirst 10 manifest entries:")

    print(
        manifest.head(10).to_string(index=False)
    )

    print("\nSaved in:", OUTPUT)

    print(
        "STATUS: COLLECTION CANDIDATE. "
        "Membership timing and ML eligibility "
        "remain provisional."
    )


if __name__ == "__main__":
    main()
