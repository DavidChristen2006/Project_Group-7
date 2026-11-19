
"""Reconstruct missing S&P 500 monthly memberships with event replay.

Original membership files and collection grid are not modified.
"""

from pathlib import Path

import pandas as pd


ROOT = Path("data/processed")

HISTORY = (
    ROOT / "historical_universe"
    / "sp500_historical_universe.parquet"
)

EVENTS = (
    ROOT / "historical_universe"
    / "sp500_membership_events.parquet"
)

GRID = (
    ROOT / "ever_member_universe"
    / "ever_member_collection_grid.parquet"
)

MASTER = (
    ROOT / "ever_member_universe"
    / "ever_member_instruments.parquet"
)

OUTPUT = (
    ROOT / "historical_universe"
    / "reconstruction_candidate"
)

INDEX = ".SPX"


def parsed_dates(df, name):
    result = df.copy()

    result[name] = pd.to_datetime(
        result[name],
        errors="raise",
    ).dt.normalize()

    if result[name].isna().any():
        raise ValueError(
            f"Missing dates in {name}."
        )

    return result


def members_by_date():
    history = parsed_dates(
        pd.read_parquet(HISTORY),
        "Decision Date",
    )

    history = history.loc[
        history["Index"].eq(INDEX)
    ].copy()

    if history.empty:
        raise ValueError(
            f"No {INDEX} snapshots found."
        )

    key = [
        "Decision Date",
        "Underlying RIC",
    ]

    if history.duplicated(key).any():
        raise ValueError(
            "Duplicate snapshot date-RIC keys."
        )

    flags = (
        history["Index Member"]
        .astype("string")
        .str.lower()
        .str.strip()
    )

    if not flags.isin(
        {"true", "false", "1", "0"}
    ).all():
        raise ValueError(
            "Unknown Index Member values."
        )

    history["Is Member"] = flags.isin(
        {"true", "1"}
    )

    snapshots = {
        date: set(
            part.loc[
                part["Is Member"],
                "Underlying RIC",
            ].astype(str)
        )
        for date, part in history.groupby(
            "Decision Date"
        )
    }

    return snapshots


def replay(state, events, start, end):
    """Apply events after start and on/before end."""

    state = set(state)

    subset = events.loc[
        events["Change Date"].gt(start)
        & events["Change Date"].le(end)
    ]

    redundant = []

    columns = [
        "Change Date",
        "RIC",
        "Change",
    ]

    for day, ric, change in subset[
        columns
    ].itertuples(index=False, name=None):

        if change == "Joiner":
            if ric in state:
                redundant.append(
                    (day, ric, change)
                )

            state.add(ric)

        else:
            if ric not in state:
                redundant.append(
                    (day, ric, change)
                )

            state.discard(ric)

    return state, redundant, subset


def main():
    snapshots = members_by_date()

    grid = parsed_dates(
        pd.read_parquet(GRID),
        "Decision Date",
    )

    all_dates = [
        pd.Timestamp(value)
        for value in sorted(
            grid["Decision Date"].unique()
        )
    ]

    master = pd.read_parquet(MASTER)

    allowed = set(
        master["Underlying RIC"].astype(str)
    )

    events = parsed_dates(
        pd.read_parquet(EVENTS),
        "Change Date",
    )

    events["RIC"] = (
        events["RIC"]
        .astype("string")
        .str.strip()
    )

    events["Change"] = (
        events["Change"]
        .astype("string")
        .str.strip()
    )

    if events[
        ["Change Date", "RIC", "Change"]
    ].isna().any().any():
        raise ValueError(
            "Missing event dates, RICs, or change types."
        )

    if not events["Change"].isin(
        {"Joiner", "Leaver"}
    ).all():
        raise ValueError(
            "Unexpected event type."
        )

    events = (
        events
        .sort_values(
            "Change Date",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    # Identify missing decision dates.
    missing_i = [
        index
        for index, date in enumerate(all_dates)
        if date not in snapshots
    ]

    if not missing_i:
        print(
            "All decision dates already have snapshots."
        )
        return

    expected_indices = list(
        range(
            missing_i[0],
            missing_i[-1] + 1,
        )
    )

    if missing_i != expected_indices:
        raise ValueError(
            "Multiple noncontiguous membership gaps."
        )

    if (
        missing_i[0] == 0
        or missing_i[-1] == len(all_dates) - 1
    ):
        raise ValueError(
            "Missing snapshot anchor on one side."
        )

    gap = [
        all_dates[index]
        for index in missing_i
    ]

    left = all_dates[
        missing_i[0] - 1
    ]

    right = all_dates[
        missing_i[-1] + 1
    ]

    print("\n--- MEMBERSHIP GAP ---")

    print(
        "Missing dates:",
        len(gap),
        "from",
        gap[0].date(),
        "to",
        gap[-1].date(),
    )

    print(
        "Left anchor:",
        left.date(),
        "members:",
        len(snapshots[left]),
    )

    print(
        "Right anchor:",
        right.date(),
        "members:",
        len(snapshots[right]),
    )

    bracket_events = events.loc[
        events["Change Date"].gt(left)
        & events["Change Date"].le(right)
    ]

    if bracket_events.duplicated(
        ["Change Date", "RIC"]
    ).any():
        raise ValueError(
            "Multiple same-day changes for a RIC "
            "inside the missing interval."
        )

    print(
        "Events in bracket:",
        len(bracket_events),
    )

    print(
        "Events by year:",
        bracket_events.groupby(
            bracket_events["Change Date"].dt.year
        ).size().to_dict(),
    )

    # Reconstruct forward from the left anchor.
    current = set(snapshots[left])
    previous = left

    reconstructed = {}
    issues = []
    warnings = []

    for decision_date in [*gap, right]:

        current, redundant, used = replay(
            current,
            events,
            previous,
            decision_date,
        )

        for day, ric, change in redundant:
            warnings.append({
                "Type": "Redundant event",
                "Date": day,
                "RIC": ric,
                "Detail": change,
            })

        unknown = sorted(
            current - allowed
        )

        for ric in unknown:
            issues.append({
                "Type": "RIC absent from equity master",
                "Date": decision_date,
                "RIC": ric,
                "Detail": "",
            })

        # Plausibility check, not an exact
        # S&P 500 constituent-count requirement.
        if not (450 <= len(current) <= 550):
            issues.append({
                "Type": "Implausible member count",
                "Date": decision_date,
                "RIC": "",
                "Detail": str(len(current)),
            })

        if decision_date != right:
            reconstructed[
                decision_date
            ] = set(current)

        previous = decision_date

    # Validate against the existing right anchor.
    missing_from_replay = sorted(
        snapshots[right] - current
    )

    extra_in_replay = sorted(
        current - snapshots[right]
    )

    anchor_match = not (
        missing_from_replay
        or extra_in_replay
    )

    print("\n--- RIGHT-ANCHOR VALIDATION ---")

    print(
        "Exact match:",
        anchor_match,
    )

    print(
        "Missing in replay:",
        missing_from_replay,
    )

    print(
        "Extra in replay:",
        extra_in_replay,
    )

    print(
        "Blocking integrity issues:",
        len(issues),
    )

    print(
        "Redundant-event warnings:",
        len(warnings),
    )

    # Diagnose the separate 2023 discrepancy.
    a = pd.Timestamp("2023-08-01")
    b = pd.Timestamp("2023-09-01")

    diagnostic_rows = []

    if a in snapshots and b in snapshots:
        predicted, redundant, involved = replay(
            snapshots[a],
            events,
            a,
            b,
        )

        expected_only = sorted(
            predicted - snapshots[b]
        )

        observed_only = sorted(
            snapshots[b] - predicted
        )

        print(
            "\n--- SEPTEMBER 2023 MISMATCH ---"
        )

        print(
            "Predicted but not observed:",
            expected_only,
        )

        print(
            "Observed but not predicted:",
            observed_only,
        )

        print("\nEvents in interval:")

        print(
            involved[
                ["Change Date", "RIC", "Change"]
            ].to_string(index=False)
        )

        for ric in expected_only:
            diagnostic_rows.append({
                "Issue": "Predicted not observed",
                "RIC": ric,
            })

        for ric in observed_only:
            diagnostic_rows.append({
                "Issue": "Observed not predicted",
                "RIC": ric,
            })

    # Stop if reconstruction cannot be validated.
    if not anchor_match or issues:
        print(
            "\nReconstruction NOT saved: "
            "anchor or integrity validation failed."
        )

        if issues:
            print(
                pd.DataFrame(issues)
                .head(15)
                .to_string(index=False)
            )

        return

    # Construct the provisional full-period panel.
    provenance = []

    for decision_date in all_dates:
        state = snapshots.get(
            decision_date
        )

        source = "OBSERVED_SNAPSHOT"

        if state is None:
            state = reconstructed[
                decision_date
            ]

            source = "PROVISIONAL_EVENT_REPLAY"

        for ric in sorted(state):
            provenance.append((
                decision_date,
                INDEX,
                ric,
                True,
                source,
            ))

    panel = pd.DataFrame(
        provenance,
        columns=[
            "Decision Date",
            "Index",
            "Underlying RIC",
            "Index Member",
            "Membership Source",
        ],
    )

    if (
        panel["Decision Date"].nunique()
        != len(all_dates)
    ):
        raise ValueError(
            "Incomplete reconstructed decision dates."
        )

    if panel.duplicated(
        [
            "Decision Date",
            "Index",
            "Underlying RIC",
        ]
    ).any():
        raise ValueError(
            "Duplicate reconstructed membership keys."
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    panel.to_parquet(
        OUTPUT / "sp500_membership_provisional.parquet",
        index=False,
    )

    monthly_counts = pd.DataFrame(
        [
            (date, len(reconstructed[date]))
            for date in gap
        ],
        columns=[
            "Decision Date",
            "Reconstructed Member Count",
        ],
    )

    monthly_counts.to_csv(
        OUTPUT / "reconstructed_monthly_counts.csv",
        index=False,
    )

    pd.DataFrame(
        diagnostic_rows,
        columns=["Issue", "RIC"],
    ).to_csv(
        OUTPUT / "2023_september_mismatch.csv",
        index=False,
    )

    same_day = events.loc[
        events["Change Date"].isin(gap),
        ["Change Date", "RIC", "Change"],
    ]

    same_day.to_csv(
        OUTPUT / "same_day_events_require_timing_review.csv",
        index=False,
    )

    pd.DataFrame(
        warnings,
        columns=["Type", "Date", "RIC", "Detail"],
    ).to_csv(
        OUTPUT / "event_replay_warnings.csv",
        index=False,
    )

    print("\n--- PROVISIONAL OUTPUT ---")

    print(
        "Total decision dates:",
        panel["Decision Date"].nunique(),
    )

    print(
        "Provisional reconstructed dates:",
        len(gap),
    )

    print(
        "Candidate member rows:",
        len(panel),
    )

    print(
        "Reconstructed count range:",
        min(map(len, reconstructed.values())),
        "to",
        max(map(len, reconstructed.values())),
    )

    print(
        "Events on missing decision dates:",
        len(same_day),
    )

    print("Saved under:", OUTPUT)

    print(
        "NOT approved for point-in-time ML training. "
        "Investigate the 2023 mismatch and "
        "event effective times first."
    )


if __name__ == "__main__":
    main()
