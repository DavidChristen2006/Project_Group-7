
"""Reconcile S&P 500 monthly membership using historical events.

Creates a provisional membership candidate and audit files.
Original datasets are never modified.
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

MAPPING = (
    ROOT / "ever_member_universe/instrument_mapping"
    / "instrument_mapping.parquet"
)

OUTPUT = (
    ROOT / "historical_universe"
    / "reconciliation_candidate"
)

INDEX = ".SPX"

KVUE_RIC = "KVUE.N^K26"
KVUE_EFFECTIVE_DATE = pd.Timestamp("2023-08-25")


def parse_dates(frame, column):
    frame = frame.copy()

    frame[column] = pd.to_datetime(
        frame[column],
        errors="raise",
    ).dt.normalize()

    if frame[column].isna().any():
        raise ValueError(
            f"Missing {column} values."
        )

    return frame


def clean_ids(frame, column):
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
            f"Blank {column} values."
        )

    return frame


def main():
    hist = parse_dates(
        pd.read_parquet(HISTORY),
        "Decision Date",
    )

    hist = clean_ids(
        hist.loc[
            hist["Index"].eq(INDEX)
        ].copy(),
        "Underlying RIC",
    )

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
            "Invalid Index Member flags."
        )

    hist = hist.loc[
        flags.isin(["true", "1"])
    ].copy()

    if hist.duplicated(
        ["Decision Date", "Underlying RIC"]
    ).any():
        raise ValueError(
            "Duplicate historical membership keys."
        )

    grid = parse_dates(
        pd.read_parquet(GRID),
        "Decision Date",
    )

    grid = clean_ids(
        grid,
        "Underlying RIC",
    )

    if grid.duplicated(
        ["Decision Date", "Underlying RIC"]
    ).any():
        raise ValueError(
            "Duplicate collection-grid keys."
        )

    decision_dates = sorted(
        pd.Timestamp(value)
        for value in grid["Decision Date"].unique()
    )

    if not decision_dates:
        raise ValueError(
            "Empty research grid."
        )

    snapshots = {
        pd.Timestamp(day): set(
            part["Underlying RIC"].astype(str)
        )
        for day, part in hist.groupby(
            "Decision Date"
        )
    }

    anchor = decision_dates[0]

    if anchor not in snapshots:
        raise ValueError(
            "First research date has no anchor snapshot."
        )

    if not set(snapshots).issubset(
        set(decision_dates)
    ):
        raise ValueError(
            "Historical snapshots contain out-of-grid dates."
        )

    master = clean_ids(
        pd.read_parquet(MASTER),
        "Underlying RIC",
    )

    known_rics = set(
        master["Underlying RIC"].astype(str)
    )

    if KVUE_RIC not in known_rics:
        raise ValueError(
            "Kenvue historical RIC absent from equity master."
        )

    mapping = pd.read_parquet(MAPPING)

    kvue = mapping.loc[
        mapping["Underlying RIC"].eq(KVUE_RIC)
    ]

    if (
        len(kvue) != 1
        or "kenvue" not in str(
            kvue.iloc[0]["Common Name"]
        ).lower()
    ):
        raise ValueError(
            "Kenvue RIC does not match mapped company."
        )

    if (
        kvue.iloc[0]["Mapping Status"]
        != "IDENTIFIER_MAPPED"
    ):
        raise ValueError(
            "Kenvue identifier mapping not verified."
        )

    events = parse_dates(
        pd.read_parquet(EVENTS),
        "Change Date",
    )

    events = clean_ids(
        events,
        "RIC",
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
            "Unexpected membership change type."
        )

    events = (
        events.loc[
            events["Change Date"].gt(anchor)
            & events["Change Date"].le(
                decision_dates[-1]
            )
        ]
        .sort_values(
            "Change Date",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if events.duplicated(
        ["Change Date", "RIC"]
    ).any():
        raise ValueError(
            "Same-day duplicate events require review."
        )

    kvue_event = events.loc[
        events["RIC"].eq(KVUE_RIC)
        & events["Change Date"].eq(
            KVUE_EFFECTIVE_DATE
        )
        & events["Change"].eq("Joiner")
    ]

    if len(kvue_event) != 1:
        raise ValueError(
            "Cannot uniquely verify Kenvue's 2023 addition."
        )

    state = set(snapshots[anchor])

    replay_rows = []
    diagnostics = []
    mismatches = []
    same_day_events = []
    redundant_events = []

    event_pos = 0

    for decision_date in decision_dates:
        applied = []

        # Replay all membership changes up to this date.
        while (
            event_pos < len(events)
            and events.at[
                event_pos, "Change Date"
            ] <= decision_date
        ):
            event = events.iloc[event_pos]

            ric = str(event["RIC"])
            action = str(event["Change"])

            if action == "Joiner":
                if ric in state:
                    redundant_events.append(
                        (decision_date, ric, action)
                    )

                state.add(ric)

            else:
                if ric not in state:
                    redundant_events.append(
                        (decision_date, ric, action)
                    )

                state.discard(ric)

            applied.append(event)
            event_pos += 1

        unknown = state - known_rics

        if unknown:
            raise ValueError(
                f"Unknown RICs on {decision_date.date()}: "
                f"{sorted(unknown)[:10]}"
            )

        if not (450 <= len(state) <= 550):
            raise ValueError(
                f"Implausible membership count "
                f"on {decision_date.date()}: {len(state)}"
            )

        observed = snapshots.get(
            decision_date
        )

        if observed is not None:
            missing_from_snapshot = sorted(
                state - observed
            )

            extra_in_snapshot = sorted(
                observed - state
            )

        else:
            missing_from_snapshot = []
            extra_in_snapshot = []

        # Decide whether the replay agrees with
        # the existing historical snapshot.
        if observed is None:
            source = "PROVISIONAL_EVENT_REPLAY"

        elif (
            not missing_from_snapshot
            and not extra_in_snapshot
        ):
            source = "OBSERVED_SNAPSHOT_MATCH"

        elif (
            decision_date
            >= pd.Timestamp("2023-09-01")
            and missing_from_snapshot == [KVUE_RIC]
            and not extra_in_snapshot
        ):
            source = (
                "OFFICIAL_EVENT_CORRECTION_CANDIDATE"
            )

        else:
            source = (
                "UNEXPLAINED_SNAPSHOT_MISMATCH"
            )

        today_events = [
            event
            for event in applied
            if event["Change Date"] == decision_date
        ]

        for event in today_events:
            same_day_events.append({
                "Decision Date": decision_date,
                "RIC": str(event["RIC"]),
                "Change": str(event["Change"]),
                "Effective Time Verified": False,
            })

        diagnostics.append({
            "Decision Date": decision_date,
            "Status": source,
            "Observed Member Count": (
                len(observed)
                if observed is not None
                else pd.NA
            ),
            "Replayed Member Count": len(state),
            "Missing From Snapshot": ", ".join(
                missing_from_snapshot
            ),
            "Extra In Snapshot": ", ".join(
                extra_in_snapshot
            ),
            "Same-Day Events": len(today_events),
        })

        for ric in missing_from_snapshot:
            mismatches.append({
                "Decision Date": decision_date,
                "RIC": ric,
                "Issue": (
                    "In replay, absent from snapshot"
                ),
                "Source": source,
            })

        for ric in extra_in_snapshot:
            mismatches.append({
                "Decision Date": decision_date,
                "RIC": ric,
                "Issue": (
                    "In snapshot, absent from replay"
                ),
                "Source": source,
            })

        for ric in sorted(state):
            replay_rows.append((
                decision_date,
                INDEX,
                ric,
                True,
                source,
                bool(today_events),
            ))

    audit = pd.DataFrame(diagnostics)

    mismatches_df = pd.DataFrame(
        mismatches,
        columns=[
            "Decision Date",
            "RIC",
            "Issue",
            "Source",
        ],
    )

    same_day_df = pd.DataFrame(
        same_day_events,
        columns=[
            "Decision Date",
            "RIC",
            "Change",
            "Effective Time Verified",
        ],
    )

    print("\n--- FULL MEMBERSHIP REPLAY AUDIT ---")

    print(
        audit["Status"]
        .value_counts()
        .to_string()
    )

    print(
        "Total decision dates:",
        len(decision_dates),
    )

    unexplained = int(
        audit["Status"]
        .eq("UNEXPLAINED_SNAPSHOT_MISMATCH")
        .sum()
    )

    corrections = int(
        audit["Status"]
        .eq("OFFICIAL_EVENT_CORRECTION_CANDIDATE")
        .sum()
    )

    print(
        "Unexplained discrepancies:",
        unexplained,
    )

    print(
        "Corrected-snapshot candidate dates:",
        corrections,
    )

    print(
        "Same-day event records:",
        len(same_day_df),
    )

    print(
        "Redundant event operations:",
        len(redundant_events),
    )

    if not mismatches_df.empty:
        print("\n--- SNAPSHOT DIFFERENCES ---")

        print(
            mismatches_df.groupby(
                ["RIC", "Issue"]
            ).size().to_string()
        )

        print(
            mismatches_df
            .head(6)
            .to_string(index=False)
        )

        print(
            "Last mismatch date:",
            mismatches_df[
                "Decision Date"
            ].max().date(),
        )

    if not same_day_df.empty:
        print(
            "\n--- SAME-DAY EVENTS "
            "REQUIRING TIMING VALIDATION ---"
        )

        print(
            same_day_df.to_string(
                index=False
            )
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        OUTPUT / "full_replay_diagnostics.csv",
        index=False,
    )

    mismatches_df.to_csv(
        OUTPUT / "membership_discrepancies.csv",
        index=False,
    )

    same_day_df.to_csv(
        OUTPUT / "same_day_timing_review.csv",
        index=False,
    )

    if unexplained or redundant_events:
        print(
            "\nCandidate not written: "
            "unresolved mismatches or redundant events."
        )

        return

    # Save a new provisional candidate.
    panel = pd.DataFrame(
        replay_rows,
        columns=[
            "Decision Date",
            "Index",
            "Underlying RIC",
            "Index Member",
            "Membership Source",
            "Same-Day Event Timing Review",
        ],
    )

    keys = [
        "Decision Date",
        "Index",
        "Underlying RIC",
    ]

    if panel.duplicated(keys).any():
        raise ValueError(
            "Duplicate candidate membership keys."
        )

    if (
        panel["Decision Date"].nunique()
        != len(decision_dates)
    ):
        raise ValueError(
            "Incomplete candidate decision dates."
        )

    panel.to_parquet(
        OUTPUT / "sp500_membership_reconciled_candidate.parquet",
        index=False,
    )

    print("\n--- RECONCILED CANDIDATE ---")

    print(
        "Saved candidate rows:",
        len(panel),
    )

    print(
        "Decision dates:",
        panel["Decision Date"].nunique(),
    )

    print("Saved in:", OUTPUT)

    print(
        "STATUS: PROVISIONAL. "
        "Event effective times and source completeness "
        "still require independent validation."
    )

    print(
        "Retired-RIC suffixes must not be used "
        "as predictive ML features."
    )


if __name__ == "__main__":
    main()
