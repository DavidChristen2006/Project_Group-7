
"""Audit reconstructed S&P 500 historical membership.

Inputs:
    sp500_historical_universe.parquet
    sp500_membership_events.parquet
    sp500_membership_events_raw.parquet

No LSEG session is needed.
No source files are modified.
"""

from pathlib import Path

import pandas as pd


ROOT = Path("data/processed/historical_universe")

UNIVERSE_FILE = (
    ROOT / "sp500_historical_universe.parquet"
)

EVENTS_FILE = (
    ROOT / "sp500_membership_events.parquet"
)

RAW_EVENTS_FILE = (
    ROOT / "sp500_membership_events_raw.parquet"
)


def base_ric(ric):
    """Remove the historical suffix for comparison only.

    This does NOT establish that two RICs identify
    the same security.
    """
    return str(ric).split("^", 1)[0]


def load_data():
    universe = pd.read_parquet(UNIVERSE_FILE)
    events = pd.read_parquet(EVENTS_FILE)
    raw_events = pd.read_parquet(RAW_EVENTS_FILE)

    universe["Decision Date"] = pd.to_datetime(
        universe["Decision Date"]
    ).dt.normalize()

    events["Change Date"] = pd.to_datetime(
        events["Change Date"]
    ).dt.normalize()

    raw_events["Date"] = pd.to_datetime(
        raw_events["Date"]
    ).dt.normalize()

    return universe, events, raw_events


def audit_identifier_collisions(universe):
    print("\n--- HISTORICAL RIC COLLISIONS ---")

    frame = universe[
        ["Decision Date", "Underlying RIC"]
    ].copy()

    frame["Base RIC"] = (
        frame["Underlying RIC"].map(base_ric)
    )

    counts = (
        frame.groupby(
            ["Decision Date", "Base RIC"]
        )["Underlying RIC"]
        .nunique()
    )

    collisions = counts[
        counts > 1
    ].reset_index(
        name="Historical RIC Count"
    )

    if collisions.empty:
        print("No base-RIC collisions detected.")
        return collisions

    details = frame.merge(
        collisions[
            ["Decision Date", "Base RIC"]
        ],
        on=["Decision Date", "Base RIC"],
        how="inner",
        validate="many_to_one",
    )

    print(
        "Ticker-date collision groups:",
        len(collisions),
    )

    print(
        "Unique affected base RICs:",
        collisions["Base RIC"].nunique(),
    )

    print("\nMost frequently affected base RICs:")

    print(
        collisions["Base RIC"]
        .value_counts()
        .head(20)
        .to_string()
    )

    print("\nExample collisions:")

    print(
        details.head(40).to_string(
            index=False
        )
    )

    print("\nAGN history, if present:")

    agn = details.loc[
        details["Base RIC"].eq("AGN.N")
    ]

    if agn.empty:
        print("No AGN collision found.")
    else:
        print(
            agn.head(30).to_string(
                index=False
            )
        )

    return collisions


def audit_reverse_events(universe, events):
    print("\n--- REVERSE EVENT CONSISTENCY ---")

    # Begin at the latest reconstructed date.
    # Moving backward reverses later events.
    dates = sorted(
        universe["Decision Date"].unique(),
        reverse=True,
    )

    latest_date = pd.Timestamp(dates[0])

    membership = set(
        universe.loc[
            universe["Decision Date"].eq(latest_date),
            "Underlying RIC",
        ]
    )

    event_groups = {
        date: group
        for date, group in events.groupby(
            "Change Date"
        )
    }

    event_dates = sorted(
        [
            event_date
            for event_date in event_groups
            if event_date <= latest_date
        ],
        reverse=True,
    )

    cursor = 0
    anomalies = []

    for decision_date in dates[1:]:
        decision_date = pd.Timestamp(
            decision_date
        )

        while (
            cursor < len(event_dates)
            and event_dates[cursor] > decision_date
            and event_dates[cursor] <= latest_date
        ):
            event_date = event_dates[cursor]
            group = event_groups[event_date]

            # Reverse joins before reversing removals.
            # Same-date transitions may still require
            # additional event sequencing information.
            for change in ("Joiner", "Leaver"):
                subset = group.loc[
                    group["Change"].eq(change)
                ]

                for _, row in subset.iterrows():
                    ric = str(row["RIC"])
                    present = ric in membership

                    if change == "Joiner":
                        if not present:
                            anomalies.append({
                                "Date": event_date,
                                "RIC": ric,
                                "Change": change,
                                "Issue": (
                                    "Joiner absent before reversal"
                                ),
                            })

                        membership.discard(ric)

                    else:
                        if present:
                            anomalies.append({
                                "Date": event_date,
                                "RIC": ric,
                                "Change": change,
                                "Issue": (
                                    "Leaver present before reversal"
                                ),
                            })

                        membership.add(ric)

            cursor += 1

        expected = set(
            universe.loc[
                universe["Decision Date"].eq(
                    decision_date
                ),
                "Underlying RIC",
            ]
        )

        if membership != expected:
            raise ValueError(
                "Reconstructed membership does not "
                "match the saved historical universe "
                f"on {decision_date.date()}."
            )

    report = pd.DataFrame(
        anomalies,
        columns=[
            "Date",
            "RIC",
            "Change",
            "Issue",
        ],
    )

    print(
        "Reverse-event state anomalies:",
        len(report),
    )

    if not report.empty:
        print(
            report.head(40).to_string(
                index=False
            )
        )

        print("\nAnomalies by type:")

        print(
            report["Issue"]
            .value_counts()
            .to_string()
        )

    return report


def audit_same_day_events(raw_events):
    print("\n--- SAME-DAY JOINER/LEAVER EVENTS ---")

    frame = raw_events.rename(
        columns={
            "Date": "Change Date",
            "Constituent RIC": "RIC",
        }
    )

    counts = (
        frame.groupby(
            ["Change Date", "RIC"]
        )["Change"]
        .nunique()
    )

    conflicts = counts[
        counts > 1
    ]

    print(
        "Same-day conflicting RICs:",
        len(conflicts),
    )

    if not conflicts.empty:
        details = frame.merge(
            conflicts.reset_index()[
                ["Change Date", "RIC"]
            ],
            on=["Change Date", "RIC"],
            how="inner",
        )

        print(
            details[
                ["Change Date", "RIC", "Change"]
            ].to_string(index=False)
        )


def main():
    universe, events, raw_events = load_data()

    print("\n--- UNIVERSE AUDIT ---")

    print(
        "Ticker-date records:",
        len(universe),
    )

    print(
        "Historical RICs:",
        universe["Underlying RIC"].nunique(),
    )

    print(
        "Decision dates:",
        universe["Decision Date"].nunique(),
    )

    collisions = audit_identifier_collisions(
        universe
    )

    anomalies = audit_reverse_events(
        universe,
        events,
    )

    audit_same_day_events(
        raw_events
    )

    print("\n--- AUDIT SUMMARY ---")

    print(
        "Base-RIC collision groups:",
        len(collisions),
    )

    print(
        "Reverse-event anomalies:",
        len(anomalies),
    )

    if len(collisions) or len(anomalies):
        print(
            "Manual identifier/event validation "
            "is required before full-scale training."
        )
    else:
        print(
            "Internal audit checks passed. "
            "Independent historical verification "
            "is still required."
        )

    print("\nNo files modified.")


if __name__ == "__main__":
    main()
