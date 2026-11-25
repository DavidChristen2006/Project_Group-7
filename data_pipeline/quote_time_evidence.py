
"""Explain UTC quote timing from retained LSEG snapshots.

No LSEG calls, trading orders, or model inference.

QUOTIM contains only a UTC time of day. The true
event date and feed delay are not independently verified.
"""

from __future__ import annotations

from datetime import time

import pandas as pd


def _utc_clock_seconds(value: object) -> float | None:

    """Handle the internal utc clock seconds step.

    Args:
        value: Input to the _utc_clock_seconds operation.

    Returns:
        Computed result or status for the caller.
    """
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None

    if isinstance(value, time):
        return (
            value.hour * 3600
            + value.minute * 60
            + value.second
            + value.microsecond / 1_000_000
        )

    components = str(value).strip().split(":")

    if len(components) != 3:
        return None

    try:
        hours = int(components[0])
        minutes = int(components[1])
        seconds = float(components[2])

    except (ValueError, TypeError):
        return None

    if not (
        0 <= hours < 24
        and 0 <= minutes < 60
        and 0 <= seconds < 60
    ):
        return None

    return (
        hours * 3600
        + minutes * 60
        + seconds
    )


def selected_option_quote_evidence(
    liquid_options: pd.DataFrame,
    option_rics: list[str],
) -> pd.DataFrame:
    """Return timing evidence for selected option RICs.

    A UTC clock lag is not proof of the actual quote age.
    QUOTIM and QuoteRetrievedUTC must come from the
    same pricing request.
    """

    required = {"RIC", "BID", "ASK"}

    missing = required - set(
        liquid_options.columns
    )

    if missing:
        raise ValueError(
            f"Missing quote fields: {sorted(missing)}"
        )

    quotes = liquid_options.copy()

    quotes["RIC"] = (
        quotes["RIC"].astype(str)
    )

    if quotes["RIC"].duplicated().any():
        raise ValueError(
            "Duplicate option RICs in quote snapshot"
        )

    lookup = quotes.set_index("RIC")

    rows = []

    for ric in dict.fromkeys(option_rics):

        if ric not in lookup.index:
            raise ValueError(
                f"Selected option missing: {ric}"
            )

        quote = lookup.loc[ric]

        quote_time = quote.get(
            "QUOTIM",
            pd.NA,
        )

        retrieved_raw = quote.get(
            "QuoteRetrievedUTC",
            pd.NA,
        )

        received = pd.to_datetime(
            retrieved_raw,
            utc=True,
            errors="coerce",
        )

        clock_sec = _utc_clock_seconds(
            quote_time
        )

        lag_minutes = None
        status = "TIME_UNAVAILABLE"

        if (
            clock_sec is not None
            and pd.notna(received)
        ):

            received_sec = (
                received.hour * 3600
                + received.minute * 60
                + received.second
                + received.microsecond / 1_000_000
            )

            # Handles clock wrap at midnight,
            # but does not establish the event date.
            difference = (
                received_sec - clock_sec
            ) % 86400

            lag_minutes = round(
                difference / 60.0,
                2,
            )

            status = "UTC_CLOCK_LAG_ONLY"

        elif clock_sec is not None:
            status = "NO_RETRIEVAL_TIMESTAMP"

        elif pd.notna(received):
            status = "NO_QUOTE_TIME"

        rows.append({
            "Option RIC": ric,
            "BID": float(quote["BID"]),
            "ASK": float(quote["ASK"]),
            "Quote time UTC (no date)": (
                str(quote_time)
                if clock_sec is not None
                else "Unknown"
            ),
            "Quote retrieved UTC": (
                received.isoformat()
                if pd.notna(received)
                else "Unknown"
            ),
            "Observed UTC clock lag (min)": (
                lag_minutes
            ),
            "Evidence": status,
        })

    return pd.DataFrame(rows)
