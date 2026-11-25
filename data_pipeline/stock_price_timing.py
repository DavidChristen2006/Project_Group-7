
"""Stock price with same-snapshot trade/quote clock evidence.

Uses the existing delayed LSEG pricing helper.
Falls back to the existing stock-price reference if needed.

Clock fields contain UTC time of day, not verified event dates.
No model training, historical data changes or orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from math import isfinite

import pandas as pd


@dataclass(frozen=True)
class StockPriceTiming:
    """Represent StockPriceTiming within the hedge research workflow.
    """
    value: float
    source: str
    price_date: date | None
    is_close_fallback: bool
    retrieved_utc: str | None = None
    trade_clock_utc: str | None = None
    quote_clock_utc: str | None = None
    evidence_status: str = "UNAVAILABLE"


def _clock(value: object) -> str | None:
    """Validate time-of-day without assigning an event date."""

    try:
        if value is None or bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        return None

    if isinstance(value, time):
        return value.isoformat()

    text = str(value).strip()

    try:
        bits = text.split(":")

        if len(bits) != 3:
            return None

        hour = int(bits[0])
        minute = int(bits[1])
        second = float(bits[2])

        if (
            0 <= hour < 24
            and 0 <= minute < 60
            and 0 <= second < 60
        ):
            return text

    except (TypeError, ValueError):
        pass

    return None


def get_reference_price_with_timing(
    ric: str,
) -> StockPriceTiming:
    """Get stock price and clocks from the same pricing request."""

    from lseg_api.price_reference import (
        get_reference_price,
    )

    from lseg_api.pricing import (
        get_delayed_streaming_snapshot,
    )

    try:
        frame = get_delayed_streaming_snapshot(
            universe=[ric],
            fields=[
                "TRDPRC_1",
                "SALTIM",
                "QUOTIM",
            ],
        )

        retrieved = datetime.now(
            timezone.utc
        ).isoformat()

        if frame is None or frame.empty:
            raise ValueError(
                "Empty stock pricing snapshot"
            )

        required = {
            "Instrument",
            "TRDPRC_1",
        }

        if not required.issubset(frame.columns):
            raise ValueError(
                "Stock RIC or price field missing"
            )

        desired = (
            str(ric)
            .strip()
            .upper()
            .lstrip("/")
        )

        matching = frame.loc[
            frame["Instrument"]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.lstrip("/")
            .eq(desired)
        ]

        if len(matching) != 1:
            raise ValueError(
                "Stock snapshot does not uniquely "
                "match the requested RIC"
            )

        record = matching.iloc[0]

        price = float(
            record["TRDPRC_1"]
        )

        if not isfinite(price) or price <= 0:
            raise ValueError(
                "Invalid stock reference price"
            )

        trade_clock = _clock(
            record.get("SALTIM")
        )

        quote_clock = _clock(
            record.get("QUOTIM")
        )

        return StockPriceTiming(
            value=price,
            source="delayed_stream_with_clock",
            price_date=None,
            is_close_fallback=False,
            retrieved_utc=retrieved,
            trade_clock_utc=trade_clock,
            quote_clock_utc=quote_clock,
            evidence_status=(
                "UTC_CLOCKS_ONLY"
                if trade_clock or quote_clock
                else "NO_EVENT_CLOCKS"
            ),
        )

    except Exception:
        # Preserve the original price-retrieval behavior.
        # Do not attach timestamps from a different quote.
        original = get_reference_price(ric)

        return StockPriceTiming(
            value=original.value,
            source=original.source,
            price_date=original.price_date,
            is_close_fallback=original.is_close_fallback,
            evidence_status="UNAVAILABLE_FALLBACK_SOURCE",
        )


def stock_time_evidence(
    meta: dict,
) -> pd.DataFrame:
    """Produce a one-row stock timestamp evidence table."""

    received = pd.to_datetime(
        meta.get(
            "underlying_stock_retrieved_utc"
        ),
        utc=True,
        errors="coerce",
    )

    trade_clock = _clock(
        meta.get(
            "underlying_stock_trade_clock_utc"
        )
    )

    quote_clock = _clock(
        meta.get(
            "underlying_stock_quote_clock_utc"
        )
    )

    def clock_lag(
        clock: str | None,
    ) -> float | None:

        """Perform the clock lag operation for this research workflow.

        Args:
            clock: Input to the clock_lag operation.

        Returns:
            Computed result or status for the caller.
        """
        if clock is None or pd.isna(received):
            return None

        hour, minute, second = (
            clock.split(":")
        )

        event_seconds = (
            int(hour) * 3600
            + int(minute) * 60
            + float(second)
        )

        retrieval_seconds = (
            received.hour * 3600
            + received.minute * 60
            + received.second
            + received.microsecond / 1_000_000
        )

        # Clock-only comparison. The event date
        # cannot be established from SALTIM/QUOTIM.
        difference = (
            retrieval_seconds - event_seconds
        ) % 86400

        return round(
            difference / 60.0,
            2,
        )

    return pd.DataFrame([{
        "Stock RIC": meta.get(
            "ric",
            "Unknown",
        ),
        "Price reference": meta.get(
            "underlying_spot"
        ),
        "Trade clock UTC (no date)": (
            trade_clock or "Unknown"
        ),
        "Quote clock UTC (no date)": (
            quote_clock or "Unknown"
        ),
        "Retrieved UTC": (
            received.isoformat()
            if pd.notna(received)
            else "Unknown"
        ),
        "Trade clock lag (min)": (
            clock_lag(trade_clock)
        ),
        "Quote clock lag (min)": (
            clock_lag(quote_clock)
        ),
        "Evidence": meta.get(
            "underlying_stock_evidence_status",
            "UNAVAILABLE",
        ),
    }])
