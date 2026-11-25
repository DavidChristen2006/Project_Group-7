
"""Use a closing-price reference if the current price is unavailable.

The caller must have an open LSEG session.

TR.PriceClose is a research reference, not a live quote.
No prices are invented or estimated.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isfinite

import pandas as pd


@dataclass(frozen=True)
class PriceReference:
    """Represent PriceReference within the hedge research workflow.
    """
    value: float
    source: str
    price_date: date | None
    is_close_fallback: bool


def get_reference_price(ric: str) -> PriceReference:
    """Return a valid current-wrapper price or closing reference."""

    import lseg.data as ld

    from lseg_api.equity import (
        get_current_equity_price,
    )

    raw = get_current_equity_price(ric)

    # First retain the existing working price retrieval method.
    if raw is not None:

        try:
            value = float(raw)

        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid LSEG price for {ric}: {raw!r}"
            ) from exc

        if not isfinite(value) or value <= 0:
            raise ValueError(
                f"Invalid LSEG price for {ric}: {value!r}"
            )

        return PriceReference(
            value=value,
            source="existing_price_wrapper",
            price_date=None,
            is_close_fallback=False,
        )

    # If the wrapper returns None, use an observed LSEG
    # closing price rather than ending the whole analysis.
    frame = ld.get_data(
        universe=[ric],
        fields=["TR.PriceClose"],
    )

    if frame is None or frame.empty:
        raise ValueError(
            f"No current price or TR.PriceClose "
            f"available for {ric}"
        )

    # Ensure the response matches the requested instrument.
    if "Instrument" in frame.columns:

        frame = frame.loc[
            frame["Instrument"]
            .astype(str)
            .str.strip()
            .str.upper()
            == ric.strip().upper()
        ]

        if frame.empty:
            raise ValueError(
                f"TR.PriceClose response did not match {ric}"
            )

    field = next(
        (
            column
            for column in frame.columns
            if str(column).strip().lower()
            in {
                "price close",
                "tr.priceclose",
            }
        ),
        None,
    )

    if field is None:
        raise ValueError(
            f"TR.PriceClose field missing for {ric}; "
            f"columns={list(frame.columns)}"
        )

    values = pd.to_numeric(
        frame[field],
        errors="coerce",
    ).dropna()

    if len(values) != 1:
        raise ValueError(
            f"Expected exactly one numeric "
            f"TR.PriceClose for {ric}"
        )

    value = float(values.iloc[0])

    if not isfinite(value) or value <= 0:
        raise ValueError(
            f"Invalid TR.PriceClose for {ric}: {value!r}"
        )

    # Retrieve the observation date if available.
    price_date = None

    try:
        date_frame = ld.get_data(
            universe=[ric],
            fields=["TR.PriceClose.date"],
        )

        if date_frame is not None and not date_frame.empty:

            if "Instrument" in date_frame.columns:

                date_frame = date_frame.loc[
                    date_frame["Instrument"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    == ric.strip().upper()
                ]

            if len(date_frame) == 1:

                date_column = next(
                    (
                        col
                        for col in date_frame.columns
                        if col != "Instrument"
                        and "date" in str(col).lower()
                    ),
                    None,
                )

                if date_column is not None:

                    parsed = pd.to_datetime(
                        date_frame.iloc[0][date_column],
                        errors="coerce",
                    )

                    if pd.notna(parsed):
                        price_date = parsed.date()

    except Exception:
        # The date is optional. If it cannot be confirmed,
        # report it as unknown, never as today's date.
        pass

    return PriceReference(
        value=value,
        source="TR.PriceClose",
        price_date=price_date,
        is_close_fallback=True,
    )
