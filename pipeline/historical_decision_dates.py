
import pandas as pd

from pipeline.historical_data_pipeline import (
    load_price_history
)


def generate_monthly_decision_dates(
    ric,
    start_date,
    end_date,
    hedge_horizons=(21, 63, 126)
):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    horizons = sorted(
        set(int(h) for h in hedge_horizons)
    )

    if start > end:
        raise ValueError(
            "Start date must not exceed end date."
        )

    if not horizons or any(h <= 0 for h in horizons):
        raise ValueError("Invalid hedge horizons.")

    history = load_price_history(ric).sort_index()

    history.index = pd.to_datetime(
        history.index
    ).normalize()

    if history.index.has_duplicates:
        raise ValueError(
            "Duplicate stock trading dates."
        )

    close = pd.to_numeric(
        history["Close"],
        errors="coerce"
    )

    if close.isna().any() or (close <= 0).any():
        raise ValueError("Invalid stock price history.")

    dates = history.index

    candidates = dates[
        (dates >= start) & (dates <= end)
    ]

    selected_dates = []

    for month, monthly_dates in (
        pd.Series(candidates, index=candidates)
        .groupby(candidates.to_period("M"))
    ):
        first_date = monthly_dates.iloc[0]

        position = dates.get_loc(first_date)

        # At least one requested horizon must be observable.
        if any(
            position + h < len(dates)
            for h in horizons
        ):
            selected_dates.append(first_date)

    return pd.DatetimeIndex(
        selected_dates,
        name="Decision Date"
    )
