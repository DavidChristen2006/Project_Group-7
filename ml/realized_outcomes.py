
import numpy as np
import pandas as pd

from pipeline.historical_data_pipeline import load_price_history


DEFAULT_HEDGE_HORIZONS = [21, 63, 126]


def build_unhedged_realized_outcomes(
    ric,
    hedge_horizons=None
):
    if hedge_horizons is None:
        hedge_horizons = DEFAULT_HEDGE_HORIZONS

    horizons = sorted(set(int(h) for h in hedge_horizons))

    if not horizons or any(h <= 0 for h in horizons):
        raise ValueError("Hedge horizons must be positive.")

    history = load_price_history(ric).copy()
    history = history.sort_index()

    history["Close"] = pd.to_numeric(
        history["Close"],
        errors="coerce"
    )

    if (
        history["Close"].isna().any()
        or (history["Close"] <= 0).any()
        or history.index.has_duplicates
    ):
        raise ValueError("Invalid historical price series.")

    dates = pd.to_datetime(history.index)
    prices = history["Close"].to_numpy(dtype=float)

    rows = []

    for horizon in horizons:
        for i in range(len(prices) - horizon):
            # Entry close plus the following H trading closes.
            path = prices[i:i + horizon + 1]

            normalized_values = path / path[0]

            running_peak = np.maximum.accumulate(
                normalized_values
            )

            drawdowns = (
                normalized_values / running_peak - 1.0
            )

            rows.append({
                "RIC": ric,
                "Decision Date": dates[i],
                "Outcome Date": dates[i + horizon],
                "Hedge Horizon Days": horizon,
                "Unhedged Final Return": float(
                    normalized_values[-1] - 1.0
                ),
                "Unhedged Maximum Drawdown": float(
                    drawdowns.min()
                )
            })

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError("Insufficient historical data.")

    result = result.sort_values(
        ["Decision Date", "Hedge Horizon Days"]
    ).reset_index(drop=True)

    return result
