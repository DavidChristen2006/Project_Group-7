
import numpy as np
import pandas as pd

from pipeline.historical_data_pipeline import load_price_history
from pipeline.historical_option_pipeline import load_option_quote_history


def build_option_leg_outcomes(
    underlying_ric,
    option_ric,
    expiry,
    side,
    hedge_horizons,
    max_entry_spread_pct=0.10
):
    if side not in (-1, 1):
        raise ValueError("side must be +1 (long) or -1 (short).")

    horizons = sorted(set(int(h) for h in hedge_horizons))

    if not horizons or any(h <= 0 for h in horizons):
        raise ValueError("Hedge horizons must be positive.")

    stock = load_price_history(underlying_ric).copy()
    quotes = load_option_quote_history(option_ric).copy()

    stock.index = pd.to_datetime(stock.index).normalize()
    quotes.index = pd.to_datetime(quotes.index).normalize()

    stock = stock.sort_index()
    quotes = quotes.sort_index()

    if stock.index.has_duplicates or quotes.index.has_duplicates:
        raise ValueError("Duplicate trading dates detected.")

    stock["Close"] = pd.to_numeric(
        stock["Close"], errors="coerce"
    )

    for column in ["BID", "ASK"]:
        quotes[column] = pd.to_numeric(
            quotes[column], errors="coerce"
        )

    quotes = quotes[
        (quotes["BID"] >= 0)
        & (quotes["ASK"] > 0)
        & (quotes["ASK"] >= quotes["BID"])
    ].copy()

    quotes["Spread Pct"] = (
        (quotes["ASK"] - quotes["BID"])
        / ((quotes["ASK"] + quotes["BID"]) / 2)
    )

    expiry_date = pd.Timestamp(expiry).normalize()

    stock_dates = stock.index
    rows = []

    for entry_date, entry in quotes.iterrows():
        if entry_date >= expiry_date:
            continue

        if entry["Spread Pct"] > max_entry_spread_pct:
            continue

        if entry_date not in stock_dates:
            continue

        entry_position = stock_dates.get_loc(entry_date)

        for horizon in horizons:
            exit_position = entry_position + horizon

            if exit_position >= len(stock_dates):
                continue

            exit_date = stock_dates[exit_position]

            # For this first version, close before expiration.
            if exit_date >= expiry_date:
                continue

            # Require an actual quote on the exit date.
            if exit_date not in quotes.index:
                continue

            exit_quote = quotes.loc[exit_date]

            entry_price = (
                entry["ASK"] if side == 1
                else entry["BID"]
            )

            exit_price = (
                exit_quote["BID"] if side == 1
                else exit_quote["ASK"]
            )

            pnl_per_share = (
                exit_price - entry_price
                if side == 1
                else entry_price - exit_price
            )

            entry_stock = float(
                stock.loc[entry_date, "Close"]
            )

            exit_stock = float(
                stock.loc[exit_date, "Close"]
            )

            if not np.isfinite(
                [entry_price, exit_price, entry_stock, exit_stock]
            ).all():
                continue

            rows.append({
                "Underlying RIC": underlying_ric,
                "Option RIC": option_ric,
                "Decision Date": entry_date,
                "Outcome Date": exit_date,
                "Expiry": expiry_date,
                "Hedge Horizon Days": horizon,
                "Side": side,
                "Entry Price": float(entry_price),
                "Exit Price": float(exit_price),
                "Option P&L Per Share": float(pnl_per_share),
                "Stock Entry Price": entry_stock,
                "Stock Exit Price": exit_stock,
                "Stock Return": exit_stock / entry_stock - 1
            })

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            ["Decision Date", "Hedge Horizon Days"]
        ).reset_index(drop=True)

    return result
