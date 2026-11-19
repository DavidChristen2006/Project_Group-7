
import numpy as np
import pandas as pd

from pipeline.historical_data_pipeline import load_price_history
from pipeline.historical_option_pipeline import load_option_quote_history
from pipeline.historical_corporate_actions import (
    normalize_historical_stock_prices,
    get_aapl_historical_split_factor,
)


LEG_SIDES = {
    "Long Put": 1,
    "Short Put": -1,
    "Short Call": -1
}


def build_historical_strategy_path(
    selection,
    strategy_name,
    shares=1000,
    option_multiplier=100,
    stock_window_override=None,
):
    legs = selection["Strategies"].get(strategy_name)

    if legs is None:
        raise ValueError(f"{strategy_name} is unavailable.")

    if shares <= 0 or option_multiplier <= 0:
        raise ValueError("Invalid position size.")

    contracts = shares / option_multiplier

    if not float(contracts).is_integer():
        raise ValueError("Whole option contracts required.")

    start = pd.Timestamp(
        selection["Decision Date"]
    ).normalize()

    end = pd.Timestamp(
        selection["Exit Date"]
    ).normalize()

    underlying_ric = selection["Underlying RIC"]

    if stock_window_override is None:
        # Legacy path is unchanged for reproducibility.
        stock = load_price_history(underlying_ric).copy()
        stock.index = pd.to_datetime(stock.index).normalize()
        stock = stock.sort_index()
        if stock.index.has_duplicates:
            raise ValueError("Duplicate stock dates.")
        path = stock.loc[start:end, ["Close"]].copy()
        if underlying_ric == "AAPL.O":
            if (
                get_aapl_historical_split_factor(start)
                != get_aapl_historical_split_factor(end)
            ):
                raise ValueError(
                    "Stock split inside holding period: "
                    "split-aware position accounting required."
                )
            path["Close"] = normalize_historical_stock_prices(
                prices=path["Close"],
                underlying_ric=underlying_ric,
            )
    else:
        # Override must have been verified with load_v2_stock_window().
        # No additional AAPL-specific normalization may be applied.
        if not isinstance(stock_window_override, pd.Series):
            raise TypeError("stock_window_override must be a validated Series")
        if stock_window_override.index.has_duplicates:
            raise ValueError("Duplicate override stock dates")
        if not isinstance(stock_window_override.index, pd.DatetimeIndex):
            raise TypeError("Override requires a DatetimeIndex")
        if stock_window_override.index.tz is not None:
            raise ValueError("Override stock dates must be timezone naive")
        if not stock_window_override.index.equals(
            stock_window_override.index.normalize()
        ):
            raise ValueError("Override stock dates must be normalized")
        path = stock_window_override.loc[start:end].to_frame(name="Close")

    if path.empty or path.index[0] != start or path.index[-1] != end:
        raise ValueError("Incomplete stock holding period.")

    if len(path) != selection["Hedge Horizon Days"] + 1:
        raise ValueError("Unexpected trading-day count.")

    path["Close"] = pd.to_numeric(
        path["Close"],
        errors="coerce"
    ).astype(float)

    if (
        not np.isfinite(path["Close"]).all()
        or (path["Close"] <= 0).any()
    ):
        raise ValueError("Invalid stock prices.")

    initial_stock_price = float(path["Close"].iloc[0])
    initial_value = shares * initial_stock_price

    path["Stock P&L"] = shares * (
        path["Close"] - initial_stock_price
    )

    path["Option P&L"] = 0.0

    for leg_name, contract in legs.items():
        side = LEG_SIDES[leg_name]

        expiry = pd.Timestamp(
            contract["Expiry"]
        ).normalize()

        if expiry <= end:
            raise ValueError(
                f"{leg_name} expires before the hedge closes."
            )

        quotes = load_option_quote_history(
            contract["RIC"]
        ).copy()

        quotes.index = pd.to_datetime(
            quotes.index
        ).normalize()

        if quotes.index.has_duplicates:
            raise ValueError(
                f"Duplicate quote dates: {contract['RIC']}"
            )

        quotes = quotes.reindex(path.index)

        quotes["BID"] = pd.to_numeric(
            quotes["BID"],
            errors="coerce"
        )

        quotes["ASK"] = pd.to_numeric(
            quotes["ASK"],
            errors="coerce"
        )
        
        # Validate the executable quotation for each position.
        #
        # Long option:
        #   Enter at ASK, liquidate at BID.
        #
        # Short option:
        #   Enter at BID, liquidate at ASK.
        #
        # A missing non-executable side does not invalidate
        # a historical liquidation observation.

        bid = quotes["BID"]
        ask = quotes["ASK"]

        if side == 1:
            # Selling a long option requires a valid BID.
            valid = (
                bid.notna()
                & np.isfinite(bid)
                & (bid >= 0)
            )

            # Opening the long option requires a valid ASK.
            entry_valid = (
                pd.notna(ask.iloc[0])
                and np.isfinite(ask.iloc[0])
                and ask.iloc[0] > 0
            )

        elif side == -1:
            # Buying back a short option requires a valid ASK.
            valid = (
                ask.notna()
                & np.isfinite(ask)
                & (ask > 0)
            )

            # Opening the short option requires a valid BID.
            entry_valid = (
                pd.notna(bid.iloc[0])
                and np.isfinite(bid.iloc[0])
                and bid.iloc[0] > 0
            )

        else:
            raise ValueError(f"Invalid option side: {side}")

        # If both sides exist, the market must not be crossed.
        both_available = bid.notna() & ask.notna()

        valid &= (
            ~both_available
            | (
                np.isfinite(bid)
                & np.isfinite(ask)
                & (bid >= 0)
                & (ask > 0)
                & (ask >= bid)
            )
        )

        # Both entry execution and immediate liquidation
        # must be valid on the decision date.
        valid.iloc[0] = bool(valid.iloc[0] and entry_valid)

        if not valid.all():
            missing_dates = path.index[~valid].strftime(
                "%Y-%m-%d"
            ).tolist()

            raise ValueError(
                f"Incomplete {leg_name} executable quotes: "
                f"{missing_dates}"
            )

        entry_price = float(
            quotes["ASK"].iloc[0]
            if side == 1
            else quotes["BID"].iloc[0]
        )

        # Liquidation value at each subsequent date.
        liquidation_prices = (
            quotes["BID"]
            if side == 1
            else quotes["ASK"]
        )

        leg_pnl = (
            side
            * contracts
            * option_multiplier
            * (liquidation_prices - entry_price)
        )

        path[f"{leg_name} P&L"] = leg_pnl
        path["Option P&L"] += leg_pnl

    path["Portfolio P&L"] = (
        path["Stock P&L"] + path["Option P&L"]
    )

    path["Portfolio Value"] = (
        initial_value + path["Portfolio P&L"]
    )

    if (path["Portfolio Value"] <= 0).any():
        raise ValueError("Nonpositive portfolio value.")

    path["Portfolio Return"] = (
        path["Portfolio P&L"] / initial_value
    )

    # Include initial wealth immediately before execution.
    # This captures the initial bid/ask execution cost.
    values = np.concatenate((
        [initial_value],
        path["Portfolio Value"].to_numpy(dtype=float)
    ))

    peaks = np.maximum.accumulate(values)

    drawdowns = values / peaks - 1.0

    path["Drawdown"] = drawdowns[1:]

    summary = {
        "Strategy": strategy_name,
        "Decision Date": start,
        "Outcome Date": end,
        "Hedge Horizon Days": selection["Hedge Horizon Days"],
        "Final Return": float(
            path["Portfolio Return"].iloc[-1]
        ),
        "Maximum Drawdown": float(
            drawdowns.min()
        ),
        "Final P&L": float(
            path["Portfolio P&L"].iloc[-1]
        ),
        "Observations": len(path)
    }

    path.index.name = "Date"

    return path, summary
