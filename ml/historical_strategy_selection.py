
import pandas as pd

from pipeline.historical_data_pipeline import load_price_history


def select_historical_strategy(
    underlying_ric,
    discovered_df,
    decision_date,
    hedge_horizon_days,
    put_strike_pct=0.90,
    call_strike_pct=1.10,
    short_put_strike_pct=0.80,
    max_strike_deviation_pct=0.03
):
    decision_date = pd.Timestamp(decision_date).normalize()

    if int(hedge_horizon_days) != hedge_horizon_days or hedge_horizon_days <= 0:
        raise ValueError("Invalid hedge horizon.")

    if not 0 < short_put_strike_pct < put_strike_pct < 1 < call_strike_pct:
        raise ValueError("Invalid strike target configuration.")

    if not 0 < max_strike_deviation_pct < 1:
        raise ValueError("Invalid strike tolerance.")

    stock = load_price_history(underlying_ric).sort_index()

    if decision_date not in stock.index:
        raise ValueError("Decision date not in stock history.")

    position = stock.index.get_loc(decision_date)
    exit_position = position + int(hedge_horizon_days)

    if exit_position >= len(stock):
        raise ValueError("Insufficient future stock history.")

    exit_date = stock.index[exit_position]
    spot = float(stock.loc[decision_date, "Close"])

    options = discovered_df.copy()

    if options.empty:
        raise ValueError("No historical options available.")

    options["Decision Date"] = pd.to_datetime(
        options["Decision Date"]
    ).dt.normalize()

    options["Expiry"] = pd.to_datetime(
        options["Expiry"]
    ).dt.normalize()

    options = options[
        (options["Decision Date"] == decision_date)
        & (options["Expiry"] > exit_date)
        & (options["Liquid"] == True)
    ].copy()

    def choose(option_type, target_pct, min_strike=None, max_strike=None):
        eligible = options[
            options["Option Type"] == option_type
        ].copy()

        if min_strike is not None:
            eligible = eligible[
                eligible["Strike"] > min_strike
            ]

        if max_strike is not None:
            eligible = eligible[
                eligible["Strike"] < max_strike
            ]

        target_strike = spot * target_pct

        eligible["Strike Deviation"] = (
            (eligible["Strike"] - target_strike).abs() / spot
        )

        eligible = eligible[
            eligible["Strike Deviation"] <= max_strike_deviation_pct
        ]

        if eligible.empty:
            return None

        eligible = eligible.sort_values(
            ["Strike Deviation", "SPREAD_PCT", "RIC"]
        )

        return eligible.iloc[0].to_dict()

    long_put = choose(
        "put",
        put_strike_pct,
        max_strike=spot
    )

    short_call = choose(
        "call",
        call_strike_pct,
        min_strike=spot
    )

    short_put = choose(
        "put",
        short_put_strike_pct,
        max_strike=(
            long_put["Strike"] if long_put is not None else spot
        )
    )

    strategies = {
        "Protective Put": (
            {"Long Put": long_put}
            if long_put is not None else None
        ),
        "Collar": (
            {"Long Put": long_put, "Short Call": short_call}
            if long_put is not None and short_call is not None
            else None
        ),
        "Put Spread": (
            {"Long Put": long_put, "Short Put": short_put}
            if long_put is not None and short_put is not None
            else None
        )
    }

    return {
        "Underlying RIC": underlying_ric,
        "Decision Date": decision_date,
        "Exit Date": exit_date,
        "Hedge Horizon Days": int(hedge_horizon_days),
        "Stock Price": spot,
        "Strategies": strategies
    }
