import pandas as pd

def build_ml_rows(
    position,
    metadata,
    market,
    historical_risk,
    strategy_metrics_df,
    hedge_config=None,
    protective_put=None,
    collar_put=None,
    collar_call=None,
    spread_long_put=None,
    spread_short_put=None,
    cross_hedge_ric=None):
    rows = []

    for _, strategy_row in strategy_metrics_df.iterrows():
        strategy_name = strategy_row["Strategy"]

        put_strike_pct = None
        put_delta_target = None
        put_selection_mode = None

        call_strike_pct = None
        call_delta_target = None
        call_selection_mode = None

        short_put_strike_pct = None
        short_put_delta_target = None
        short_put_selection_mode = None

        cross_hedge_ratio = None

        actual_put_strike_pct = None
        actual_put_delta = None
        actual_put_iv = None

        actual_call_strike_pct = None
        actual_call_delta = None
        actual_call_iv = None

        actual_short_put_strike_pct = None
        actual_short_put_delta = None
        actual_short_put_iv = None

        if hedge_config is not None:

            if strategy_name == "Protective Put":
                put_strike_pct = hedge_config.get(
                    "put_strike_pct"
                )

                put_delta_target = hedge_config.get(
                    "put_delta"
                )

                put_selection_mode = hedge_config.get(
                    "put_selection_mode"
                )

            elif strategy_name == "Collar":
                put_strike_pct = hedge_config.get(
                    "put_strike_pct"
                )

                put_delta_target = hedge_config.get(
                    "put_delta"
                )

                put_selection_mode = hedge_config.get(
                    "put_selection_mode"
                )

                call_strike_pct = hedge_config.get(
                    "call_strike_pct"
                )

                call_delta_target = hedge_config.get(
                    "call_delta"
                )

                call_selection_mode = hedge_config.get(
                    "call_selection_mode"
                )

            elif strategy_name == "Put Spread":
                put_strike_pct = hedge_config.get(
                    "put_strike_pct"
                )

                put_delta_target = hedge_config.get(
                    "put_delta"
                )

                put_selection_mode = hedge_config.get(
                    "put_selection_mode"
                )

                short_put_strike_pct = hedge_config.get(
                    "short_put_strike_pct"
                )

                short_put_delta_target = hedge_config.get(
                    "short_put_delta"
                )

                short_put_selection_mode = hedge_config.get(
                    "short_put_selection_mode"
                )

            elif strategy_name == "Cross Hedge":
                cross_hedge_ratio = hedge_config.get("cross_hedge_ratio")

        if strategy_name == "Protective Put" and protective_put is not None:
            actual_put_strike_pct = protective_put["StrikePrice"] / market.current_price
            actual_put_delta = protective_put["Delta"] 
            actual_put_iv = protective_put["Implied Volatility"]

        elif strategy_name == "Collar":

            if collar_put is not None:
                actual_put_strike_pct = collar_put["StrikePrice"] / market.current_price
                actual_put_delta = collar_put["Delta"]
                actual_put_iv = collar_put["Implied Volatility"]

            if collar_call is not None:
                actual_call_strike_pct = collar_call["StrikePrice"] / market.current_price
                actual_call_delta = collar_call["Delta"]
                actual_call_iv = collar_call["Implied Volatility"]

        elif strategy_name == "Put Spread":

            if spread_long_put is not None:
                actual_put_strike_pct = spread_long_put["StrikePrice"] / market.current_price
                actual_put_delta = spread_long_put["Delta"]
                actual_put_iv = spread_long_put["Implied Volatility"]

            if spread_short_put is not None:
                actual_short_put_strike_pct = spread_short_put["StrikePrice"] / market.current_price
                actual_short_put_delta = spread_short_put["Delta"]
                actual_short_put_iv = spread_short_put["Implied Volatility"]

        rows.append({
            "RIC": position.ric,
            "Shares": position.shares,
            "Hedge Horizon Days":position.hedge_horizon_days,

            "Country": metadata.country,
            "Currency": metadata.currency,
            "Economic Sector": metadata.economic_sector,
            "Industry Group": metadata.industry_group,
            "Industry": metadata.industry,

            "Current Price": market.current_price,
            "Position Value": market.current_price * position.shares,
            "Dividend Yield": market.dividend_yield,
            "Risk-Free Rate": market.risk_free_rate,
            "Historical Max Drawdown": historical_risk["max_drawdown"],

            "Strategy": strategy_name,
            "Annualized Volatility": strategy_row["Annualized Volatility"],
            "VaR": strategy_row["VaR"],
            "CVaR": strategy_row["CVaR"],
            "Maximum Drawdown": strategy_row["Maximum Drawdown"],
            "Worst Return": strategy_row["Worst Return"],
            "Average Return": strategy_row["Average Return"],
            "Hedge Cost %": strategy_row["Hedge Cost %"],
            "Correlation": strategy_row["Correlation"],
            "Beta": strategy_row["Beta"],
            "R Squared": strategy_row["R Squared"],

            "Long Put Strike %":
                put_strike_pct,

            "Long Put Delta Target":
                put_delta_target,

            "Long Put Selection Mode":
                put_selection_mode,

            "Call Strike %":
                call_strike_pct,

            "Call Delta Target":
                call_delta_target,

            "Call Selection Mode":
                call_selection_mode,

            "Short Put Strike %":
                short_put_strike_pct,

            "Short Put Delta Target":
                short_put_delta_target,

            "Short Put Selection Mode":
                short_put_selection_mode,

            "Cross Hedge Ratio":
                cross_hedge_ratio,

            "Cross Hedge RICs":
                cross_hedge_ric,

            "Actual Put Strike %": actual_put_strike_pct,
            "Actual Put Delta": actual_put_delta,
            "Actual Put IV": actual_put_iv,
            "Actual Call Strike %": actual_call_strike_pct,
            "Actual Call Delta": actual_call_delta,
            "Actual Call IV": actual_call_iv,
            "Actual Short Put Strike %": actual_short_put_strike_pct,
            "Actual Short Put Delta": actual_short_put_delta,
            "Actual Short Put IV": actual_short_put_iv})
    return pd.DataFrame(rows)