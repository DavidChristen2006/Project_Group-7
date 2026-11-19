from analytics.strategy_backtest import calculate_rolling_returns, calculate_rolling_compounded_returns, protective_put_backtest, collar_backtest, put_spread_backtest
from analytics.risk_metrics import calculate_return_metrics

def analyze_strategy_backtests(
    position,
    history,
    current_price,

    protective_put_strike,
    protective_put_premium_per_share,

    collar_put_strike,
    collar_put_premium_per_share,
    collar_call_strike,
    collar_call_premium_per_share,

    spread_long_put_strike,
    spread_long_put_premium_per_share,
    spread_short_put_strike,
    spread_short_put_premium_per_share,

    cross_hedge_returns):

    historical_horizon_returns = (
        calculate_rolling_returns(
            history=history,
            horizon_days=position.hedge_horizon_days))

    protective_put_returns = protective_put_backtest(
        stock_returns=historical_horizon_returns,
        current_price=current_price,
        put_strike=protective_put_strike,
        put_premium_per_share=protective_put_premium_per_share)

    collar_returns = collar_backtest(
        stock_returns=historical_horizon_returns,
        current_price=current_price,
        put_strike=collar_put_strike,
        put_premium_per_share=collar_put_premium_per_share,
        call_strike=collar_call_strike,
        call_premium_per_share=collar_call_premium_per_share)

    put_spread_returns = put_spread_backtest(
        stock_returns=historical_horizon_returns,
        current_price=current_price,
        long_put_strike=spread_long_put_strike,
        long_put_premium_per_share=spread_long_put_premium_per_share,
        short_put_strike=spread_short_put_strike,
        short_put_premium_per_share=spread_short_put_premium_per_share)

    cross_hedge_horizon_returns = (
        calculate_rolling_compounded_returns(
            returns=cross_hedge_returns["Hedged Return"],
            horizon_days=position.hedge_horizon_days))

    periods_per_year = 252 / position.hedge_horizon_days

    unhedged_horizon_metrics = calculate_return_metrics(
        historical_horizon_returns,
        periods_per_year=periods_per_year,
        calculate_drawdown=False
    )

    protective_put_risk_metrics = calculate_return_metrics(
        protective_put_returns,
        periods_per_year=periods_per_year,
        calculate_drawdown=False
    )

    collar_risk_metrics = calculate_return_metrics(
        collar_returns,
        periods_per_year=periods_per_year,
        calculate_drawdown=False
    )

    put_spread_risk_metrics = calculate_return_metrics(
        put_spread_returns,
        periods_per_year=periods_per_year,
        calculate_drawdown=False
    )

    cross_hedge_horizon_metrics = calculate_return_metrics(
        cross_hedge_horizon_returns,
        periods_per_year=periods_per_year,
        calculate_drawdown=False
    )

    return {
        "historical_horizon_returns": historical_horizon_returns,
        "protective_put_returns": protective_put_returns,
        "collar_returns": collar_returns,
        "put_spread_returns": put_spread_returns,
        "cross_hedge_horizon_returns": cross_hedge_horizon_returns,
        "unhedged_horizon_metrics": unhedged_horizon_metrics,
        "protective_put_risk_metrics": protective_put_risk_metrics,
        "collar_risk_metrics": collar_risk_metrics,
        "put_spread_risk_metrics": put_spread_risk_metrics,
        "cross_hedge_horizon_metrics": cross_hedge_horizon_metrics}