from hedging.protective_put import protective_put_scenarios
from hedging.collar import collar_scenarios
from hedging.put_spread import put_spread_scenarios
from hedging.comparison import compare_strategies
from analytics.strategy_metrics import create_strategy_summary, build_strategy_table

def analyze_strategies(
    position,
    current_price,
    drawdown_scenarios,

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

    unhedged_horizon_metrics,
    protective_put_risk_metrics,
    collar_risk_metrics,
    put_spread_risk_metrics,
    cross_hedge_horizon_metrics,
    cross_hedge_correlation,
    cross_hedge_beta,
    cross_hedge_r_squared):

    put_df = protective_put_scenarios(
        current_price=current_price,
        shares=position.shares,
        scenarios=drawdown_scenarios,
        put_strike=protective_put_strike,
        put_premium_per_share=(protective_put_premium_per_share))

    collar_df = collar_scenarios(
        current_price=current_price,
        shares=position.shares,
        scenarios=drawdown_scenarios,
        put_strike=collar_put_strike,
        put_premium_per_share=(collar_put_premium_per_share),
        call_strike=collar_call_strike,
        call_premium_per_share=(collar_call_premium_per_share))

    put_spread_df = put_spread_scenarios(
        current_price=current_price,
        shares=position.shares,
        scenarios=drawdown_scenarios,
        long_put_strike=spread_long_put_strike,
        long_put_premium_per_share=(spread_long_put_premium_per_share),
        short_put_strike=spread_short_put_strike,
        short_put_premium_per_share=(spread_short_put_premium_per_share))

    comparison_df = compare_strategies(
        put_df=put_df,
        collar_df=collar_df,
        put_spread_df=put_spread_df)

    protective_put_cost_pct = protective_put_premium_per_share / current_price
    collar_cost_pct = (collar_put_premium_per_share - collar_call_premium_per_share) / current_price
    put_spread_cost_pct = (spread_long_put_premium_per_share - spread_short_put_premium_per_share) / current_price

    strategy_summaries = []

    strategy_summaries.append(
        create_strategy_summary(
            strategy_name="Unhedged",
            risk_metrics=unhedged_horizon_metrics))

    strategy_summaries.append(
        create_strategy_summary(
            strategy_name="Cross Hedge",
            risk_metrics=cross_hedge_horizon_metrics,
            correlation=cross_hedge_correlation,
            beta=cross_hedge_beta,
            r_squared=cross_hedge_r_squared))

    strategy_summaries.append(
        create_strategy_summary(
            strategy_name="Protective Put",
            risk_metrics=protective_put_risk_metrics,
            hedge_cost_pct=protective_put_cost_pct))

    strategy_summaries.append(
        create_strategy_summary(
            strategy_name="Collar",
            risk_metrics=collar_risk_metrics,
            hedge_cost_pct=collar_cost_pct))

    strategy_summaries.append(
        create_strategy_summary(
            strategy_name="Put Spread",
            risk_metrics=put_spread_risk_metrics,
            hedge_cost_pct=put_spread_cost_pct))

    strategy_metrics_df = build_strategy_table(strategy_summaries)

    return {
        "put_df": put_df,
        "collar_df": collar_df,
        "put_spread_df": put_spread_df,
        "comparison_df": comparison_df,
        "strategy_metrics_df": strategy_metrics_df,
        "protective_put_cost_pct": protective_put_cost_pct,
        "collar_cost_pct": collar_cost_pct,
        "put_spread_cost_pct": put_spread_cost_pct}