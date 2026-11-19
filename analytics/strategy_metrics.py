import pandas as pd

def create_strategy_summary(strategy_name, risk_metrics, hedge_cost_pct=None, correlation=None, beta=None, r_squared=None):
    return {
        "Strategy": strategy_name,
        "Annualized Volatility":
            risk_metrics["annualized_volatility"],
        "VaR": risk_metrics["var"],
        "CVaR": risk_metrics["cvar"],
        "Maximum Drawdown": risk_metrics["max_drawdown"],
        "Worst Return": risk_metrics["worst_return"],
        "Average Return": risk_metrics["average_return"],
        "Hedge Cost %": hedge_cost_pct,
        "Correlation": correlation,
        "Beta": beta,
        "R Squared": r_squared}

def build_strategy_table(strategy_summaries):
    return pd.DataFrame(strategy_summaries)