"""Organise strategy metrics into summaries and a comparasion table 

Input:
Strategy names, calculated risk metrics, and optional hedge statistics 

Output:
Strategy summary dictionaries and a DataFrame for comparing strategies 
"""

import pandas as pd

def create_strategy_summary(strategy_name, risk_metrics, hedge_cost_pct=None, correlation=None, beta=None, r_squared=None):
    """Organise existing metrics into a summary for one strategy 
    
    Input:
    Strategy name, risk metric dictionary, and optional hedge cost, correlation, beta and R-squared values 
    
    Output:
    Dictionary containing the strategy name and supplied metrics with None for optional values that were not provided
    """
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
    """Combine strategy summaries into a comparison table
    
    Input:
    List of strategy summary dictionaries 

    Output:
    DataFrame with one row per strategy and one column per metric 
    """
    return pd.DataFrame(strategy_summaries)