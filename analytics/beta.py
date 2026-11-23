"""Estimate beta and related statistic using linear regression

The model fits an ordinarly least-squared regression to aligned return series and calculated a beta-adjusted monetary hedge notional
"""

import pandas as pd
import statsmodels.api as sm

def estimate_beta(stock_returns, hedge_returns):
    """Estimates a stock's sensitivity to a hedge asset using OLS
    
    Fit Stock = alpha + beta * hedge + residual, including an intercept.
    Align the two series by index and remove rows missing either return. 

    Parameters:
    stock_returns : pandas.Series 
        Numeric stock returns indexed by observation date or period
    hedge_returns : pandas.Series 
        Numeric hedge-asset returns with the same frequency and return units as stock_returns

    Returns: 
    dict
        dictionary containing following values:
            - alpha: (intercept), in return units per observation period
            - beta: sensitivity of stock returns to hedge returns 
            - r_squared: proportion of stock-return variation explained by the fitted regression within the sample 
            - correlation: pearson correlation between the aligned returns 
            - residual_volatility: sample standard deviation of the regression residuals, at the input return frequency 
            - observations: number of nomissing aligned return observations 
    """
    data = pd.concat(
            [stock_returns.rename("Stock"),
            hedge_returns.rename("Hedge")],
        axis=1).dropna()
    if len(data) < 30:
        raise ValueError("At least 30 overlapping return observations are required to estimate beta.")
    x = sm.add_constant(data["Hedge"])
    model = sm.OLS(data["Stock"], x).fit()
    alpha = model.params["const"]
    beta = model.params["Hedge"]
    r_squared = model.rsquared
    correlation = data["Stock"].corr(data["Hedge"])
    residual_volatility = model.resid.std()
    return {
        "alpha": alpha,
        "beta": beta,
        "r_squared": r_squared,
        "correlation": correlation,
        "residual_volatility": residual_volatility,
        "observations": len(data)}

def calculate_hedge_notional(position_value, beta, hedge_ratio=1.0):
    """Calculate the beta-adjusted monetary exposure to hedge 

Parameters:
position_value : float 
    Monetary value of the stock position
beta : float
    Estimated sensitivity of the stock's returns to the hedge asset
hedge_ratio : float, optional 
    fraction of the beta-adjusted exposure to hedge. The default is 1.0 for a full hedge -> 0.5 represtes a half hedge

Returns: 
float 
    position_value * beta * hedge_ratio, expressed in the same currency as position_value. 
"""
    hedge_notional = position_value * beta * hedge_ratio
    return hedge_notional
