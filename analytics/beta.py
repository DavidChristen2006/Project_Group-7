import pandas as pd
import statsmodels.api as sm

def estimate_beta(stock_returns, hedge_returns):
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
    hedge_notional = position_value * beta * hedge_ratio
    return hedge_notional