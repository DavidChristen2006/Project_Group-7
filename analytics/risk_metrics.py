"""Calculates historical risk and return metrics 

Input:
Return series and settings for confidence level and annualisation

Output:
VaR, CVaR, maximum drawdown, volatiltiy, and return summaries
"""

import numpy as np

def calculate_var(
    returns,
    confidence_level=0.95
):
    """Calculate the historical Value at Risk (VaR) return threshold 
    
    Input:
    Return Series and confidence level (default: 0.95)

    Output:
    Lower-tail return percentile (at 95% confidence)
    """
    clean_returns = returns.dropna()

    percentile = (
        1 - confidence_level
    ) * 100

    return np.percentile(
        clean_returns,
        percentile
    )

def calculate_cvar(
    returns,
    confidence_level=0.95
):
    """Calculate the average return in the lower tail/part 
    
    Input:
    Return Series and confidence level (default: 0.95)

    Output:
    Mean return at or below the VaR threshold, or VaR if the tail/lower part is empty 
    """
    clean_returns = returns.dropna()

    var = calculate_var(
        clean_returns,
        confidence_level=confidence_level
    )

    tail_returns = clean_returns[
        clean_returns <= var
    ]

    if tail_returns.empty:
        return var

    return tail_returns.mean()


def calculate_max_drawdown(
    returns
):
    """Calculate the largest decline from previous cumulative wealth peak
    
    Input:
    Chronologically ordered return Series, expressed as decimals

    Output:
    Maximum drawdown as a nonpositive decimal 
    """
    clean_returns = returns.dropna()

    wealth_index = (
        1 + clean_returns
    ).cumprod()

    running_max = (
        wealth_index.cummax()
    )

    drawdown = (
        wealth_index
        / running_max
        - 1
    )

    return drawdown.min()


def calculate_return_metrics(
    returns,
    confidence_level=0.95,
    periods_per_year=252,
    calculate_drawdown=True
):
    """Combine historical risk and return measures. 
    
    Input:
    Return Series, confidence level, periods per year, and a flag controlling whether maximum drawdown is calculated 

    Output: 
    Dictionary containing annualised volatility, VaR, CVaR, maximum drawdonw, worst return, and average return. Maximum drawdown is NaN when its calculation is disabled 
    """
    clean_returns = returns.dropna()

    annualized_volatility = (
        clean_returns.std()
        * np.sqrt(periods_per_year)
    )

    var = calculate_var(
        clean_returns,
        confidence_level=confidence_level
    )

    cvar = calculate_cvar(
        clean_returns,
        confidence_level=confidence_level
    )

    if calculate_drawdown:
        max_drawdown = (
            calculate_max_drawdown(
                clean_returns
            )
        )
    else:
        max_drawdown = np.nan

    worst_return = (
        clean_returns.min()
    )

    average_return = (
        clean_returns.mean()
    )

    return {
        "annualized_volatility":
            annualized_volatility,

        "var":
            var,

        "cvar":
            cvar,

        "max_drawdown":
            max_drawdown,

        "worst_return":
            worst_return,

        "average_return":
            average_return
    }